"""
NIXL transfer time regression from exp5b direct measurements.

Reads exp5b CSV (per-request NIXL bytes and time deltas), runs OLS
regression, and extracts protocol_ms (intercept) and eff_bw (slope).

These two constants govern NIXL KV transfer time:
    T_transfer = protocol_ms + nixl_bytes / (eff_bw * 1e6)

Can update advisor/plan.py constants from measured data.
"""

import csv
import json
import math
import os
import sys


def _linreg(xs, ys):
    """OLS linear regression. Returns (intercept, slope, r_sq)."""
    n = len(xs)
    if n < 2:
        return 0, 0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_xx = sum((x - mx) ** 2 for x in xs)
    ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    ss_yy = sum((y - my) ** 2 for y in ys)
    if ss_xx == 0:
        return my, 0, 0
    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r_sq = 1 - sse / ss_yy if ss_yy > 0 else 0
    return intercept, slope, r_sq


def _stats(vals):
    n = len(vals)
    if not n:
        return 0, 0
    mean = sum(vals) / n
    if n < 2:
        return mean, 0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var)


def load_exp5b(data_dir):
    """Load exp5b-results.csv. Returns list of valid rows as dicts."""
    csv_path = os.path.join(data_dir, "exp5b-results.csv")
    if not os.path.exists(csv_path):
        return None

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    valid = []
    for row in rows:
        if row.get("status_code") != "200":
            continue
        bytes_d = row.get("nixl_bytes_delta", "")
        time_d = row.get("nixl_xfer_time_delta_ms", "")
        transfers = row.get("nixl_transfers_delta", "")
        if not bytes_d or not time_d or not transfers:
            continue
        try:
            valid.append({
                "prompt_tokens_target": int(float(row["prompt_tokens_target"])),
                "prompt_tokens_actual": int(float(row["prompt_tokens_actual"])),
                "ttft_ms": float(row["ttft_ms"]),
                "nixl_bytes": float(bytes_d),
                "nixl_time_ms": float(time_d),
                "nixl_transfers": int(float(transfers)),
            })
        except (ValueError, KeyError):
            continue

    return valid


def analyze_nixl_transfer(data_dir, kv_bytes_per_token=None):
    """Extract NIXL transfer model constants from exp5b data.

    Returns dict with:
        protocol_ms, eff_bw_gbs, r_sq, n_points,
        by_seq_len (per-length summary), gate_checks (verification results)
    """
    rows = load_exp5b(data_dir)
    if not rows:
        return None

    # Gate 1: filter to transfers_delta == 1
    clean = [r for r in rows if r["nixl_transfers"] == 1]
    other_transfers = [r for r in rows if r["nixl_transfers"] != 1]

    gate_checks = {}
    gate_checks["total_rows"] = len(rows)
    gate_checks["single_transfer_rows"] = len(clean)
    gate_checks["multi_transfer_rows"] = len(other_transfers)

    if not clean:
        gate_checks["fatal"] = "No rows with nixl_transfers_delta == 1"
        return {"gate_checks": gate_checks}

    # Per-sequence-length summary
    by_len = {}
    for r in clean:
        L = r["prompt_tokens_target"]
        by_len.setdefault(L, []).append(r)

    summary_by_len = {}
    for L in sorted(by_len.keys()):
        rr = by_len[L]
        bytes_vals = [r["nixl_bytes"] for r in rr]
        time_vals = [r["nixl_time_ms"] for r in rr]
        ttft_vals = [r["ttft_ms"] for r in rr]
        tokens_vals = [r["prompt_tokens_actual"] for r in rr]

        bytes_mean, bytes_sd = _stats(bytes_vals)
        time_mean, time_sd = _stats(time_vals)
        ttft_mean, ttft_sd = _stats(ttft_vals)
        tokens_mean, _ = _stats(tokens_vals)

        summary_by_len[L] = {
            "n": len(rr),
            "bytes_mean": bytes_mean,
            "bytes_sd": bytes_sd,
            "time_ms_mean": time_mean,
            "time_ms_sd": time_sd,
            "ttft_ms_mean": ttft_mean,
            "ttft_ms_sd": ttft_sd,
            "tokens_mean": tokens_mean,
        }

        # Cache-miss check
        if kv_bytes_per_token and tokens_mean > 0:
            expected = kv_bytes_per_token * tokens_mean
            ratio = bytes_mean / expected if expected > 0 else 0
            summary_by_len[L]["cache_miss_ratio"] = ratio

    # Gate 2: regression on (bytes, time)
    xs = [r["nixl_bytes"] for r in clean]
    ys = [r["nixl_time_ms"] for r in clean]

    intercept, slope, r_sq = _linreg(xs, ys)

    # slope has units ms/byte. eff_bw = 1/slope in bytes/ms = 10^6 bytes/ms per GB/s
    # 1 byte/ms = 10^3 bytes/s. To get GB/s: (1/slope) * 10^3 / 10^9 = 1/(slope * 10^6)
    eff_bw_gbs = 1.0 / (slope * 1e6) if slope > 0 else 0

    gate_checks["r_squared"] = r_sq
    gate_checks["eff_bw_plausible"] = 0.01 < eff_bw_gbs < 10.0

    # Residual analysis
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    res_mean, res_sd = _stats(residuals)

    return {
        "protocol_ms": intercept,
        "eff_bw_gbs": eff_bw_gbs,
        "r_sq": r_sq,
        "n_points": len(clean),
        "slope_ms_per_byte": slope,
        "residual_sd_ms": res_sd,
        "by_seq_len": summary_by_len,
        "gate_checks": gate_checks,
    }


def print_nixl_transfer(result):
    """Print the NIXL transfer regression analysis."""
    if not result:
        print("  No exp5b data found.")
        return

    gc = result.get("gate_checks", {})
    if "fatal" in gc:
        print(f"  FATAL: {gc['fatal']}")
        return

    print(f"\n{'='*70}")
    print(f"  NIXL TRANSFER MODEL (exp5b direct measurement)")
    print(f"{'='*70}")
    print()
    print(f"  Data points:      {result['n_points']}")
    print(f"  R²:               {result['r_sq']:.4f}")
    print()
    print(f"  protocol_ms:      {result['protocol_ms']:.2f} ms  (regression intercept)")
    print(f"  eff_bw:           {result['eff_bw_gbs']:.4f} GB/s  (1/slope)")
    print(f"  slope:            {result['slope_ms_per_byte']:.4e} ms/byte")
    print(f"  residual σ:       {result['residual_sd_ms']:.2f} ms")
    print()

    # Per-sequence-length table
    by_len = result.get("by_seq_len", {})
    if by_len:
        print(f"  {'L':>6} | {'n':>3} | {'bytes (MB)':>12} | {'time (ms)':>12} | "
              f"{'TTFT (ms)':>12} | {'tokens':>7} | {'cache_miss':>10}")
        print(f"  {'-'*6}-+-{'-'*3}-+-{'-'*12}-+-{'-'*12}-+-"
              f"{'-'*12}-+-{'-'*7}-+-{'-'*10}")
        for L in sorted(by_len.keys()):
            s = by_len[L]
            cm = s.get("cache_miss_ratio", "")
            cm_str = f"{cm:.2f}" if cm else "--"
            print(f"  {L:>6} | {s['n']:>3} | {s['bytes_mean']/1e6:>9.2f} MB | "
                  f"{s['time_ms_mean']:>8.2f} ms | "
                  f"{s['ttft_ms_mean']:>8.1f} ms | {s['tokens_mean']:>7.0f} | "
                  f"{cm_str:>10}")

    print()

    # Gate checks
    r2 = result["r_sq"]
    bw = result["eff_bw_gbs"]
    if r2 > 0.9:
        print(f"  R² = {r2:.3f} > 0.9: linear model fits well")
    else:
        print(f"  WARNING: R² = {r2:.3f} < 0.9: poor linear fit — check for nonlinearity")

    if 0.1 <= bw <= 1.25:
        print(f"  eff_bw = {bw:.3f} GB/s: plausible for T4 10 Gbps NIC")
    elif bw > 1.25:
        print(f"  WARNING: eff_bw = {bw:.3f} GB/s > 1.25: exceeds NIC speed — cache contamination?")
    elif bw < 0.1:
        print(f"  WARNING: eff_bw = {bw:.3f} GB/s < 0.1: unexpectedly slow transfer")

    proto = result["protocol_ms"]
    if proto > 0:
        print(f"  protocol_ms = {proto:.1f} ms > 0: overhead outside NIXL metric")
    elif proto < -5:
        print(f"  WARNING: protocol_ms = {proto:.1f} ms < 0: model misspecification")

    gc = result.get("gate_checks", {})
    if gc.get("multi_transfer_rows", 0) > 0:
        print(f"  NOTE: {gc['multi_transfer_rows']} rows had transfers_delta != 1 (excluded)")

    print(f"\n{'='*70}")

    # Advisor update recommendation
    print()
    print(f"  ADVISOR UPDATE:")
    print(f"    NIXL_PROTOCOL_MS = {proto:.1f}")
    print(f"    NIXL_EFF_BW_GBS  = {bw:.3f}")
    print(f"    (in advisor/plan.py, lines 149-150)")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract NIXL transfer model from exp5b data")
    parser.add_argument("data_dir",
                        help="Directory containing exp5b-results.csv")
    parser.add_argument("--kv-bytes", type=int, default=None,
                        help="KV bytes per token for cache-miss check")
    args = parser.parse_args()

    result = analyze_nixl_transfer(args.data_dir, args.kv_bytes)
    print_nixl_transfer(result)
