"""
Extract advisor baselines from experiment data.

Reads exp11 and exp14 CSVs, computes the metrics the advisor needs
(TTFT, throughput, overhead, rates), and saves them as baselines.json.
This replaces the need to manually add entries to MEASURED_BASELINES.

Usage:
    python3 advisor/calibrate.py <data_dir> [--gpu-type t4]

Example:
    python3 advisor/calibrate.py clusters/my-cluster/data --gpu-type t4
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

_advisor_dir = os.path.dirname(__file__)
_toolkit_dir = os.path.join(os.path.dirname(_advisor_dir), "toolkit")
sys.path.insert(0, _advisor_dir)
sys.path.insert(0, _toolkit_dir)

from analyze import get_status, load_csv, safe_float, safe_int, stats  # noqa: E402

from plan import _linreg, fetch_model_profile  # noqa: E402


def _load_run_info(data_dir: str) -> dict:
    path = os.path.join(data_dir, "run-info.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def extract_baselines(data_dir: str, gpu_type: str | None = None) -> dict | None:
    """Extract baseline metrics from experiment CSVs.

    Returns a dict matching the MEASURED_BASELINES entry format,
    or None if insufficient data.
    """
    info = _load_run_info(data_dir)
    toolkit = info.get("toolkit", {})
    model = toolkit.get("model")
    if not model:
        print("ERROR: run-info.json missing or has no toolkit.model")
        return None

    gpu = gpu_type or toolkit.get("gpu_type", "t4")

    # Load exp11 data: TTFT by (config, concurrency, seq_len)
    exp11_path = os.path.join(data_dir, "exp11-results.csv")
    rows = load_csv(exp11_path)
    if not rows:
        print(f"No exp11 data found in {data_dir}")
        return None

    groups: dict = defaultdict(list)
    for r in rows:
        if get_status(r) != 200:
            continue
        key = (r["config"], safe_int(r["concurrency"]),
               safe_int(r["prompt_tokens_target"]))
        groups[key].append(safe_float(r["ttft_ms"]))

    grouped = {k: stats(v) for k, v in groups.items()}

    # Extract mono/disagg TTFT at c=1 per seq_len
    seq_lens = sorted(set(k[2] for k in grouped if k[1] == 1))
    if not seq_lens:
        print("No c=1 data found in exp11")
        return None

    mono_by_s: dict = {}
    disagg_by_s: dict = {}
    for sl in seq_lens:
        bl = grouped.get(("BASELINE", 1, sl))
        if bl and bl["n"] >= 3:
            mono_by_s[sl] = bl["median"]
        for cfg in ["DISAGG-1D", "DISAGG-2D"]:
            dg = grouped.get((cfg, 1, sl))
            if dg and dg["n"] >= 3 and sl not in disagg_by_s:
                disagg_by_s[sl] = dg["median"]

    if not mono_by_s:
        print("No BASELINE c=1 data with sufficient samples (n>=3)")
        return None

    ref_seq_len = min(mono_by_s.keys())
    mono_ttft = round(mono_by_s[ref_seq_len])
    disagg_ttft = round(disagg_by_s.get(ref_seq_len, mono_ttft))

    # Fit linear TTFT(s) = base + rate*s if enough seq_lens
    baselines: dict = {}
    if len(mono_by_s) >= 2:
        mono_pts = [(float(s), t) for s, t in sorted(mono_by_s.items())]
        mono_base, mono_rate = _linreg(mono_pts)
        mono_r2 = _r_squared(mono_pts, mono_base, mono_rate)

        if mono_r2 >= 0.8 and mono_rate > 0:
            baselines["mono_ttft_base_ms"] = round(max(mono_base, 0), 1)
            baselines["mono_ttft_rate"] = round(mono_rate, 3)
            baselines["ref_seq_len"] = ref_seq_len

    if len(disagg_by_s) >= 2:
        disagg_pts = [(float(s), t) for s, t in sorted(disagg_by_s.items())]
        disagg_base, disagg_rate = _linreg(disagg_pts)
        disagg_r2 = _r_squared(disagg_pts, disagg_base, disagg_rate)

        if disagg_r2 >= 0.8 and disagg_rate > 0:
            baselines["disagg_ttft_base_ms"] = round(max(disagg_base, 0), 1)
            baselines["disagg_ttft_rate"] = round(disagg_rate, 3)
            if "ref_seq_len" not in baselines:
                baselines["ref_seq_len"] = ref_seq_len

    # Throughput: use c=1 total_ms to estimate req/s
    mono_tput = _estimate_throughput(rows, "BASELINE", ref_seq_len)
    disagg_tput = _estimate_throughput(rows, None, ref_seq_len)

    # NIXL overhead from exp14 (D-config minus C-config)
    nixl_ms = _estimate_nixl_from_exp14(data_dir, seq_lens)

    # Model properties from HuggingFace
    try:
        profile = fetch_model_profile(model)
        params_b = round(profile.params_b, 1)
        is_moe = profile.is_moe
    except Exception:
        params_b = 0.0
        is_moe = False

    overhead_ms = round(disagg_ttft - mono_ttft)
    overhead_pct = round((disagg_ttft - mono_ttft) / max(mono_ttft, 1) * 100, 1)

    baselines.update({
        "params_b": params_b,
        "is_moe": is_moe,
        "mono_ttft_ms": mono_ttft,
        "disagg_ttft_ms": disagg_ttft,
        "mono_throughput": mono_tput,
        "disagg_throughput": disagg_tput,
        "overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
    })
    if nixl_ms is not None:
        baselines["nixl_ms"] = nixl_ms

    n_seq = len(mono_by_s)
    rate_info = ""
    if "mono_ttft_rate" in baselines:
        mono_r2_val = _r_squared(
            [(float(s), t) for s, t in sorted(mono_by_s.items())],
            baselines.get("mono_ttft_base_ms", 0),
            baselines["mono_ttft_rate"])
        rate_info = f", rate fitted from {n_seq} seq_lens (R²={mono_r2_val:.2f})"

    print(f"Calibrated {model} on {gpu.upper()}: "
          f"mono={mono_ttft}ms, disagg={disagg_ttft}ms, "
          f"overhead={overhead_ms:+d}ms ({overhead_pct:+.1f}%){rate_info}")

    return {"model": model, "gpu_type": gpu, "baselines": baselines}


def _r_squared(points: list, intercept: float, slope: float) -> float:
    if len(points) < 2:
        return 0.0
    y_mean = sum(y for _, y in points) / len(points)
    ss_tot = sum((y - y_mean) ** 2 for _, y in points)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _estimate_throughput(rows: list, config: str | None,
                         ref_seq_len: int) -> float:
    """Estimate requests/second from c=1, total_ms at reference seq_len."""
    total_ms_vals = []
    for r in rows:
        if get_status(r) != 200:
            continue
        if safe_int(r["concurrency"]) != 1:
            continue
        if safe_int(r["prompt_tokens_target"]) != ref_seq_len:
            continue
        if config and r["config"] != config:
            continue
        if not config and not r["config"].startswith("DISAGG"):
            continue
        t = safe_float(r["total_ms"])
        if t > 0:
            total_ms_vals.append(t)
    if not total_ms_vals:
        return 0.0
    median_ms = sorted(total_ms_vals)[len(total_ms_vals) // 2]
    return round(1000.0 / median_ms, 2) if median_ms > 0 else 0.0


def _estimate_nixl_from_exp14(data_dir: str, seq_lens: list) -> int | None:
    """Extract NIXL transfer overhead from exp14 data."""
    exp14_path = os.path.join(data_dir, "exp14-results.csv")
    rows = load_csv(exp14_path)
    if not rows:
        return None

    groups: dict = defaultdict(list)
    for r in rows:
        if get_status(r) != 200:
            continue
        key = (r["config"], safe_int(r["prompt_tokens_target"]))
        groups[key].append(safe_float(r["ttft_ms"]))

    grouped = {k: stats(v) for k, v in groups.items()}

    nixl_values = []
    for sl in seq_lens:
        cfg_c = next((k for k in grouped if k[0].startswith("C-") and k[1] == sl), None)
        cfg_d = next((k for k in grouped if k[0].startswith("D-") and k[1] == sl), None)
        if cfg_c and cfg_d:
            diff = grouped[cfg_d]["median"] - grouped[cfg_c]["median"]
            nixl_values.append(diff)

    if not nixl_values:
        return None
    return round(sorted(nixl_values)[len(nixl_values) // 2])


def save_baselines(data_dir: str, result: dict) -> str:
    """Save calibration result to baselines.json."""
    output = {**result, "calibrated_at": datetime.now(timezone.utc).isoformat()}  # noqa: UP017
    path = os.path.join(data_dir, "baselines.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"Saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Extract advisor baselines from experiment data")
    parser.add_argument("data_dir", help="Path to data directory with exp results")
    parser.add_argument("--gpu-type", default=None,
                        help="GPU type (auto-detected from run-info.json, fallback: t4)")
    args = parser.parse_args()

    result = extract_baselines(args.data_dir, args.gpu_type)
    if not result:
        sys.exit(1)

    save_baselines(args.data_dir, result)


if __name__ == "__main__":
    main()
