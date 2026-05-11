"""
KV head ratio sweep analysis for disaggregated inference.

Loads exp5 sequence-length sweep data across multiple models, fits
regression per model to extract NIXL transfer constants (protocol overhead
and effective bandwidth), and prints a comparison table showing how
KV head count affects disaggregation overhead.

Thesis: models with fewer KV heads transfer less data per token, so
disagg overhead grows slower with sequence length. At short prompts,
fixed protocol overhead dominates and all models look the same.

Part of the llm-d-diagnostics advisory layer.
"""

import argparse
import csv
import json
import math
import os
import sys


# Model architecture for KV bytes calculation
MODEL_ARCH = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "short": "TinyLlama 1.1B", "params_b": 1.1,
        "n_layers": 22, "n_kv_heads": 4, "d_head": 64,
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "short": "Qwen2.5 0.5B", "params_b": 0.5,
        "n_layers": 24, "n_kv_heads": 2, "d_head": 64,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "short": "Qwen2.5 1.5B", "params_b": 1.5,
        "n_layers": 28, "n_kv_heads": 2, "d_head": 64,
    },
    "stabilityai/stablelm-2-1_6b-chat": {
        "short": "StableLM 1.6B", "params_b": 1.6,
        "n_layers": 24, "n_kv_heads": 32, "d_head": 64,
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "short": "SmolLM2 1.7B", "params_b": 1.7,
        "n_layers": 24, "n_kv_heads": 32, "d_head": 64,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "short": "Qwen2.5 3B", "params_b": 3.0,
        "n_layers": 36, "n_kv_heads": 2, "d_head": 128,
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "short": "Phi-3 3.8B", "params_b": 3.8,
        "n_layers": 32, "n_kv_heads": 32, "d_head": 96,
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "short": "Phi-3 3.8B", "params_b": 3.8,
        "n_layers": 32, "n_kv_heads": 32, "d_head": 96,
    },
    "allenai/OLMoE-1B-7B-0924-Instruct": {
        "short": "OLMoE 7B", "params_b": 7.0,
        "n_layers": 16, "n_kv_heads": 16, "d_head": 64,
    },
}


def _kv_bytes(arch):
    return 2 * arch["n_layers"] * arch["n_kv_heads"] * arch["d_head"] * 2


def _load_exp5(data_dir, model_override=None):
    """Load exp5-results.csv and compute paired T_transfer = D - C per run."""
    csv_path = os.path.join(data_dir, "exp5-results.csv")
    if not os.path.exists(csv_path):
        return None, None

    model_name = model_override or "unknown"
    if not model_override:
        run_info_path = os.path.join(data_dir, "run-info.json")
        if os.path.exists(run_info_path):
            try:
                with open(run_info_path) as f:
                    info = json.load(f)
                model_name = info.get("toolkit", {}).get("model", "unknown")
            except (ValueError, KeyError):
                pass

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    by_run = {}
    for row in rows:
        if row.get("status_code", "0") != "200":
            continue
        config = row["config"]
        seq_len = int(float(row.get("prompt_tokens_target", "0")))
        run = int(float(row.get("run", "0")))
        ttft = float(row.get("ttft_ms", "0"))
        if seq_len > 0 and ttft > 0:
            by_run.setdefault((seq_len, run), {})[config] = ttft

    transfer_by_len = {}
    for (seq_len, run), configs in sorted(by_run.items()):
        c = configs.get("C-sidecar-only")
        d = configs.get("D-disaggregated")
        if c is not None and d is not None:
            transfer_by_len.setdefault(seq_len, []).append(d - c)

    return transfer_by_len, model_name


def _linreg(xs, ys):
    """Linear regression with R-squared. Returns (intercept, slope, r_sq)."""
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


def analyze(data_dirs, models=None):
    """Analyze exp5 data from multiple directories.

    models: optional list of model IDs parallel to data_dirs.
    """
    results = []

    for i, data_dir in enumerate(data_dirs):
        model_override = models[i] if models and i < len(models) else None
        transfer_by_len, model_name = _load_exp5(data_dir, model_override)
        if transfer_by_len is None:
            print(f"  SKIP: no exp5-results.csv in {data_dir}")
            continue

        arch = MODEL_ARCH.get(model_name)
        if not arch:
            print(f"  SKIP: unknown model {model_name}")
            continue

        kv = _kv_bytes(arch)

        xs, ys = [], []
        by_len = {}
        for seq_len in sorted(transfer_by_len.keys()):
            vals = transfer_by_len[seq_len]
            mean, sd = _stats(vals)
            by_len[seq_len] = (mean, sd, len(vals))
            for v in vals:
                xs.append(seq_len)
                ys.append(v)

        intercept, slope, r_sq = _linreg(xs, ys)

        eff_bw = (kv / (slope / 1000)) / 1e9 if slope > 0 else 0

        results.append({
            "model": model_name,
            "short": arch["short"],
            "params_b": arch["params_b"],
            "kv_heads": arch["n_kv_heads"],
            "kv_bytes": kv,
            "protocol_ms": intercept,
            "slope_ms_per_tok": slope,
            "eff_bw_gbs": eff_bw,
            "r_sq": r_sq,
            "by_len": by_len,
            "n_points": len(xs),
        })

    return results


def print_summary(results):
    """Print the presentation-ready comparison table."""
    if not results:
        print("  No data to analyze.")
        return

    results.sort(key=lambda r: r["kv_bytes"])

    print(f"\n{'='*80}")
    print(f"  KV HEAD RATIO SWEEP: NIXL Transfer Physics")
    print(f"{'='*80}")
    print()

    print(f"  {'Model':>18} | {'Params':>6} | {'KV':>4} | {'KV/tok':>8} | "
          f"{'Protocol':>10} | {'Slope':>12} | {'Eff BW':>8} | {'R²':>5}")
    print(f"  {'':>18} | {'':>6} | {'heads':>4} | {'(KB)':>8} | "
          f"{'(ms)':>10} | {'(ms/tok)':>12} | {'(GB/s)':>8} | {'':>5}")
    print(f"  {'-'*18}-+-{'-'*6}-+-{'-'*4}-+-{'-'*8}-+-"
          f"{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*5}")

    for r in results:
        print(f"  {r['short']:>18} | {r['params_b']:>5.1f}B | {r['kv_heads']:>4} | "
              f"{r['kv_bytes']//1024:>6} KB | {r['protocol_ms']:>8.1f}ms | "
              f"{r['slope_ms_per_tok']:>9.4f}ms/t | {r['eff_bw_gbs']:>6.2f} | "
              f"{r['r_sq']:>5.3f}")

    print()

    protocols = [r["protocol_ms"] for r in results]
    bws = [r["eff_bw_gbs"] for r in results if r["eff_bw_gbs"] > 0]
    p_mean, p_sd = _stats(protocols)
    b_mean, b_sd = _stats(bws)

    print(f"  Protocol overhead: {p_mean:.1f}ms +/- {p_sd:.1f}ms "
          f"(CV={p_sd/p_mean*100:.0f}%)" if p_mean > 0 else "")
    print(f"  Effective NIC BW:  {b_mean:.2f} GB/s +/- {b_sd:.2f} GB/s "
          f"(CV={b_sd/b_mean*100:.0f}%)" if b_mean > 0 else "")

    if p_mean > 0 and p_sd / p_mean < 0.3:
        print(f"  -> Protocol overhead is CONSISTENT across models (as predicted)")
    else:
        print(f"  -> Protocol overhead VARIES by model (unexpected -- investigate)")

    if b_mean > 0 and b_sd / b_mean < 0.3:
        print(f"  -> Effective bandwidth is CONSISTENT (linear model valid)")
    else:
        print(f"  -> Effective bandwidth VARIES (may indicate non-linear transfer)")

    print()

    seq_lens = sorted(set(L for r in results for L in r["by_len"]))
    if seq_lens:
        print(f"  T_transfer by sequence length (mean ms):")
        print()
        header = f"  {'Model':>18} |"
        for L in seq_lens:
            header += f" {'L='+str(L):>8} |"
        print(header)
        print(f"  {'-'*18}-+" + ("-" * 8 + "-+-") * (len(seq_lens) - 1) + "-" * 8 + "-+")

        for r in results:
            row = f"  {r['short']:>18} |"
            for L in seq_lens:
                if L in r["by_len"]:
                    mean, sd, n = r["by_len"][L]
                    row += f" {mean:>6.0f}ms |"
                else:
                    row += f" {'--':>7} |"
            print(row)

    print(f"\n{'='*80}")

    print()
    print("  THESIS CHECK:")
    if len(results) >= 2:
        lo_kv = results[0]
        hi_kv = results[-1]
        short_lens = [L for L in seq_lens if L <= 50]
        long_lens = [L for L in seq_lens if L >= 1000]

        if short_lens:
            lo_short = [lo_kv["by_len"].get(L, (0, 0, 0))[0] for L in short_lens if L in lo_kv["by_len"]]
            hi_short = [hi_kv["by_len"].get(L, (0, 0, 0))[0] for L in short_lens if L in hi_kv["by_len"]]
            if lo_short and hi_short:
                diff = abs(sum(hi_short)/len(hi_short) - sum(lo_short)/len(lo_short))
                print(f"  Short prompt (L<=50): {lo_kv['short']} vs {hi_kv['short']} "
                      f"differ by {diff:.0f}ms {'(< noise, as expected)' if diff < 10 else '(unexpected gap)'}")

        if long_lens:
            lo_long = [lo_kv["by_len"].get(L, (0, 0, 0))[0] for L in long_lens if L in lo_kv["by_len"]]
            hi_long = [hi_kv["by_len"].get(L, (0, 0, 0))[0] for L in long_lens if L in hi_kv["by_len"]]
            if lo_long and hi_long:
                diff = sum(hi_long)/len(hi_long) - sum(lo_long)/len(lo_long)
                ratio = (sum(hi_long)/len(hi_long)) / (sum(lo_long)/len(lo_long)) if sum(lo_long) > 0 else 0
                print(f"  Long prompt (L>=1000): {hi_kv['short']} is {diff:.0f}ms slower "
                      f"({ratio:.1f}x) than {lo_kv['short']} "
                      f"{'-> GQA effect CONFIRMED' if diff > 30 else '-> weak signal'}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KV head ratio sweep analysis for disaggregated inference")
    parser.add_argument("data_dirs", nargs="+",
                        help="Paths to exp5 data directories")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model IDs for each data dir (when run-info.json missing)")
    args = parser.parse_args()

    results = analyze(args.data_dirs, args.models)
    print_summary(results)
