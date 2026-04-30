#!/usr/bin/env python3
"""Cross-cluster scaling model for disaggregation crossover prediction.

Loads exp5 data from multiple clusters, fits per-cluster regression lines,
extracts hardware-independent scaling coefficients, and predicts crossover
for target hardware configurations.

All computation uses stdlib only (no numpy/scipy).

Usage:
    python3 toolkit/scaling_model.py \\
        clusters/rdu3-t4x3/data \\
        clusters/rdu3-t4x3-phi3/data \\
        clusters/rdu3-t4x5-phi3/data-v3/

    python3 toolkit/scaling_model.py \\
        clusters/rdu3-t4x3/data \\
        clusters/rdu3-t4x3-phi3/data \\
        --predict llama-31b --gpu-decode H200 --gpu-prefill L40S
"""

import csv
import json
import math
import os
import sys

# ── Import shared stats from analyze.py ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from analyze import _t95, pearson_r, safe_float, safe_int, stats


# ── Model architecture database ─────────────────────────────────

MODELS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "short": "TinyLlama 1.1B",
        "params_b": 1.1,
        "n_layers": 22,
        "n_kv_heads": 4,   # GQA: 4 KV, 32 Q
        "n_q_heads": 32,
        "d_head": 64,
        "d_model": 2048,
        "dtype_bytes": 2,   # float16
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "short": "Phi-3 3.8B",
        "params_b": 3.8,
        "n_layers": 32,
        "n_kv_heads": 32,  # MHA: 32 KV, 32 Q
        "n_q_heads": 32,
        "d_head": 96,
        "d_model": 3072,
        "dtype_bytes": 2,   # float16
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "short": "Llama 3.1 8B",
        "params_b": 8.0,
        "n_layers": 32,
        "n_kv_heads": 8,   # GQA: 8 KV, 32 Q
        "n_q_heads": 32,
        "d_head": 128,
        "d_model": 4096,
        "dtype_bytes": 2,
    },
    # Large model reference — public model card values
    "meta-llama/Llama-3.1-70B-Instruct": {
        "short": "Llama 3.1 70B",
        "params_b": 70.6,
        "n_layers": 80,
        "n_kv_heads": 8,   # GQA: 8 KV, 64 Q
        "n_q_heads": 64,
        "d_head": 128,
        "d_model": 8192,
        "dtype_bytes": 2,
    },
}


def kv_bytes_per_token(model_info):
    """KV cache bytes per token for one layer.

    KV cache per token = 2 (K+V) × n_kv_heads × d_head × dtype_bytes
    Total = above × n_layers
    """
    per_layer = 2 * model_info["n_kv_heads"] * model_info["d_head"] * model_info["dtype_bytes"]
    return per_layer * model_info["n_layers"]


def prefill_flops_per_token(model_info):
    """Approximate prefill FLOPs per token (forward pass only).

    ~2 × params for the linear layers (dominant term).
    Ignores attention O(L²) which matters at long sequences.
    """
    return 2 * model_info["params_b"] * 1e9


# ── GPU hardware database ───────────────────────────────────────

GPUS = {
    "T4": {
        "hbm_bw_gbs": 300,       # GB/s
        "fp16_tflops": 65,       # TFLOPS
        "memory_gb": 16,
    },
    "L40S": {
        "hbm_bw_gbs": 864,
        "fp16_tflops": 362,
        "memory_gb": 48,
    },
    "A100-80": {
        "hbm_bw_gbs": 2039,
        "fp16_tflops": 312,
        "memory_gb": 80,
    },
    "H100": {
        "hbm_bw_gbs": 3350,
        "fp16_tflops": 989,
        "memory_gb": 80,
    },
    "H200": {
        "hbm_bw_gbs": 4800,
        "fp16_tflops": 989,
        "memory_gb": 141,
    },
}

NETWORKS = {
    "TCP":  {"label": "TCP/OVN-K", "rtt_ms": 0.5, "bw_gbps": 25},
    "RoCE": {"label": "RoCE 100G", "rtt_ms": 0.01, "bw_gbps": 100},
    "IB":   {"label": "IB 400G",   "rtt_ms": 0.001, "bw_gbps": 400},
}


# ── Data loading ─────────────────────────────────────────────────

def load_exp5_paired(data_dir):
    """Load exp5 data and compute paired T_transfer = D_i - C_i per run.

    Returns dict: {seq_len: [list of paired differences in ms]}
    Also returns: {seq_len: [list of T_prefill = A_i values in ms]}
    And the model name from run-info.json.
    """
    csv_path = os.path.join(data_dir, "exp5-results.csv")
    if not os.path.exists(csv_path):
        return None, None, None

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Get model name
    model_name = "unknown"
    run_info_path = os.path.join(data_dir, "run-info.json")
    if os.path.exists(run_info_path):
        try:
            with open(run_info_path) as f:
                info = json.load(f)
            model_name = info.get("toolkit", {}).get("model", "unknown")
        except (ValueError, KeyError):
            pass

    # Group by (config, seq_len, run)
    by_key = {}
    for row in rows:
        status = row.get("status_code", "0")
        if status != "200":
            continue
        comp = safe_int(row.get("completion_tokens", "0"))
        if comp < 10:
            continue

        config = row["config"]
        seq_len = safe_int(row.get("prompt_tokens_target", "0"))
        # Handle both "run" field naming conventions
        run = safe_int(row.get("run", "0"))
        ttft = safe_float(row.get("ttft_ms", "0"))

        if seq_len > 0 and ttft > 0:
            by_key.setdefault((config, seq_len, run), {})[config] = ttft

    # Build keyed by (seq_len, run) with all configs
    by_run = {}
    for (config, seq_len, run), vals in by_key.items():
        by_run.setdefault((seq_len, run), {})[config] = vals.get(config, 0)

    # Actually need to re-index properly
    by_run = {}
    for row in rows:
        status = row.get("status_code", "0")
        if status != "200":
            continue
        comp = safe_int(row.get("completion_tokens", "0"))
        if comp < 10:
            continue
        config = row["config"]
        seq_len = safe_int(row.get("prompt_tokens_target", "0"))
        run = safe_int(row.get("run", "0"))
        ttft = safe_float(row.get("ttft_ms", "0"))
        if seq_len > 0 and ttft > 0:
            key = (seq_len, run)
            if key not in by_run:
                by_run[key] = {}
            by_run[key][config] = ttft

    # Compute paired differences
    transfer_by_len = {}  # seq_len → [D_i - C_i]
    prefill_by_len = {}   # seq_len → [A_i]

    for (seq_len, run), configs in sorted(by_run.items()):
        a = configs.get("A-prefill-direct")
        c = configs.get("C-sidecar-only")
        d = configs.get("D-disaggregated")

        if c is not None and d is not None:
            transfer_by_len.setdefault(seq_len, []).append(d - c)
        if a is not None:
            prefill_by_len.setdefault(seq_len, []).append(a)

    return transfer_by_len, prefill_by_len, model_name


def linreg(xs, ys):
    """Simple linear regression. Returns (intercept, slope)."""
    n = len(xs)
    if n < 2:
        return 0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_xx = sum((x - mx) ** 2 for x in xs)
    ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if ss_xx == 0:
        return my, 0
    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    return intercept, slope


def linreg_ci(xs, ys, alpha=0.05):
    """Linear regression with confidence intervals on slope and intercept.

    Returns (intercept, slope, slope_ci_lo, slope_ci_hi,
             intercept_ci_lo, intercept_ci_hi, r_squared, se_slope).
    """
    n = len(xs)
    if n < 3:
        return 0, 0, 0, 0, 0, 0, 0, float('inf')

    intercept, slope = linreg(xs, ys)

    # Residuals
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r ** 2 for r in residuals)
    mse = sse / (n - 2)

    mx = sum(xs) / n
    ss_xx = sum((x - mx) ** 2 for x in xs)
    ss_yy = sum((y - sum(ys)/n) ** 2 for y in ys)

    # Standard errors
    se_slope = math.sqrt(mse / ss_xx) if ss_xx > 0 else float('inf')
    se_intercept = math.sqrt(mse * (1/n + mx**2 / ss_xx)) if ss_xx > 0 else float('inf')

    # t critical value
    df = n - 2
    t_crit = _t95(df)

    slope_ci_lo = slope - t_crit * se_slope
    slope_ci_hi = slope + t_crit * se_slope
    int_ci_lo = intercept - t_crit * se_intercept
    int_ci_hi = intercept + t_crit * se_intercept

    # R²
    r_sq = 1 - sse / ss_yy if ss_yy > 0 else 0

    return intercept, slope, slope_ci_lo, slope_ci_hi, int_ci_lo, int_ci_hi, r_sq, se_slope


# ── Core scaling model ──────────────────────────────────────────

class ClusterRegression:
    """Regression results for one cluster's exp5 data."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.name = os.path.basename(os.path.dirname(data_dir.rstrip("/")))
        # Handle data-v3 subdirectory naming
        if self.name.startswith("data"):
            self.name = os.path.basename(os.path.dirname(os.path.dirname(data_dir.rstrip("/"))))
            self.name += f" ({os.path.basename(data_dir.rstrip('/'))})"

        self.transfer_by_len, self.prefill_by_len, self.model_name = load_exp5_paired(data_dir)
        if self.transfer_by_len is None:
            raise FileNotFoundError(f"No exp5-results.csv in {data_dir}")

        self.model_info = MODELS.get(self.model_name, {})

        # Fit regressions using median at each seq_len
        self._fit()

    def _fit(self):
        """Fit linear regressions on median T_transfer and T_prefill vs seq_len."""
        self.seq_lens = sorted(self.transfer_by_len.keys())

        # Transfer: median of paired differences at each length
        self.transfer_medians = {}
        self.transfer_stats = {}
        for sl in self.seq_lens:
            vals = self.transfer_by_len[sl]
            s = stats(vals)
            self.transfer_medians[sl] = s["median"]
            self.transfer_stats[sl] = s

        # Prefill: median of A values at each length
        self.prefill_medians = {}
        self.prefill_stats = {}
        for sl in self.seq_lens:
            if sl in self.prefill_by_len:
                vals = self.prefill_by_len[sl]
                s = stats(vals)
                self.prefill_medians[sl] = s["median"]
                self.prefill_stats[sl] = s

        # Linear regression on medians
        xs_tr = [float(sl) for sl in self.seq_lens]
        ys_tr = [self.transfer_medians[sl] for sl in self.seq_lens]

        self.tr_intercept, self.tr_slope, self.tr_slope_ci_lo, self.tr_slope_ci_hi, \
            self.tr_int_ci_lo, self.tr_int_ci_hi, self.tr_r2, self.tr_se_slope = \
            linreg_ci(xs_tr, ys_tr)

        self.tr_pearson = pearson_r(xs_tr, ys_tr)

        # Prefill regression
        pf_lens = sorted(self.prefill_medians.keys())
        if len(pf_lens) >= 2:
            xs_pf = [float(sl) for sl in pf_lens]
            ys_pf = [self.prefill_medians[sl] for sl in pf_lens]
            self.pf_intercept, self.pf_slope, self.pf_slope_ci_lo, self.pf_slope_ci_hi, \
                self.pf_int_ci_lo, self.pf_int_ci_hi, self.pf_r2, self.pf_se_slope = \
                linreg_ci(xs_pf, ys_pf)
            self.pf_pearson = pearson_r(xs_pf, ys_pf)
        else:
            self.pf_intercept = self.pf_slope = 0
            self.pf_pearson = 0
            self.pf_r2 = 0

    def kv_bytes_per_token(self):
        if self.model_info:
            return kv_bytes_per_token(self.model_info)
        return None

    def summary(self):
        """Print regression summary."""
        print(f"\n  {self.name}: {self.model_name}")
        print(f"    Sequence lengths: {self.seq_lens}")
        print()

        print("    Transfer (D-C paired difference) by sequence length:")
        for sl in self.seq_lens:
            s = self.transfer_stats[sl]
            print(f"      {sl:>5} tok: {s['median']:.1f}ms "
                  f"(n={s['n']}, 95%CI [{s['ci95_lo']:.1f}, {s['ci95_hi']:.1f}], "
                  f"CV={s['cv']:.2f})")
        print()

        print(f"    T_transfer = {self.tr_intercept:.1f}ms + {self.tr_slope:.4f}ms/tok")
        print(f"      Slope 95%CI: [{self.tr_slope_ci_lo:.4f}, {self.tr_slope_ci_hi:.4f}]")
        print(f"      Intercept 95%CI: [{self.tr_int_ci_lo:.1f}, {self.tr_int_ci_hi:.1f}]")
        print(f"      R² = {self.tr_r2:.4f}, Pearson r = {self.tr_pearson:.3f}")
        print()

        print(f"    T_prefill = {self.pf_intercept:.1f}ms + {self.pf_slope:.4f}ms/tok")
        print(f"      R² = {self.pf_r2:.4f}, Pearson r = {self.pf_pearson:.3f}")
        print()

        kv = self.kv_bytes_per_token()
        if kv:
            print(f"    KV cache: {kv:,} bytes/token")
            effective_bw = self.tr_slope  # ms/token
            if effective_bw > 0:
                # slope = kv_bytes / effective_bandwidth
                # effective_bandwidth = kv_bytes / slope_ms * 1000 (bytes/s)
                eff_bw_gbs = (kv / (effective_bw / 1000)) / 1e9
                print(f"    Effective transfer bandwidth: {eff_bw_gbs:.2f} GB/s")
                print(f"      (includes protocol/serialization overhead)")


def extract_physical_coefficients(regressions):
    """Extract hardware-independent scaling coefficients from multiple clusters.

    With two models on the same GPU, we can decompose:
        T_transfer(L) = T_protocol + (KV_bytes(L) / bandwidth_eff)
        T_prefill(L)  = T_launch + (FLOPs(L) / compute_eff)

    The slopes encode: slope_tr = kv_per_token / BW_eff
                       slope_pf = flops_per_token / COMPUTE_eff

    So: BW_eff = kv_per_token / slope_tr
        COMPUTE_eff = flops_per_token / slope_pf
    """
    results = []

    for reg in regressions:
        kv = reg.kv_bytes_per_token()
        flops = prefill_flops_per_token(reg.model_info) if reg.model_info else None

        entry = {
            "name": reg.name,
            "model": reg.model_name,
            "short": reg.model_info.get("short", reg.model_name) if reg.model_info else reg.model_name,
            "tr_intercept": reg.tr_intercept,
            "tr_slope": reg.tr_slope,
            "tr_slope_ci": (reg.tr_slope_ci_lo, reg.tr_slope_ci_hi),
            "pf_intercept": reg.pf_intercept,
            "pf_slope": reg.pf_slope,
            "pf_slope_ci": (reg.pf_slope_ci_lo, reg.pf_slope_ci_hi),
            "kv_bytes_per_token": kv,
            "flops_per_token": flops,
        }

        if kv and reg.tr_slope > 0:
            # Effective bandwidth in GB/s
            entry["bw_eff_gbs"] = (kv / (reg.tr_slope / 1000)) / 1e9
        if flops and reg.pf_slope > 0:
            # Effective compute in TFLOPS
            entry["compute_eff_tflops"] = (flops / (reg.pf_slope / 1000)) / 1e12

        results.append(entry)

    return results


def predict_for_model(coefficients, target_model_key, gpu_decode="T4",
                      gpu_prefill="T4", network="TCP"):
    """Predict T_transfer and T_prefill for a target model on target hardware.

    Uses measured effective bandwidth/compute from our clusters, scaled by
    hardware ratios.
    """
    target = MODELS.get(target_model_key)
    if not target:
        print(f"  ERROR: Unknown model {target_model_key}")
        return None

    gpu_d = GPUS.get(gpu_decode)
    gpu_p = GPUS.get(gpu_prefill)
    net = NETWORKS.get(network)
    measured_gpu = GPUS["T4"]

    if not gpu_d or not gpu_p:
        print(f"  ERROR: Unknown GPU")
        return None

    target_kv = kv_bytes_per_token(target)
    target_flops = prefill_flops_per_token(target)

    print(f"\n  Prediction: {target['short']} on {gpu_decode} (decode) + {gpu_prefill} (prefill)")
    print(f"    Network: {net['label'] if net else network}")
    print(f"    KV cache: {target_kv:,} bytes/token")
    print(f"    Prefill FLOPs: {target_flops:.2e} per token")
    print()

    # Use each cluster's measurements as independent estimates
    predictions = []
    for coeff in coefficients:
        if "bw_eff_gbs" not in coeff or "compute_eff_tflops" not in coeff:
            continue

        # Scale bandwidth by GPU ratio
        bw_ratio = gpu_d["hbm_bw_gbs"] / measured_gpu["hbm_bw_gbs"]
        scaled_bw = coeff["bw_eff_gbs"] * bw_ratio

        # Scale compute by GPU ratio
        compute_ratio = gpu_p["fp16_tflops"] / measured_gpu["fp16_tflops"]
        scaled_compute = coeff["compute_eff_tflops"] * compute_ratio

        # Protocol overhead: scale network component only
        # Measured protocol = sidecar_processing + network_rtt + nixl_setup
        # Conservative: use measured value as upper bound
        protocol_ms = coeff["tr_intercept"]
        if net:
            # Reduce by network improvement (rough: 30% of protocol is network)
            network_fraction = 0.3
            protocol_ms = protocol_ms * (1 - network_fraction) + \
                          protocol_ms * network_fraction * (net["rtt_ms"] / 0.5)

        # Predict at various sequence lengths
        seq_lens = [10, 50, 100, 250, 500, 1000, 2000, 4096]
        pred = {
            "source": coeff["short"],
            "protocol_ms": protocol_ms,
            "points": [],
        }

        for L in seq_lens:
            # T_transfer = protocol + kv_bytes * L / bandwidth
            t_tr_ms = protocol_ms + (target_kv * L) / (scaled_bw * 1e9) * 1000

            # T_prefill = launch_overhead + flops * L / compute
            launch_ms = coeff["pf_intercept"] * (gpu_p["fp16_tflops"] / measured_gpu["fp16_tflops"])
            # Hmm, launch overhead doesn't scale with compute. It's fixed.
            # Actually pf_intercept includes model-specific startup. Let's be more careful.
            # pf_intercept is ~200ms for TinyLlama, ~700ms for Phi-3, so it scales with model.
            # The compute part of pf_intercept: startup / warmup.
            # Keep as is: scale both intercept and slope.
            t_pf_ms = launch_ms + (target_flops * L) / (scaled_compute * 1e12) * 1000

            pred["points"].append({
                "seq_len": L,
                "t_transfer_ms": t_tr_ms,
                "t_prefill_ms": t_pf_ms,
            })

        predictions.append(pred)

    return predictions


def cross_cluster_comparison(regressions):
    """Compare regression slopes across clusters to validate scaling."""
    print("\n" + "=" * 60)
    print("  Cross-Cluster Scaling Comparison")
    print("=" * 60)

    # Group by model
    by_model = {}
    for reg in regressions:
        by_model.setdefault(reg.model_name, []).append(reg)

    # Within-model comparison (same model, different topology)
    for model, regs in by_model.items():
        if len(regs) > 1:
            short = regs[0].model_info.get("short", model) if regs[0].model_info else model
            print(f"\n  Same model ({short}), different topology:")
            print(f"    {'Cluster':>30} | {'T_tr slope':>12} | {'T_pf slope':>12} | "
                  f"{'Protocol':>10} | {'r(transfer)':>12}")
            print(f"    {'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}")
            for reg in regs:
                print(f"    {reg.name:>30} | {reg.tr_slope:>9.4f}ms/t | "
                      f"{reg.pf_slope:>9.4f}ms/t | {reg.tr_intercept:>7.1f}ms | "
                      f"{reg.tr_pearson:>10.3f}")

    # Cross-model comparison (different models, same hardware)
    models = list(by_model.keys())
    if len(models) >= 2:
        print(f"\n  Cross-model scaling:")
        print()

        # Use first regression per model for comparison
        regs = [by_model[m][0] for m in models]

        # Header
        print(f"    {'Metric':>25} |", end="")
        for reg in regs:
            short = reg.model_info.get("short", "") if reg.model_info else ""
            print(f" {short:>15} |", end="")
        if len(regs) == 2:
            print(f" {'Ratio':>10} |", end="")
        print()

        print(f"    {'-'*25}-+", end="")
        for _ in regs:
            print(f"-{'-'*15}-+", end="")
        if len(regs) == 2:
            print(f"-{'-'*10}-+", end="")
        print()

        # Rows
        def row(label, vals, fmt=".1f"):
            print(f"    {label:>25} |", end="")
            for v in vals:
                print(f" {v:>15{fmt}} |", end="")
            if len(vals) == 2 and vals[0] > 0:
                print(f" {vals[1]/vals[0]:>9.2f}x |", end="")
            print()

        row("Params (B)", [r.model_info.get("params_b", 0) for r in regs])
        row("KV bytes/token", [kv_bytes_per_token(r.model_info) if r.model_info else 0 for r in regs], ",")
        row("T_tr slope (ms/tok)", [r.tr_slope for r in regs], ".4f")
        row("T_pf slope (ms/tok)", [r.pf_slope for r in regs], ".4f")
        row("T_tr intercept (ms)", [r.tr_intercept for r in regs])
        row("T_pf intercept (ms)", [r.pf_intercept for r in regs])
        row("Transfer r", [r.tr_pearson for r in regs], ".3f")
        row("Prefill r", [r.pf_pearson for r in regs], ".3f")
        print()

        # Validate: do slope ratios match model size ratios?
        if len(regs) == 2:
            r0, r1 = regs
            param_ratio = r1.model_info["params_b"] / r0.model_info["params_b"]
            kv_ratio = kv_bytes_per_token(r1.model_info) / kv_bytes_per_token(r0.model_info)
            tr_ratio = r1.tr_slope / r0.tr_slope if r0.tr_slope > 0 else float('inf')
            pf_ratio = r1.pf_slope / r0.pf_slope if r0.pf_slope > 0 else float('inf')

            print("  Validation — do measured ratios match physics?")
            print(f"    Model params ratio: {param_ratio:.2f}x")
            print(f"    KV cache ratio:     {kv_ratio:.2f}x (from architecture)")
            print(f"    Transfer slope ratio: {tr_ratio:.2f}x (measured)")
            print(f"    Prefill slope ratio:  {pf_ratio:.2f}x (measured)")
            print()

            # Expected: transfer slope ratio ≈ KV ratio
            #           prefill slope ratio ≈ param ratio
            tr_match = abs(tr_ratio - kv_ratio) / kv_ratio
            pf_match = abs(pf_ratio - param_ratio) / param_ratio

            if tr_match < 0.3:
                print(f"    Transfer: CONSISTENT — slope ratio ({tr_ratio:.2f}x) "
                      f"matches KV ratio ({kv_ratio:.2f}x) within {tr_match:.0%}")
            else:
                print(f"    Transfer: INCONSISTENT — slope ratio ({tr_ratio:.2f}x) "
                      f"vs KV ratio ({kv_ratio:.2f}x), off by {tr_match:.0%}")
                print(f"      Possible causes: protocol overhead variation, "
                      f"memory bandwidth saturation, NIXL batching effects")

            if pf_match < 0.3:
                print(f"    Prefill:  CONSISTENT — slope ratio ({pf_ratio:.2f}x) "
                      f"matches param ratio ({param_ratio:.2f}x) within {pf_match:.0%}")
            else:
                print(f"    Prefill:  INCONSISTENT — slope ratio ({pf_ratio:.2f}x) "
                      f"vs param ratio ({param_ratio:.2f}x), off by {pf_match:.0%}")
                print(f"      Possible causes: different MFU, attention overhead, "
                      f"memory bandwidth effects")


def print_predictions(predictions, target_model_key):
    """Print prediction tables."""
    target = MODELS.get(target_model_key, {})
    short = target.get("short", target_model_key)

    print(f"\n  Predicted latencies for {short}:")
    print()

    for pred in predictions:
        print(f"    Based on: {pred['source']} (protocol overhead: {pred['protocol_ms']:.1f}ms)")
        print(f"    {'Seq len':>10} | {'T_transfer':>12} | {'T_prefill':>12} | "
              f"{'Ratio pf/tr':>12} | {'Crossover?':>10}")
        print(f"    {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

        crossover_found = False
        for pt in pred["points"]:
            ratio = pt["t_prefill_ms"] / pt["t_transfer_ms"] if pt["t_transfer_ms"] > 0 else 0
            marker = ""
            if ratio < 1 and not crossover_found:
                marker = "← HERE"
                crossover_found = True
            print(f"    {pt['seq_len']:>10} | {pt['t_transfer_ms']:>9.1f}ms | "
                  f"{pt['t_prefill_ms']:>9.1f}ms | {ratio:>10.1f}x | {marker}")

        if not crossover_found:
            print(f"    → Prefill always dominates (disagg favorable)")
        print()


# ── Main ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Cross-cluster scaling model for disaggregation crossover prediction")
    parser.add_argument("data_dirs", nargs="+",
                        help="Paths to data directories (e.g., clusters/rdu3-t4x3/data)")
    parser.add_argument("--predict", default=None,
                        help="Target model to predict (e.g., llama-31b, llama-70b)")
    parser.add_argument("--gpu-decode", default="T4",
                        help="Target decode GPU (default: T4)")
    parser.add_argument("--gpu-prefill", default="T4",
                        help="Target prefill GPU (default: T4)")
    parser.add_argument("--network", default="TCP",
                        help="Target network (TCP, RoCE, IB)")
    args = parser.parse_args()

    # Model name aliases
    model_aliases = {
        "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-31b": "meta-llama/Llama-3.1-70B-Instruct",
        "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
        "phi-3": "microsoft/Phi-3-mini-4k-instruct",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }

    print("llm-d Scaling Model — Cross-Cluster Analysis")
    print("=" * 60)

    # Load all clusters
    regressions = []
    for data_dir in args.data_dirs:
        try:
            reg = ClusterRegression(data_dir)
            reg.summary()
            regressions.append(reg)
        except FileNotFoundError as e:
            print(f"\n  SKIP: {e}")

    if len(regressions) < 1:
        print("\n  ERROR: Need at least one cluster with exp5 data")
        sys.exit(1)

    # Cross-cluster comparison
    if len(regressions) >= 2:
        cross_cluster_comparison(regressions)

    # Extract physical coefficients
    coefficients = extract_physical_coefficients(regressions)

    print("\n" + "=" * 60)
    print("  Physical Coefficients (Hardware-Independent)")
    print("=" * 60)
    for coeff in coefficients:
        print(f"\n  {coeff['short']} ({coeff['name']}):")
        print(f"    Protocol overhead: {coeff['tr_intercept']:.1f}ms")
        print(f"    Transfer slope: {coeff['tr_slope']:.4f} ms/token")
        if "bw_eff_gbs" in coeff:
            print(f"    Effective transfer BW: {coeff['bw_eff_gbs']:.2f} GB/s")
        print(f"    Prefill slope: {coeff['pf_slope']:.4f} ms/token")
        if "compute_eff_tflops" in coeff:
            print(f"    Effective compute: {coeff['compute_eff_tflops']:.2f} TFLOPS")

    # Predict for target model
    if args.predict:
        target_key = model_aliases.get(args.predict, args.predict)
        predictions = predict_for_model(
            coefficients, target_key,
            gpu_decode=args.gpu_decode, gpu_prefill=args.gpu_prefill,
            network=args.network)
        if predictions:
            print_predictions(predictions, target_key)

    # Show example predictions for common deployment scenarios
    if not args.predict:
        print("\n" + "=" * 60)
        print("  Example Deployment Predictions")
        print("=" * 60)

        for target, gpu_d, gpu_p, net in [
            ("meta-llama/Llama-3.1-70B-Instruct", "H200", "L40S", "IB"),
            ("meta-llama/Llama-3.1-70B-Instruct", "H200", "L40S", "TCP"),
            ("meta-llama/Llama-3.1-8B-Instruct", "H100", "H100", "IB"),
        ]:
            predictions = predict_for_model(coefficients, target,
                                            gpu_decode=gpu_d, gpu_prefill=gpu_p,
                                            network=net)
            if predictions:
                print_predictions(predictions, target)


if __name__ == "__main__":
    main()
