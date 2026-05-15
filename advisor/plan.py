"""
Deployment advisor for disaggregated inference.

Pre-experiment: feasibility check and cost estimate at c=1. Reports the
overhead threshold — how much contention advantage disagg needs to justify
its transfer cost.

Post-experiment (--data-dir): data-backed mono vs disagg recommendation
using measured crossover points from exp11 throughput sweeps. Detects
crossovers at both p50 and p90 and reports variance asymmetry between
topologies under concurrent load.
"""

import argparse
import json
import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

try:
    from ._cluster import SCALING_GPUS
    from .pricing import GPU_MEM_BW_GBS, GPU_NIC_BW_GBPS, GPU_VRAM_GB, get_cheapest, get_price
except ImportError:
    from _cluster import SCALING_GPUS  # type: ignore[no-redef]
    from pricing import (  # type: ignore[no-redef]
        GPU_MEM_BW_GBS,
        GPU_NIC_BW_GBPS,
        GPU_VRAM_GB,
        get_cheapest,
        get_price,
    )


DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


@dataclass
class ModelProfile:
    model_id: str
    num_params: int = 0
    num_layers: int = 0
    hidden_size: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    torch_dtype: str = "float16"
    architecture: str = ""
    is_moe: bool = False
    weight_gb: float = 0.0
    max_position_embeddings: int = 4096


def fetch_model_profile(model_id: str) -> ModelProfile:
    """Fetch model architecture info from HuggingFace (no cluster dependency)."""
    p = ModelProfile(model_id=model_id)
    try:
        url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
        req = urllib.request.Request(url, headers={"User-Agent": "llm-d-diagnostics/1.0"})
        config = json.loads(urllib.request.urlopen(req, timeout=15).read())
        p.num_layers = config.get("num_hidden_layers", 32)
        p.hidden_size = config.get("hidden_size", 4096)
        p.num_kv_heads = config.get("num_key_value_heads", config.get("num_attention_heads", 32))
        p.head_dim = config.get("head_dim", p.hidden_size // max(config.get("num_attention_heads", 32), 1))
        p.torch_dtype = config.get("torch_dtype", "float16")
        p.max_position_embeddings = config.get("max_position_embeddings", 4096)
        p.architecture = (config.get("architectures") or [""])[0]
        p.is_moe = config.get("num_local_experts", 0) > 0
    except Exception:
        pass
    try:
        url = f"https://huggingface.co/api/models/{model_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "llm-d-diagnostics/1.0"})
        info = json.loads(urllib.request.urlopen(req, timeout=10).read())
        p.num_params = info.get("safetensors", {}).get("total", 0)
    except Exception:
        pass
    if p.num_params:
        p.weight_gb = p.num_params * DTYPE_BYTES.get(p.torch_dtype, 2) / (1024**3)
    return p


# Measured baseline data from 8-model study on T4 GPUs
MEASURED_BASELINES = {
    ("Qwen/Qwen2.5-0.5B-Instruct", "t4"): {
        "params_b": 0.5, "is_moe": False,
        "mono_ttft_ms": 58, "disagg_ttft_ms": 91,
        "mono_throughput": 0.20, "disagg_throughput": 0.22,
        "overhead_ms": 33, "overhead_pct": 56.9,
    },
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "t4"): {
        "params_b": 1.1, "is_moe": False,
        "mono_ttft_ms": 73, "disagg_ttft_ms": 109,
        "mono_throughput": 0.21, "disagg_throughput": 0.21,
        "overhead_ms": 36, "overhead_pct": 49.3,
        "nixl_ms": 17,
    },
    ("Qwen/Qwen2.5-1.5B-Instruct", "t4"): {
        "params_b": 1.5, "is_moe": False,
        "mono_ttft_ms": 112, "disagg_ttft_ms": 161,
        "mono_throughput": 0.12, "disagg_throughput": 0.20,
        "overhead_ms": 49, "overhead_pct": 43.8,
    },
    ("stabilityai/stablelm-2-1_6b-chat", "t4"): {
        "params_b": 1.6, "is_moe": False,
        "mono_ttft_ms": 105, "disagg_ttft_ms": 157,
        "mono_throughput": 0.21, "disagg_throughput": 0.21,
        "overhead_ms": 52, "overhead_pct": 49.5,
    },
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "t4"): {
        "params_b": 1.7, "is_moe": False,
        "mono_ttft_ms": 104, "disagg_ttft_ms": 155,
        "mono_throughput": 0.21, "disagg_throughput": 0.23,
        "overhead_ms": 51, "overhead_pct": 49.0,
    },
    ("Qwen/Qwen2.5-3B-Instruct", "t4"): {
        "params_b": 3.0, "is_moe": False,
        "mono_ttft_ms": 177, "disagg_ttft_ms": 239,
        "mono_throughput": 0.21, "disagg_throughput": 0.18,
        "overhead_ms": 62, "overhead_pct": 35.0,
        "nixl_ms": 25,
    },
    ("microsoft/Phi-3.5-mini-instruct", "t4"): {
        "params_b": 3.8, "is_moe": False,
        "mono_ttft_ms": 173, "disagg_ttft_ms": 247,
        "mono_throughput": 0.21, "disagg_throughput": 0.21,
        "overhead_ms": 74, "overhead_pct": 42.8,
    },
    ("allenai/OLMoE-1B-7B-0924-Instruct", "t4"): {
        "params_b": 7.0, "is_moe": True,
        "mono_ttft_ms": 2999, "disagg_ttft_ms": 3278,
        "mono_throughput": 0.12, "disagg_throughput": 0.20,
        "overhead_ms": 279, "overhead_pct": 9.3,
        "nixl_ms": 267,
    },
    ("microsoft/Phi-3-mini-4k-instruct", "t4"): {
        "params_b": 3.8, "is_moe": False,
        "mono_ttft_ms": 770, "disagg_ttft_ms": 1022,
        "mono_ttft_base_ms": 560, "mono_ttft_rate": 1.70,
        "disagg_ttft_base_ms": 636, "disagg_ttft_rate": 3.25,
        "ref_seq_len": 100,
        "mono_throughput": 1.30, "disagg_throughput": 0.98,
        "overhead_ms": 252, "overhead_pct": 32.7,
        "nixl_ms": 252,
    },
}

# KV cache bytes per token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
KV_BYTES_PER_TOKEN = {
    "Qwen/Qwen2.5-0.5B-Instruct": 2 * 24 * 2 * 64 * 2,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2 * 22 * 4 * 64 * 2,
    "Qwen/Qwen2.5-1.5B-Instruct": 2 * 28 * 2 * 64 * 2,
    "stabilityai/stablelm-2-1_6b-chat": 2 * 24 * 32 * 64 * 2,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 2 * 24 * 32 * 64 * 2,
    "Qwen/Qwen2.5-3B-Instruct": 2 * 36 * 2 * 128 * 2,
    "microsoft/Phi-3.5-mini-instruct": 2 * 32 * 32 * 96 * 2,
    "allenai/OLMoE-1B-7B-0924-Instruct": 2 * 16 * 16 * 128 * 2,
    "microsoft/Phi-3-mini-4k-instruct": 2 * 32 * 32 * 96 * 2,
}

# NIXL transfer model: T_transfer = protocol_ms + (kv_bytes * seq_len) / eff_bw
# Measured via exp5b direct NIXL prometheus scraping on T4 cluster (R²=0.999).
# protocol_ms is fixed overhead (NIXL handshake, buffer setup) — model-independent.
# eff_bw is effective NIC throughput in GB/s — hardware-dependent.
NIXL_PROTOCOL_MS = 4.3         # regression intercept (exp5b, 180 points, Phi-3 on T4)
NIXL_EFF_BW_GBS = 0.299        # regression slope -> effective bandwidth (10 Gbps OVN/TCP)
MOE_NIXL_CORRECTION = 1.0      # MoE uses dense attention → KV transfer is identical to dense

# Contention model: R(c,s) = c^Δγ(s), Δγ(s) = -C_A·(T(∞)-1) + C_B·ln(s)
# Power-law scaling: α_mono ~ c^γ_mono(s), α_disagg ~ c^γ_disagg (constant).
# Δγ = γ_mono - γ_disagg captures two competing effects:
#   C_A: NIXL serialization (hurts disagg at short s, per unit T(∞) overhead)
#   C_B: decode interference (helps disagg at long s, O(s) attention)
# Calibrated from Phi-3/T4/TCP (N=1). Valid for c ≤ ~32.
CONTENTION_SCALE_A = 0.889   # NIXL serialization rate (Phi-3/T4/TCP, N=1)
CONTENTION_SCALE_B = 0.144   # decode interference rate per ln(s)


@dataclass
class CapacityPlan:
    model: str
    gpu_type: str
    target_throughput: float
    target_ttft_ms: float
    seq_len: int = 128

    mono_gpus_per_instance: int = 1
    mono_instances: int = 1
    mono_total_gpus: int = 1
    mono_est_ttft_ms: float = 0
    mono_est_throughput: float = 0
    mono_cost_per_hr: float = 0

    disagg_prefill_gpus: int = 1
    disagg_decode_gpus: int = 2
    disagg_total_gpus: int = 3
    disagg_est_ttft_ms: float = 0
    disagg_est_throughput: float = 0
    disagg_cost_per_hr: float = 0

    recommendation: str = ""
    confidence: str = ""
    reasoning: list = field(default_factory=list)
    crossovers: list = field(default_factory=list)
    contention_table: list = field(default_factory=list)
    overhead_asymptote: float = 0.0
    measured_conditions: str = ""
    overhead_thresholds: list = field(default_factory=list)
    critical_concurrency: int = 0
    mono_cv_by_concurrency: list = field(default_factory=list)
    predicted_delta_gamma: float = 0.0
    predicted_crossover_c: float = 0.0
    predicted_s_cross: float = 0.0
    measured_delta_gamma: float = 0.0
    measured_delta_gamma_by_s: dict = field(default_factory=dict)
    measured_fit_a: float = 0.0
    measured_fit_b: float = 0.0
    measured_fit_r2: float = 0.0


def plan_capacity(
    model_id: str,
    target_throughput: float,
    target_ttft_ms: float = 500,
    gpu_type: str = "t4",
    provider: str = "aws",
    seq_len: int = 128,
    data_dir: str = "",
) -> CapacityPlan:
    """Generate a capacity plan for the given model and requirements."""

    profile = fetch_model_profile(model_id)
    price = get_price(provider, gpu_type)
    if not price:
        provider, price = get_cheapest(gpu_type)

    # Prefer toolkit's hardware DB (more precise), fall back to pricing tables.
    # SCALING_GPUS uses keys like "T4", "A100-80"; pricing uses "t4", "a100_80".
    _gpu_key = gpu_type.upper().replace("_", "-")
    _hw = SCALING_GPUS.get(_gpu_key, {})
    gpu_vram = _hw.get("memory_gb", GPU_VRAM_GB.get(gpu_type, 16))

    plan = CapacityPlan(
        model=model_id, gpu_type=gpu_type,
        target_throughput=target_throughput,
        target_ttft_ms=target_ttft_ms,
        seq_len=seq_len,
    )

    plan.reasoning.extend(_validate_profile(profile))

    weight_gb = profile.weight_gb or (profile.num_params * 2 / (1024**3))

    max_tp_per_node = 8
    tp = max(1, math.ceil(weight_gb / (gpu_vram * 0.8)))
    pp = 1
    if tp > max_tp_per_node:
        pp = math.ceil(tp / max_tp_per_node)
        tp = max_tp_per_node
    plan.mono_gpus_per_instance = tp * pp

    kv_bytes = _estimate_kv_bytes(profile)
    if weight_gb > 0:
        _check_vram_feasibility(weight_gb, kv_bytes, seq_len, plan.mono_gpus_per_instance, gpu_vram, plan)

    if pp > 1:
        plan.reasoning.append(
            f"Model requires pipeline parallelism (PP={pp}, TP={tp}). "
            f"Each instance uses {tp * pp} GPUs."
        )

    exp11_path = os.path.join(data_dir, "exp11-results.csv") if data_dir else ""
    if data_dir and os.path.exists(exp11_path):
        plan.confidence = "experiment"
        _plan_from_experiments(plan, data_dir, seq_len, target_throughput, price)
    elif (model_id, gpu_type) in MEASURED_BASELINES:
        plan.confidence = "measured"
        _plan_from_measured(plan, MEASURED_BASELINES[(model_id, gpu_type)],
                           target_throughput, price, seq_len)
    else:
        _plan_from_extrapolation(plan, profile, gpu_type, target_throughput, price, tp, seq_len)

    if plan.overhead_asymptote == 0 and plan.overhead_thresholds and len(plan.overhead_thresholds) >= 2:
        ts = sorted(plan.overhead_thresholds, key=lambda t: t["seq_len"])
        plan.overhead_asymptote = round(ts[-1]["threshold"], 2)
    elif plan.overhead_asymptote == 0:
        measured = MEASURED_BASELINES.get((model_id, gpu_type), {})
        prefill_rate = measured.get("mono_ttft_rate", 0)
        if prefill_rate <= 0:
            prefill_rate = _estimate_prefill_rate(profile, gpu_type)
        if prefill_rate > 0:
            kv_bytes = _estimate_kv_bytes(profile)
            plan.overhead_asymptote = round(
                _estimate_overhead_asymptote(kv_bytes, prefill_rate, gpu_type,
                                            profile.is_moe), 2)

    if plan.overhead_asymptote > 0:
        plan.predicted_delta_gamma = round(
            _predict_delta_gamma(seq_len, plan.overhead_asymptote), 3)
        plan.predicted_s_cross = round(
            _predict_s_cross(plan.overhead_asymptote), 0)
        for t in plan.overhead_thresholds:
            dg = _predict_delta_gamma(t["seq_len"], plan.overhead_asymptote)
            if "predicted_crossover_c" not in t:
                c_cross = _predict_crossover_c(t["threshold"], dg)
                t["predicted_crossover_c"] = round(c_cross, 1)
        if plan.predicted_crossover_c == 0:
            if plan.overhead_thresholds:
                t0 = plan.overhead_thresholds[0]
                dg0 = _predict_delta_gamma(t0["seq_len"], plan.overhead_asymptote)
                plan.predicted_crossover_c = round(
                    _predict_crossover_c(t0["threshold"], dg0), 1)
            elif plan.mono_est_ttft_ms > 0 and plan.disagg_est_ttft_ms > 0:
                t_s = plan.disagg_est_ttft_ms / plan.mono_est_ttft_ms
                plan.predicted_crossover_c = round(
                    _predict_crossover_c(t_s, plan.predicted_delta_gamma), 1)

    _generate_recommendation(plan)

    return plan


def _baseline_ttft(baseline, seq_len, field="mono"):
    """Get TTFT from a baseline, seq_len-aware if rate data available."""
    base_key = f"{field}_ttft_base_ms"
    rate_key = f"{field}_ttft_rate"
    if base_key in baseline and rate_key in baseline:
        return baseline[base_key] + baseline[rate_key] * seq_len
    return baseline[f"{field}_ttft_ms"]


def _plan_from_experiments(plan, data_dir, seq_len, target_throughput, price):
    """Plan from exp11 experiment data: detect crossovers at p50 and p90."""
    import sys
    _advisor_dir = os.path.dirname(__file__)
    _toolkit_dir = os.path.join(os.path.dirname(_advisor_dir), "toolkit")
    if _advisor_dir not in sys.path:
        sys.path.insert(0, _advisor_dir)
    if _toolkit_dir not in sys.path:
        sys.path.insert(0, _toolkit_dir)
    from validate import load_exp_baselines

    measurements = load_exp_baselines(data_dir)
    exp11 = measurements.get("exp11", {})
    if not exp11:
        plan.confidence = "extrapolated"
        plan.reasoning.append("No exp11 data found in data dir; falling back")
        return

    measured_seqs = sorted(set(k[2] for k in exp11 if k[0] == "BASELINE" and k[1] == 1))
    measured_concs = sorted(set(k[1] for k in exp11))

    plan.measured_conditions = (
        f"c∈{{{','.join(str(c) for c in measured_concs)}}}, "
        f"s∈{{{','.join(str(s) for s in measured_seqs)}}}")

    if seq_len in measured_seqs:
        target_s = seq_len
    else:
        target_s = min(measured_seqs, key=lambda s: abs(s - seq_len))
        plan.reasoning.append(
            f"Requested seq_len={seq_len} not measured; using nearest: {target_s}")

    bl_c1 = exp11.get(("BASELINE", 1, target_s))
    if bl_c1:
        plan.mono_est_ttft_ms = round(bl_c1["median"])

    dg_c1 = exp11.get(("DISAGG-1D", 1, target_s))
    if not dg_c1:
        dg_c1 = exp11.get(("DISAGG-2D", 1, target_s))
    if dg_c1:
        plan.disagg_est_ttft_ms = round(dg_c1["median"])

    for s in measured_seqs:
        bl_c1_s = exp11.get(("BASELINE", 1, s))
        dg_c1_s = exp11.get(("DISAGG-1D", 1, s)) or exp11.get(("DISAGG-2D", 1, s))
        if bl_c1_s and dg_c1_s and bl_c1_s["median"] > 0:
            t = dg_c1_s["median"] / bl_c1_s["median"]
            plan.overhead_thresholds.append({
                "seq_len": s,
                "threshold": round(t, 2),
                "overhead_pct": round((t - 1) * 100),
            })

    if len(plan.overhead_thresholds) >= 3:
        mono_pts = [(s, exp11[("BASELINE", 1, s)]["median"])
                    for s in measured_seqs if ("BASELINE", 1, s) in exp11]
        overhead_pts = []
        for s in measured_seqs:
            dg = exp11.get(("DISAGG-1D", 1, s)) or exp11.get(("DISAGG-2D", 1, s))
            bl = exp11.get(("BASELINE", 1, s))
            if dg and bl:
                overhead_pts.append((s, dg["median"] - bl["median"]))
        seq_range = [p[0] for p in mono_pts]
        wide_enough = max(seq_range) / max(min(seq_range), 1) >= 3 if seq_range else False
        if wide_enough and len(mono_pts) >= 3 and len(overhead_pts) >= 3:
            _, prefill_rate = _linreg(mono_pts)
            _, overhead_rate = _linreg(overhead_pts)
            if prefill_rate > 0 and overhead_rate > 0:
                plan.overhead_asymptote = round(1 + overhead_rate / prefill_rate, 2)

    for (cfg, conc, pt), dg_stats in exp11.items():
        if not cfg.startswith("DISAGG"):
            continue
        if conc <= 1:
            continue
        bl_cn = exp11.get(("BASELINE", conc, pt))
        if not bl_cn or bl_cn["median"] <= 0 or bl_cn["p90"] <= 0:
            continue
        bl_c1_pt = exp11.get(("BASELINE", 1, pt))
        dg_c1_pt = exp11.get((cfg, 1, pt))
        if not bl_c1_pt or not dg_c1_pt:
            continue
        if bl_c1_pt["median"] <= 0 or dg_c1_pt["median"] <= 0:
            continue

        alpha_mono = bl_cn["median"] / bl_c1_pt["median"]
        alpha_disagg = dg_stats["median"] / dg_c1_pt["median"]
        contention_ratio = alpha_mono / alpha_disagg if alpha_disagg > 0 else 0
        threshold = dg_c1_pt["median"] / bl_c1_pt["median"]

        p50_delta = (dg_stats["median"] - bl_cn["median"]) / bl_cn["median"] * 100
        p90_delta = (dg_stats["p90"] - bl_cn["p90"]) / bl_cn["p90"] * 100
        p50_cross = dg_stats["median"] < bl_cn["median"]
        p90_cross = dg_stats["p90"] < bl_cn["p90"]

        alpha_mono_p90 = bl_cn["p90"] / bl_c1_pt["p90"] if bl_c1_pt["p90"] > 0 else 0
        alpha_disagg_p90 = dg_stats["p90"] / dg_c1_pt["p90"] if dg_c1_pt["p90"] > 0 else 0
        contention_ratio_p90 = alpha_mono_p90 / alpha_disagg_p90 if alpha_disagg_p90 > 0 else 0
        has_p90 = bl_c1_pt["p90"] > 0 and dg_c1_pt["p90"] > 0
        threshold_p90 = (dg_c1_pt["p90"] / bl_c1_pt["p90"]) if has_p90 else threshold

        mono_cv = bl_cn.get("cv")
        disagg_cv = dg_stats.get("cv")
        gap = contention_ratio - threshold
        r_uncertainty = contention_ratio * (disagg_cv or 0)
        significant = abs(gap) > r_uncertainty if r_uncertainty > 0 else None
        entry = {
            "seq_len": pt, "concurrency": conc, "config": cfg,
            "p50_delta_pct": round(p50_delta, 1),
            "p90_delta_pct": round(p90_delta, 1),
            "p50_cross": p50_cross, "p90_cross": p90_cross,
            "mono_cv": round(mono_cv, 3) if mono_cv is not None else None,
            "disagg_cv": round(disagg_cv, 3) if disagg_cv is not None else None,
            "alpha_mono": round(alpha_mono, 2),
            "alpha_disagg": round(alpha_disagg, 2),
            "contention_ratio": round(contention_ratio, 2),
            "threshold": round(threshold, 2),
            "contention_ratio_p90": round(contention_ratio_p90, 2),
            "threshold_p90": round(threshold_p90, 2),
            "significant": significant,
        }
        plan.contention_table.append(entry)
        if p50_cross or p90_cross:
            plan.crossovers.append(entry)

    if max(measured_concs) <= 1:
        plan.reasoning.append(
            "Only c=1 measured — crossovers require concurrent load. "
            "Re-run with concurrency > 1 to detect crossover")

    cv_threshold = 0.10
    mono_cv_by_c = {}
    for (cfg, conc, _pt), st in exp11.items():
        if cfg != "BASELINE" or conc < 1:
            continue
        cv = st.get("cv")
        if cv is not None:
            mono_cv_by_c.setdefault(conc, []).append(cv)

    for conc in sorted(mono_cv_by_c):
        median_cv = sorted(mono_cv_by_c[conc])[len(mono_cv_by_c[conc]) // 2]
        plan.mono_cv_by_concurrency.append({
            "concurrency": conc, "median_cv": round(median_cv, 3),
        })

    prev_c, prev_cv = 0, 0.0
    for entry in plan.mono_cv_by_concurrency:
        c, cv = entry["concurrency"], entry["median_cv"]
        if cv >= cv_threshold and prev_cv < cv_threshold and prev_c > 0:
            plan.critical_concurrency = prev_c
            break
        prev_c, prev_cv = c, cv

    fit_a, fit_b, fit_r2, dg_by_s = _fit_measured_delta_gamma(plan.contention_table)
    plan.measured_fit_a = round(fit_a, 3)
    plan.measured_fit_b = round(fit_b, 3)
    plan.measured_fit_r2 = round(fit_r2, 2)
    plan.measured_delta_gamma_by_s = dg_by_s
    if fit_b != 0 or fit_a != 0:
        plan.measured_delta_gamma = round(
            fit_a + fit_b * math.log(max(seq_len, 1)), 3)

    gpus_per = plan.mono_gpus_per_instance
    mono_rps = 1000 / max(plan.mono_est_ttft_ms, 1)
    plan.mono_instances = max(1, math.ceil(target_throughput / mono_rps))
    plan.mono_total_gpus = plan.mono_instances * gpus_per
    plan.mono_cost_per_hr = plan.mono_total_gpus * price

    plan.mono_est_throughput = round(mono_rps * plan.mono_instances, 2)
    plan.disagg_prefill_gpus = gpus_per
    plan.disagg_decode_gpus = plan.mono_instances * gpus_per
    plan.disagg_total_gpus = plan.disagg_prefill_gpus + plan.disagg_decode_gpus
    plan.disagg_est_throughput = plan.mono_est_throughput
    plan.disagg_cost_per_hr = plan.disagg_total_gpus * price

    plan.reasoning.append(
        f"Based on {sum(v['n'] for v in exp11.values())} measurements "
        f"({plan.measured_conditions})")


def _plan_from_measured(plan, measured, target_throughput, price, seq_len=128):
    """Plan using actual measured data from our experiments."""
    mono_rps = measured["mono_throughput"]
    disagg_rps = measured["disagg_throughput"]
    gpus_per = plan.mono_gpus_per_instance

    mono_ttft = _baseline_ttft(measured, seq_len)
    disagg_ttft = _baseline_ttft(measured, seq_len, "disagg")

    plan.mono_instances = max(1, math.ceil(target_throughput / mono_rps)) if mono_rps > 0 else 1
    plan.mono_total_gpus = plan.mono_instances * gpus_per
    plan.mono_est_ttft_ms = round(mono_ttft)
    plan.mono_est_throughput = mono_rps * plan.mono_instances
    plan.mono_cost_per_hr = plan.mono_total_gpus * price

    disagg_instances = max(1, math.ceil(target_throughput / disagg_rps)) if disagg_rps > 0 else 1
    prefill_capacity = 1000 / max(mono_ttft, 1)
    min_prefill = max(1, math.ceil(target_throughput / prefill_capacity))
    plan.disagg_prefill_gpus = min_prefill * gpus_per
    plan.disagg_decode_gpus = disagg_instances * gpus_per
    plan.disagg_total_gpus = plan.disagg_prefill_gpus + plan.disagg_decode_gpus
    plan.disagg_est_ttft_ms = round(disagg_ttft)
    plan.disagg_est_throughput = disagg_rps * disagg_instances
    plan.disagg_cost_per_hr = plan.disagg_total_gpus * price

    has_rates = "mono_ttft_base_ms" in measured
    if has_rates:
        plan.reasoning.append(
            f"TTFT scaled for {seq_len} tokens "
            f"(linear model, valid 50-1000 tokens)")
    elif seq_len > 150 or seq_len < 50:
        ref = measured.get("ref_seq_len", 100)
        plan.reasoning.append(
            f"TTFT measured at ~{ref} tokens; prediction at {seq_len} "
            f"tokens may diverge — run experiments to calibrate")
    plan.reasoning.append(f"Based on measured data: mono {mono_rps:.2f} req/s, disagg {disagg_rps:.2f} req/s")


def _find_nearest_baselines(params_b, gpu_type, is_moe=False):
    """Find bracketing baselines for interpolation, or two nearest for extrapolation."""
    candidates = []
    for (model_id, gtype), data in MEASURED_BASELINES.items():
        if gtype != gpu_type:
            continue
        if is_moe != data.get("is_moe", False):
            continue
        candidates.append((data["params_b"], data, model_id))
    if not candidates:
        return []
    below = [(b, d, m) for b, d, m in candidates if b <= params_b]
    above = [(b, d, m) for b, d, m in candidates if b > params_b]
    if below and above:
        lo = max(below, key=lambda x: x[0])
        hi = min(above, key=lambda x: x[0])
        return [(lo[1], lo[2]), (hi[1], hi[2])]
    candidates.sort(key=lambda x: abs(x[0] - params_b))
    return [(c[1], c[2]) for c in candidates[:2]]


def _estimate_kv_bytes(profile) -> int:
    """Estimate KV cache bytes per token from model profile."""
    known = KV_BYTES_PER_TOKEN.get(profile.model_id)
    if known:
        return known
    if profile.num_kv_heads and profile.head_dim and profile.num_layers:
        dtype_bytes = DTYPE_BYTES.get(profile.torch_dtype, 2)
        return 2 * profile.num_layers * profile.num_kv_heads * profile.head_dim * dtype_bytes
    return KV_BYTES_PER_TOKEN["microsoft/Phi-3.5-mini-instruct"]


def _estimate_nixl_ms(kv_bytes_per_token: int, seq_len: int = 128,
                      gpu_type: str = "t4", is_moe: bool = False) -> float:
    """Estimate NIXL transfer time from KV cache size and NIC bandwidth.

    T = protocol_ms + (kv_bytes_per_token * ceil(seq_len/16)*16) / eff_bw
    NIXL transfers KV in 16-token blocks; block alignment eliminates
    quantization error at short sequences (exp5b: 180/180 exact multiples).
    """
    t4_nic = GPU_NIC_BW_GBPS.get("t4", 25)
    target_nic = GPU_NIC_BW_GBPS.get(gpu_type, t4_nic)
    scaled_bw = NIXL_EFF_BW_GBS * (target_nic / t4_nic)
    block_aligned = math.ceil(seq_len / 16) * 16
    data_ms = (kv_bytes_per_token * block_aligned) / (scaled_bw * 1e9) * 1000
    if is_moe:
        data_ms *= MOE_NIXL_CORRECTION
    return NIXL_PROTOCOL_MS + data_ms


def _estimate_overhead_asymptote(kv_bytes_per_token: int, prefill_rate: float,
                                 gpu_type: str = "t4", is_moe: bool = False) -> float:
    """T(∞) = 1 + nixl_rate / prefill_rate.

    As seq_len → ∞, protocol overhead and base TTFT become negligible.
    T(∞) is the per-token overhead ratio — the minimum contention advantage
    disagg needs at very long prompts.
    """
    if prefill_rate <= 0:
        return 0.0
    t4_nic = GPU_NIC_BW_GBPS.get("t4", 25)
    target_nic = GPU_NIC_BW_GBPS.get(gpu_type, t4_nic)
    scaled_bw = NIXL_EFF_BW_GBS * (target_nic / t4_nic)
    nixl_rate_ms = kv_bytes_per_token / (scaled_bw * 1e9) * 1000
    if is_moe:
        nixl_rate_ms *= MOE_NIXL_CORRECTION
    return 1 + nixl_rate_ms / prefill_rate


PREFILL_RATE_CONSTANT = 143.2  # ms/token × GB/s / B_params, from Phi-3/T4 (N=1)


def _estimate_prefill_rate(profile, gpu_type: str = "t4") -> float:
    """Estimate marginal prefill cost (ms/token) from model size and GPU bandwidth.

    Calibrated from Phi-3/T4: 1.70 ms/token at 3.8B params, 320 GB/s HBM.
    Scales linearly with model size, inversely with memory bandwidth.
    """
    _gpu_key = gpu_type.upper().replace("_", "-")
    _hw = SCALING_GPUS.get(_gpu_key, {})
    mem_bw = _hw.get("hbm_bw_gbs", GPU_MEM_BW_GBS.get(gpu_type, 320))
    params_b = profile.num_params / 1e9 if profile.num_params else 3.8
    return PREFILL_RATE_CONSTANT * params_b / mem_bw


def _linreg(points):
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return sy / n, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return intercept, slope


def _predict_delta_gamma(seq_len: int, overhead_asymptote: float) -> float:
    t_inf = max(overhead_asymptote, 1.0)
    return -CONTENTION_SCALE_A * (t_inf - 1) + CONTENTION_SCALE_B * math.log(max(seq_len, 1))


def _predict_contention_ratio(delta_gamma: float, concurrency: int) -> float:
    if concurrency <= 1:
        return 1.0
    return concurrency ** delta_gamma


def _predict_crossover_c(threshold: float, delta_gamma: float) -> float:
    if delta_gamma <= 0:
        return float('inf')
    if threshold <= 1:
        return 1.0
    return threshold ** (1.0 / delta_gamma)


def _predict_s_cross(overhead_asymptote: float) -> float:
    t_inf = max(overhead_asymptote, 1.0)
    if CONTENTION_SCALE_B <= 0:
        return float('inf')
    return math.exp(CONTENTION_SCALE_A * (t_inf - 1) / CONTENTION_SCALE_B)


def _fit_measured_delta_gamma(contention_table: list) -> tuple:
    by_s = {}
    by_cfg_s = {}
    for e in contention_table:
        c, sl = e["concurrency"], e["seq_len"]
        cfg = e.get("config", "")
        if c <= 1:
            continue
        r = e["contention_ratio"]
        if r <= 0:
            continue
        by_s.setdefault(sl, []).append((math.log(c), math.log(r)))
        by_cfg_s.setdefault(cfg, {}).setdefault(sl, []).append(
            (math.log(c), math.log(r)))

    dg_points = []
    dg_by_s = {}
    for sl, pts in sorted(by_s.items()):
        if len(pts) < 2:
            continue
        _, gamma = _linreg(pts)
        dg_by_s[sl] = round(gamma, 3)
        dg_points.append((math.log(sl), gamma))

    if len(dg_points) < 2:
        return 0.0, 0.0, 0.0, dg_by_s

    intercept, slope = _linreg(dg_points)
    y_mean = sum(y for _, y in dg_points) / len(dg_points)
    ss_tot = sum((y - y_mean) ** 2 for _, y in dg_points)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in dg_points)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    for cfg, cfg_data in by_cfg_s.items():
        for sl, pts in sorted(cfg_data.items()):
            if len(pts) < 2:
                continue
            _, gamma = _linreg(pts)
            dg_by_s[(cfg, sl)] = round(gamma, 3)

    return intercept, slope, max(r_sq, 0.0), dg_by_s


def _validate_profile(profile):
    """Flag plans built on missing or implausible profile data."""
    issues = []
    if not profile.num_params:
        issues.append("WARNING: could not fetch model params from HuggingFace. "
                       "GPU count and TTFT estimates may be wrong.")
    if not profile.num_kv_heads:
        issues.append("WARNING: could not determine KV head count. "
                       "NIXL transfer estimate uses Phi-3 fallback (may be 30x wrong).")
    return issues


def _check_vram_feasibility(weight_gb, kv_bytes_per_token, seq_len, gpus_per_instance, gpu_vram, plan):
    """Check if model + KV cache fits in VRAM."""
    weight_per_gpu = weight_gb / max(gpus_per_instance, 1)
    kv_per_gpu = (kv_bytes_per_token * seq_len) / (max(gpus_per_instance, 1) * 1024**3)
    total_per_gpu = weight_per_gpu + kv_per_gpu
    usable_vram = gpu_vram * 0.85
    if total_per_gpu > usable_vram:
        plan.reasoning.append(
            f"VRAM WARNING: {total_per_gpu:.1f} GB/GPU needed "
            f"(weights {weight_per_gpu:.1f} + KV {kv_per_gpu:.1f}) "
            f"exceeds {usable_vram:.0f} GB usable on {plan.gpu_type.upper()}. "
            f"Increase TP or use a larger GPU.")


def _proportional_scale(ref, params_b, plan, seq_len=128):
    """Scale TTFT and throughput proportionally from a reference baseline."""
    scale = params_b / ref["params_b"] if ref["params_b"] > 0 else 1
    if scale > 5:
        plan.reasoning.append(
            f"LOW CONFIDENCE: extrapolating {scale:.0f}x beyond nearest baseline "
            f"({ref['params_b']:.1f}B -> {params_b:.1f}B). TTFT estimate is unreliable.")
        plan.confidence = "low"
    return _baseline_ttft(ref, seq_len) * scale, ref["mono_throughput"] / max(scale, 0.5)


def _plan_from_extrapolation(plan, profile, gpu_type, target_throughput, price, tp, seq_len=128):
    """Plan by extrapolating from the 8-model dataset + scaling model."""

    params_b = profile.num_params / 1e9 if profile.num_params else 1
    is_moe = profile.is_moe

    _gpu_key = gpu_type.upper().replace("_", "-")
    _hw = SCALING_GPUS.get(_gpu_key, {})
    t4_bw = SCALING_GPUS.get("T4", {}).get("hbm_bw_gbs", GPU_MEM_BW_GBS.get("t4", 320))
    target_bw = _hw.get("hbm_bw_gbs", GPU_MEM_BW_GBS.get(gpu_type, t4_bw))
    bw_speedup = target_bw / t4_bw

    t4_nic = GPU_NIC_BW_GBPS.get("t4", 25)
    target_nic = GPU_NIC_BW_GBPS.get(gpu_type, t4_nic)
    nic_speedup = target_nic / t4_nic

    nearest = _find_nearest_baselines(params_b, gpu_type, is_moe)
    used_cross_gpu = not nearest
    if not nearest:
        nearest = _find_nearest_baselines(params_b, "t4", is_moe)

    interpolated = False

    if nearest:
        if len(nearest) >= 2:
            if nearest[0][0]["params_b"] > nearest[1][0]["params_b"]:
                nearest[0], nearest[1] = nearest[1], nearest[0]
            (lo, _), (hi, _) = nearest[0], nearest[1]
            lo_b, hi_b = lo["params_b"], hi["params_b"]
            range_b = hi_b - lo_b if hi_b != lo_b else 1

            if lo_b <= params_b <= hi_b:
                t = (params_b - lo_b) / range_b
                lo_ttft = _baseline_ttft(lo, seq_len)
                hi_ttft = _baseline_ttft(hi, seq_len)
                mono_ttft = lo_ttft + t * (hi_ttft - lo_ttft)
                mono_rps = lo["mono_throughput"] + t * (hi["mono_throughput"] - lo["mono_throughput"])
                interpolated = True
                plan.reasoning.append(
                    f"Interpolated between {lo_b:.1f}B and {hi_b:.1f}B baselines")
            else:
                ref = lo if abs(lo_b - params_b) < abs(hi_b - params_b) else hi
                mono_ttft, mono_rps = _proportional_scale(ref, params_b, plan, seq_len)
                plan.reasoning.append(
                    f"Proportional scaling from {ref['params_b']:.1f}B baseline "
                    f"(target {params_b:.1f}B is outside measured range)")
        else:
            (ref, _) = nearest[0]
            mono_ttft, mono_rps = _proportional_scale(ref, params_b, plan, seq_len)
            plan.reasoning.append(f"Scaled from single baseline ({ref['params_b']:.1f}B)")

        if gpu_type != "t4":
            mono_ttft = mono_ttft / bw_speedup
            plan.reasoning.append(
                f"GPU scaling: {bw_speedup:.1f}x memory BW ({t4_bw} -> {target_bw} GB/s)")

        tp_eff = max(0.6, 0.85 - 0.05 * (tp - 2)) if tp > 1 else 1.0
        if tp > 1:
            mono_ttft = mono_ttft / (tp * tp_eff)
            plan.reasoning.append(f"TP={tp} parallelism: TTFT / {tp * tp_eff:.1f}")

        pp = plan.mono_gpus_per_instance // max(tp, 1)
        if pp > 1:
            pp_factor = 1 + 0.3 * (pp - 1)
            mono_ttft = mono_ttft * pp_factor
            plan.reasoning.append(f"PP={pp} pipeline overhead: TTFT * {pp_factor:.1f}")

        kv_bytes = _estimate_kv_bytes(profile)
        nixl_ms = _estimate_nixl_ms(kv_bytes, seq_len, gpu_type, is_moe)
        disagg_ttft = mono_ttft + nixl_ms

        plan.mono_est_ttft_ms = round(max(mono_ttft, 10))
        plan.mono_est_throughput = round(max(mono_rps * bw_speedup * tp * tp_eff, 0.01), 2)
        plan.disagg_est_ttft_ms = round(max(disagg_ttft, 10))

        kv_kb = kv_bytes / 1024
        data_ms = nixl_ms - NIXL_PROTOCOL_MS
        plan.reasoning.append(
            f"KV cache: {kv_kb:.0f} KB/token ({profile.num_kv_heads} KV heads) x {seq_len} tokens, "
            f"NIXL: {NIXL_PROTOCOL_MS:.0f}ms protocol + {data_ms:.0f}ms data")
        if gpu_type != "t4":
            plan.reasoning.append(
                f"NIC scaling: {nic_speedup:.0f}x ({t4_nic} -> {target_nic} Gbps)")
    else:
        raise RuntimeError(f"No baselines found for {params_b:.1f}B — this should be unreachable")

    plan.mono_instances = max(1, math.ceil(target_throughput / plan.mono_est_throughput)) if plan.mono_est_throughput > 0 else 1
    plan.mono_est_throughput = round(plan.mono_est_throughput * plan.mono_instances, 2)
    gpus_per = plan.mono_gpus_per_instance
    plan.mono_total_gpus = plan.mono_instances * gpus_per
    plan.mono_cost_per_hr = plan.mono_total_gpus * price

    prefill_capacity = 1000 / max(plan.mono_est_ttft_ms, 1)
    min_prefill = max(1, math.ceil(target_throughput / prefill_capacity))
    plan.disagg_prefill_gpus = min_prefill * gpus_per
    plan.disagg_decode_gpus = plan.mono_instances * gpus_per
    plan.disagg_total_gpus = plan.disagg_prefill_gpus + plan.disagg_decode_gpus
    plan.disagg_est_throughput = plan.mono_est_throughput
    plan.disagg_cost_per_hr = plan.disagg_total_gpus * price

    if plan.disagg_total_gpus > plan.mono_total_gpus:
        plan.reasoning.append(
            "Without measured disagg throughput, disagg always uses more GPUs. "
            "Run experiments to measure actual disagg throughput before deciding.")

    if plan.confidence != "low":
        plan.confidence = "interpolated" if (interpolated and not used_cross_gpu) else "extrapolated"
    plan.reasoning.append("Run experiments for empirical validation on your actual hardware")


def _generate_recommendation(plan):
    """Generate topology recommendation based on available evidence."""
    if plan.confidence == "experiment" and plan.crossovers:
        p90_crosses = [c for c in plan.crossovers if c["p90_cross"]]
        p50_crosses = [c for c in plan.crossovers if c["p50_cross"]]

        if p90_crosses:
            best = min(p90_crosses, key=lambda c: c["seq_len"])
            sig = best.get("significant")
            all_above_cstar = (plan.critical_concurrency > 0
                               and all(c["concurrency"] > plan.critical_concurrency
                                       for c in p90_crosses))
            if all_above_cstar:
                plan.recommendation = "MONOLITHIC"
                plan.reasoning.append(
                    f"p90 crossover exists at c≥{best['concurrency']}, "
                    f"s≥{best['seq_len']} — but only above c*={plan.critical_concurrency} "
                    f"where mono is already unstable")
                plan.reasoning.append(
                    "Both topologies degrade above c*; "
                    "disagg wins by default, not by advantage")
            else:
                plan.recommendation = "DISAGGREGATE" if sig is not False else "DISAGGREGATE (within noise)"
                r = best.get("contention_ratio", 0)
                t = best.get("threshold", 0)
                if best["p50_cross"]:
                    plan.reasoning.append(
                        f"Contention advantage R={r:.2f} exceeds threshold "
                        f"T={t:.2f} at p50 AND p90 "
                        f"(c≥{best['concurrency']}, s≥{best['seq_len']})")
                else:
                    plan.reasoning.append(
                        f"Contention advantage R_p90={best.get('contention_ratio_p90', 0):.2f} "
                        f"exceeds threshold T_p90={best.get('threshold_p90', 0):.2f} at p90 "
                        f"(c≥{best['concurrency']}, s≥{best['seq_len']}; "
                        f"median {best['p50_delta_pct']:+.0f}%)")
            if sig is False:
                plan.reasoning.append(
                    "Gap is within measurement noise (R × CV_disagg > |R - T|) — "
                    "increase sample size or concurrency to confirm")
        elif p50_crosses:
            best = max(p50_crosses, key=lambda c: abs(c["p50_delta_pct"]))
            plan.recommendation = "MONOLITHIC"
            r = best.get("contention_ratio", 0)
            t = best.get("threshold", 0)
            r_p90 = best.get("contention_ratio_p90", 0)
            t_p90 = best.get("threshold_p90", 0)
            sig = best.get("significant")
            sig_note = "" if sig is not False else " (within noise)"
            plan.reasoning.append(
                f"p50: R={r:.2f} > T={t:.2f} — disagg wins by "
                f"{abs(best['p50_delta_pct']):.0f}%{sig_note} "
                f"(c={best['concurrency']}, s={best['seq_len']})")
            plan.reasoning.append(
                f"p90: R={r_p90:.2f} < T={t_p90:.2f} — mono wins by "
                f"{abs(best['p90_delta_pct']):.0f}%")
            mono_cv = best["mono_cv"]
            disagg_cv = best["disagg_cv"]
            if mono_cv is not None and disagg_cv is not None and mono_cv > 0:
                cv_ratio = disagg_cv / mono_cv
                plan.reasoning.append(
                    f"Disagg variance {cv_ratio:.0f}x higher — "
                    f"R drops at tail while T rises")
            plan.reasoning.append(
                "For SLO compliance (p90/p99): mono wins everywhere")

    elif plan.confidence == "experiment":
        plan.recommendation = "MONOLITHIC"
        plan.reasoning.append(
            "Measured: mono wins at all tested conditions (p50 and p90)")

    else:
        overhead_pct = round(
            (plan.disagg_est_ttft_ms - plan.mono_est_ttft_ms)
            / max(plan.mono_est_ttft_ms, 1) * 100)
        plan.recommendation = "MONOLITHIC (c=1 estimate)"
        dg = plan.predicted_delta_gamma
        if dg <= 0:
            plan.reasoning.append(
                f"At c=1, disagg adds {overhead_pct}% overhead")
            plan.reasoning.append(
                f"Δγ={dg:.2f} ≤ 0 — disagg scales worse than mono at this s")
            if plan.predicted_s_cross < 1e6:
                plan.reasoning.append(
                    f"Need s > {plan.predicted_s_cross:.0f} tokens for disagg to scale better")
        elif plan.predicted_crossover_c > 32:
            plan.reasoning.append(
                f"At c=1, disagg adds {overhead_pct}% overhead")
            plan.reasoning.append(
                f"Predicted crossover at c≈{plan.predicted_crossover_c:.0f} — "
                f"unlikely to reach in practice (Δγ={dg:.2f})")
        elif plan.predicted_crossover_c > 1:
            plan.reasoning.append(
                f"At c=1, disagg adds {overhead_pct}% overhead")
            plan.reasoning.append(
                f"Predicted crossover at c≈{plan.predicted_crossover_c:.0f} — "
                f"may win under load (Δγ={dg:.2f})")
        else:
            # Δγ > 0 and c_cross ≤ 1 implies T(s) ≤ 1 — disagg already wins at c=1
            plan.recommendation = "DISAGGREGATE (c=1 estimate)"
            plan.reasoning.append(
                f"Disagg already faster at c=1 by {-overhead_pct}% (Δγ={dg:.2f})")
        plan.reasoning.append(
            "To measure: ./toolkit/run.sh <cluster> tput-seqlen")

    if plan.confidence == "experiment" and plan.critical_concurrency > 0:
        c_star = plan.critical_concurrency
        cv_entries = {e["concurrency"]: e["median_cv"]
                      for e in plan.mono_cv_by_concurrency}
        below = cv_entries.get(c_star, 0)
        above_c = min((c for c in cv_entries if c > c_star), default=0)
        above = cv_entries.get(above_c, 0) if above_c else 0
        plan.reasoning.append(
            f"Mono stability threshold c*={c_star}: "
            f"CV={below:.0%} at c={c_star}, "
            f"CV={above:.0%} at c={above_c}. "
            f"Below c* mono is deterministic; above c* tail latency explodes")


def print_plan(plan: CapacityPlan):
    is_experiment = plan.confidence == "experiment"
    header = "DEPLOYMENT RECOMMENDATION" if is_experiment else "FEASIBILITY ESTIMATE"

    print(f"\n{'='*60}")
    print(f"  {header}: {plan.model}")
    print(f"{'='*60}")

    if is_experiment:
        print(f"  Data: {plan.measured_conditions}")
        print()

        if plan.mono_cv_by_concurrency:
            if plan.critical_concurrency > 0:
                c_star = plan.critical_concurrency
                print(f"  STABILITY THRESHOLD: c* = {c_star}")
                cv_parts = [f"c={e['concurrency']}: {e['median_cv']:.0%}"
                            for e in plan.mono_cv_by_concurrency]
                print(f"    Mono CV:  {',  '.join(cv_parts)}")
                print("    Below c*: mono is deterministic and always wins")
                print("    Above c*: mono tail latency explodes")
            else:
                max_c = max(e["concurrency"] for e in plan.mono_cv_by_concurrency)
                max_cv = max(e["median_cv"] for e in plan.mono_cv_by_concurrency)
                print(f"  STABILITY THRESHOLD: none detected (c* > {max_c})")
                cv_parts = [f"c={e['concurrency']}: {e['median_cv']:.0%}"
                            for e in plan.mono_cv_by_concurrency]
                print(f"    Mono CV:  {',  '.join(cv_parts)}")
                print(f"    Mono stays deterministic through c={max_c} "
                      f"(CV {max_cv:.0%})")
            print()

        print("  OVERHEAD AT c=1")
        print(f"    Mono:   {plan.mono_est_ttft_ms}ms    "
              f"Disagg: {plan.disagg_est_ttft_ms}ms")
        if plan.mono_est_ttft_ms > 0:
            overhead = plan.disagg_est_ttft_ms - plan.mono_est_ttft_ms
            overhead_pct = overhead / plan.mono_est_ttft_ms * 100
            print(f"    Overhead: {overhead:.0f}ms ({overhead_pct:+.0f}%)")
        if plan.overhead_asymptote > 0:
            asym_pct = round((plan.overhead_asymptote - 1) * 100)
            print(f"    T(∞) = {plan.overhead_asymptote:.2f} — "
                  f"long-prompt floor ({asym_pct}% overhead)")
        if plan.overhead_thresholds:
            parts = [f"s={t['seq_len']}:{t['threshold']:.2f}"
                     for t in plan.overhead_thresholds]
            print(f"    T(s):  {',  '.join(parts)}")
        print()

        if plan.measured_fit_r2 > 0 and plan.predicted_delta_gamma != 0:
            pred_dg = plan.predicted_delta_gamma
            meas_dg = plan.measured_delta_gamma
            err = ((pred_dg - meas_dg) / abs(meas_dg) * 100) if meas_dg != 0 else 0
            print("  CONTENTION MODEL: R(c,s) = c^Δγ(s)")
            print(f"    Δγ(s) = {plan.measured_fit_a:.3f} + "
                  f"{plan.measured_fit_b:.3f}·ln(s)")
            print(f"    Fit R² = {plan.measured_fit_r2:.2f}")
            print(f"    At s={plan.seq_len}: predicted Δγ={pred_dg:.3f}  "
                  f"measured Δγ={meas_dg:.3f}  (error: {err:+.0f}%)")

            pooled = {k: v for k, v in plan.measured_delta_gamma_by_s.items()
                      if isinstance(k, int)}
            per_cfg = {}
            for k, v in plan.measured_delta_gamma_by_s.items():
                if isinstance(k, tuple):
                    cfg, sl = k
                    per_cfg.setdefault(sl, {})[cfg] = v

            if pooled:
                print("    Per-seq_len Δγ (pooled fit → per-point residual):")
                a, b = plan.measured_fit_a, plan.measured_fit_b
                for sl in sorted(pooled):
                    fitted = a + b * math.log(sl)
                    resid = pooled[sl] - fitted
                    cfg_parts = ""
                    if sl in per_cfg and len(per_cfg[sl]) > 1:
                        parts = [f"{c}={v:.3f}" for c, v in
                                 sorted(per_cfg[sl].items())]
                        cfg_parts = f"  [{', '.join(parts)}]"
                    print(f"      s={sl:>5}: Δγ={pooled[sl]:+.3f}  "
                          f"fit={fitted:+.3f}  resid={resid:+.3f}{cfg_parts}")

                if per_cfg:
                    max_spread = 0.0
                    for _sl, cfgs in per_cfg.items():
                        if len(cfgs) > 1:
                            vals = list(cfgs.values())
                            max_spread = max(max_spread,
                                             max(vals) - min(vals))
                    if max_spread > 0.1:
                        print(f"    WARNING: config spread {max_spread:.2f} — "
                              f"1D and 2D may have different scaling")

            max_c = max((e["concurrency"] for e in plan.contention_table), default=16)
            for c in [2, 4, 8, 16]:
                if c > max_c * 2:
                    break
                r_pred = _predict_contention_ratio(pred_dg, c)
                r_meas = _predict_contention_ratio(meas_dg, c)
                print(f"      c={c:>2}: predicted R={r_pred:.2f}  measured R={r_meas:.2f}")
            if plan.predicted_s_cross < 1e6:
                print(f"    s_cross ≈ {plan.predicted_s_cross:.0f} tokens "
                      f"(below this, disagg scales worse)")
            print()

        crosses = sorted(
            [c for c in plan.contention_table if c["p50_cross"] or c["p90_cross"]],
            key=lambda x: (x["concurrency"], x["seq_len"]))
        n_total = len(plan.contention_table)
        n_cross = len(crosses)
        n_no = n_total - n_cross

        print("  CROSSOVER ANALYSIS")
        if crosses:
            for c in crosses:
                sig = c.get("significant")
                sig_str = "" if sig is None else (" sig" if sig else " ~noise")
                p50_w = "CROSS" if c["p50_cross"] else "no"
                p90_w = "CROSS" if c["p90_cross"] else "no"
                mono_cv = c["mono_cv"]
                disagg_cv = c["disagg_cv"]
                cv_str = ""
                if mono_cv is not None and disagg_cv is not None:
                    cv_str = f"  CV: {disagg_cv:.0%} vs {mono_cv:.0%}"
                above_cstar = ""
                if plan.critical_concurrency > 0 and c["concurrency"] > plan.critical_concurrency:
                    above_cstar = " [above c*]"
                print(f"    c={c['concurrency']}, s={c['seq_len']} ({c['config']}){above_cstar}:")
                print(f"      R={c['contention_ratio']:.2f} vs T={c['threshold']:.2f}{sig_str}  "
                      f"p50: {p50_w}  p90: {p90_w}{cv_str}")
            if crosses and all(not c["p90_cross"] for c in crosses):
                print("    All crossovers are p50-only — disagg tail latency")
                print("    is higher under concurrent load")
            if n_no > 0:
                print(f"    No crossover: {n_no}/{n_total} conditions")
        else:
            print(f"    No crossover at any condition ({n_total} tested)")
        print()

    else:
        print(f"  Mono TTFT:   {plan.mono_est_ttft_ms}ms (c=1)")
        print(f"  Disagg TTFT: {plan.disagg_est_ttft_ms}ms (c=1)")
        if plan.mono_est_ttft_ms > 0:
            overhead = plan.disagg_est_ttft_ms - plan.mono_est_ttft_ms
            overhead_pct = overhead / plan.mono_est_ttft_ms * 100
            print(f"  Overhead:    {overhead:.0f}ms ({overhead_pct:+.0f}%)")
            if plan.overhead_asymptote > 0:
                asym_pct = round((plan.overhead_asymptote - 1) * 100)
                if plan.overhead_asymptote > 2.0:
                    print(f"  T(∞) = {plan.overhead_asymptote:.2f} — disagg overhead "
                          f"stays >{asym_pct}% even at infinite prompt length")
                    print("  Disagg is unlikely to help on this hardware/network")
                elif plan.overhead_asymptote < 1.15:
                    print(f"  T(∞) = {plan.overhead_asymptote:.2f} — overhead drops "
                          f"to {asym_pct}% at long prompts")
                    print("  Disagg likely wins under moderate concurrent load")
                else:
                    print(f"  T(∞) = {plan.overhead_asymptote:.2f} — long-prompt "
                          f"floor is {asym_pct}% overhead")
                    print(f"  Disagg needs >{asym_pct}% contention advantage to win")
        if plan.predicted_delta_gamma != 0:
            dg = plan.predicted_delta_gamma
            print("  Contention model: R(c,s) = c^Δγ(s)")
            print(f"    Δγ(s) = -{CONTENTION_SCALE_A:.3f}·(T(∞)-1) + "
                  f"{CONTENTION_SCALE_B:.3f}·ln(s)")
            print(f"    At s={plan.seq_len}: Δγ = {dg:.3f}")
            if dg > 0:
                for c in [2, 4, 8, 16]:
                    r = _predict_contention_ratio(dg, c)
                    print(f"      c={c:>2}: predicted R = {r:.2f}")
                if 1 < plan.predicted_crossover_c <= 64:
                    print(f"    Predicted crossover: c ≈ {plan.predicted_crossover_c:.0f}")
                elif plan.predicted_crossover_c > 64:
                    print("    Predicted crossover: c > 64 (disagg unlikely to help)")
            else:
                print("    Δγ ≤ 0 — disagg scales worse than mono at this s")
            if plan.predicted_s_cross < 1e6:
                print(f"    s_cross ≈ {plan.predicted_s_cross:.0f} tokens "
                      f"(need s > s_cross for disagg advantage)")
            print("    (N=1 calibration — valid for c ≤ ~32)")
        print()

    print(f"  GPU: {plan.mono_total_gpus} mono vs "
          f"{plan.disagg_total_gpus} disagg "
          f"({plan.disagg_prefill_gpus}P+{plan.disagg_decode_gpus}D)")
    print(f"  Cost: ${plan.mono_cost_per_hr:.2f}/hr vs "
          f"${plan.disagg_cost_per_hr:.2f}/hr")
    print()

    print(f"  RECOMMENDATION: {plan.recommendation}")
    for r in plan.reasoning:
        print(f"    - {r}")
    print(f"{'='*60}\n")


def sweep_seq_lens(
    model_id: str,
    gpu_type: str = "t4",
    provider: str = "aws",
    data_dir: str = "",
    seq_lens: list | None = None,
) -> dict:
    """Run plan_capacity across multiple seq_lens to map the phase boundary."""
    ref_plan = plan_capacity(
        model_id, target_throughput=1.0, gpu_type=gpu_type,
        provider=provider, seq_len=128, data_dir=data_dir)

    if seq_lens is None:
        seq_lens = [50, 100, 200, 500, 1000, 2000, 4000]

    t_inf = ref_plan.overhead_asymptote
    s_cross = _predict_s_cross(t_inf) if t_inf > 0 else float('inf')

    results = []
    for s in seq_lens:
        plan = plan_capacity(
            model_id, target_throughput=1.0, gpu_type=gpu_type,
            provider=provider, seq_len=s, data_dir=data_dir)
        mono = plan.mono_est_ttft_ms
        disagg = plan.disagg_est_ttft_ms
        t_s = disagg / mono if mono > 0 else 0
        dg = _predict_delta_gamma(s, t_inf) if t_inf > 0 else 0
        c_cross = _predict_crossover_c(t_s, dg) if dg != 0 else float('inf')
        results.append({
            "seq_len": s,
            "mono_ttft_ms": round(mono),
            "disagg_ttft_ms": round(disagg),
            "T_s": round(t_s, 2),
            "delta_gamma": round(dg, 3),
            "c_cross": round(c_cross, 1) if c_cross < 1e6 else float('inf'),
            "confidence": plan.confidence,
        })

    return {"model": model_id, "gpu_type": gpu_type,
            "t_inf": t_inf, "s_cross": round(s_cross),
            "entries": results}


def print_sweep(sweep: dict):
    """Print the sequence length sweep table."""
    print(f"\n{'='*72}")
    print(f"  SEQUENCE LENGTH SWEEP: {sweep['model']} on {sweep['gpu_type'].upper()}")
    print(f"{'='*72}")
    t_inf = sweep["t_inf"]
    s_cross = sweep["s_cross"]
    if t_inf > 0:
        print(f"  T(∞) = {t_inf:.2f}    s_cross ≈ {s_cross} tokens")
    print()

    print(f"  {'Seq_len':>7} | {'Mono TTFT':>9} | {'Disagg TTFT':>11} | "
          f"{'T(s)':>5} | {'Δγ(s)':>6} | {'c_cross':>7} | Winner")
    print(f"  {'-'*7}-+-{'-'*9}-+-{'-'*11}-+-{'-'*5}-+-"
          f"{'-'*6}-+-{'-'*7}-+-{'-'*20}")

    for e in sweep["entries"]:
        c_cross_str = f"{e['c_cross']:.0f}" if e['c_cross'] < 1000 else "inf"
        if e["delta_gamma"] <= 0:
            winner = "MONO"
        elif e["c_cross"] <= 1:
            winner = "DISAGG"
        elif e["c_cross"] <= 32:
            winner = f"c > {e['c_cross']:.0f} → DISAGG"
        elif e["c_cross"] < 1000:
            winner = f"c > {e['c_cross']:.0f} (unlikely)"
        else:
            winner = "MONO"
        conf = "" if e["confidence"] in ("measured", "experiment") else " *"
        print(f"  {e['seq_len']:>7} | {e['mono_ttft_ms']:>7}ms | "
              f"{e['disagg_ttft_ms']:>9}ms | {e['T_s']:>5.2f} | "
              f"{e['delta_gamma']:>+6.3f} | {c_cross_str:>7} | {winner}{conf}")

    print()
    if s_cross < 1e6:
        print(f"  Phase boundary: disagg scaling advantage begins at s ≈ {s_cross} tokens")
        short = [e for e in sweep["entries"] if e["seq_len"] < s_cross]
        long_win = [e for e in sweep["entries"]
                    if e["delta_gamma"] > 0 and e["c_cross"] <= 32]
        if short:
            print(f"  Short prompts (s < {s_cross}): mono always wins regardless of load")
        if long_win:
            c_range = sorted(set(int(e["c_cross"]) for e in long_win))
            s_range = sorted(e["seq_len"] for e in long_win)
            print(f"  Long prompts (s ≥ {min(s_range)}): disagg wins above "
                  f"c ≈ {min(c_range)}-{max(c_range)}")
    else:
        print("  Disagg scaling advantage does not emerge at any tested seq_len")
    print(f"{'='*72}\n")


WORKLOAD_PROFILES = {
    "chat": [(50, 0.20), (100, 0.35), (200, 0.25), (500, 0.15), (1000, 0.05)],
    "summarization": [(500, 0.10), (1000, 0.30), (2000, 0.35), (4000, 0.25)],
    "rag": [(200, 0.15), (500, 0.35), (1000, 0.35), (2000, 0.15)],
    "code": [(100, 0.20), (200, 0.30), (500, 0.30), (1000, 0.15), (2000, 0.05)],
}


def parse_workload(spec: str) -> tuple:
    """Parse workload spec: either a profile name or 'tokens:weight,...'."""
    if spec in WORKLOAD_PROFILES:
        return spec, WORKLOAD_PROFILES[spec]
    pairs = []
    for part in spec.split(","):
        tokens_s, weight_s = part.strip().split(":")
        pairs.append((int(tokens_s), float(weight_s)))
    total = sum(w for _, w in pairs)
    if total > 0:
        pairs = [(s, w / total) for s, w in pairs]
    return "custom", pairs


@dataclass
class WorkloadResult:
    model: str
    gpu_type: str
    workload_name: str
    workload_dist: list
    buckets: list
    weighted_mono_ttft: float
    weighted_disagg_ttft: float
    weighted_overhead_pct: float
    s_cross: float
    frac_above_scross: float
    concurrency_analysis: list
    recommendation: str
    reasoning: list = field(default_factory=list)


def analyze_workload(
    model_id: str,
    workload_dist: list,
    workload_name: str = "custom",
    gpu_type: str = "t4",
    provider: str = "aws",
    data_dir: str = "",
) -> WorkloadResult:
    """Analyze a workload distribution across sequence lengths."""
    ref_plan = plan_capacity(
        model_id, target_throughput=1.0, gpu_type=gpu_type,
        provider=provider, seq_len=128, data_dir=data_dir)
    t_inf = ref_plan.overhead_asymptote
    s_cross = _predict_s_cross(t_inf) if t_inf > 0 else float('inf')

    buckets = []
    w_mono = 0.0
    w_disagg = 0.0
    frac_above = 0.0

    for s, weight in workload_dist:
        plan = plan_capacity(
            model_id, target_throughput=1.0, gpu_type=gpu_type,
            provider=provider, seq_len=s, data_dir=data_dir)
        mono = plan.mono_est_ttft_ms
        disagg = plan.disagg_est_ttft_ms
        t_s = disagg / mono if mono > 0 else 0
        dg = _predict_delta_gamma(s, t_inf) if t_inf > 0 else 0
        c_cross = _predict_crossover_c(t_s, dg) if dg > 0 else float('inf')
        buckets.append({
            "seq_len": s, "weight": weight,
            "mono_ttft_ms": round(mono), "disagg_ttft_ms": round(disagg),
            "T_s": round(t_s, 2), "delta_gamma": round(dg, 3),
            "c_cross": round(c_cross, 1) if c_cross < 1e6 else float('inf'),
            "confidence": plan.confidence,
        })
        w_mono += weight * mono
        w_disagg += weight * disagg
        if s > s_cross:
            frac_above += weight

    w_overhead_pct = (w_disagg - w_mono) / w_mono * 100 if w_mono > 0 else 0

    conc_analysis = []
    for c in [2, 4, 8, 16, 32]:
        mono_at_c = 0.0
        frac_wins = 0.0
        for b in buckets:
            r = _predict_contention_ratio(b["delta_gamma"], c)
            mono_at_c += b["weight"] * b["mono_ttft_ms"] * r
            if r > b["T_s"]:
                frac_wins += b["weight"]
        advantage_pct = (mono_at_c - w_disagg) / w_disagg * 100 if w_disagg > 0 else 0
        verdict = "DISAGG" if mono_at_c > w_disagg else "MONO"
        conc_analysis.append({
            "concurrency": c,
            "mono_ttft_at_c": round(mono_at_c),
            "disagg_ttft": round(w_disagg),
            "advantage_pct": round(advantage_pct),
            "frac_disagg_wins": round(frac_wins, 2),
            "verdict": verdict,
        })

    first_disagg = next((ca for ca in conc_analysis if ca["verdict"] == "DISAGG"), None)
    if first_disagg:
        recommendation = (f"DISAGGREGATE at c ≥ {first_disagg['concurrency']} "
                          f"for {workload_name} workloads")
    else:
        recommendation = f"MONOLITHIC for {workload_name} workloads"

    result = WorkloadResult(
        model=model_id, gpu_type=gpu_type,
        workload_name=workload_name, workload_dist=workload_dist,
        buckets=buckets,
        weighted_mono_ttft=round(w_mono),
        weighted_disagg_ttft=round(w_disagg),
        weighted_overhead_pct=round(w_overhead_pct),
        s_cross=round(s_cross) if s_cross < 1e6 else float('inf'),
        frac_above_scross=round(frac_above, 2),
        concurrency_analysis=conc_analysis,
        recommendation=recommendation,
    )

    result.reasoning.append(
        f"{round(frac_above * 100)}% of requests (s > {round(s_cross)}) "
        f"benefit from disagg scaling" if s_cross < 1e6
        else "No requests benefit from disagg scaling at any s")
    if first_disagg:
        result.reasoning.append(
            f"At c={first_disagg['concurrency']}, mono contention exceeds "
            f"disagg overhead ({first_disagg['advantage_pct']:+d}% differential)")
    else:
        worst = conc_analysis[-1]
        result.reasoning.append(
            f"Even at c={worst['concurrency']}, mono contention does not "
            f"overcome disagg overhead ({worst['advantage_pct']:+d}%)")

    return result


def print_workload(result: WorkloadResult):
    """Print workload analysis results."""
    dist_str = ", ".join(f"{s}:{w:.0%}" for s, w in result.workload_dist)
    print(f"\n{'='*72}")
    print(f"  WORKLOAD ANALYSIS: {result.model} on {result.gpu_type.upper()}")
    print(f"{'='*72}")
    print(f"  Workload: {result.workload_name} ({dist_str})")
    if result.s_cross < 1e6:
        print(f"  s_cross ≈ {result.s_cross:.0f} tokens    "
              f"{result.frac_above_scross:.0%} of traffic above s_cross")
    print()

    print("  Per-bucket breakdown:")
    for b in result.buckets:
        c_str = (f"c>{b['c_cross']:.0f}" if 1 < b["c_cross"] < 1000
                 else "DISAGG" if b["c_cross"] <= 1
                 else "MONO")
        print(f"    s={b['seq_len']:>5} ({b['weight']:>4.0%}): "
              f"mono {b['mono_ttft_ms']:>5}ms  disagg {b['disagg_ttft_ms']:>5}ms  "
              f"T={b['T_s']:.2f}  Δγ={b['delta_gamma']:+.2f}  → {c_str}")
    print()

    print("  Workload aggregate (c=1):")
    print(f"    Weighted mono TTFT:   {result.weighted_mono_ttft:>5}ms")
    print(f"    Weighted disagg TTFT: {result.weighted_disagg_ttft:>5}ms")
    print(f"    Weighted overhead:    {result.weighted_overhead_pct:+.0f}%")
    print()

    print("  Concurrency scaling (mono contention vs disagg baseline):")
    for ca in result.concurrency_analysis:
        c = ca["concurrency"]
        print(f"    c={c:>2}: {ca['advantage_pct']:+d}% differential contention  "
              f"({ca['frac_disagg_wins']:.0%} of traffic benefits)  "
              f"→ {ca['verdict']}")
    print()

    print(f"  RECOMMENDATION: {result.recommendation}")
    for r in result.reasoning:
        print(f"    - {r}")
    print(f"{'='*72}\n")


def save_plan(plan: CapacityPlan, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": plan.model, "gpu_type": plan.gpu_type,
        "target_throughput": plan.target_throughput, "target_ttft_ms": plan.target_ttft_ms,
        "mono": {"gpus": plan.mono_total_gpus, "ttft_ms": plan.mono_est_ttft_ms,
                 "throughput": plan.mono_est_throughput, "cost_hr": plan.mono_cost_per_hr},
        "disagg": {"gpus": plan.disagg_total_gpus, "ttft_ms": plan.disagg_est_ttft_ms,
                   "throughput": plan.disagg_est_throughput, "cost_hr": plan.disagg_cost_per_hr},
        "recommendation": plan.recommendation, "confidence": plan.confidence,
        "reasoning": plan.reasoning, "crossovers": plan.crossovers,
        "contention_table": plan.contention_table,
        "overhead_asymptote": plan.overhead_asymptote,
        "overhead_thresholds": plan.overhead_thresholds,
        "measured_conditions": plan.measured_conditions,
        "critical_concurrency": plan.critical_concurrency,
        "mono_cv_by_concurrency": plan.mono_cv_by_concurrency,
        "seq_len": plan.seq_len,
        "predicted_delta_gamma": plan.predicted_delta_gamma,
        "predicted_crossover_c": plan.predicted_crossover_c,
        "predicted_s_cross": plan.predicted_s_cross,
        "measured_delta_gamma": plan.measured_delta_gamma,
        "measured_delta_gamma_by_s": {
            str(k): v for k, v in plan.measured_delta_gamma_by_s.items()},
        "measured_fit_a": plan.measured_fit_a,
        "measured_fit_b": plan.measured_fit_b,
        "measured_fit_r2": plan.measured_fit_r2,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU capacity planner for disaggregated inference")
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model ID")
    parser.add_argument("--gpu-type", default="t4", help="GPU type (default: t4)")
    parser.add_argument("--throughput", type=float, default=1.0, help="Target throughput in req/s (default: 1.0)")
    parser.add_argument("--ttft-slo", type=float, default=500, help="Target TTFT in ms (default: 500)")
    parser.add_argument("--provider", default="aws", help="Cloud provider for pricing (default: aws)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Prompt sequence length in tokens (default: 128)")
    parser.add_argument("--data-dir", default="",
                        help="Path to experiment data dir (enables data-backed recommendations)")
    parser.add_argument("--save", default="", help="Path to save plan JSON")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep across sequence lengths to show phase boundary")
    parser.add_argument("--workload",
                        help="Workload profile: chat|summarization|rag|code or 'tokens:weight,...'")
    args = parser.parse_args()

    if args.sweep and args.workload:
        parser.error("--sweep and --workload are mutually exclusive")

    if args.sweep:
        sweep = sweep_seq_lens(
            args.model, gpu_type=args.gpu_type,
            provider=args.provider, data_dir=args.data_dir)
        print_sweep(sweep)
    elif args.workload:
        name, dist = parse_workload(args.workload)
        result = analyze_workload(
            args.model, workload_dist=dist, workload_name=name,
            gpu_type=args.gpu_type, provider=args.provider,
            data_dir=args.data_dir)
        print_workload(result)
    else:
        plan = plan_capacity(
            args.model,
            target_throughput=args.throughput,
            target_ttft_ms=args.ttft_slo,
            gpu_type=args.gpu_type,
            provider=args.provider,
            seq_len=args.seq_len,
            data_dir=args.data_dir,
        )
        print_plan(plan)

        if args.save:
            save_plan(plan, Path(args.save))
            print(f"  Plan saved to {args.save}")
