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
        "sidecar_ms": 2, "nixl_ms": 17,
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
        "sidecar_ms": 21, "nixl_ms": 25,
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
        "sidecar_ms": 25, "nixl_ms": 267,
    },
    ("microsoft/Phi-3-mini-4k-instruct", "t4"): {
        "params_b": 3.8, "is_moe": False,
        "mono_ttft_ms": 770, "disagg_ttft_ms": 1022,
        "mono_ttft_base_ms": 560, "mono_ttft_rate": 1.70,
        "disagg_ttft_base_ms": 636, "disagg_ttft_rate": 3.25,
        "ref_seq_len": 100,
        "mono_throughput": 1.30, "disagg_throughput": 0.98,
        "overhead_ms": 252, "overhead_pct": 32.7,
        "sidecar_ms": 0, "nixl_ms": 252,
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

# Scaling constants from exp5 regression (Pearson r=0.987)
SCALING_SIDECAR_MS = 12

# NIXL transfer model: T_transfer = protocol_ms + (kv_bytes * seq_len) / eff_bw
# Measured via exp5b direct NIXL prometheus scraping on T4 cluster (R²=0.999).
# protocol_ms is fixed overhead (NIXL handshake, buffer setup) — model-independent.
# eff_bw is effective NIC throughput in GB/s — hardware-dependent.
NIXL_PROTOCOL_MS = 4.3         # regression intercept (exp5b, 180 points, Phi-3 on T4)
NIXL_EFF_BW_GBS = 0.299        # regression slope -> effective bandwidth (10 Gbps OVN/TCP)
MOE_NIXL_CORRECTION = 1.0      # MoE uses dense attention → KV transfer is identical to dense


@dataclass
class CapacityPlan:
    model: str
    gpu_type: str
    target_throughput: float
    target_ttft_ms: float

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
    measured_conditions: str = ""
    overhead_thresholds: list = field(default_factory=list)


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

    threshold_by_s = {t["seq_len"]: t["threshold"] for t in plan.overhead_thresholds}

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
        threshold = threshold_by_s.get(pt, 0)

        p50_delta = (dg_stats["median"] - bl_cn["median"]) / bl_cn["median"] * 100
        p90_delta = (dg_stats["p90"] - bl_cn["p90"]) / bl_cn["p90"] * 100
        p50_cross = dg_stats["median"] < bl_cn["median"]
        p90_cross = dg_stats["p90"] < bl_cn["p90"]

        alpha_mono_p90 = bl_cn["p90"] / bl_c1_pt["p90"] if bl_c1_pt["p90"] > 0 else 0
        alpha_disagg_p90 = dg_stats["p90"] / dg_c1_pt["p90"] if dg_c1_pt["p90"] > 0 else 0
        contention_ratio_p90 = alpha_mono_p90 / alpha_disagg_p90 if alpha_disagg_p90 > 0 else 0
        threshold_p90_val = bl_cn["p90"] and dg_c1_pt["p90"] and bl_c1_pt["p90"]
        threshold_p90 = (dg_c1_pt["p90"] / bl_c1_pt["p90"]) if threshold_p90_val else threshold

        if p50_cross or p90_cross:
            mono_cv = bl_cn.get("cv")
            disagg_cv = dg_stats.get("cv")
            plan.crossovers.append({
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
            })

    if max(measured_concs) <= 1:
        plan.reasoning.append(
            "Only c=1 measured — crossovers require concurrent load. "
            "Re-run with concurrency > 1 to detect crossover")

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
        disagg_ttft = mono_ttft + SCALING_SIDECAR_MS + nixl_ms

        plan.mono_est_ttft_ms = round(max(mono_ttft, 10))
        plan.mono_est_throughput = round(max(mono_rps * bw_speedup * tp * tp_eff, 0.01), 2)
        plan.disagg_est_ttft_ms = round(max(disagg_ttft, 10))

        kv_kb = kv_bytes / 1024
        data_ms = nixl_ms - NIXL_PROTOCOL_MS
        plan.reasoning.append(
            f"KV cache: {kv_kb:.0f} KB/token ({profile.num_kv_heads} KV heads) x {seq_len} tokens, "
            f"NIXL: {NIXL_PROTOCOL_MS:.0f}ms protocol + {data_ms:.0f}ms data + "
            f"sidecar: {SCALING_SIDECAR_MS}ms")
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
            plan.recommendation = "DISAGGREGATE"
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
        elif p50_crosses:
            best = max(p50_crosses, key=lambda c: abs(c["p50_delta_pct"]))
            plan.recommendation = "MONOLITHIC"
            r = best.get("contention_ratio", 0)
            t = best.get("threshold", 0)
            r_p90 = best.get("contention_ratio_p90", 0)
            t_p90 = best.get("threshold_p90", 0)
            plan.reasoning.append(
                f"p50: R={r:.2f} > T={t:.2f} — disagg wins by "
                f"{abs(best['p50_delta_pct']):.0f}% "
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
        plan.reasoning.append(
            f"At c=1, mono always faster "
            f"(disagg adds {overhead_pct}% overhead)")
        plan.reasoning.append(
            f"Disagg needs >{overhead_pct}% contention advantage under load — "
            f"run experiments to measure")
        plan.reasoning.append(
            "To measure: ./toolkit/run.sh <cluster> tput-seqlen")


def print_plan(plan: CapacityPlan):
    is_experiment = plan.confidence == "experiment"
    header = "DEPLOYMENT RECOMMENDATION" if is_experiment else "FEASIBILITY ESTIMATE"

    print(f"\n{'='*60}")
    print(f"  {header}: {plan.model}")
    print(f"{'='*60}")

    if is_experiment:
        print(f"  Data: {plan.measured_conditions}")
        print()

        if plan.overhead_thresholds:
            print("  Overhead threshold T(s) — disagg needs this much contention advantage:")
            parts = [f"s={t['seq_len']}:{t['threshold']:.2f} ({t['overhead_pct']}%)"
                     for t in plan.overhead_thresholds]
            for i in range(0, len(parts), 3):
                print(f"    {',  '.join(parts[i:i+3])}")
            print()

        p50_crosses = [c for c in plan.crossovers if c["p50_cross"]]
        p90_crosses = [c for c in plan.crossovers if c["p90_cross"]]

        if p50_crosses or p90_crosses:
            print("  Contention analysis (R = α_mono/α_disagg, crosses when R > T):")
            for c in sorted(plan.crossovers, key=lambda x: (x["concurrency"], x["seq_len"])):
                p50_w = "CROSS" if c["p50_cross"] else "no"
                p90_w = "CROSS" if c["p90_cross"] else "no"
                print(f"    c={c['concurrency']}, s={c['seq_len']} ({c['config']}):")
                print(f"      α_mono={c['alpha_mono']:.2f}  "
                      f"α_disagg={c['alpha_disagg']:.2f}  "
                      f"R={c['contention_ratio']:.2f} vs T={c['threshold']:.2f}  "
                      f"p50: {p50_w}")
                print(f"      R_p90={c['contention_ratio_p90']:.2f} vs "
                      f"T_p90={c['threshold_p90']:.2f}  p90: {p90_w}")
                mono_cv = c["mono_cv"]
                disagg_cv = c["disagg_cv"]
                if mono_cv is not None and disagg_cv is not None and mono_cv > 0:
                    cv_ratio = disagg_cv / mono_cv
                    print(f"      Variance: disagg {cv_ratio:.0f}x higher "
                          f"(CV {disagg_cv*100:.0f}% vs {mono_cv*100:.0f}%)")
        else:
            print("  No crossover detected: mono wins at all conditions.")
        print()

    print(f"  Mono TTFT:   {plan.mono_est_ttft_ms}ms (c=1)")
    print(f"  Disagg TTFT: {plan.disagg_est_ttft_ms}ms (c=1)")
    if plan.mono_est_ttft_ms > 0:
        overhead = plan.disagg_est_ttft_ms - plan.mono_est_ttft_ms
        overhead_pct = overhead / plan.mono_est_ttft_ms * 100
        print(f"  Overhead:    {overhead:.0f}ms ({overhead_pct:+.0f}%)")
        if not is_experiment:
            print(f"  Threshold:   disagg needs >{overhead_pct:.0f}% contention "
                  f"advantage to justify overhead")
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
        "overhead_thresholds": plan.overhead_thresholds,
        "measured_conditions": plan.measured_conditions,
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
    args = parser.parse_args()

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
