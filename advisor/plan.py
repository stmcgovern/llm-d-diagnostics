"""
GPU capacity planner for disaggregated inference.

Answers "how many GPUs do I need?" given a model, target throughput,
TTFT SLO, and GPU type. Uses measured data when available, otherwise
extrapolates using the scaling model and known GPU performance ratios.

Integrates with ``toolkit/scaling_model.py`` for the hardware database
(MODELS, GPUS) and keeps ``fetch_model_profile`` locally since it only
talks to HuggingFace (no cluster dependency).
"""

import argparse
import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .pricing import GPU_TFLOPS_FP16, GPU_VRAM_GB, get_price, get_cheapest
    from ._cluster import SCALING_GPUS
except ImportError:
    from pricing import GPU_TFLOPS_FP16, GPU_VRAM_GB, get_price, get_cheapest
    from _cluster import SCALING_GPUS


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
}

# Scaling model from exp5 regression (Pearson r=0.987 on NIXL transfer vs seq length)
SCALING_NIXL_BASE_MS = 20
SCALING_SIDECAR_MS = 12
MOE_NIXL_CORRECTION = 2.73


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
    confidence: str = "low"
    reasoning: list = field(default_factory=list)


def plan_capacity(
    model_id: str,
    target_throughput: float,
    target_ttft_ms: float = 500,
    gpu_type: str = "t4",
    provider: str = "aws",
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
    gpu_tflops = _hw.get("fp16_tflops", GPU_TFLOPS_FP16.get(gpu_type, 65))

    plan = CapacityPlan(
        model=model_id, gpu_type=gpu_type,
        target_throughput=target_throughput,
        target_ttft_ms=target_ttft_ms,
    )

    weight_gb = profile.weight_gb or (profile.num_params * 2 / (1024**3))

    max_tp_per_node = 8
    tp = max(1, math.ceil(weight_gb / (gpu_vram * 0.8)))
    pp = 1
    if tp > max_tp_per_node:
        pp = math.ceil(tp / max_tp_per_node)
        tp = max_tp_per_node
    plan.mono_gpus_per_instance = tp * pp

    if pp > 1:
        plan.reasoning.append(
            f"Model requires pipeline parallelism (PP={pp}, TP={tp}). "
            f"Each instance uses {tp * pp} GPUs. PP adds ~30% TTFT overhead per stage."
        )

    measured = MEASURED_BASELINES.get((model_id, gpu_type))
    if measured:
        plan.confidence = "measured"
        _plan_from_measured(plan, measured, target_throughput, price, tp)
    else:
        _plan_from_extrapolation(plan, profile, gpu_type, gpu_tflops, target_throughput, price, tp)

    _generate_recommendation(plan)

    return plan


def _plan_from_measured(plan, measured, target_throughput, price, tp):
    """Plan using actual measured data from our experiments."""
    mono_rps = measured["mono_throughput"]
    disagg_rps = measured["disagg_throughput"]

    plan.mono_instances = max(1, math.ceil(target_throughput / mono_rps)) if mono_rps > 0 else 1
    plan.mono_total_gpus = plan.mono_instances * tp
    plan.mono_est_ttft_ms = measured["mono_ttft_ms"]
    plan.mono_est_throughput = mono_rps * plan.mono_instances
    plan.mono_cost_per_hr = plan.mono_total_gpus * price

    disagg_instances = max(1, math.ceil(target_throughput / disagg_rps)) if disagg_rps > 0 else 1
    plan.disagg_prefill_gpus = max(1, disagg_instances // 3 + 1) * tp
    plan.disagg_decode_gpus = disagg_instances * tp
    plan.disagg_total_gpus = plan.disagg_prefill_gpus + plan.disagg_decode_gpus
    plan.disagg_est_ttft_ms = measured["disagg_ttft_ms"]
    plan.disagg_est_throughput = disagg_rps * disagg_instances
    plan.disagg_cost_per_hr = plan.disagg_total_gpus * price

    plan.reasoning.append(f"Based on measured data: mono {mono_rps:.2f} req/s, disagg {disagg_rps:.2f} req/s")


def _find_nearest_baselines(params_b, gpu_type, is_moe=False):
    """Find the two nearest measured baselines by parameter count for interpolation."""
    candidates = []
    for (model_id, gtype), data in MEASURED_BASELINES.items():
        if gtype != gpu_type:
            continue
        if is_moe != data.get("is_moe", False):
            continue
        candidates.append((abs(data["params_b"] - params_b), data))
    candidates.sort(key=lambda x: x[0])
    return [c[1] for c in candidates[:2]]


def _plan_from_extrapolation(plan, profile, gpu_type, gpu_tflops, target_throughput, price, tp):
    """Plan by interpolating from the 8-model dataset + scaling model."""

    params_b = profile.num_params / 1e9 if profile.num_params else 1
    is_moe = profile.is_moe
    t4_tflops = GPU_TFLOPS_FP16["t4"]
    gpu_speedup = gpu_tflops / t4_tflops

    nearest = _find_nearest_baselines(params_b, gpu_type, is_moe)
    if not nearest:
        nearest = _find_nearest_baselines(params_b, "t4", is_moe)

    if nearest:
        if len(nearest) >= 2:
            lo, hi = nearest[0], nearest[1]
            lo_b, hi_b = lo["params_b"], hi["params_b"]
            if hi_b != lo_b:
                t = (params_b - lo_b) / (hi_b - lo_b)
                t = max(0, min(2, t))
            else:
                t = 0
            mono_ttft = lo["mono_ttft_ms"] + t * (hi["mono_ttft_ms"] - lo["mono_ttft_ms"])
            disagg_ttft = lo["disagg_ttft_ms"] + t * (hi["disagg_ttft_ms"] - lo["disagg_ttft_ms"])
            mono_rps = lo["mono_throughput"] + t * (hi["mono_throughput"] - lo["mono_throughput"])
        else:
            ref = nearest[0]
            scale = params_b / ref["params_b"] if ref["params_b"] > 0 else 1
            mono_ttft = ref["mono_ttft_ms"] * scale
            disagg_ttft = ref["disagg_ttft_ms"] * scale
            mono_rps = ref["mono_throughput"] / max(scale, 0.5)

        if gpu_type != "t4":
            mono_ttft = mono_ttft / gpu_speedup
            disagg_ttft = disagg_ttft / gpu_speedup

        plan.mono_est_ttft_ms = round(max(mono_ttft, 10))
        plan.mono_est_throughput = round(max(mono_rps * gpu_speedup, 0.01), 2)
        plan.disagg_est_ttft_ms = round(max(disagg_ttft, 10))

        plan.reasoning.append(
            f"Interpolated from 8-model dataset (nearest: {nearest[0]['params_b']:.1f}B"
            + (f", {nearest[1]['params_b']:.1f}B" if len(nearest) >= 2 else "")
            + f") on {gpu_type}")
    else:
        plan.mono_est_ttft_ms = round(200 * params_b / gpu_speedup)
        plan.mono_est_throughput = round(1000 / max(plan.mono_est_ttft_ms, 1) * 0.7, 2)
        nixl_ms = SCALING_NIXL_BASE_MS * (MOE_NIXL_CORRECTION if is_moe else 1)
        overhead_ms = SCALING_SIDECAR_MS + nixl_ms
        plan.disagg_est_ttft_ms = round(plan.mono_est_ttft_ms + overhead_ms)
        plan.reasoning.append("Pure heuristic (no matching baseline found); run experiments to validate")

    plan.mono_instances = max(1, math.ceil(target_throughput / plan.mono_est_throughput)) if plan.mono_est_throughput > 0 else 1
    plan.mono_total_gpus = plan.mono_instances * tp
    plan.mono_cost_per_hr = plan.mono_total_gpus * price

    plan.disagg_prefill_gpus = max(1, plan.mono_instances // 2) * tp
    plan.disagg_decode_gpus = plan.mono_instances * tp
    plan.disagg_total_gpus = plan.disagg_prefill_gpus + plan.disagg_decode_gpus
    plan.disagg_est_throughput = round(plan.mono_est_throughput * plan.mono_instances, 2)
    plan.disagg_cost_per_hr = plan.disagg_total_gpus * price

    if plan.confidence != "measured":
        plan.confidence = "interpolated" if nearest else "low"
        plan.reasoning.append("Run experiments for empirical validation on your actual hardware")


def _generate_recommendation(plan):
    """Generate YES/NO recommendation from the plan."""
    mono_cost_eff = plan.mono_est_throughput / max(plan.mono_total_gpus, 1)
    disagg_cost_eff = plan.disagg_est_throughput / max(plan.disagg_total_gpus, 1)

    if plan.disagg_total_gpus <= plan.mono_total_gpus and plan.disagg_est_ttft_ms < plan.mono_est_ttft_ms:
        plan.recommendation = "DISAGGREGATE"
        plan.reasoning.append("Same or fewer GPUs with lower latency")
    elif plan.disagg_est_ttft_ms < plan.target_ttft_ms * 0.7 and plan.mono_est_ttft_ms > plan.target_ttft_ms * 0.9:
        plan.recommendation = "DISAGGREGATE"
        plan.reasoning.append("Disagg has more SLO headroom")
    elif plan.mono_total_gpus < plan.disagg_total_gpus and mono_cost_eff > disagg_cost_eff * 1.2:
        plan.recommendation = "MONOLITHIC"
        plan.reasoning.append("Mono uses fewer GPUs with better per-GPU efficiency")
    else:
        plan.recommendation = "RUN EXPERIMENTS TO DECIDE"
        plan.reasoning.append("Close call -- empirical benchmark needed")
        plan.reasoning.append(f"Run: ./toolkit/run.sh <cluster> characterize")


def print_plan(plan: CapacityPlan):
    print(f"\n{'='*60}")
    print(f"  CAPACITY PLAN: {plan.model}")
    print(f"{'='*60}")
    print(f"  Target: {plan.target_throughput} req/s, TTFT < {plan.target_ttft_ms}ms")
    print(f"  GPU: {plan.gpu_type.upper()}    Confidence: {plan.confidence}")
    print()
    print(f"  Option A: MONOLITHIC")
    print(f"    TP={plan.mono_gpus_per_instance}, Instances={plan.mono_instances}")
    print(f"    GPUs: {plan.mono_total_gpus}    Cost: ${plan.mono_cost_per_hr:.2f}/hr")
    print(f"    Est. TTFT: {plan.mono_est_ttft_ms}ms    Throughput: {plan.mono_est_throughput:.2f} req/s")
    slo_status = "MEETS SLO" if plan.mono_est_ttft_ms < plan.target_ttft_ms else "EXCEEDS SLO"
    print(f"    SLO: {slo_status}")
    print()
    print(f"  Option B: DISAGGREGATED ({plan.disagg_prefill_gpus}P + {plan.disagg_decode_gpus}D)")
    print(f"    GPUs: {plan.disagg_total_gpus}    Cost: ${plan.disagg_cost_per_hr:.2f}/hr")
    print(f"    Est. TTFT: {plan.disagg_est_ttft_ms}ms    Throughput: {plan.disagg_est_throughput:.2f} req/s")
    slo_status = "MEETS SLO" if plan.disagg_est_ttft_ms < plan.target_ttft_ms else "EXCEEDS SLO"
    print(f"    SLO: {slo_status}")
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
        "reasoning": plan.reasoning,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU capacity planner for disaggregated inference")
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model ID")
    parser.add_argument("--gpu-type", default="t4", help="GPU type (default: t4)")
    parser.add_argument("--throughput", type=float, default=1.0, help="Target throughput in req/s (default: 1.0)")
    parser.add_argument("--ttft-slo", type=float, default=500, help="Target TTFT in ms (default: 500)")
    parser.add_argument("--provider", default="aws", help="Cloud provider for pricing (default: aws)")
    parser.add_argument("--save", default="", help="Path to save plan JSON")
    args = parser.parse_args()

    plan = plan_capacity(
        args.model,
        target_throughput=args.throughput,
        target_ttft_ms=args.ttft_slo,
        gpu_type=args.gpu_type,
        provider=args.provider,
    )
    print_plan(plan)

    if args.save:
        save_plan(plan, Path(args.save))
        print(f"  Plan saved to {args.save}")
