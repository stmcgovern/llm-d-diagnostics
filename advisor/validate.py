"""
Validate advisor predictions against measured experiment data.

Loads exp11 and exp14 results, runs the advisor's plan_capacity() at each
measured prompt length, and reports prediction accuracy. Exposes where
the advisor's model is accurate and where it breaks down.

Usage:
    python3 advisor/validate.py <data_dir> [--gpu-type t4]

Example:
    python3 advisor/validate.py clusters/my-cluster/data --gpu-type t4
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

_advisor_dir = os.path.dirname(__file__)
_toolkit_dir = os.path.join(os.path.dirname(_advisor_dir), "toolkit")
sys.path.insert(0, _advisor_dir)
sys.path.insert(0, _toolkit_dir)

from analyze import get_status, load_csv, safe_float, safe_int, stats  # noqa: E402

from plan import (  # noqa: E402
    MEASURED_BASELINES,
    _estimate_kv_bytes,
    _estimate_nixl_ms,
    _predict_contention_ratio,
    _predict_delta_gamma,
    fetch_model_profile,
    plan_capacity,
)


@dataclass
class ValidationResult:
    metric: str
    seq_len: int
    concurrency: int
    predicted: float
    measured: float
    error_pct: float
    grade: str
    unit: str = "ms"


def _grade(error_pct: float) -> str:
    abs_err = abs(error_pct)
    if abs_err < 20:
        return "GOOD"
    if abs_err < 50:
        return "FAIR"
    if abs_err < 100:
        return "POOR"
    return "WRONG"


def load_run_info(data_dir: str) -> dict:
    path = os.path.join(data_dir, "run-info.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _grouped_stats(path, key_fn):
    """Load CSV, group rows by key_fn, return {key: stats(ttft_ms)}."""
    rows = load_csv(path)
    groups = defaultdict(list)
    for r in rows:
        if get_status(r) != 200:
            continue
        groups[key_fn(r)].append(safe_float(r["ttft_ms"]))
    return {k: stats(v) for k, v in groups.items()}


def load_exp_baselines(data_dir: str) -> dict:
    """Extract measured TTFT and overhead from exp11 + exp14 data.

    Returns dict keyed by metric type with measured values per condition.
    """
    return {
        "exp11": _grouped_stats(
            os.path.join(data_dir, "exp11-results.csv"),
            lambda r: (r["config"], safe_int(r["concurrency"]),
                       safe_int(r["prompt_tokens_target"]))),
        "exp14": _grouped_stats(
            os.path.join(data_dir, "exp14-results.csv"),
            lambda r: (r["config"], safe_int(r["prompt_tokens_target"]))),
    }


def get_predictions(model: str, gpu_type: str, seq_lens: list) -> dict:
    """Run advisor predictions at each sequence length."""
    preds = {}
    for sl in seq_lens:
        plan = plan_capacity(
            model, target_throughput=1.0, target_ttft_ms=99999,
            gpu_type=gpu_type, seq_len=sl,
        )
        profile = fetch_model_profile(model)
        kv_bytes = _estimate_kv_bytes(profile)
        nixl_ms = _estimate_nixl_ms(kv_bytes, sl, gpu_type, profile.is_moe)

        overhead_asym = (plan.overhead_asymptote if plan.overhead_asymptote > 0
                         else 1 + nixl_ms / max(plan.mono_est_ttft_ms, 1))
        preds[sl] = {
            "mono_ttft_ms": plan.mono_est_ttft_ms,
            "disagg_ttft_ms": plan.disagg_est_ttft_ms,
            "confidence": plan.confidence,
            "nixl_ms": nixl_ms,
            "predicted_delta_gamma": _predict_delta_gamma(sl, overhead_asym),
        }
    return preds


def compare(predictions: dict, measurements: dict) -> list:
    """Compare predictions against measurements, return ValidationResults."""
    results = []

    exp11 = measurements.get("exp11", {})
    exp14 = measurements.get("exp14", {})

    seq_lens = sorted(predictions.keys())

    for sl in seq_lens:
        pred = predictions[sl]

        bl_key = ("BASELINE", 1, sl)
        if bl_key in exp11:
            measured = exp11[bl_key]["median"]
            predicted = pred["mono_ttft_ms"]
            err = (predicted - measured) / measured * 100 if measured > 0 else 0
            results.append(ValidationResult(
                "mono_ttft", sl, 1, predicted, measured, err, _grade(err)))

        for disagg_cfg in ["DISAGG-1D", "DISAGG-2D"]:
            dg_key = (disagg_cfg, 1, sl)
            if dg_key in exp11:
                measured = exp11[dg_key]["median"]
                predicted = pred["disagg_ttft_ms"]
                err = (predicted - measured) / measured * 100 if measured > 0 else 0
                results.append(ValidationResult(
                    f"disagg_ttft({disagg_cfg})", sl, 1,
                    predicted, measured, err, _grade(err)))

        cfg_c = next((k for k in exp14 if k[0].startswith("C-") and k[1] == sl), None)
        cfg_d = next((k for k in exp14 if k[0].startswith("D-") and k[1] == sl), None)

        if cfg_d and cfg_c:
            measured_nixl = exp14[cfg_d]["median"] - exp14[cfg_c]["median"]
            predicted_nixl = pred["nixl_ms"]
            err = ((predicted_nixl - measured_nixl) / max(abs(measured_nixl), 1)
                   * 100)
            conc = 8
            results.append(ValidationResult(
                "nixl_ms", sl, conc, predicted_nixl,
                measured_nixl, err, _grade(err)))

        for disagg_cfg in ["DISAGG-1D", "DISAGG-2D"]:
            for conc in sorted(set(k[1] for k in exp11 if k[1] > 1)):
                bl_c1 = exp11.get(("BASELINE", 1, sl))
                dg_c1 = exp11.get((disagg_cfg, 1, sl))
                bl_cn = exp11.get(("BASELINE", conc, sl))
                dg_cn = exp11.get((disagg_cfg, conc, sl))
                if not all([bl_c1, dg_c1, bl_cn, dg_cn]):
                    continue
                if bl_c1["median"] <= 0 or dg_c1["median"] <= 0:
                    continue
                alpha_mono = bl_cn["median"] / bl_c1["median"]
                alpha_disagg = dg_cn["median"] / dg_c1["median"]
                if alpha_disagg <= 0:
                    continue
                measured_r = alpha_mono / alpha_disagg
                predicted_r = _predict_contention_ratio(
                    pred["predicted_delta_gamma"], conc)
                err = (predicted_r - measured_r) / measured_r * 100
                results.append(ValidationResult(
                    f"R(c={conc},{disagg_cfg})", sl, conc,
                    predicted_r, measured_r, err, _grade(err),
                    unit=""))

    return results


def print_report(model: str, gpu_type: str, results: list, measurements: dict):
    """Print formatted validation report."""
    print(f"\n{'='*70}")
    print(f"  ADVISOR VALIDATION: {model} on {gpu_type.upper()}")
    print(f"{'='*70}")

    is_measured = (model, gpu_type) in MEASURED_BASELINES
    print(f"  Advisor mode: {'measured baseline' if is_measured else 'extrapolated'}")
    print()

    print(f"  {'Metric':>25} | {'Seq':>5} | {'c':>2} | {'Predicted':>10} | "
          f"{'Measured':>10} | {'Error':>8} | {'Grade':>5}")
    print(f"  {'-'*25}-+-{'-'*5}-+-{'-'*2}-+-{'-'*10}-+-"
          f"{'-'*10}-+-{'-'*8}-+-{'-'*5}")

    for r in results:
        if r.unit == "ms":
            pred_s = f"{r.predicted:>8.0f}ms"
            meas_s = f"{r.measured:>8.0f}ms"
        else:
            pred_s = f"{r.predicted:>9.3f}"
            meas_s = f"{r.measured:>9.3f}"
        print(f"  {r.metric:>25} | {r.seq_len:>5} | {r.concurrency:>2} | "
              f"{pred_s:>10} | {meas_s:>10} | "
              f"{r.error_pct:>+7.0f}% | {r.grade:>5}")
    print()

    grades = [r.grade for r in results]
    good = grades.count("GOOD")
    fair = grades.count("FAIR")
    poor = grades.count("POOR")
    wrong = grades.count("WRONG")
    total = len(grades)

    print(f"  Summary: {good}/{total} GOOD, {fair}/{total} FAIR, "
          f"{poor}/{total} POOR, {wrong}/{total} WRONG")

    if wrong > total * 0.3:
        print("  VERDICT: Advisor predictions are UNRELIABLE for this model/GPU.")
        print("           Run experiments for accurate data.")
    elif poor + wrong > total * 0.3:
        print("  VERDICT: Advisor predictions have SIGNIFICANT errors.")
        print("           Use measured baselines (now added) for accuracy.")
    elif good + fair >= total * 0.7:
        print("  VERDICT: Advisor predictions are USABLE.")
    print()

    exp11 = measurements.get("exp11", {})
    crossovers = []
    seq_lens = sorted(set(k[2] for k in exp11 if k[1] == 1))
    for sl in seq_lens:
        bl = exp11.get(("BASELINE", 1, sl))
        for dc in ["DISAGG-1D", "DISAGG-2D"]:
            dg = exp11.get((dc, 1, sl))
            if bl and dg and dg["median"] < bl["median"]:
                crossovers.append((sl, dc, 1))

    concurrencies = sorted(set(k[1] for k in exp11))
    for c in concurrencies:
        if c == 1:
            continue
        for sl in sorted(set(k[2] for k in exp11 if k[1] == c)):
            bl = exp11.get(("BASELINE", c, sl))
            for dc in ["DISAGG-1D", "DISAGG-2D"]:
                dg = exp11.get((dc, c, sl))
                if bl and dg and dg["median"] < bl["median"]:
                    crossovers.append((sl, dc, c))

    if crossovers:
        print("  CROSSOVER POINTS (disagg faster than mono):")
        for sl, dc, c in crossovers:
            bl = exp11[("BASELINE", c, sl)]["median"]
            dg = exp11[(dc, c, sl)]["median"]
            delta = (dg - bl) / bl * 100
            print(f"    c={c}, {sl} tokens, {dc}: {delta:+.1f}% "
                  f"(mono={bl:.0f}ms, disagg={dg:.0f}ms)")
        print("  NOTE: Advisor cannot predict concurrency-dependent crossovers.")
    else:
        print("  No crossover detected: mono wins at all measured conditions.")
    print()

    sparse = []
    for key, st in sorted(exp11.items()):
        cfg, conc, sl = key
        if st["n"] < 10:
            sparse.append((sl, conc, cfg, st["n"]))
    if sparse:
        print("  DATA QUALITY WARNING — sparse samples (n < 10):")
        for sl, conc, cfg, n in sparse:
            print(f"    s={sl}, c={conc}, {cfg}: n={n}")
        print("  Median from <10 samples is unreliable. "
              "Grades at these points should be discounted.")
        print()

    cal_seq_lens = sorted(set(k[2] for k in exp11 if k[1] == 1))
    if cal_seq_lens:
        cal_max = cal_seq_lens[-1]
        extrap = [r for r in results if r.seq_len > cal_max * 0.8
                  and r.metric.startswith(("mono_", "disagg_"))]
        if extrap:
            print(f"  VALIDITY NOTE: Calibration data extends to s={cal_max}.")
            print(f"  Prefill time is superlinear in seq_len (attention O(s*d)).")
            print(f"  TTFT predictions degrade outside the calibration range.")
            print()

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate advisor predictions against experiment data")
    parser.add_argument("data_dir", help="Path to data directory with exp results")
    parser.add_argument("--gpu-type", default="t4", help="GPU type (default: t4)")
    args = parser.parse_args()

    info = load_run_info(args.data_dir)
    model = info.get("toolkit", {}).get("model")
    if not model:
        print("ERROR: run-info.json missing or has no toolkit.model")
        sys.exit(1)

    print(f"Validating advisor for: {model} on {args.gpu_type.upper()}")

    measurements = load_exp_baselines(args.data_dir)
    if not measurements["exp11"] and not measurements["exp14"]:
        print("ERROR: No exp11 or exp14 data found in", args.data_dir)
        sys.exit(1)

    seq_lens = sorted(set(
        k[2] for k in measurements["exp11"] if k[1] == 1
    ) | set(
        k[1] for k in measurements["exp14"]
    ))
    if not seq_lens:
        print("ERROR: Could not determine sequence lengths from data")
        sys.exit(1)

    predictions = get_predictions(model, args.gpu_type, seq_lens)
    results = compare(predictions, measurements)
    print_report(model, args.gpu_type, results, measurements)


if __name__ == "__main__":
    main()
