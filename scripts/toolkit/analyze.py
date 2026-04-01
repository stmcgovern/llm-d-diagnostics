#!/usr/bin/env python3
"""
llm-d Diagnostics Toolkit — Analysis

Reads CSV data from experiments and produces rigorous statistical summaries.
Flags data quality issues, computes confidence intervals, and separates
findings from predictions.

Usage:
    python3 scripts/toolkit/analyze.py [data_dir]

Default data_dir: data/
"""

import csv
import math
import os
import sys
from collections import defaultdict


def load_csv(filepath):
    """Load a CSV file and return list of dicts."""
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return list(csv.DictReader(f))


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def get_status(row):
    """Get HTTP status code from a row, handling both 'status_code' and 'status' fields."""
    return safe_int(row.get("status_code", row.get("status", 0)))


# ── Statistics ───────────────────────────────────────────────────────────────

# Two-tailed t critical values at 95% for common df (scipy-free)
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045,
    30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def _t95(df):
    """Return t critical value for 95% CI. Uses table for small df, 1.96 for large."""
    if df in _T95:
        return _T95[df]
    # Interpolate from nearest keys
    keys = sorted(_T95.keys())
    if df > keys[-1]:
        return 1.96
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return _T95[lo]
    frac = (df - lo) / (hi - lo)
    return _T95[lo] * (1 - frac) + _T95[hi] * frac


def stats(values):
    """Compute summary statistics for a list of floats.

    Returns dict with: n, mean, median, std, min, max, p10, p25, p75, p90,
    iqr, ci95_lo, ci95_hi (95% confidence interval for the mean).
    """
    if not values:
        return {"n": 0}

    s = sorted(values)
    n = len(s)
    mean = sum(s) / n

    if n >= 2:
        variance = sum((x - mean) ** 2 for x in s) / (n - 1)
        std = math.sqrt(variance)
        # 95% CI for the mean (t-approximation for small n)
        t_val = _t95(n - 1)
        ci_half = t_val * std / math.sqrt(n)
    else:
        std = 0
        ci_half = 0

    def percentile(p):
        idx = p / 100 * (n - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    return {
        "n": n,
        "mean": round(mean, 1),
        "median": round(percentile(50), 1),
        "std": round(std, 1),
        "min": round(s[0], 1),
        "max": round(s[-1], 1),
        "p10": round(percentile(10), 1),
        "p25": round(percentile(25), 1),
        "p50": round(percentile(50), 1),
        "p75": round(percentile(75), 1),
        "p90": round(percentile(90), 1),
        "p99": round(percentile(99), 1),
        "iqr": round(percentile(75) - percentile(25), 1),
        "ci95_lo": round(mean - ci_half, 1),
        "ci95_hi": round(mean + ci_half, 1),
    }


def fmt_stats(s):
    """Format stats dict for display.

    Reports median with IQR (robust), plus mean with 95% CI (parametric).
    These are separate: the CI describes uncertainty about the mean, not the median.
    """
    if s["n"] == 0:
        return "no data"
    return (f'{s["median"]:.1f}ms (n={s["n"]}, '
            f'IQR=[{s["p25"]:.1f},{s["p75"]:.1f}], '
            f'mean={s["mean"]:.1f}, '
            f'95%CI=[{s["ci95_lo"]:.1f},{s["ci95_hi"]:.1f}])')


# ── Data quality ─────────────────────────────────────────────────────────────

def check_data_quality(rows, max_tokens_field="max_tokens",
                       ct_field="completion_tokens"):
    """Check for truncated responses and errors."""
    issues = []
    total = len(rows)
    if total == 0:
        return issues

    errors = [r for r in rows if get_status(r) not in (0, 200)]
    if errors:
        issues.append(f"  {len(errors)}/{total} non-200 responses ({100*len(errors)/total:.1f}%)")

    if ct_field in rows[0]:
        truncated = [r for r in rows if safe_int(r.get(ct_field, 0)) < 10
                     and get_status(r) == 200]
        if truncated:
            issues.append(f"  {len(truncated)}/{total} truncated responses "
                          f"(<10 completion tokens, {100*len(truncated)/total:.1f}%)")

    error_field = [r for r in rows if r.get("error", "")]
    if error_field:
        issues.append(f"  {len(error_field)}/{total} requests with error field set")

    return issues


def validate_completeness(rows, group_fields, expected_per_group=None):
    """Check that each (config, prompt_length, etc.) group has consistent row counts.

    Args:
        rows: List of dicts from CSV.
        group_fields: List of field names to group by (e.g., ["config", "prompt_tokens_target"]).
        expected_per_group: Expected rows per group (optional). If None, uses the mode.

    Returns:
        List of warning strings (empty if data is complete).
    """
    groups = defaultdict(int)
    for r in rows:
        key = tuple(r.get(f, "") for f in group_fields)
        groups[key] += 1

    if not groups:
        return ["No data rows found"]

    counts = list(groups.values())
    if expected_per_group is None:
        # Mode of group counts. Ties broken by largest count (most complete).
        expected_per_group = max(set(counts), key=lambda c: (counts.count(c), c))

    warnings = []
    for key, count in sorted(groups.items()):
        if count != expected_per_group:
            label = "/".join(f"{f}={v}" for f, v in zip(group_fields, key))
            warnings.append(f"  {label}: {count} rows (expected {expected_per_group})")

    error_rate = sum(1 for r in rows if get_status(r) != 200) / len(rows) if rows else 0
    if error_rate > 0.1:
        warnings.append(f"  High error rate: {error_rate:.0%} of requests returned non-200")

    return warnings


# ── Experiment analyses ──────────────────────────────────────────────────────

def analyze_exp1(data_dir):
    """Experiment 1: Single-Request Latency."""
    rows = load_csv(os.path.join(data_dir, "exp1-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config", "prompt_tokens_target"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")
    print()

    # Group by (config, prompt_tokens_target)
    print(f"  {'Prompt':>6} | {'Config':>10} | {'Median TTFT':>12} | "
          f"{'Mean':>8} | {'Std':>6} | {'IQR':>15} | {'95% CI':>17} | n")
    print(f"  {'':->6}-+-{'':->10}-+-{'':->12}-+-"
          f"{'':->8}-+-{'':->6}-+-{'':->15}-+-{'':->17}-+---")

    prompt_targets = sorted(set(r["prompt_tokens_target"] for r in rows), key=lambda x: safe_int(x))
    configs = sorted(set(r["config"] for r in rows))

    for pt in prompt_targets:
        for cfg in configs:
            v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg and r["prompt_tokens_target"] == pt
                 and get_status(r) == 200]
            s = stats(v)
            if s["n"] == 0:
                continue
            print(f'  {pt:>6} | {cfg:>10} | {s["median"]:>10.1f}ms | '
                  f'{s["mean"]:>6.1f} | {s["std"]:>5.1f} | '
                  f'[{s["p25"]:>5.1f},{s["p75"]:>5.1f}] | '
                  f'[{s["ci95_lo"]:>6.1f},{s["ci95_hi"]:>6.1f}] | {s["n"]}')

    # Overhead analysis: compare each non-baseline config to baseline
    bl_name = "BASELINE"
    disagg_configs = [c for c in configs if c != bl_name]
    if not disagg_configs:
        return

    print()
    print(f"  Overhead (disagg median - {bl_name} median):")
    overheads = {c: [] for c in disagg_configs}
    for pt in prompt_targets:
        bl = stats([safe_float(r["ttft_ms"]) for r in rows
                    if r["config"] == bl_name and r["prompt_tokens_target"] == pt
                    and get_status(r) == 200])
        if bl["n"] == 0:
            continue

        parts = []
        skip = False
        for dc in disagg_configs:
            ds = stats([safe_float(r["ttft_ms"]) for r in rows
                        if r["config"] == dc and r["prompt_tokens_target"] == pt
                        and get_status(r) == 200])
            if ds["n"] == 0:
                skip = True
                break
            oh = ds["median"] - bl["median"]
            overheads[dc].append(oh)
            parts.append(f"{dc}=+{oh:.1f}ms")

        if skip:
            print(f"    {pt:>5} tok: insufficient data")
        else:
            print(f"    {pt:>5} tok: {'  '.join(parts)}")

    any_data = False
    for dc in disagg_configs:
        s = stats(overheads[dc])
        if s["n"] == 0:
            continue
        any_data = True
        print(f"    {dc} overhead: {s['mean']:.1f} +/- {s['std']:.1f}ms "
              f"(range {s['min']:.1f}-{s['max']:.1f})")

    if not any_data:
        print("    No overhead data to summarize")
        return

    # Flag variance asymmetry between disagg configs
    oh_stats = {dc: stats(overheads[dc]) for dc in disagg_configs if overheads[dc]}
    std_vals = [(dc, oh_stats[dc]["std"]) for dc in oh_stats if oh_stats[dc]["std"] > 0]
    if len(std_vals) >= 2:
        min_dc, min_std = min(std_vals, key=lambda x: x[1])
        max_dc, max_std = max(std_vals, key=lambda x: x[1])
        if max_std > min_std * 1.5:
            print(f"    WARNING: {max_dc} overhead variance ({max_std:.1f}ms) is "
                  f"{max_std/min_std:.1f}x {min_dc} ({min_std:.1f}ms). "
                  f"Investigate node asymmetry.")


def analyze_exp1b(data_dir):
    """Experiment 1b: Latency Decomposition."""
    rows = load_csv(os.path.join(data_dir, "exp1b-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")
    print()

    # Filter: only full responses (completion_tokens >= 10)
    configs = {}
    for cfg in ["A-prefill-direct", "B-decode-direct",
                "C-sidecar-only", "D-disaggregated"]:
        all_v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg and get_status(r) == 200]
        full_v = [safe_float(r["ttft_ms"]) for r in rows
                  if r["config"] == cfg and get_status(r) == 200
                  and safe_int(r.get("completion_tokens", 0)) >= 10]
        trimmed = len(all_v) - len(full_v)

        s_all = stats(all_v)
        s_full = stats(full_v)
        configs[cfg] = s_full

        trim_note = f" ({trimmed} truncated excluded)" if trimmed else ""
        print(f"  {cfg}:{trim_note}")
        print(f"    All data: {fmt_stats(s_all)}")
        if trimmed:
            print(f"    Trimmed:  {fmt_stats(s_full)}")
        print()

    # Decomposition
    if any(configs[c]["n"] == 0 for c in configs):
        print("  Decomposition: insufficient data (one or more configs have no valid rows)")
        return

    a = configs["A-prefill-direct"]["median"]
    b = configs["B-decode-direct"]["median"]
    c = configs["C-sidecar-only"]["median"]
    d = configs["D-disaggregated"]["median"]

    print("  Decomposition (difference of medians):")
    print(f"    A (prefill direct):  {a:.1f}ms")
    print(f"    B (decode direct):   {b:.1f}ms")
    print(f"    C (sidecar only):    {c:.1f}ms")
    print(f"    D (disaggregated):   {d:.1f}ms")
    print()
    print(f"    T_sidecar    = C - B = {c-b:.1f}ms")
    print(f"    T_prefill_rt = D - C = {d-c:.1f}ms  (includes NIXL transfer)")
    print(f"    T_overhead   = D - A = {d-a:.1f}ms")
    print(f"    Sum check: {c-b:.1f} + {d-c:.1f} = {(c-b)+(d-c):.1f}ms "
          f"vs T_overhead = {d-a:.1f}ms "
          f"(residual: {(d-a) - ((c-b)+(d-c)):.1f}ms)")

    # Paired-difference decomposition: median(D_i - C_i) instead of median(D) - median(C).
    # More precise when runs are sequential and share environmental noise.
    run_data = defaultdict(dict)
    for r in rows:
        if get_status(r) == 200 and safe_int(r.get("completion_tokens", 0)) >= 10:
            run_data[r["run"]][r["config"]] = safe_float(r["ttft_ms"])

    paired_runs = [run for run, cfgs in run_data.items()
                   if all(c in cfgs for c in ["A-prefill-direct", "B-decode-direct",
                                              "C-sidecar-only", "D-disaggregated"])]

    if len(paired_runs) >= 5:
        diffs_sidecar = []
        diffs_prefill_rt = []
        diffs_overhead = []
        for run in paired_runs:
            d_ = run_data[run]
            diffs_sidecar.append(d_["C-sidecar-only"] - d_["B-decode-direct"])
            diffs_prefill_rt.append(d_["D-disaggregated"] - d_["C-sidecar-only"])
            diffs_overhead.append(d_["D-disaggregated"] - d_["A-prefill-direct"])

        s_sc = stats(diffs_sidecar)
        s_pr = stats(diffs_prefill_rt)
        s_oh = stats(diffs_overhead)

        print()
        print(f"  Paired-difference decomposition (n={len(paired_runs)} matched runs):")
        print(f"    T_sidecar    = mean(C_i - B_i) = {s_sc['mean']:.1f}ms "
              f"(95%CI [{s_sc['ci95_lo']:.1f}, {s_sc['ci95_hi']:.1f}], "
              f"median={s_sc['median']:.1f})")
        print(f"    T_prefill_rt = mean(D_i - C_i) = {s_pr['mean']:.1f}ms "
              f"(95%CI [{s_pr['ci95_lo']:.1f}, {s_pr['ci95_hi']:.1f}], "
              f"median={s_pr['median']:.1f})")
        print(f"    T_overhead   = mean(D_i - A_i) = {s_oh['mean']:.1f}ms "
              f"(95%CI [{s_oh['ci95_lo']:.1f}, {s_oh['ci95_hi']:.1f}], "
              f"median={s_oh['median']:.1f})")
    else:
        print()
        print(f"  Paired-difference decomposition: insufficient matched runs "
              f"({len(paired_runs)} found, need >= 5)")

    # Check A ≈ B (same model, different GPU)
    a_mean = configs["A-prefill-direct"]["mean"]
    b_mean = configs["B-decode-direct"]["mean"]
    a_std = configs["A-prefill-direct"]["std"]
    b_std = configs["B-decode-direct"]["std"]
    print()
    print(f"    Validation: A ≈ B (same model, different GPU)")
    print(f"      A: {a_mean:.1f} +/- {a_std:.1f}ms")
    print(f"      B: {b_mean:.1f} +/- {b_std:.1f}ms")
    diff_pct = abs(a - b) / min(a, b) * 100 if min(a, b) > 0 else float('inf')
    print(f"      Difference: {abs(a-b):.1f}ms ({diff_pct:.1f}%) — "
          f"{'OK' if diff_pct < 5 else 'WARNING: significant'}")


def analyze_exp2(data_dir):
    """Experiment 2: Throughput Under Load."""
    rows = load_csv(os.path.join(data_dir, "exp2-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows, ct_field="_none_"):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config", "concurrency"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")

    suspicious = sum(1 for r in rows
                     if safe_float(r["total_ms"]) < 100
                     and get_status(r) == 200)
    if suspicious:
        print(f"    {suspicious}/{len(rows)} requests with <100ms total latency "
              f"(likely truncated responses)")
    print()

    print(f"  {'C':>3} | {'Config':>10} | {'Med Total':>10} | {'P90':>8} | "
          f"{'Mean':>8} | {'Std':>6} | n | {'Latency Ratio':>14}")
    print(f"  {'':->3}-+-{'':->10}-+-{'':->10}-+-{'':->8}-+-"
          f"{'':->8}-+-{'':->6}-+---+-{'':->14}")

    concurrency_levels = sorted(set(r["concurrency"] for r in rows), key=lambda x: safe_int(x))

    for conc in concurrency_levels:
        bl_s = None
        for cfg in ["BASELINE", "DISAGG-1D", "DISAGG-2D"]:
            v = [safe_float(r["total_ms"]) for r in rows
                 if r["config"] == cfg and r["concurrency"] == conc
                 and get_status(r) == 200]
            s = stats(v)
            if s["n"] == 0:
                continue

            if cfg == "BASELINE":
                bl_s = s
                ratio = "1.00x (ref)"
            else:
                ratio = f'{s["median"]/bl_s["median"]:.2f}x' if bl_s else "N/A"

            print(f'  {conc:>3} | {cfg:>10} | {s["median"]:>8.0f}ms | '
                  f'{s["p90"]:>6.0f}ms | {s["mean"]:>6.0f}ms | '
                  f'{s["std"]:>5.0f} | {s["n"]:>2} | {ratio:>14}')

    print()
    print("  Note: Latency ratio >1 means disagg is slower than baseline.")
    print("  Throughput numbers require wall-clock timing to be precise.")


def analyze_exp3(data_dir):
    """Experiment 3: Prefill Isolation."""
    rows = load_csv(os.path.join(data_dir, "exp3-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config", "weight"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")
    print()

    for cfg in ["BASELINE", "DISAGG-2D"]:
        for weight in ["heavy", "light"]:
            v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg and r["weight"] == weight
                 and get_status(r) == 200]
            s = stats(v)
            print(f"  {cfg:>10} {weight:>5}: {fmt_stats(s)}")

    # Isolation ratio
    bl_light = [safe_float(r["ttft_ms"]) for r in rows
                if r["config"] == "BASELINE" and r["weight"] == "light"
                and get_status(r) == 200]
    dg_light = [safe_float(r["ttft_ms"]) for r in rows
                if r["config"] == "DISAGG-2D" and r["weight"] == "light"
                and get_status(r) == 200]

    bl_s = stats(bl_light)
    dg_s = stats(dg_light)

    if bl_s["n"] > 0 and dg_s["n"] > 0 and dg_s["median"] > 0:
        ratio_median = bl_s["median"] / dg_s["median"]
        ratio_mean = bl_s["mean"] / dg_s["mean"] if dg_s["mean"] > 0 else 0
        print()
        print(f"  Isolation ratio (BL / DG for light requests):")
        print(f"    Median: {ratio_median:.3f}   Mean: {ratio_mean:.3f}")
        print(f"    >1 = disagg protects light requests")
        print(f"    <1 = disagg makes light requests worse")
        print(f"    Observed: {'disagg helps' if ratio_median > 1 else 'disagg hurts'}")

    # ITL analysis (requires streaming data with itl_ms column)
    if "itl_ms" in rows[0]:
        all_itl = [safe_float(r["itl_ms"]) for r in rows
                   if get_status(r) == 200 and safe_float(r["itl_ms"]) > 0]
        if all_itl and max(all_itl) >= 1.0:
            print()
            print("  Inter-Token Latency (from streaming, mean of per-request avg ITL):")
            configs = sorted(set(r["config"] for r in rows))
            weights = sorted(set(r["weight"] for r in rows))
            for cfg in configs:
                for weight in weights:
                    v = [safe_float(r["itl_ms"]) for r in rows
                         if r["config"] == cfg and r["weight"] == weight
                         and get_status(r) == 200 and safe_float(r["itl_ms"]) > 0]
                    s = stats(v)
                    if s["n"] > 0:
                        print(f"    {cfg:>10} {weight:>5}: {fmt_stats(s)}")
        elif all_itl:
            print()
            print("  ITL: all values <1ms (model generates faster than measurement resolution)")


def analyze_exp4(data_dir):
    """Experiment 4: Fault Tolerance."""
    rows = load_csv(os.path.join(data_dir, "exp4-results.csv"))
    if not rows:
        print("  No data found")
        return

    # Group rows by sub-experiment
    subs = defaultdict(list)
    for r in rows:
        subs[r["sub"]].append(r)

    # Display results per sub-experiment with detection gap analysis
    for sub in sorted(subs.keys()):
        sub_rows = subs[sub]
        print(f"  --- {sub} ---")

        kill_epoch = None
        for r in sub_rows:
            status = r.get("status_code", "")
            ttft = safe_float(r.get("ttft_ms", 0))
            total = safe_float(r.get("total_ms", 0))
            epoch = safe_float(r.get("epoch_ms", 0))
            note = r.get("note", "")
            error = r.get("error", "")
            phase = r.get("phase", "")

            # Track kill/partition events for detection gap
            if phase in ("kill", "partition-on"):
                kill_epoch = epoch
                print(f"    [{phase}] {note}")
                continue
            if phase in ("partition-off", "netem-on", "netem-off", "skip"):
                print(f"    [{phase}] {note}")
                continue

            ok = "200" in str(status)
            symbol = "OK" if ok else "FAIL"
            err_msg = f" [{error}]" if error else ""

            # Detection gap: time from kill/partition to this request's epoch
            gap_str = ""
            if kill_epoch and epoch and phase.startswith("during"):
                gap_ms = epoch - kill_epoch
                gap_str = f" (detection gap: {gap_ms/1000:.1f}s)"

            print(f"    [{phase}] {symbol} "
                  f"ttft={ttft:.0f}ms total={total:.0f}ms — {note}{err_msg}{gap_str}")

        # Sub-experiment summary
        request_rows = [r for r in sub_rows
                        if r.get("phase", "") not in
                        ("kill", "partition-on", "partition-off",
                         "netem-on", "netem-off", "skip")]
        if request_rows:
            ok_count = sum(1 for r in request_rows if r.get("status_code") == "200")
            fail_count = len(request_rows) - ok_count
            print(f"    Summary: {ok_count}/{len(request_rows)} OK, "
                  f"{fail_count} failed")

        # Recovery analysis: compare pre vs post TTFT
        pre_rows = [r for r in request_rows if r.get("phase", "").startswith("pre")]
        post_rows = [r for r in request_rows if r.get("phase", "").startswith("post")]
        if pre_rows and post_rows:
            pre_ttft = safe_float(pre_rows[0].get("ttft_ms", 0))
            post_ttft = safe_float(post_rows[0].get("ttft_ms", 0))
            if pre_ttft > 0:
                ratio = post_ttft / pre_ttft
                label = "cold start" if ratio > 3.0 else "warm" if ratio < 1.5 else "elevated"
                print(f"    Recovery latency: {post_ttft:.0f}ms vs pre-kill {pre_ttft:.0f}ms "
                      f"({ratio:.1f}x — {label})")
        print()

    # Log analysis — scan exp4-logs/ for NIXL/transport events
    log_dir = os.path.join(data_dir, "exp4-logs")
    if os.path.isdir(log_dir):
        _analyze_exp4_logs(log_dir)


# Patterns to search for in exp4 pod logs, grouped by category.
_LOG_PATTERNS = {
    "NIXL connection": [
        "nixl", "kv_transfer", "kv transfer", "pull_async", "push_async",
        "handshake", "register_agent", "deregister",
    ],
    "ZMQ discovery": [
        "zmq", "discovery", "new peer", "peer lost", "endpoint",
        "service_discovery",
    ],
    "Transport errors": [
        "ECONNRESET", "ECONNREFUSED", "ETIMEDOUT", "broken pipe",
        "connection reset", "connection refused", "timed out",
    ],
    "Reconnection": [
        "reconnect", "retry", "backoff", "re-establish",
        "connection restored", "recovered",
    ],
    "Sidecar routing": [
        "prefill.*fail", "prefill.*error", "fallback", "no backend",
        "upstream.*error", "502", "503", "504",
    ],
}


def _analyze_exp4_logs(log_dir):
    """Scan exp4 pod logs for NIXL, ZMQ, and transport events."""
    import glob
    import re

    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not log_files:
        return

    print("  --- Log Analysis ---")

    for log_path in log_files:
        filename = os.path.basename(log_path)
        try:
            with open(log_path) as f:
                lines = f.readlines()
        except OSError:
            continue

        if not lines:
            continue

        hits = defaultdict(list)
        for i, line in enumerate(lines):
            lower = line.lower()
            for category, patterns in _LOG_PATTERNS.items():
                for pat in patterns:
                    if re.search(pat, lower):
                        hits[category].append((i + 1, line.rstrip()))
                        break  # one match per category per line

        if hits:
            print(f"    {filename}:")
            for category, matches in sorted(hits.items()):
                print(f"      {category}: {len(matches)} events")
                # Show first and last match for context
                if len(matches) <= 3:
                    for lineno, text in matches:
                        print(f"        L{lineno}: {text[:120]}")
                else:
                    lineno, text = matches[0]
                    print(f"        L{lineno}: {text[:120]}")
                    print(f"        ... ({len(matches) - 2} more)")
                    lineno, text = matches[-1]
                    print(f"        L{lineno}: {text[:120]}")
    print()


def pearson_r(xs, ys):
    """Pearson correlation coefficient (stdlib-only)."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den > 0 else 0.0


def analyze_exp5(data_dir):
    """Experiment 5: Sequence Length Sweep — Transfer Tax Crossover."""
    rows = load_csv(os.path.join(data_dir, "exp5-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config", "prompt_tokens_target"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")
    print()

    # Check actual vs target prompt tokens.
    # build_prompt() systematically produces ~1.2x the target token count
    # (tokenizer behavior). Only warn if the ratio varies within a given
    # target length, which would indicate a data quality issue.
    by_target = defaultdict(set)
    for r in rows:
        target = safe_int(r.get("prompt_tokens_target", 0))
        actual = safe_int(r.get("prompt_tokens_actual", 0))
        if actual > 0 and target > 0:
            by_target[target].add(actual)
    variable_targets = {t: actuals for t, actuals in by_target.items()
                        if len(actuals) > 1}
    if variable_targets:
        print("  WARNING: prompt token counts vary within same target length:")
        for target, actuals in sorted(variable_targets.items()):
            print(f"    target={target}, actual values={sorted(actuals)}")
        print()
    elif by_target:
        # Report the systematic ratio as a note, not a warning
        sample = next(iter(by_target.items()))
        ratio = next(iter(sample[1])) / sample[0] if sample[0] > 0 else 0
        if ratio > 1.1:
            print(f"  Note: build_prompt produces ~{ratio:.1f}x target token count "
                  f"(consistent across all lengths)")
            print()

    # Prompt lengths and configs
    prompt_lengths = sorted(set(safe_int(r["prompt_tokens_target"]) for r in rows))
    config_names = ["A-prefill-direct", "B-decode-direct",
                    "C-sidecar-only", "D-disaggregated"]

    # Per-length decomposition table
    print(f"  {'Length':>8} | {'T_sidecar':>10} | {'T_transfer':>10} | "
          f"{'T_overhead':>10} | {'A(prefill)':>10} | {'B(decode)':>10} | "
          f"{'C(sidecar)':>10} | {'D(disagg)':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-"
          f"{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    # Collect per-length stats for crossover analysis
    length_data = {}  # prompt_length -> {config -> stats_dict}
    transfer_points = []  # (prompt_length, T_transfer_mean, T_prefill_median)

    for pt in prompt_lengths:
        cfg_stats = {}
        for cfg in config_names:
            v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg
                 and safe_int(r["prompt_tokens_target"]) == pt
                 and get_status(r) == 200
                 and safe_int(r.get("completion_tokens", 0)) >= 10]
            cfg_stats[cfg] = stats(v)
        length_data[pt] = cfg_stats

        a = cfg_stats["A-prefill-direct"]
        b = cfg_stats["B-decode-direct"]
        c = cfg_stats["C-sidecar-only"]
        d = cfg_stats["D-disaggregated"]

        if all(s["n"] > 0 for s in [a, b, c, d]):
            t_sc = c["median"] - b["median"]
            t_tr = d["median"] - c["median"]
            t_oh = d["median"] - a["median"]
            print(f"  {pt:>8} | {t_sc:>9.1f}ms | {t_tr:>9.1f}ms | "
                  f"{t_oh:>9.1f}ms | {a['median']:>9.1f}ms | {b['median']:>9.1f}ms | "
                  f"{c['median']:>9.1f}ms | {d['median']:>9.1f}ms")
            transfer_points.append((pt, t_tr, a["median"]))
        else:
            missing = [cfg for cfg in config_names if cfg_stats[cfg]["n"] == 0]
            # Check if ALL configs have 0 successful responses at this length
            all_rows_at_pt = [r for r in rows
                              if safe_int(r["prompt_tokens_target"]) == pt]
            errors_at_pt = [r for r in all_rows_at_pt if get_status(r) != 200]
            if all_rows_at_pt and len(errors_at_pt) == len(all_rows_at_pt):
                # Check for HTTP 400 (context window exceeded)
                status_codes = set(get_status(r) for r in errors_at_pt)
                if 400 in status_codes:
                    print(f"  {pt:>8} | (all requests failed — HTTP 400: "
                          f"prompt likely exceeds model context window)")
                else:
                    print(f"  {pt:>8} | (all requests failed — "
                          f"status codes: {', '.join(str(s) for s in sorted(status_codes))})")
            else:
                print(f"  {pt:>8} | (insufficient data: {', '.join(missing)})")

    print()

    # Paired-difference analysis at each length
    print("  Paired-difference T_transfer = mean(D_i - C_i) at each length:")
    paired_transfers = []  # (prompt_length, mean, ci_lo, ci_hi)

    for pt in prompt_lengths:
        run_data = defaultdict(dict)
        for r in rows:
            if (safe_int(r["prompt_tokens_target"]) == pt
                    and get_status(r) == 200
                    and safe_int(r.get("completion_tokens", 0)) >= 10):
                run_data[r["run"]][r["config"]] = safe_float(r["ttft_ms"])

        paired_runs = [run for run, cfgs in run_data.items()
                       if "C-sidecar-only" in cfgs and "D-disaggregated" in cfgs]

        if len(paired_runs) >= 5:
            diffs = [run_data[run]["D-disaggregated"] - run_data[run]["C-sidecar-only"]
                     for run in paired_runs]
            s = stats(diffs)
            paired_transfers.append((pt, s["mean"], s["ci95_lo"], s["ci95_hi"]))
            print(f"    {pt:>6} tokens: {s['mean']:.1f}ms "
                  f"(95%CI [{s['ci95_lo']:.1f}, {s['ci95_hi']:.1f}], "
                  f"n={s['n']})")
        else:
            print(f"    {pt:>6} tokens: insufficient paired runs ({len(paired_runs)})")
    print()

    # Crossover detection
    if len(transfer_points) >= 2:
        print("  Crossover analysis (T_transfer vs T_prefill):")
        crossover_found = False
        for i in range(len(transfer_points) - 1):
            pt_lo, t_tr_lo, t_pf_lo = transfer_points[i]
            pt_hi, t_tr_hi, t_pf_hi = transfer_points[i + 1]
            if t_tr_lo <= t_pf_lo and t_tr_hi > t_pf_hi:
                print(f"    CROSSOVER: T_transfer overtakes T_prefill between "
                      f"{pt_lo} and {pt_hi} tokens")
                print(f"      At {pt_lo}: T_transfer={t_tr_lo:.1f}ms, "
                      f"T_prefill={t_pf_lo:.1f}ms")
                print(f"      At {pt_hi}: T_transfer={t_tr_hi:.1f}ms, "
                      f"T_prefill={t_pf_hi:.1f}ms")
                crossover_found = True
        if not crossover_found:
            if all(t_tr > t_pf for _, t_tr, t_pf in transfer_points):
                print(f"    T_transfer > T_prefill at ALL measured lengths "
                      f"(transfer-dominated)")
            elif all(t_tr <= t_pf for _, t_tr, t_pf in transfer_points):
                print(f"    T_transfer < T_prefill at ALL measured lengths "
                      f"(compute-dominated)")
                print(f"    Crossover may occur beyond {transfer_points[-1][0]} tokens")
            else:
                print(f"    Non-monotonic behavior — no clean crossover detected")
        print()

    # Linearity check: is T_transfer proportional to sequence length?
    if len(transfer_points) >= 3:
        xs = [float(pt) for pt, _, _ in transfer_points]
        ys = [t_tr for _, t_tr, _ in transfer_points]
        r = pearson_r(xs, ys)
        label = "linear (bandwidth-limited)" if r > 0.95 else "non-linear"
        print(f"  Linearity: Pearson r(seq_len, T_transfer) = {r:.3f} — {label}")
        if r <= 0.95:
            print(f"    Transfer cost is not purely bandwidth-limited; "
                  f"investigate protocol overhead or memory allocation")
        print()

    # Validation: A ≈ B at each length
    print("  Validation: A ≈ B (same model, different GPU)")
    for pt in prompt_lengths:
        a = length_data[pt]["A-prefill-direct"]
        b = length_data[pt]["B-decode-direct"]
        if a["n"] > 0 and b["n"] > 0:
            diff_pct = abs(a["median"] - b["median"]) / min(a["median"], b["median"]) * 100 \
                if min(a["median"], b["median"]) > 0 else float('inf')
            flag = "" if diff_pct < 10 else " ← WARNING"
            print(f"    {pt:>6}: A={a['median']:.1f}ms, B={b['median']:.1f}ms, "
                  f"diff={diff_pct:.1f}%{flag}")
    print()


def analyze_exp6(data_dir):
    """Experiment 6: Saturation Profiling — Scaling Dividend."""
    rows = load_csv(os.path.join(data_dir, "exp6-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows, ct_field="_none_"):
        print(f"    {issue}")

    # Exp6 has different expected row counts per QPS level (QPS × duration_s),
    # so we validate per-group rather than using the mode-based generic check.
    duration_s = 30  # default
    run_info_path = os.path.join(data_dir, "run-info.json")
    if os.path.exists(run_info_path):
        try:
            import json
            with open(run_info_path) as f:
                info = json.load(f)
            duration_s = info.get("experiments", {}).get("exp6", {}).get("duration_s", 30)
        except (ValueError, KeyError):
            pass

    completeness_warnings = []
    groups = defaultdict(int)
    for r in rows:
        key = (r.get("config", ""), r.get("qps_target", ""))
        groups[key] += 1
    for (cfg, qps_str), count in sorted(groups.items()):
        expected = int(safe_float(qps_str) * duration_s)
        if expected > 0 and count != expected:
            completeness_warnings.append(
                f"  config={cfg}/qps={qps_str}: {count} rows (expected {expected})")
    if completeness_warnings:
        print("  Completeness warnings:")
        for w in completeness_warnings:
            print(f"    {w}")
    print()

    configs = sorted(set(r["config"] for r in rows))
    qps_levels = sorted(set(safe_float(r["qps_target"]) for r in rows))

    # Latency vs QPS table
    print(f"  {'Config':>10} | {'QPS':>5} | {'n':>4} | {'p50':>8} | "
          f"{'p90':>8} | {'p99':>8} | {'err%':>5} | {'delay_p50':>10} | "
          f"{'delay_p99':>10}")
    print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-"
          f"{'-'*8}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}")

    # Collect per-(config, qps) stats for saturation analysis
    cfg_qps_stats = {}  # (config, qps) -> stats_dict
    cfg_baselines = {}  # config -> p50 at lowest QPS

    for cfg in configs:
        for qps in qps_levels:
            v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg
                 and safe_float(r["qps_target"]) == qps
                 and get_status(r) == 200]
            delays = [safe_float(r["depart_delay_ms"]) for r in rows
                      if r["config"] == cfg
                      and safe_float(r["qps_target"]) == qps]
            errors = [r for r in rows
                      if r["config"] == cfg
                      and safe_float(r["qps_target"]) == qps
                      and get_status(r) != 200]

            s = stats(v)
            d = stats(delays) if delays else stats([])
            cfg_qps_stats[(cfg, qps)] = s

            # Track baseline (lowest QPS level)
            if qps == qps_levels[0] and s["n"] > 0:
                cfg_baselines[cfg] = s["p50"]

            total_n = len(v) + len(errors)
            err_pct = len(errors) / total_n * 100 if total_n > 0 else 0

            if s["n"] > 0:
                n_warn = "*" if s["n"] < 100 else " "
                delay_warn = " !" if d["n"] > 0 and d["p50"] > 50 else "  "
                print(f"  {cfg:>10} | {qps:>5.0f} | {s['n']:>3}{n_warn}| "
                      f"{s['p50']:>7.1f}ms | {s['p90']:>7.1f}ms | "
                      f"{s['p99']:>7.1f}ms | {err_pct:>4.0f}% | "
                      f"{d['p50']:>9.1f}ms | {d['p99']:>9.1f}ms{delay_warn}")
            else:
                print(f"  {cfg:>10} | {qps:>5.0f} | {0:>4} | "
                      f"{'no data':>8} | {'':>8} | {'':>8} | "
                      f"{err_pct:>4.0f}% |")

    print()
    print("  * = n<100 (p99 estimate is approximate)  ! = depart delay >50ms (harness saturated)")
    print()

    # Saturation point detection
    # Read SLO_MULT from run-info.json if available, otherwise default 2.0
    slo_mult = 2.0
    run_info_path = os.path.join(data_dir, "run-info.json")
    if os.path.exists(run_info_path):
        try:
            import json
            with open(run_info_path) as f:
                info = json.load(f)
            slo_mult = info.get("experiments", {}).get("exp6", {}).get("slo_mult", 2.0)
        except (ValueError, KeyError):
            pass

    print(f"  Saturation Analysis (SLO: p99 > {slo_mult}x baseline p50):")
    saturation_points = {}

    for cfg in configs:
        baseline_p50 = cfg_baselines.get(cfg)
        if baseline_p50 is None or baseline_p50 <= 0:
            print(f"    {cfg}: no baseline data")
            continue

        slo_threshold = baseline_p50 * slo_mult
        sat_qps = None

        for qps in qps_levels:
            s = cfg_qps_stats.get((cfg, qps))
            if s and s["n"] > 0 and s["p99"] > slo_threshold:
                sat_qps = qps
                break

        saturation_points[cfg] = sat_qps
        if sat_qps is not None:
            s = cfg_qps_stats[(cfg, sat_qps)]
            print(f"    {cfg}: saturates at QPS={sat_qps} "
                  f"(p99={s['p99']:.1f}ms > threshold={slo_threshold:.1f}ms)")
        else:
            print(f"    {cfg}: no saturation observed up to QPS={qps_levels[-1]}")

    print()

    # Scaling dividend
    baseline_sat = saturation_points.get("BASELINE")
    if baseline_sat:
        for cfg in configs:
            if cfg == "BASELINE":
                continue
            disagg_sat = saturation_points.get(cfg)
            if disagg_sat:
                dividend = disagg_sat / baseline_sat
                print(f"  Scaling dividend ({cfg} vs BASELINE): "
                      f"{dividend:.1f}x (saturates at QPS={disagg_sat} vs {baseline_sat})")
            else:
                print(f"  Scaling dividend ({cfg} vs BASELINE): "
                      f">={qps_levels[-1] / baseline_sat:.1f}x "
                      f"(disagg did not saturate)")
        print()
    elif baseline_sat is None and any(saturation_points.get(c) for c in configs):
        print("  Scaling dividend: BASELINE did not saturate, cannot compute ratio")
        print()

    # Depart delay warnings
    delay_issues = []
    for cfg in configs:
        for qps in qps_levels:
            delays = [safe_float(r["depart_delay_ms"]) for r in rows
                      if r["config"] == cfg
                      and safe_float(r["qps_target"]) == qps]
            if delays:
                d = stats(delays)
                if d["p50"] > 50:
                    delay_issues.append((cfg, qps, d["p50"]))
    if delay_issues:
        print("  WARNING: Test harness saturation detected (depart delay > 50ms):")
        for cfg, qps, delay in delay_issues:
            print(f"    {cfg} at QPS={qps}: median depart delay = {delay:.1f}ms")
        print(f"    Results at these QPS levels reflect closed-loop behavior.")
        print()


def analyze_exp7(data_dir):
    """Experiment 7: Mixed Workload — The Essential Characterization.

    Distills disaggregated inference to three numbers:
        1. TTFT overhead: what does disagg cost per request?
        2. ITL stability: does disagg protect decode quality under load?
        3. Throughput: how many requests can each topology sustain?
    """
    rows = load_csv(os.path.join(data_dir, "exp7-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows, ct_field="_none_"):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config", "workload_class"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")
    print()

    configs = sorted(set(r["config"] for r in rows))
    classes = sorted(set(r.get("workload_class", "unknown") for r in rows))

    # Workload mix summary
    for cfg in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg]
        total = len(cfg_rows)
        for wl in classes:
            wl_rows = [r for r in cfg_rows if r.get("workload_class") == wl]
            ok = sum(1 for r in wl_rows if get_status(r) == 200)
            print(f"  {cfg} / {wl}: {ok}/{len(wl_rows)} OK ({len(wl_rows)/total*100:.0f}% of mix)")
    print()

    # Per-(config, workload_class) TTFT and ITL stats
    print(f"  {'Config':>10} | {'Class':>6} | {'n':>4} | {'TTFT p50':>9} | "
          f"{'TTFT p99':>9} | {'ITL mean':>9} | {'ITL p99':>9}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*4}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")

    cfg_wl_stats = {}  # (config, workload_class) -> (ttft_stats, itl_stats)

    for cfg in configs:
        for wl in classes:
            ok_rows = [r for r in rows
                       if r["config"] == cfg
                       and r.get("workload_class") == wl
                       and get_status(r) == 200]

            ttfts = [safe_float(r["ttft_ms"]) for r in ok_rows]
            itl_means = [safe_float(r.get("itl_mean_ms", 0)) for r in ok_rows
                         if safe_float(r.get("itl_mean_ms", 0)) > 0]
            itl_p99s = [safe_float(r.get("itl_p99_ms", 0)) for r in ok_rows
                        if safe_float(r.get("itl_p99_ms", 0)) > 0]

            s_ttft = stats(ttfts)
            s_itl_mean = stats(itl_means)
            s_itl_p99 = stats(itl_p99s)
            cfg_wl_stats[(cfg, wl)] = (s_ttft, s_itl_mean, s_itl_p99)

            if s_ttft["n"] > 0:
                itl_m = f"{s_itl_mean['median']:.1f}ms" if s_itl_mean["n"] > 0 else "n/a"
                itl_p = f"{s_itl_p99['median']:.1f}ms" if s_itl_p99["n"] > 0 else "n/a"
                print(f"  {cfg:>10} | {wl:>6} | {s_ttft['n']:>4} | "
                      f"{s_ttft['p50']:>8.1f}ms | {s_ttft['p99']:>8.1f}ms | "
                      f"{itl_m:>9} | {itl_p:>9}")
    print()

    # ── The Three Essential Numbers ──────────────────────────────────────

    print("  ═══════════════════════════════════════════════════")
    print("  THE VERDICT")
    print("  ═══════════════════════════════════════════════════")
    print()

    baseline = "BASELINE"
    disagg = [c for c in configs if c != baseline]
    if not disagg:
        print("  Cannot compute: need both BASELINE and DISAGG configs")
        return
    disagg_cfg = disagg[0]  # typically DISAGG-2D

    # 1. TTFT Overhead
    bl_ttft = cfg_wl_stats.get((baseline, "short"), (stats([]),))[0]
    dg_ttft = cfg_wl_stats.get((disagg_cfg, "short"), (stats([]),))[0]
    if bl_ttft["n"] > 0 and dg_ttft["n"] > 0:
        overhead = dg_ttft["median"] - bl_ttft["median"]
        overhead_pct = overhead / bl_ttft["median"] * 100 if bl_ttft["median"] > 0 else 0
        print(f"  1. TTFT OVERHEAD (short requests):")
        print(f"     Baseline:     {bl_ttft['median']:.1f}ms (p50)")
        print(f"     Disaggregated: {dg_ttft['median']:.1f}ms (p50)")
        print(f"     Overhead:     {overhead:+.1f}ms ({overhead_pct:+.0f}%)")
        if overhead < 0:
            print(f"     → Disagg is FASTER (likely noise or routing benefit)")
        elif overhead_pct < 20:
            print(f"     → Acceptable overhead")
        else:
            print(f"     → Significant overhead — investigate sidecar/transfer cost")
    else:
        print(f"  1. TTFT OVERHEAD: insufficient data")
    print()

    # 2. ITL Stability (the core value proposition)
    bl_itl_short = cfg_wl_stats.get((baseline, "short"), (stats([]), stats([]), stats([])))[1]
    dg_itl_short = cfg_wl_stats.get((disagg_cfg, "short"), (stats([]), stats([]), stats([])))[1]
    bl_itl_p99_short = cfg_wl_stats.get((baseline, "short"), (stats([]), stats([]), stats([])))[2]
    dg_itl_p99_short = cfg_wl_stats.get((disagg_cfg, "short"), (stats([]), stats([]), stats([])))[2]

    if bl_itl_short["n"] > 0 and dg_itl_short["n"] > 0:
        print(f"  2. ITL STABILITY (short requests — decode quality under mixed load):")
        print(f"     Baseline     ITL: mean={bl_itl_short['median']:.1f}ms, "
              f"p99={bl_itl_p99_short['median']:.1f}ms")
        print(f"     Disaggregated ITL: mean={dg_itl_short['median']:.1f}ms, "
              f"p99={dg_itl_p99_short['median']:.1f}ms")

        if bl_itl_p99_short["median"] > 0 and dg_itl_p99_short["median"] > 0:
            ratio = dg_itl_p99_short["median"] / bl_itl_p99_short["median"]
            print(f"     ITL p99 ratio: {ratio:.2f}x")
            if ratio < 0.7:
                print(f"     → Disagg SIGNIFICANTLY improves decode quality")
            elif ratio < 1.0:
                print(f"     → Disagg improves decode quality")
            elif ratio < 1.3:
                print(f"     → Similar decode quality (disagg not helping here)")
            else:
                print(f"     → Disagg DEGRADES decode quality — investigate")
    else:
        print(f"  2. ITL STABILITY: insufficient data (need streaming responses with >=2 tokens)")
    print()

    # 3. Goodput (successful requests / offered duration)
    # Use duration_s from run-info.json for a stable denominator
    exp7_duration = None
    run_info_path = os.path.join(data_dir, "run-info.json")
    if os.path.exists(run_info_path):
        try:
            import json
            with open(run_info_path) as f:
                info = json.load(f)
            exp7_duration = info.get("experiments", {}).get("exp7", {}).get("duration_s")
        except (ValueError, KeyError):
            pass

    bl_qps = 0.0
    dg_qps = 0.0
    for cfg in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg]
        ok = sum(1 for r in cfg_rows if get_status(r) == 200)

        # Use configured duration; fall back to schedule span
        if exp7_duration:
            duration = float(exp7_duration)
        else:
            departures = sorted(safe_float(r.get("scheduled_at_s", 0)) for r in cfg_rows)
            duration = departures[-1] - departures[0] if len(departures) >= 2 else 0

        goodput = ok / duration if duration > 0 else 0

        if cfg == baseline:
            bl_qps = goodput
        else:
            dg_qps = goodput

    if bl_qps > 0 and dg_qps > 0:
        print(f"  3. GOODPUT (successful req/s at offered load):")
        print(f"     Baseline:      {bl_qps:.1f} req/s")
        print(f"     Disaggregated: {dg_qps:.1f} req/s")
        ratio = dg_qps / bl_qps
        print(f"     Ratio: {ratio:.2f}x")
        if ratio > 1.2:
            print(f"     → Disagg handles {ratio:.1f}x more load")
        elif ratio > 0.9:
            print(f"     → Similar goodput (neither wins at this QPS)")
        else:
            print(f"     → Disagg goodput is LOWER — overhead exceeds benefit at this QPS")
    else:
        print(f"  3. GOODPUT: insufficient data")
    print()

    # Overall verdict
    print("  ─────────────────────────────────────────────────")
    # Only give a verdict if we have enough data
    if bl_ttft["n"] > 0 and dg_ttft["n"] > 0:
        wins = 0
        if overhead_pct < 20:
            wins += 1
        if (bl_itl_p99_short["n"] > 0 and dg_itl_p99_short["n"] > 0
                and bl_itl_p99_short["median"] > 0
                and dg_itl_p99_short["median"] / bl_itl_p99_short["median"] < 0.85):
            wins += 1
        if dg_qps > bl_qps * 1.1:
            wins += 1

        if wins >= 2:
            print("  VERDICT: Disaggregation is BENEFICIAL for this workload/QPS.")
        elif wins == 1:
            print("  VERDICT: Disaggregation shows MARGINAL benefit. "
                  "Consider higher QPS or longer prompts.")
        else:
            print("  VERDICT: Disaggregation is NOT justified at this operating point. "
                  "Overhead exceeds benefit.")
        print("  ─────────────────────────────────────────────────")
    print()

    # GPU utilization (if available)
    gpu_rows = load_csv(os.path.join(data_dir, "exp7-gpu.csv"))
    if gpu_rows:
        print("  GPU Utilization Summary:")
        for cfg in configs:
            cfg_gpu = [r for r in gpu_rows if r.get("config") == cfg]
            gpus = sorted(set(r.get("gpu_index", "") for r in cfg_gpu))
            for gpu in gpus:
                utils = [safe_float(r["gpu_util_pct"]) for r in cfg_gpu
                         if r.get("gpu_index") == gpu]
                mems = [safe_float(r["mem_util_pct"]) for r in cfg_gpu
                        if r.get("gpu_index") == gpu]
                if utils:
                    s_u = stats(utils)
                    s_m = stats(mems)
                    print(f"    {cfg} GPU {gpu}: "
                          f"compute={s_u['mean']:.0f}% (p90={s_u['p90']:.0f}%), "
                          f"memory={s_m['mean']:.0f}% (p90={s_m['p90']:.0f}%)")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        print(f"Usage: python3 analyze.py [data_dir]")
        sys.exit(1)

    experiments = [
        ("Experiment 1: Single-Request Latency", "exp1-results.csv", analyze_exp1),
        ("Experiment 1b: Latency Decomposition", "exp1b-results.csv", analyze_exp1b),
        ("Experiment 2: Throughput Under Load", "exp2-results.csv", analyze_exp2),
        ("Experiment 3: Prefill Isolation", "exp3-results.csv", analyze_exp3),
        ("Experiment 4: Fault Tolerance", "exp4-results.csv", analyze_exp4),
        ("Experiment 5: Sequence Length Sweep", "exp5-results.csv", analyze_exp5),
        ("Experiment 6: Saturation Profiling", "exp6-results.csv", analyze_exp6),
        ("Experiment 7: Mixed Workload", "exp7-results.csv", analyze_exp7),
    ]

    print(f"llm-d Diagnostics — Analysis")
    print(f"Data directory: {data_dir}")
    print()

    for title, filename, analyzer in experiments:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"{'=' * 60}")
            print(f"  {title}")
            print(f"{'=' * 60}")
            print()
            analyzer(data_dir)
            print()
        else:
            print(f"  {title}: {filename} not found, skipping")
            print()


if __name__ == "__main__":
    main()
