#!/usr/bin/env python3
"""
Experiment 7: Mixed Workload — The Essential Characterization

Answers the fundamental question: is the disaggregation bet worth it?

Sends a realistic workload mix (short + long prompts, Poisson arrivals,
streaming responses) to both monolithic and disaggregated deployments,
then compares TTFT, inter-token latency (ITL), and throughput.

What this measures that other experiments don't:
    1. ITL (inter-token latency) — the user-perceived streaming speed.
       Disaggregation's value is keeping ITL stable under load by isolating
       decode from prefill. If ITL degrades, disaggregation isn't working.
    2. Mixed prompt lengths — disaggregation's benefit is workload-dependent.
       Short prompts = pure overhead. Long prompts = transfer tax. Mixed
       workloads reveal whether the system handles heterogeneity.
    3. Poisson arrivals — realistic, not batch. Captures queueing behavior.

Configs:
    BASELINE:  all requests to monolithic vLLM
    DISAGG-2D: through sidecars, round-robin decode-1/decode-2

Workload mix (configurable):
    80% short prompts (10 tokens, 20 max output)
    20% long prompts  (500 tokens, 50 max output)

Usage:
    python3 scripts/toolkit/exp7_mixed_workload.py
    QPS=4 DURATION_S=60 python3 scripts/toolkit/exp7_mixed_workload.py

Additional env vars:
    QPS             Target arrival rate (default: 4)
    DURATION_S      Duration of each config run (default: 60)
    SHORT_TOKENS    Short prompt token count (default: 10)
    LONG_TOKENS     Long prompt token count (default: 500)
    SHORT_MAX       Max output tokens for short prompts (default: 20)
    LONG_MAX        Max output tokens for long prompts (default: 50)
    LONG_PCT        Percentage of requests that are long (default: 20)
    GPU_SAMPLE_INTERVAL  GPU utilization sampling interval in seconds (default: 2)
"""

import math
import os
import random
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    BASELINE_URL, DISAGG_D1_URL, DISAGG_D2_URL, MODEL,
    WARMUP, DATA_DIR, PREFILL_HOST,
    build_prompt, send_streaming, send_request, CSVWriter,
    progress, dot, print_config, env, write_run_info,
)

QPS = float(env("QPS", "4"))
DURATION_S = int(env("DURATION_S", "60"))
SHORT_TOKENS = int(env("SHORT_TOKENS", "10"))
LONG_TOKENS = int(env("LONG_TOKENS", "500"))
SHORT_MAX = int(env("SHORT_MAX", "20"))
LONG_MAX = int(env("LONG_MAX", "50"))
LONG_PCT = int(env("LONG_PCT", "20"))
GPU_SAMPLE_INTERVAL = float(env("GPU_SAMPLE_INTERVAL", "2"))

SHORT_PROMPT = build_prompt(SHORT_TOKENS)
LONG_PROMPT = build_prompt(LONG_TOKENS)
DISAGG_HEADERS = {"x-prefiller-host-port": PREFILL_HOST}

FIELDS = [
    "experiment", "config", "seq", "workload_class",
    "prompt_tokens_target", "max_tokens",
    "ttft_ms", "total_ms", "itl_mean_ms", "itl_p99_ms",
    "status_code", "completion_tokens",
    "scheduled_at_s", "depart_delay_ms",
    "error",
]

GPU_FIELDS = [
    "experiment", "config", "sample_time_s",
    "gpu_index", "gpu_util_pct", "mem_util_pct",
]


def poisson_intervals(rate, duration_s, seed=None):
    """Generate Poisson-distributed inter-arrival times.

    Returns list of arrival times (seconds from start) for a Poisson
    process with the given rate, truncated to duration_s.
    """
    rng = random.Random(seed)
    arrivals = []
    t = 0.0
    while t < duration_s:
        # Exponential inter-arrival time
        interval = -math.log(1.0 - rng.random()) / rate
        t += interval
        if t < duration_s:
            arrivals.append(t)
    return arrivals


def pick_workload(seq, long_pct):
    """Deterministic workload assignment based on sequence number.

    Uses a deterministic pattern rather than random sampling so that
    the workload mix is reproducible. Exact when long_pct divides 100
    evenly (e.g., 10, 20, 25, 50); approximate otherwise.
    """
    # Every 100/long_pct requests, one is long
    if long_pct <= 0:
        return "short"
    period = max(1, 100 // long_pct)
    return "long" if seq % period == 0 else "short"


def compute_itl(token_times):
    """Compute inter-token latency statistics from token_times.

    Args:
        token_times: list of timestamps (seconds from request start)
    Returns:
        (itl_mean_ms, itl_p99_ms) or (0, 0) if insufficient tokens.
    """
    if len(token_times) < 2:
        return 0.0, 0.0
    gaps = [(token_times[i + 1] - token_times[i]) * 1000
            for i in range(len(token_times) - 1)]
    gaps.sort()
    n = len(gaps)
    mean = sum(gaps) / n

    # p99: interpolate
    idx = (n - 1) * 0.99
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    p99 = gaps[lo] + frac * (gaps[hi] - gaps[lo])

    return round(mean, 2), round(p99, 2)


def run_config(config_name, writer, gpu_writer):
    """Run the mixed workload for one config."""
    max_workers = min(max(int(QPS * 8), 16), 128)

    def pick_url(seq):
        if config_name == "BASELINE":
            return BASELINE_URL
        return DISAGG_D1_URL if seq % 2 == 1 else DISAGG_D2_URL

    def headers():
        return None if config_name == "BASELINE" else DISAGG_HEADERS

    # GPU utilization sampler (runs in background thread)
    gpu_stop = threading.Event()
    gpu_data = []

    def sample_gpu():
        """Sample GPU utilization via nvidia-smi.

        This works if the experiment runs on a node with GPUs, or if
        the user has configured SSH/oc access. If nvidia-smi is not
        available, silently produces no data.
        """
        import subprocess
        config_start = time.monotonic()
        while not gpu_stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=index,utilization.gpu,utilization.memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    sample_t = round(time.monotonic() - config_start, 2)
                    for line in result.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) == 3:
                            gpu_writer.write({
                                "experiment": "exp7",
                                "config": config_name,
                                "sample_time_s": sample_t,
                                "gpu_index": parts[0],
                                "gpu_util_pct": parts[1],
                                "mem_util_pct": parts[2],
                            })
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass  # nvidia-smi not available — skip silently
            gpu_stop.wait(GPU_SAMPLE_INTERVAL)

    gpu_thread = threading.Thread(target=sample_gpu, daemon=True)
    gpu_thread.start()

    results_lock = threading.Lock()
    all_results = []

    def do_request(scheduled_time, seq):
        wl = pick_workload(seq, LONG_PCT)
        prompt = LONG_PROMPT if wl == "long" else SHORT_PROMPT
        max_tok = LONG_MAX if wl == "long" else SHORT_MAX
        url = pick_url(seq)

        # Wait until scheduled departure
        now = time.monotonic()
        if scheduled_time > now:
            time.sleep(scheduled_time - now)

        actual_depart = time.monotonic()
        depart_delay = round((actual_depart - scheduled_time) * 1000, 2)

        r = send_streaming(url, prompt, max_tok, extra_headers=headers())

        itl_mean, itl_p99 = compute_itl(r.token_times)

        row = {
            "experiment": "exp7",
            "config": config_name,
            "seq": seq,
            "workload_class": wl,
            "prompt_tokens_target": LONG_TOKENS if wl == "long" else SHORT_TOKENS,
            "max_tokens": max_tok,
            "ttft_ms": r.ttft_ms,
            "total_ms": r.total_ms,
            "itl_mean_ms": itl_mean,
            "itl_p99_ms": itl_p99,
            "status_code": r.status,
            "completion_tokens": r.completion_tokens,
            "scheduled_at_s": round(scheduled_time - start_mono, 3),
            "depart_delay_ms": depart_delay,
            "error": r.error,
        }
        writer.write(row)
        dot()

        with results_lock:
            all_results.append((wl, r))

        return r

    # Warm-up
    for i in range(WARMUP):
        url = pick_url(i + 1)
        send_request(url, SHORT_PROMPT, SHORT_MAX, extra_headers=headers())

    # Generate Poisson arrivals — same seed for all configs so both
    # BASELINE and DISAGG see identical arrival patterns. This controls
    # for arrival-time variance and isolates the deployment topology effect.
    arrival_times = poisson_intervals(QPS, DURATION_S, seed=42)
    progress(f"    {len(arrival_times)} requests over {DURATION_S}s "
             f"(target QPS={QPS}, actual={len(arrival_times)/DURATION_S:.1f})")

    start_mono = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, arr_t in enumerate(arrival_times):
            scheduled = start_mono + arr_t
            # Stagger submission
            now = time.monotonic()
            if scheduled - now > 0.1:
                time.sleep(max(0, scheduled - now - 0.05))
            futures.append(pool.submit(do_request, scheduled, i + 1))

        for f in futures:
            f.result()

    wall_s = time.monotonic() - start_mono
    actual_qps = len(arrival_times) / wall_s if wall_s > 0 else 0

    # Stop GPU sampling
    gpu_stop.set()
    gpu_thread.join(timeout=5)

    # Summary
    ok = sum(1 for wl, r in all_results if r.status == 200)
    short_results = [r for wl, r in all_results if wl == "short" and r.status == 200]
    long_results = [r for wl, r in all_results if wl == "long" and r.status == 200]

    progress(f" {ok}/{len(all_results)} OK (actual {actual_qps:.1f} req/s)")

    if short_results:
        short_itls = [compute_itl(r.token_times)[0] for r in short_results
                      if len(r.token_times) >= 2]
        short_ttfts = [r.ttft_ms for r in short_results]
        if short_itls:
            mean_itl = sum(short_itls) / len(short_itls)
            progress(f"    Short: n={len(short_results)}, "
                     f"TTFT p50={sorted(short_ttfts)[len(short_ttfts)//2]:.0f}ms, "
                     f"ITL mean={mean_itl:.1f}ms")

    if long_results:
        long_itls = [compute_itl(r.token_times)[0] for r in long_results
                     if len(r.token_times) >= 2]
        long_ttfts = [r.ttft_ms for r in long_results]
        if long_itls:
            mean_itl = sum(long_itls) / len(long_itls)
            progress(f"    Long:  n={len(long_results)}, "
                     f"TTFT p50={sorted(long_ttfts)[len(long_ttfts)//2]:.0f}ms, "
                     f"ITL mean={mean_itl:.1f}ms")


def main():
    outfile = os.path.join(DATA_DIR, "exp7-results.csv")
    gpu_file = os.path.join(DATA_DIR, "exp7-gpu.csv")
    write_run_info("exp7", {
        "qps": QPS, "duration_s": DURATION_S,
        "short_tokens": SHORT_TOKENS, "long_tokens": LONG_TOKENS,
        "short_max": SHORT_MAX, "long_max": LONG_MAX,
        "long_pct": LONG_PCT,
        "gpu_sample_interval": GPU_SAMPLE_INTERVAL,
    })
    writer = CSVWriter(outfile, FIELDS)
    gpu_writer = CSVWriter(gpu_file, GPU_FIELDS)

    progress("=== Experiment 7: Mixed Workload ===")
    print_config()
    progress(f"  Target QPS: {QPS}")
    progress(f"  Duration: {DURATION_S}s per config")
    progress(f"  Workload: {100-LONG_PCT}% short ({SHORT_TOKENS} tok) / "
             f"{LONG_PCT}% long ({LONG_TOKENS} tok)")
    progress(f"  Output: {outfile}")
    progress(f"  GPU data: {gpu_file}")
    progress("")

    configs = [
        ("BASELINE",  "monolithic vLLM"),
        ("DISAGG-2D", "disaggregated, 2 decode replicas"),
    ]

    for config_name, desc in configs:
        progress(f"  Config: {config_name} ({desc})")
        run_config(config_name, writer, gpu_writer)
        progress("")

        # Cooldown between configs
        progress("  Cooling down (10s)...")
        time.sleep(10)
        progress("")

    writer.close()
    gpu_writer.close()
    progress("")
    progress(f"=== Experiment 7 Complete === ({outfile})")


if __name__ == "__main__":
    main()
