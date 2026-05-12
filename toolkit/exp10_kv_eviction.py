#!/usr/bin/env python3
"""
Experiment 10: KV Cache Retention / Eviction Timing

Measures how long a decode pod retains KV cache entries after a request
completes, and at what point re-sending the same prompt costs as much as
a cold start.  This matters for routing: if the cache lifetime is shorter
than inter-request gaps for a given user, the router gains nothing by
pinning sessions to a specific decode pod.

Method:
    1. Establish cold and warm TTFT distributions (10 samples each).
       The decision boundary is the midpoint between warm p75 and cold p25,
       providing a statistically grounded threshold rather than an arbitrary
       midpoint from single measurements.
    2. Delay sweep: for each delay in EVICTION_DELAYS:
       a. Warm the cache by sending the prompt.
       b. Sleep for `delay` seconds.
       c. Resend the same prompt. Classify as hit/miss vs threshold.
       d. Repeat EVICTION_RUNS times.
    3. Optionally (BG_LOAD=1) repeat the sweep while a background thread
       sends unrelated prompts at 1 QPS.
    4. Pressure test: warm the cache, then send PRESSURE_PROMPTS_N distinct
       prompts (filling cache slots), then resend the original. This tests
       eviction under memory pressure — what production workloads experience
       — rather than just time-based decay.

Note: long delays (300s+) make this experiment slow -- total runtime
scales with sum(EVICTION_DELAYS) * EVICTION_RUNS * (1 + BG_LOAD).

Usage:
    python3 toolkit/exp10_kv_eviction.py
    EVICTION_DELAYS=10,30,60,120,300 EVICTION_RUNS=10 python3 toolkit/exp10_kv_eviction.py

Additional env vars:
    EVICTION_DELAYS      Comma-separated delays in seconds (default: 10,30,60)
    EVICTION_RUNS        Repetitions per delay (default: 5)
    CACHE_PROMPT_TOKENS  Prompt size in tokens (default: 500)
    BG_LOAD              Set to 1 to repeat sweep under background load (default: 0)
    PRESSURE_PROMPTS_N   Distinct prompts for pressure test (default: 20)
"""

import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    DATA_DIR,
    MAX_TOKENS,
    PinnedConnection,
    build_prompt,
    decode_direct_url_by_ip,
    discover_pod_ips,
    dot,
    env,
    print_config,
    progress,
    write_run_info,
)
from schemas import CacheHit, EvictionPhase, Exp10Row, TypedCSVWriter

EVICTION_DELAYS = [int(x) for x in env("EVICTION_DELAYS", "10,30,60").split(",")]
EVICTION_RUNS = int(env("EVICTION_RUNS", "5"))
CACHE_PROMPT_TOKENS = int(env("CACHE_PROMPT_TOKENS", "500"))
BG_LOAD = env("BG_LOAD", "0") == "1"

CACHE_PROMPT = build_prompt(CACHE_PROMPT_TOKENS)
# Background prompt must have truly distinct content (not just different
# length) to avoid prefix cache contamination. build_prompt() repeats the
# same sentence, so longer prompts share the same prefix.
BG_PROMPT = "Background load request: " + build_prompt(CACHE_PROMPT_TOKENS)
PRESSURE_PROMPTS_N = int(env("PRESSURE_PROMPTS_N", "20"))  # number of distinct prompts for pressure test



def record(writer, pod, run, delay, bg, phase, r, cache_hit=CacheHit.NA):
    writer.write({
        "experiment": "exp10",
        "pod": pod,
        "run": run,
        "delay_s": delay,
        "background_load": int(bg),
        "phase": phase,
        "ttft_ms": r.ttft_ms,
        "total_ms": r.total_ms,
        "status_code": r.status,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "cache_hit": cache_hit,
        "error": r.error,
    })


def bg_sender(bg_conn, stop_event):
    """Send background requests at ~1 QPS until stopped."""
    try:
        while not stop_event.is_set():
            bg_conn.send(BG_PROMPT, MAX_TOKENS)
            stop_event.wait(1.0)
    finally:
        bg_conn.close()


def establish_distributions(conn, writer, n=10):
    """Measure cold and warm TTFT distributions for proper classification.

    Returns (cold_ttfts, warm_ttfts, threshold) where threshold is the
    midpoint between the warm p75 and cold p25 — a statistically grounded
    decision boundary rather than an arbitrary midpoint of single values.
    """
    progress("  Establishing cold/warm distributions...")
    cold_ttfts = []
    warm_ttfts = []
    # Cold prompts must have TRULY distinct content — not just different
    # lengths. build_prompt(N) repeats BASE_SENTENCE, so longer prompts
    # share the same prefix as shorter ones. We add a unique suffix to
    # each cold prompt to prevent prefix cache contamination.
    for i in range(n):
        # Cold: unique prompt content (different suffix forces cache miss)
        cold_base = build_prompt(CACHE_PROMPT_TOKENS)
        cold_prompt = f"Cold measurement {i}: {cold_base}"
        r = conn.send(cold_prompt, MAX_TOKENS)
        cold_ttfts.append(r.ttft_ms)
        record(writer, conn.pod_name, i + 1, 0, False, EvictionPhase.DIST_COLD, r)

        # Warm: send CACHE_PROMPT, then immediately resend (guaranteed hit)
        conn.send(CACHE_PROMPT, MAX_TOKENS)  # prime
        r = conn.send(CACHE_PROMPT, MAX_TOKENS)  # hit
        warm_ttfts.append(r.ttft_ms)
        record(writer, conn.pod_name, i + 1, 0, False, EvictionPhase.DIST_WARM, r)
        dot()

    cold_ttfts.sort()
    warm_ttfts.sort()
    # Decision boundary: midpoint between cold p25 and warm p75
    # This separates the distributions where they're closest
    cold_p25 = cold_ttfts[len(cold_ttfts) // 4]
    warm_p75 = warm_ttfts[3 * len(warm_ttfts) // 4]
    threshold = (cold_p25 + warm_p75) / 2.0

    progress(" done")
    progress(f"    Cold  TTFT: median={cold_ttfts[len(cold_ttfts)//2]:.1f}ms, "
             f"p25={cold_ttfts[len(cold_ttfts)//4]:.1f}ms")
    progress(f"    Warm  TTFT: median={warm_ttfts[len(warm_ttfts)//2]:.1f}ms, "
             f"p75={warm_ttfts[3*len(warm_ttfts)//4]:.1f}ms")
    progress(f"    Threshold: {threshold:.1f}ms")
    return cold_ttfts, warm_ttfts, threshold


def run_sweep(conn, writer, bg_load, threshold):
    """Run the delay sweep, optionally under background load.

    Delays are randomized to prevent order effects: running delays in
    ascending order could bias short delays (cache warm from setup) or
    long delays (GPU thermal state drift). Each (delay, run) pair is
    independent — we re-warm the cache before each measurement.
    """
    phase_warm = EvictionPhase.BG_WARM if bg_load else EvictionPhase.WARM
    phase_after = EvictionPhase.BG_AFTER_DELAY if bg_load else EvictionPhase.AFTER_DELAY
    stop_event = threading.Event()

    if bg_load:
        bg_conn = PinnedConnection(conn.url, pod_name=conn.pod_name + "-bg")
        bg_thread = threading.Thread(target=bg_sender, args=(bg_conn, stop_event), daemon=True)
        bg_thread.start()

    # Build all (delay, run) pairs and randomize order
    trials = [(delay, run)
              for delay in EVICTION_DELAYS
              for run in range(1, EVICTION_RUNS + 1)]
    random.shuffle(trials)

    progress(f"    {len(trials)} trials (randomized order)")
    try:
        for trial_idx, (delay, run) in enumerate(trials):
            progress(f"    [{trial_idx+1}/{len(trials)}] delay={delay}s run={run}: ",
                     end="")

            # Warm the cache (fresh for each trial)
            r_warm = conn.send(CACHE_PROMPT, MAX_TOKENS)
            record(writer, conn.pod_name, run, delay, bg_load, phase_warm, r_warm)

            # Wait
            time.sleep(delay)

            # Resend same prompt
            r_after = conn.send(CACHE_PROMPT, MAX_TOKENS)
            hit = CacheHit.YES if r_after.ttft_ms < threshold else CacheHit.NO
            record(writer, conn.pod_name, run, delay, bg_load, phase_after, r_after, cache_hit=hit)
            progress(f"{'HIT' if hit == CacheHit.YES else 'MISS'} "
                     f"({r_after.ttft_ms:.1f}ms)")
    finally:
        if bg_load:
            stop_event.set()


def run_pressure_test(conn, writer, threshold):
    """Evict by filling cache with other prompts, not by waiting.

    Tests real-world eviction: warm the cache, then send N distinct prompts
    (filling cache slots), then resend the original. This measures eviction
    under memory pressure, which is what production workloads experience.
    """
    progress("  Cache pressure test:")
    # Generate N prompts with truly distinct content (not just length)
    base = build_prompt(CACHE_PROMPT_TOKENS)
    pressure_prompts = [f"Pressure prompt {i}: {base}" for i in range(PRESSURE_PROMPTS_N)]

    for run in range(1, EVICTION_RUNS + 1):
        # Warm the target cache entry
        conn.send(CACHE_PROMPT, MAX_TOKENS)
        r_warm = conn.send(CACHE_PROMPT, MAX_TOKENS)
        record(writer, conn.pod_name, run, 0, False, EvictionPhase.PRESSURE_WARM, r_warm)

        # Fill cache with different prompts
        for p in pressure_prompts:
            conn.send(p, MAX_TOKENS)
            dot()

        # Resend original — was it evicted?
        r_after = conn.send(CACHE_PROMPT, MAX_TOKENS)
        hit = CacheHit.YES if r_after.ttft_ms < threshold else CacheHit.NO
        record(writer, conn.pod_name, run, 0, False, EvictionPhase.PRESSURE_AFTER, r_after, cache_hit=hit)
        progress(f"    run {run}: {PRESSURE_PROMPTS_N} eviction prompts → "
                 f"{'HIT' if hit == CacheHit.YES else 'MISS'} "
                 f"({r_after.ttft_ms:.1f}ms vs threshold {threshold:.1f}ms)")


def main():
    outfile = os.path.join(DATA_DIR, "exp10-results.csv")
    write_run_info("exp10", {
        "eviction_delays": EVICTION_DELAYS,
        "eviction_runs": EVICTION_RUNS,
        "cache_prompt_tokens": CACHE_PROMPT_TOKENS,
        "bg_load": BG_LOAD,
    })
    writer = TypedCSVWriter(outfile, Exp10Row)

    progress("=== Experiment 10: KV Cache Retention / Eviction ===")
    print_config()
    progress(f"  Delays: {EVICTION_DELAYS}")
    progress(f"  Runs per delay: {EVICTION_RUNS}")
    progress(f"  Cache prompt tokens: {CACHE_PROMPT_TOKENS}")
    progress(f"  Background load: {'ON' if BG_LOAD else 'OFF'}")
    progress(f"  Output: {outfile}")
    progress("")

    # Discover decode pod
    pods = discover_pod_ips("app=vllm-decode")
    if not pods:
        progress("ERROR: no decode pods found (label=app=vllm-decode)")
        sys.exit(1)
    pod_name, pod_ip = pods[0]
    url = decode_direct_url_by_ip(pod_ip)
    conn = PinnedConnection(url, pod_name=pod_name)
    progress(f"  Target pod: {pod_name} ({pod_ip})")
    progress(f"  URL: {url}")
    progress("")

    # Warm-up
    progress("  Warming up...")
    conn.warmup(CACHE_PROMPT, MAX_TOKENS)

    # Establish cold/warm distributions for proper classification
    _cold_ttfts, _warm_ttfts, threshold = establish_distributions(conn, writer)

    # Delay-based eviction sweep
    progress("  Delay sweep (no background load):")
    run_sweep(conn, writer, False, threshold)

    # Sweep with background load (if enabled)
    if BG_LOAD:
        progress("  Delay sweep (with background load):")
        run_sweep(conn, writer, True, threshold)

    # Pressure-based eviction test (fill cache with other prompts)
    run_pressure_test(conn, writer, threshold)

    conn.close()
    writer.close()
    progress(f"\n=== Experiment 10 Complete === ({outfile})")


if __name__ == "__main__":
    main()
