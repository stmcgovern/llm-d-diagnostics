#!/usr/bin/env python3
"""
Experiment 1b: Latency Decomposition

Isolates each component of the disaggregated request path:
    A. Baseline:        client -> prefill vLLM (8100)
    B. Direct decode:   client -> decode vLLM (8001), no sidecar
    C. Sidecar-only:    client -> sidecar (8000) -> decode vLLM (8001), no disagg
    D. Disaggregated:   client -> sidecar (8000) -> prefill (8100) -> NIXL -> decode

Derived:
    T_sidecar    = C - B
    T_prefill_rt = D - C  (includes NIXL)
    T_overhead   = D - A

Usage: python3 /scripts/toolkit/exp1b_decompose.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    BASELINE_URL, DISAGG_D1_URL, DECODE_DIRECT_URL, MODEL,
    MAX_TOKENS, WARMUP, RUNS, DATA_DIR, PREFILL_HOST,
    build_prompt, send_request, CSVWriter,
    progress, dot, print_config, env, write_run_info,
)

DECOMPOSE_RUNS = int(env("RUNS", "30"))

DECOMPOSE_PROMPT = int(env("DECOMPOSE_PROMPT", "50"))
PROMPT = build_prompt(DECOMPOSE_PROMPT)

CONFIGS = [
    ("A-prefill-direct",  BASELINE_URL,     None,
     "client -> prefill:8100 (baseline)"),
    ("B-decode-direct",   DECODE_DIRECT_URL, None,
     "client -> decode:8001 (bypass sidecar)"),
    ("C-sidecar-only",    DISAGG_D1_URL,    None,
     "client -> sidecar:8000 -> decode:8001 (no disagg)"),
    ("D-disaggregated",   DISAGG_D1_URL,    {"x-prefiller-host-port": PREFILL_HOST},
     "client -> sidecar -> prefill -> NIXL -> decode"),
]

FIELDS = [
    "experiment", "config", "run",
    "ttft_ms", "total_ms", "status_code", "completion_tokens", "error",
]


def main():
    outfile = os.path.join(DATA_DIR, "exp1b-results.csv")
    write_run_info("exp1b", {"decompose_prompt_tokens": DECOMPOSE_PROMPT,
                             "runs": DECOMPOSE_RUNS})
    writer = CSVWriter(outfile, FIELDS)

    progress("=== Experiment 1b: Latency Decomposition ===")
    print_config()
    progress(f"  Runs per config: {DECOMPOSE_RUNS}")
    progress(f"  Output: {outfile}")
    progress("")

    for config_name, url, headers, desc in CONFIGS:
        progress(f"  {config_name}: {desc}")

        # Warm-up
        for _ in range(WARMUP):
            send_request(url, PROMPT, MAX_TOKENS, extra_headers=headers)

        # Measured runs
        for run in range(1, DECOMPOSE_RUNS + 1):
            r = send_request(url, PROMPT, MAX_TOKENS, extra_headers=headers)

            writer.write({
                "experiment": "exp1b",
                "config": config_name,
                "run": run,
                "ttft_ms": r.ttft_ms,
                "total_ms": r.total_ms,
                "status_code": r.status,
                "completion_tokens": r.completion_tokens,
                "error": r.error,
            })
            dot()

        progress(" done")

    writer.close()
    progress(f"=== Experiment 1b Complete === ({outfile})")


if __name__ == "__main__":
    main()
