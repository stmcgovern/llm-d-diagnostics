#!/usr/bin/env python3
"""
Experiment 1: Single-Request Latency

Measures per-request overhead of disaggregation across prompt lengths.
Sequential requests, no concurrency.

Usage (in-pod):
    python3 /scripts/toolkit/exp1_latency.py

Override defaults:
    MODEL="meta-llama/..." RUNS=10 python3 exp1_latency.py

Additional env vars:
    PROMPT_LENGTHS   Comma-separated token targets (default: 10,50,100,500,1000)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    BASELINE_URL, DISAGG_D1_URL, DISAGG_D2_URL, MODEL,
    MAX_TOKENS, WARMUP, RUNS, DATA_DIR,
    build_prompt, send_request, send_disagg, CSVWriter,
    progress, dot, print_config, env, write_run_info,
)

PROMPT_LENGTHS = [int(x) for x in env("PROMPT_LENGTHS", "10,50,100,500,1000").split(",")]

CONFIGS = [
    ("BASELINE",  BASELINE_URL,  False),
    ("DISAGG-D1", DISAGG_D1_URL, True),
    ("DISAGG-D2", DISAGG_D2_URL, True),
]

FIELDS = [
    "experiment", "config", "run", "prompt_tokens_target", "max_tokens",
    "ttft_ms", "total_ms", "status_code",
    "prompt_tokens_actual", "completion_tokens", "error",
]


def main():
    outfile = os.path.join(DATA_DIR, "exp1-results.csv")
    write_run_info("exp1", {"prompt_lengths": PROMPT_LENGTHS})
    writer = CSVWriter(outfile, FIELDS)

    progress("=== Experiment 1: Single-Request Latency ===")
    print_config()
    progress(f"  Prompt lengths: {PROMPT_LENGTHS}")
    progress(f"  Output: {outfile}")
    progress("")

    for ptokens in PROMPT_LENGTHS:
        prompt = build_prompt(ptokens)
        progress(f"--- Prompt target: {ptokens} tokens ---")

        for config_name, url, use_disagg in CONFIGS:
            progress(f"  {config_name}: ", end="")

            # Warm-up
            for _ in range(WARMUP):
                if use_disagg:
                    send_disagg(url, prompt, MAX_TOKENS)
                else:
                    send_request(url, prompt, MAX_TOKENS)

            # Measured runs
            for run in range(1, RUNS + 1):
                if use_disagg:
                    r = send_disagg(url, prompt, MAX_TOKENS)
                else:
                    r = send_request(url, prompt, MAX_TOKENS)

                writer.write({
                    "experiment": "exp1",
                    "config": config_name,
                    "run": run,
                    "prompt_tokens_target": ptokens,
                    "max_tokens": MAX_TOKENS,
                    "ttft_ms": r.ttft_ms,
                    "total_ms": r.total_ms,
                    "status_code": r.status,
                    "prompt_tokens_actual": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "error": r.error,
                })
                dot()

            progress(" done")
        progress("")

    writer.close()
    progress(f"=== Experiment 1 Complete === ({outfile})")


if __name__ == "__main__":
    main()
