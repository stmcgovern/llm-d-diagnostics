#!/usr/bin/env python3
"""
Experiment 5: Sequence Length Sweep

Runs the exp1b 4-config decomposition at multiple prompt lengths to find the
crossover point where KV cache transfer time dominates TTFT.

KV cache size grows linearly with sequence length. At short sequences the
transfer tax is negligible; at some crossover point, transfer time exceeds
prefill compute time. Finding that crossover is the most important
characterization of a disaggregated deployment.

Configs (identical to exp1b):
    A. Baseline:        client -> prefill vLLM (8100)
    B. Direct decode:   client -> decode vLLM (8001), no sidecar
    C. Sidecar-only:    client -> sidecar (8000) -> decode vLLM (8001), no disagg
    D. Disaggregated:   client -> sidecar (8000) -> prefill (8100) -> NIXL -> decode

Derived at each sequence length:
    T_sidecar    = C - B
    T_transfer   = D - C  (includes NIXL KV transfer)
    T_overhead   = D - A

Crossover: the prompt length where T_transfer > median(A).

Usage:
    python3 scripts/toolkit/exp5_seqlen_sweep.py
    SWEEP_LENGTHS=10,100,1000,4096 python3 scripts/toolkit/exp5_seqlen_sweep.py

Additional env vars:
    SWEEP_LENGTHS   Comma-separated prompt token targets (default: 10,50,100,250,500,1000,2000,4096)
    RUNS            Runs per config per length (default: 30)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    BASELINE_URL, DISAGG_D1_URL, DECODE_DIRECT_URL, MODEL,
    MAX_TOKENS, WARMUP, DATA_DIR, PREFILL_HOST,
    build_prompt, send_request, CSVWriter,
    progress, dot, print_config, env, write_run_info,
)

SWEEP_LENGTHS = [int(x) for x in env("SWEEP_LENGTHS", "10,50,100,250,500,1000,2000,4096").split(",")]
SWEEP_RUNS = int(env("RUNS", "30"))

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
    "experiment", "config", "prompt_tokens_target", "run",
    "ttft_ms", "total_ms", "status_code",
    "prompt_tokens_actual", "completion_tokens", "error",
]


def main():
    outfile = os.path.join(DATA_DIR, "exp5-results.csv")
    write_run_info("exp5", {"sweep_lengths": SWEEP_LENGTHS, "runs": SWEEP_RUNS})
    writer = CSVWriter(outfile, FIELDS)

    progress("=== Experiment 5: Sequence Length Sweep ===")
    print_config()
    progress(f"  Sweep lengths: {SWEEP_LENGTHS}")
    progress(f"  Runs per config per length: {SWEEP_RUNS}")
    progress(f"  Output: {outfile}")
    progress("")

    for ptokens in SWEEP_LENGTHS:
        prompt = build_prompt(ptokens)
        progress(f"--- Prompt target: {ptokens} tokens ---")

        for config_name, url, headers, desc in CONFIGS:
            progress(f"  {config_name}: ", end="")

            # Warm-up
            for _ in range(WARMUP):
                send_request(url, prompt, MAX_TOKENS, extra_headers=headers)

            # Measured runs
            for run in range(1, SWEEP_RUNS + 1):
                r = send_request(url, prompt, MAX_TOKENS, extra_headers=headers)

                writer.write({
                    "experiment": "exp5",
                    "config": config_name,
                    "prompt_tokens_target": ptokens,
                    "run": run,
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
    progress("")
    progress(f"=== Experiment 5 Complete === ({outfile})")


if __name__ == "__main__":
    main()
