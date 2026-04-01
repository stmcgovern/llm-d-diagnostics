#!/bin/bash
# Cluster config: rdu3-t4x3 (3x T4 GPUs)
#
# Source this before running experiments:
#   source clusters/rdu3-t4x3/env.sh
#   ./scripts/toolkit/run.sh clusters/rdu3-t4x3 exp5

export NS=llm-d
export MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export DATA_DIR="clusters/rdu3-t4x3/data"

# In-cluster URLs are the defaults in client.py — no overrides needed.
# Experiments run inside the test-client pod via `oc exec`.
export PREFILL_HOST="vllm-prefill-svc.${NS}.svc.cluster.local:8100"
