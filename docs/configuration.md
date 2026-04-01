# Configuration

Everything is configured via environment variables so nothing is hardcoded
to a specific cluster.

## Global (all experiments)

| Variable | Default | What it does |
|----------|---------|-------------|
| `NS` | `default` | Kubernetes namespace |
| `MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Model name for API requests |
| `SIM` | (unset) | Set to `1` for inference-sim mode (HTTP, no TLS) |
| `BASELINE_URL` | `http://vllm-prefill-svc:8100/v1/completions` | Prefill direct endpoint |
| `DISAGG_D1_URL` | `https://vllm-decode-svc:8000/v1/completions` | Decode-1 through sidecar |
| `DISAGG_D2_URL` | `https://vllm-decode-2-svc:8000/v1/completions` | Decode-2 through sidecar |
| `RUNS` | `20` | Measured runs per config |
| `WARMUP` | `3` | Warmup requests (discarded) |
| `MAX_TOKENS` | `20` | Max completion tokens |
| `DATA_DIR` | `data` | Where CSV results get written |

## Per-experiment overrides

| Variable | Default | Experiment | What it does |
|----------|---------|------------|-------------|
| `PROMPT_LENGTHS` | `10,50,100,500,1000` | exp1 | Comma-separated prompt lengths |
| `CONCURRENCY_LEVELS` | `1,2,4,8,16` | exp2 | Comma-separated concurrency levels |
| `HEAVY_PROMPT_TOKENS` | `1000` | exp3 | Heavy request prompt size |
| `LIGHT_PROMPT_TOKENS` | `10` | exp3 | Light request prompt size |
| `SWEEP_LENGTHS` | `10,50,100,250,500,1000,2000,4096` | exp5 | Sequence lengths to sweep |
| `QPS_LEVELS` | `1,2,4,8,12,16,24,32` | exp6 | QPS levels to test |
| `DURATION_S` | `30` (exp6), `60` (exp7) | exp6, exp7 | Seconds per config/QPS level |
| `SLO_MULT` | `2.0` | exp6 | Saturation threshold (p99 > N × baseline p50) |
| `QPS` | `4` | exp7 | Target arrival rate |
| `LONG_PCT` | `20` | exp7 | Percentage of long prompts in mix |
| `SHORT_TOKENS` | `10` | exp7 | Short prompt token count |
| `LONG_TOKENS` | `500` | exp7 | Long prompt token count |

Full list in [`scripts/toolkit/client.py`](../scripts/toolkit/client.py) and each experiment script.

## Cluster config file

Each cluster has an `env.sh` that sets the variables for that deployment:

```bash
# clusters/my-cluster/env.sh
export NS=my-namespace
export MODEL="my-org/my-model"
export DATA_DIR="clusters/my-cluster/data"
export PREFILL_HOST="vllm-prefill-svc.${NS}.svc.cluster.local:8100"
```

Source it before running experiments directly, or let `run.sh` source it
automatically.
