# llm-d Diagnostics Toolkit

A simple toolkit for kicking the tires on an [llm-d](https://github.com/llm-d/llm-d)
disaggregated inference deployment. Point it at a cluster, run the experiments,
and get a characterization you can compare against other deployments.

Python 3.8+, no external dependencies. Runs inside any Kubernetes cluster
via a test-client pod.

## What It Does

Eight experiments that answer the questions you'd ask about any new P/D deployment:

| | Experiment | Question |
|-|------------|----------|
| 1 | Latency sweep | How much overhead does disaggregation add per request? |
| 1b | Decomposition | Where does the overhead come from? (sidecar vs prefill round-trip) |
| 2 | Throughput | How does it scale under concurrent load? |
| 3 | Isolation | Does P/D separation protect light requests from heavy prefills? |
| 4 | Fault tolerance | What happens when pods die? How long to recover? (\*) |
| 5 | Sequence length | Does transfer cost scale with prompt length? |
| 6 | Saturation | At what QPS does each topology collapse? |
| 7 | Mixed workload | Under realistic conditions, is disaggregation worth it? |

(\*) Exp 4 runs locally (it kills pods via `oc delete`), not via `run.sh`.

Each experiment writes CSV data. The analyzer computes medians, confidence
intervals, and flags data quality issues.

## Quick Start

```bash
# 1. Deploy the P/D topology
oc apply -f manifests/ -n my-namespace

# 2. Create a cluster config
mkdir -p clusters/my-cluster/data
cat > clusters/my-cluster/env.sh <<EOF
export NS=my-namespace
export MODEL="my-org/my-model"
export DATA_DIR="clusters/my-cluster/data"
export PREFILL_HOST="vllm-prefill-svc.\${NS}.svc.cluster.local:8100"
EOF

# 3. Run experiments (copies toolkit to pod, runs in-cluster, copies results back)
./scripts/toolkit/run.sh clusters/my-cluster exp1
./scripts/toolkit/run.sh clusters/my-cluster all     # run all (exp1-3, exp5-7)

# 4. Fault tolerance runs from outside (it kills and restores pods)
source clusters/my-cluster/env.sh
python3 scripts/toolkit/exp4_fault.py

# 5. Analyze
./scripts/toolkit/run.sh clusters/my-cluster analyze
```

### Simulation Mode (no GPU required)

You can validate the toolkit end-to-end without GPUs using
[llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim),
a GPU-free vLLM simulator:

```bash
# 1. Deploy sim topology
oc apply -f manifests/sim/ -n my-namespace

# 2. Wait for pods to be ready
oc get pods -n my-namespace

# 3. Create a cluster config
mkdir -p clusters/my-sim/data
cat > clusters/my-sim/env.sh <<EOF
export NS=my-namespace
export MODEL="sim"
export DATA_DIR="clusters/my-sim/data"
export PREFILL_HOST="vllm-prefill-svc.\${NS}.svc.cluster.local:8100"
EOF

# 4. Run an experiment
SIM=1 ./scripts/toolkit/run.sh clusters/my-sim exp1

# 5. Analyze
./scripts/toolkit/run.sh clusters/my-sim analyze
```

**Limitations:** inference-sim returns canned responses with configurable
latency. It exercises the sidecar routing and the toolkit's measurement
pipeline, but there is no real model, no KV cache, and no NIXL transfer.
The numbers don't reflect real inference behavior. Experiment 4 (fault
tolerance) still validates pod lifecycle and recovery mechanics.

## Cluster Results

Each cluster gets its own directory under `clusters/`. Run the same toolkit
on different hardware, compare the results.

| Cluster | Hardware | Assessment |
|---------|----------|------------|
| [rdu3-t4x3](clusters/rdu3-t4x3/) | 3x Tesla T4, OpenShift | [ASSESSMENT.md](clusters/rdu3-t4x3/ASSESSMENT.md) |

**tl;dr from rdu3-t4x3:** Don't use P/D for TinyLlama (53ms overhead,
saturation cliff at QPS=24 vs baseline handling QPS=32). But the machinery
works — fault recovery is automatic, NIXL handshakes re-establish on pod
restart, transfer cost is flat at ~30ms (protocol overhead, not bandwidth),
and the overhead structure is well understood for planning larger deployments.

## Configuration

Everything is configured via environment variables — nothing is hardcoded
to a specific cluster. See [docs/configuration.md](docs/configuration.md)
for the full reference.

## Using with Claude Code

This repo includes a Claude Code skill that walks you through the full
assessment workflow:

```
/assess-cluster my-cluster-name my-namespace
```

It checks the deployment is healthy, runs experiments, analyzes the data,
and writes the assessment. It also has guided workflows for elasticity
testing (scaling decode GPUs up/down) and fault tolerance deep dives
(network partition, cascade failure, prefill redundancy).

## Repo Layout

```
scripts/toolkit/    The toolkit (Python 3, stdlib only)
manifests/          K8s manifests for a real P/D deployment
manifests/sim/      K8s manifests for inference-sim (no GPU)
clusters/           Results from each cluster assessment
docs/               Configuration reference
.claude/skills/     Claude Code skill for guided assessment
```

## TODO

- [ ] Large-model results where P/D actually wins (8B+ parameters)
- [ ] RDMA/NVLink testing (currently TCP only)
- [ ] EPP (Endpoint Picker) integration — routing is currently manual via headers
- [ ] Multi-prefill topologies

## See Also

- [llm-d](https://github.com/llm-d/llm-d) — the project
- [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark) — official
  benchmark framework (Helm-based, CI/CD scale). This toolkit is different:
  lightweight, portable, meant for hands-on cluster validation.
- [llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) — GPU-free
  vLLM simulator
