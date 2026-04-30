---
name: cluster-reporter
description: Generate standardized ASSESSMENT.md and REPORT.md from collected experiment data
allowed-tools: Read, Bash, Grep, Glob, Write, Edit, Agent
argument-hint: <cluster-name> [--compare <other-cluster>]
---

# Report Cluster

Generate standardized assessment and report documents from experiment data
already collected in `clusters/<cluster-name>/data/`. This is the complement
to `/assess-cluster` which runs experiments; this skill writes them up.

**Arguments:** `$ARGUMENTS`
- First argument: cluster name (directory under `clusters/`)
- `--compare <name>`: optional second cluster for cross-model/cross-hardware comparison

## Prerequisites

Verify the data exists before starting:

```bash
ls clusters/<cluster-name>/data/exp*-results.csv
cat clusters/<cluster-name>/env.sh
```

Required: `env.sh` (cluster metadata) and at least some `exp*-results.csv` files.
Optional: `exp4-results.csv` (fault tolerance), `analysis.txt` (pre-computed stats).

If `analysis.txt` doesn't exist, generate it:

```bash
python3 toolkit/analyze.py clusters/<cluster-name>/data/
```

## Dimensions That Vary Across Clusters

Every report should clearly identify where this cluster sits along these axes:

| Dimension | Examples | Where to find |
|-----------|----------|---------------|
| **Hardware** | T4, A100, H100 | env.sh, node specs |
| **Model** | TinyLlama 1.1B, Phi-3 3.8B, Llama-3 70B | env.sh `$MODEL` |
| **Topology** | 1P+2D, 2P+2D, 1P+4D | manifest count |
| **Network** | TCP/OVN-Kubernetes, RDMA/InfiniBand | cluster config |
| **Software** | vLLM version, sidecar version | env.sh images |
| **Scale** | 3 GPUs, 8 GPUs, 64 GPUs | env.sh, manifests |

These dimensions determine which findings are portable (fault tolerance,
protocol behavior) vs which are specific to this deployment (absolute
latencies, saturation points).

## Document 1: ASSESSMENT.md (Executive Summary)

Write `clusters/<cluster-name>/ASSESSMENT.md`. This is the one-page version.
Someone should be able to read this in 5 minutes and know whether disagg
is worth it for this deployment.

### Template

```markdown
# <cluster-name> -- llm-d Assessment

**Date:** <month year>
**Author:** Sean McGovern, Red Hat

## Setup

<N>x <GPU-type> on <platform> (<server-model>, <CNI>, <RDMA or no RDMA>).

| Role | Node | GPU | Port |
|------|------|-----|------|
| Prefill (kv_producer) | <node> | <gpu> | 8100 |
| Decode-1 + sidecar (kv_consumer) | <node> | <gpu> | 8000 |
| ... | ... | ... | ... |

- **Model:** <model-name> (<params>, <dtype>). <one sentence on why this model>
- **vLLM:** <version>, NixlConnector, UCX over <transport>
- **Sidecar:** <image:tag>
- **Routing:** <manual or EPP>

## Performance

### Overhead: <N>ms fixed cost

<overhead table from exp1b decomposition>

### Throughput: <one-line verdict>

<concurrency table from exp2>

### Isolation: <one-line verdict>

<ITL comparison from exp3>

### Transfer cost: <one-line verdict>

<sequence length sweep from exp5>

### Saturation: <one-line verdict>

<QPS sweep from exp6>

### Mixed workload verdict: <JUSTIFIED | MARGINAL | NOT JUSTIFIED>

<key numbers from exp7: TTFT overhead, ITL ratio, goodput>

### Fault tolerance: <one-line verdict>

<table of scenarios from exp4: scenario, result, recovery time>

Key findings:
- <bullet points on what worked, what broke, what surprised>

## Assessment

<2-3 paragraphs: recommendation, what we learned, what to try next>

---

*Raw data in `data/`. Detailed analysis in [REPORT.md](REPORT.md).
Generated with the [llm-d diagnostics toolkit](../../README.md).*
```

### Writing Guidelines for ASSESSMENT.md

- **Lead with the verdict.** First sentence of Assessment section should be
  "Use/Don't use P/D disaggregation for <model>."
- **Pick the numbers that tell the story.** Don't dump all stats.
- **Every table needs a one-line interpretation** above or below it.
- **Flag anomalies.** If a number is surprising, say so and say why.
- **Separate portable from specific.** Fault tolerance findings generalize.
  Absolute latencies do not.

## Document 2: REPORT.md (Technical Detail)

Write `clusters/<cluster-name>/REPORT.md`. This is the full technical report
for someone who wants to reproduce or challenge the findings.

### Template

```markdown
# Characterizing llm-d Disaggregated Inference on <platform>

**Cluster:** <cluster-name>
**Date:** <month year>
**Author:** Sean McGovern, Red Hat

---

## Abstract

<3-5 sentences: what we deployed, what we measured, key findings>

---

## 1. System Under Test

<component table, topology diagram, why this model>

## 2. Experiments and Findings

### 2.1 Experiment 1: Single-Request Latency
<question, method, table, findings>

### 2.2 Experiment 1b: Latency Decomposition
<question, method, decomposition math, key insight>

### 2.3 Experiment 2: Throughput Under Load
<question, method, concurrency table, findings>

### 2.4 Experiment 3: Prefill Isolation
<question, method, ITL comparison, findings>

### 2.5 Experiment 4: Fault Tolerance
<per-scenario subsections: 4a, 4b, 4d, 4e, 4f, 4h, 4i, 4j, 4k, 4l>

### 2.6 Experiment 5: Sequence Length Sweep
<question, method, table, transfer cost analysis>

### 2.7 Experiment 6: Saturation Profiling
<question, method, QPS table, saturation cliff>

### 2.8 Experiment 7: Mixed Workload
<question, method, TTFT/ITL/goodput comparison, verdict>

## 3. Overhead Budget

<decomposition diagram showing where time goes>

## 4. Limitations

<numbered list of caveats, data quality issues, untested scenarios>

## 5. Predictions for Larger Models / Next Topology

<hypothesis table: what we expect to change and why>

## 6. Related Work

<links to llm-d, vLLM, inference-sim, this toolkit>
```

### Writing Guidelines for REPORT.md

- **Every experiment gets:** Question, Method, Results table, Findings list.
- **Show the math.** Decomposition formulas, sum checks, correlation values.
- **Quote data quality.** Error rates, outlier counts, filtered runs.
- **Exp4 subsections should include:** prediction, what happened, recovery
  timeline, and what it means for production.
- **Limitations are not apologies.** They scope the claims. Each limitation
  should suggest what would change if it were removed.

## Cross-Cluster Comparison (--compare)

When `--compare <other-cluster>` is specified, add a comparison section to
both ASSESSMENT.md and REPORT.md.

### What to Compare

Read both clusters' data and `env.sh`. Identify what changed between them
(model, hardware, topology, software) and what stayed the same.

#### Same hardware, different model (e.g., TinyLlama vs Phi-3 on T4x3)

Focus on:
- Which quantities are model-independent (sidecar overhead, NIXL protocol)
- Which scale with model size (prefill compute, recovery time, saturation point)
- Whether the isolation crossover happened (disagg hurts small models, helps large)
- Fault tolerance: same mechanisms, different timings

#### Same model, different hardware (e.g., T4 vs A100)

Focus on:
- Overhead budget changes (RDMA vs TCP, faster sidecar)
- Saturation point changes (GPU throughput ceiling)
- Recovery time changes (model load speed)

#### Different topology (e.g., 1P+2D vs 2P+2D)

Focus on:
- Prefill redundancy (SPOF eliminated?)
- Throughput scaling (does second prefill help?)
- Discovery behavior (ZMQ handles multi-prefill?)

### Comparison Table Template

```markdown
## Cross-Model Comparison: <Model-A> vs <Model-B>

| Metric | <Model-A> | <Model-B> | Portable? |
|--------|-----------|-----------|-----------|
| Overhead (exp1b) | Xms | Yms | Partly (sidecar yes, prefill RT scales) |
| Saturation cliff | QPS=N | QPS=M | No (model-dependent) |
| ITL isolation ratio | 0.Xx | Y.Xx | No (crossover between models) |
| Recovery time (prefill) | Xs | Ys | No (model-load dominated) |
| NIXL handshake | Automatic | Automatic | **Yes** |
| Sidecar fallback | Works | Works | **Yes** |
| Rolling update safe | No | No | **Yes** (infrastructure) |
| ZMQ cache bug (4j) | N/A | Observed | **Yes** (protocol-level) |
```

Portable findings (marked **Yes**) are the most valuable -- they apply to
any deployment regardless of model or hardware.

## Data Quality Checks

Before writing either document, verify the data:

1. **Completeness:** Which experiments have data? Flag any missing.
2. **Error rates:** Check for truncated responses (completion_tokens < max_tokens),
   non-200 status codes, timeout errors.
3. **Outliers:** Look for values >3x median in each config. Note but don't
   silently drop them.
4. **Sample size:** n >= 20 for medians, n >= 30 for confidence intervals.
   Flag configs with fewer runs.
5. **Decomposition sum check:** T_sidecar + T_prefill_rt should be within
   5% of T_overhead. If not, investigate.

## Workflow Summary

1. Read `env.sh` and all `exp*-results.csv` files
2. Run `analyze.py` if `analysis.txt` is missing
3. Check data quality (completeness, errors, outliers)
4. Write ASSESSMENT.md (executive summary, 1 page)
5. Write REPORT.md (technical detail, full experiment coverage)
6. If `--compare`: read other cluster's data, add comparison sections
7. Confirm with user before writing files
