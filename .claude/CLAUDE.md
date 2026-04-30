# llm-d Diagnostics Toolkit

## What This Is

A portable toolkit for validating llm-d disaggregated inference deployments.
Run it against a cluster, get a performance characterization and fault
tolerance assessment.

## Key Paths

- `toolkit/` — Python toolkit (zero external dependencies)
- `toolkit/client.py` — shared HTTP client, config, CSV writer
- `toolkit/analyze.py` — statistical analysis of experiment CSVs
- `manifests/` — K8s manifests for real GPU deployment
- `manifests/sim/` — K8s manifests for inference-sim (GPU-free)
- `clusters/<name>/` — per-cluster data and assessments

## Skills

- `/cluster-assessor <cluster-name> <namespace>` — run the full toolkit
  against a live cluster and produce an assessment
- `/cluster-reporter <cluster-name> [--compare <other-cluster>]` — generate
  standardized ASSESSMENT.md and REPORT.md from collected experiment data

## Conventions

- All toolkit configuration is via environment variables (see client.py)
- Never hardcode cluster-specific values (namespace, node names, URLs)
- Two main commands: `characterize` (non-destructive) and `fault-test` (destructive)
- Individual experiments: `latency`, `decompose`, `throughput`, `isolation`,
  `seqlen`, `saturation`, `mixed`, `fault`
- Always confirm before running `fault-test` or `fault` (kills pods)
- Assessments go in `clusters/<name>/ASSESSMENT.md` with three sections:
  Setup, Performance, Assessment
