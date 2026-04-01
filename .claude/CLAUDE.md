# llm-d Diagnostics Toolkit

## What This Is

A portable toolkit for validating llm-d disaggregated inference deployments.
Run it against a cluster, get a performance characterization and fault
tolerance assessment.

## Key Paths

- `scripts/toolkit/` — Python toolkit (zero external dependencies)
- `scripts/toolkit/client.py` — shared HTTP client, config, CSV writer
- `scripts/toolkit/analyze.py` — statistical analysis of experiment CSVs
- `manifests/` — K8s manifests for real GPU deployment
- `manifests/sim/` — K8s manifests for inference-sim (GPU-free)
- `clusters/<name>/` — per-cluster data and assessments

## Skills

- `/assess-cluster <cluster-name> <namespace>` — run the full toolkit
  against a live cluster and produce an assessment

## Conventions

- All toolkit configuration is via environment variables (see client.py)
- Never hardcode cluster-specific values (namespace, node names, URLs)
- Experiments 1-3 run inside the test-client pod; experiment 4 runs locally
  (it kills pods via `oc`)
- Always confirm before running experiment 4 (destructive)
- Assessments go in `clusters/<name>/ASSESSMENT.md` with three sections:
  Setup, Performance, Assessment
