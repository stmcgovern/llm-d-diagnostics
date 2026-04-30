# RDU3-Experimental-T4x3 — llm-d Assessment

**Date:** April 2026
**Author:** Sean McGovern, Red Hat

## Setup

3x NVIDIA Tesla T4 on OpenShift (Dell R740xd, OVN-Kubernetes CNI, no RDMA).

| Role | Node | GPU | Port |
|------|------|-----|------|
| Prefill (kv_producer) | node-0 | T4 | 8100 |
| Decode-1 + sidecar (kv_consumer) | node-1 | T4 | 8000 |
| Decode-2 + sidecar (kv_consumer) | node-2 | T4 | 8000 |

```
         Prefill (8100)
            |
         NIXL/TCP
        /        \
  Decode-1      Decode-2
  sidecar       sidecar
  (8000)        (8000)
```

- **Model:** TinyLlama 1.1B (FP16). Prefill compute is <15ms at 1000 tokens — too small to benefit from P/D separation.
- **vLLM:** v0.18.1, NixlConnector, UCX over TCP
- **Sidecar:** llm-d-routing-sidecar v0.6.1, TLS proxy, no EPP
- **Routing:** Manual via `x-prefiller-host-port` header

## Performance

### Overhead: 53ms fixed cost

Disaggregation adds ~53ms per request regardless of prompt length.

| Path | Median | What it measures |
|------|--------|-----------------|
| Direct to GPU | 204 ms | Raw model latency |
| Through sidecar | 225 ms | +20ms TLS proxy |
| Full disagg (P/D) | 257 ms | +33ms prefill round-trip |
| **Total overhead** | | **53ms** |

The 53ms is infrastructure cost — sidecar and network. It doesn't change with model size.

### Throughput: baseline wins at every concurrency level

| Concurrency | Baseline (1 GPU) | Disagg (3 GPUs) | Ratio |
|-------------|-----------------|-----------------|-------|
| 1 | 302 ms | 346 ms | 1.14x slower |
| 4 | 338 ms | 449 ms | 1.33x slower |
| 16 | 364 ms | 740 ms | 2.03x slower |

TinyLlama's prefill is so fast that vLLM's continuous batching on a single GPU
handles everything efficiently. The 53ms overhead per request compounds under load.

### Isolation: disagg hurts light requests

Mixed workload (1 heavy + 5 light requests simultaneously):

| Config | Light request median |
|--------|---------------------|
| Baseline | 238 ms |
| Disagg | 409 ms |

Single prefill GPU is the bottleneck — all requests still queue through it.

### Transfer cost: flat 30ms across sequence lengths

Exp 5 decomposed the overhead at 7 prompt lengths (10-2000 tokens):

| Prompt | T_sidecar | T_transfer | T_overhead |
|--------|-----------|------------|------------|
| 10 tok | 33 ms | 29 ms | 61 ms |
| 50 tok | 33 ms | 31 ms | 69 ms |
| 100 tok | 31 ms | 31 ms | 68 ms |
| 500 tok | 31 ms | 31 ms | 65 ms |
| 1000 tok | 31 ms | 35 ms | 66 ms |

T_transfer is ~30ms regardless of sequence length (Pearson r=0.925 — not bandwidth-limited).
The overhead is protocol/infrastructure cost, not data volume. 2000-token runs failed
(prompt exceeds 2048 context window after tokenizer expansion).

### Saturation: disagg collapses at QPS=24

Exp 6 ramped QPS from 1 to 32 with rate-controlled Poisson arrivals:

| QPS | Baseline p50 | DISAGG-2D p50 | DISAGG-2D status |
|-----|-------------|---------------|------------------|
| 8 | 233 ms | 267 ms | Healthy |
| 16 | 236 ms | 318 ms | Healthy |
| 24 | 253 ms | 177,316 ms | **Collapsed** |
| 32 | 258 ms | 210,606 ms | Collapsed |

Baseline handles QPS=32 without stress (258ms p50). Disagg hits a hard cliff at QPS=24
with broken pipes, SSL timeouts, and depart delays of 500+ seconds.

### Mixed workload verdict: MARGINAL

Exp 7 sent a realistic Poisson workload (80% short, 20% long prompts) at QPS=4:

| Metric | Baseline | Disagg | Delta |
|--------|----------|--------|-------|
| TTFT p50 (short) | 31 ms | 78 ms | +47 ms (+153%) |
| ITL p99 (short) | 11.3 ms | 11.0 ms | 0.97x |
| Goodput | 3.8 req/s | 3.8 req/s | 1.00x |

ITL is essentially identical. Goodput is identical. TTFT overhead is +47ms. At QPS=4
(well below saturation), disaggregation provides no measurable benefit.

### Fault tolerance: the machinery works

| Scenario | Result | Recovery |
|----------|--------|----------|
| Kill decode-1 | Decode-2 unaffected | 130s (model reload) |
| Kill prefill | **All disagg requests fail** | 167s (model reload) |
| Scale 3→2→3 GPUs | No degradation at 2 GPUs | 156s (cold start) |

- NIXL handshake re-establishes automatically after pod restart (new engine_id, new IP)
- Both decode pods reconnect to new prefill without intervention (ZMQ discovery on port 5600)
- First request after recovery: ~900-990ms (5x normal) — cold start penalty
- **Prefill is a single point of failure** in this topology

## Assessment

**Don't use P/D disaggregation for TinyLlama.** The model is too small. Prefill
compute is negligible, so there's nothing to offload — disaggregation just adds
53ms of overhead to every request and makes everything slower. The saturation data
makes this definitive: baseline handles QPS=32 while disagg collapses at QPS=24.

**The llm-d machinery works correctly.** The sidecar routes requests, NIXL
transfers KV cache, and the whole pipeline recovers automatically from pod
failures. This is what matters for larger models where prefill compute is the
actual bottleneck.

**What we learned for the next deployment:**

1. The 53ms overhead floor (20ms sidecar + 33ms prefill RT) is the tax you pay
   for disaggregation on any model. It only makes sense when prefill compute
   exceeds this — roughly 8B+ parameters.

2. Transfer cost is flat (~30ms), independent of sequence length. This confirms
   the overhead is protocol/infrastructure, not bandwidth-limited. RDMA would
   reduce this component.

3. Disagg has a hard throughput ceiling. The saturation cliff at QPS=24 shows
   that the sidecar/routing path becomes a bottleneck before the GPU does.
   This matters for capacity planning even on larger models.

4. Prefill needs redundancy. Single prefill = single point of failure.

5. Recovery time is model-load dominated (130-167s for 1.1B). Plan for longer
   on larger models.

## Cross-Model Comparison: TinyLlama 1.1B vs Phi-3 3.8B

Same hardware (3x T4), same software (vLLM v0.18.1, sidecar v0.6.1), different model.
This isolates which findings are model-dependent vs infrastructure-level.

| Metric | TinyLlama 1.1B | Phi-3 3.8B | Portable? |
|--------|---------------|------------|-----------|
| Disagg overhead | 53ms | 262ms | Partly — sidecar ~36ms is stable, transfer scales |
| Sidecar cost | 20ms | 36-42ms | **Yes** (model-independent) |
| Transfer cost (1000 tok) | 35ms (flat) | 113ms (linear) | No — scales with KV cache size |
| Saturation: baseline | QPS=32+ | QPS=16 | No — model-dependent |
| Saturation: disagg-2D | QPS=24 (collapsed) | QPS=32+ | No — model-dependent |
| Scaling dividend | 0x (disagg loses) | **2x** (disagg wins) | No — crossover between models |
| ITL isolation ratio | 0.96x (no help) | **1.38x** (disagg helps) | No — crossover between models |
| Long-prompt ITL gain | None | **1.9x** | No — model-dependent |
| Recovery time (prefill) | 167s | 127s | No — model-load dominated |
| NIXL handshake | Automatic | Automatic | **Yes** |
| Sidecar fallback | Observed | 188 events confirmed | **Yes** |
| ZMQ cache bug (4j) | Not tested | 5x slower container restart | **Yes** (protocol-level) |
| Rolling update safe | Not tested | **No** (85% failure) | **Yes** (infrastructure) |
| NIXL health check | <1s detection | <1s detection | **Yes** |

**Key insight:** The crossover happens between 1.1B and 3.8B parameters on T4.
At 1.1B, disaggregation adds overhead to everything. At 3.8B, it doubles
throughput capacity and improves decode quality. The portable findings
(sidecar cost, NIXL behavior, fault tolerance mechanisms, ZMQ bugs) apply
to any model on any hardware.

Phi-3 assessment data collected on the same cluster with a separate configuration.

---

*Raw data in `data/`. Detailed statistical analysis in [REPORT.md](REPORT.md).
Generated with the [llm-d diagnostics toolkit](../../README.md).*
