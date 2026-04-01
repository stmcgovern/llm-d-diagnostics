# Characterizing llm-d Disaggregated Inference on OpenShift

**Cluster:** RDU3-Experimental-T4x3
**Date:** April 2026
**Author:** Sean McGovern, Red Hat

---

## Abstract

We deploy llm-d's prefill/decode disaggregated inference architecture on a
3-GPU OpenShift cluster and validate its behavior using a lightweight diagnostics
toolkit. The goal is to understand the distributed inference machinery —
especially failure modes, recovery behavior, and overhead structure — so that
we can reason about it confidently in production settings.

The model (TinyLlama 1.1B) is deliberately small. It does not need or benefit
from disaggregation. We use it to isolate the infrastructure: every millisecond
of overhead is the llm-d stack, not model compute.

Key findings: NIXL handshake re-establishment is fully automatic across pod
restarts. Prefill is a single point of failure in single-prefill topologies.
Recovery time is dominated by model loading (130-167s). The disaggregation
overhead floor is ~53ms (sidecar 20ms + prefill round-trip 33ms), independent
of model size. Transfer cost is flat at ~30ms regardless of sequence length
(protocol overhead, not bandwidth). Baseline handles QPS=32 while disagg
collapses at QPS=24. Under realistic mixed workload at QPS=4, disaggregation
adds +47ms TTFT overhead with no measurable ITL or throughput benefit.

---

## 1. System Under Test

| Component | Version / Spec |
|-----------|---------------|
| Platform | OpenShift 4.x on Dell R740xd |
| GPUs | 3x NVIDIA Tesla T4 (16 GB GDDR6, PCIe 3.0) |
| Nodes | node-0, node-1, node-2 |
| Network | Cluster CNI (OVN-Kubernetes). No RDMA, no InfiniBand |
| Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0, FP16 |
| Inference engine | vLLM v0.18.1 |
| KV transfer | NixlConnector, UCX over TCP |
| Routing sidecar | llm-d-routing-sidecar v0.6.1 (nixlv2 connector) |
| KV config | max_model_len=2048, gpu_memory_utilization=0.8 |
| Diagnostics toolkit | This repository (`scripts/toolkit/`) |

### 1.1 Topology

```
                   +-----------+
                   | Prefill   |  vLLM kv_producer
                   | (node-0) |  port 8100
                   | T4 GPU    |
                   +-----+-----+
                  NIXL/TCP|
            +-------------+-------------+
            |                           |
      +-----+------+            +------+-----+
      | Decode-1   |            | Decode-2   |
      | (node-1)  |            | (node-2)  |
      | T4 GPU     |            | T4 GPU     |
      | sidecar    |            | sidecar    |
      | :8000      |            | :8000      |
      +------------+            +------------+
```

- **Prefill GPU** runs vLLM as `kv_producer`. Receives prompts, computes attention,
  transfers KV cache to decode workers via NIXL over TCP.
- **Decode GPUs** run vLLM as `kv_consumer` with an llm-d routing sidecar (TLS proxy,
  NIXL coordinator). The sidecar receives client requests, routes prefill to the
  prefill GPU, receives KV cache, then completes decode locally.
- **No EPP (Endpoint Picker).** Routing is explicit via the `x-prefiller-host-port` header.

### 1.2 Why TinyLlama

TinyLlama's prefill compute is negligible on T4 hardware (<15ms at 1000 prompt tokens).
This means disaggregation overhead dominates — the model is too small to benefit from
P/D separation. We chose it deliberately: it lets us measure the *overhead floor* of the
llm-d stack without confounding prefill/decode compute effects.

---

## 2. Experiments and Findings

All data is in `data/` (11,000+ measured data points across 9 CSV files). Statistical
analysis uses sample medians, IQR, and 95% confidence intervals (t-distribution).
Data quality: 0.7-14.3% error rates across experiments, flagged and handled per experiment.

### 2.1 Experiment 1: Single-Request Latency

**Question:** What is the per-request overhead of disaggregation, and does it scale
with prompt length?

**Method:** Sequential requests, no concurrency. 20 runs per config per prompt length.
Three configs: baseline (direct to prefill GPU), disagg-D1 (through decode-1 sidecar),
disagg-D2 (through decode-2 sidecar).

| Prompt | Baseline | Disagg-D1 | Disagg-D2 | Overhead D1 | Overhead D2 |
|--------|----------|-----------|-----------|-------------|-------------|
| 10 tok | 201 ms | 274 ms | 272 ms | +73 ms | +70 ms |
| 50 tok | 204 ms | 273 ms | 270 ms | +69 ms | +67 ms |
| 100 tok | 207 ms | 266 ms | 251 ms | +59 ms | +44 ms |
| 500 tok | 211 ms | 281 ms | 252 ms | +71 ms | +41 ms |
| 1000 tok | 212 ms | 284 ms | 274 ms | +72 ms | +62 ms |

*All values are medians over n=20. All requests returned HTTP 200.*

**Findings:**

1. **Overhead does not scale with prompt length.** Baseline TTFT increases only 11ms
   from 10 to 1000 tokens (201 to 212ms), confirming TinyLlama's prefill compute is
   negligible. The NIXL KV transfer at this model size is fast enough to absorb into
   the constant overhead.

2. **D1 overhead is stable:** 68.5 +/- 5.6ms (range 59-73ms).

3. **D2 overhead is more variable:** 56.7 +/- 13.5ms (range 41-70ms), with 2.4x the
   variance of D1. This asymmetry is unexplained — possible causes include network
   path differences between nodes (node-1 vs node-2) or thermal behavior.

### 2.2 Experiment 1b: Latency Decomposition

**Question:** Where does the 53-69ms disaggregation overhead come from?

**Method:** Four request paths at fixed 50-token prompt, 30 runs each:

| Path | What it measures | Median |
|------|-----------------|--------|
| A. Prefill direct | Raw model latency (prefill GPU) | 203.9 ms |
| B. Decode direct | Raw model latency (decode GPU, bypass sidecar) | 204.7 ms |
| C. Sidecar only | Decode GPU through sidecar, no prefill routing | 224.6 ms |
| D. Disaggregated | Full P/D path through sidecar | 257.1 ms |

**Derived components:**

```
T_sidecar    = C - B = 19.9 ms   (TLS proxy overhead)
T_prefill_rt = D - C = 32.5 ms   (prefill round-trip, includes ~3ms NIXL)
T_overhead   = D - A = 53.2 ms   (total disagg cost)

Sum check: 19.9 + 32.5 = 52.4 ms  vs  53.2 ms measured  (residual: 0.8 ms)
```

**Key insight:** The overhead decomposes into two measured components — sidecar
proxy (20ms) and prefill round-trip (33ms). Both are infrastructure costs,
independent of model size. This is the floor cost you pay for disaggregation
on any model.

**Validation:** Path A (prefill GPU) and Path B (decode GPU) differ by only 0.4%
(203.9 vs 204.7ms), confirming the T4 GPUs perform comparably.

### 2.3 Experiment 2: Throughput Under Load

**Question:** At what concurrency does disaggregation outperform a single GPU?

**Method:** Concurrent request batches via ThreadPoolExecutor, 20 requests per config
per concurrency level. Three configs: baseline (1 GPU), disagg-1D (2 GPUs),
disagg-2D (3 GPUs).

| Concurrency | Baseline | Disagg-1D | Disagg-2D | Ratio (2D/BL) |
|-------------|----------|-----------|-----------|----------------|
| 1 | 302 ms | 348 ms | 346 ms | 1.14x |
| 2 | 328 ms | 445 ms | 371 ms | 1.13x |
| 4 | 338 ms | 508 ms | 449 ms | 1.33x |
| 8 | 349 ms | 523 ms | 506 ms | 1.45x |
| 16 | 364 ms | 811 ms | 740 ms | 2.03x |

*All values are median total latency (ms). Ratio >1 means disagg is slower.*

**Findings:**

1. **No crossover observed.** Baseline wins at every concurrency level, with the gap
   widening under load. At C=16, disagg-2D is 2.03x slower.

2. **Why baseline wins:** vLLM's continuous batching efficiently interleaves prefill
   and decode on a single T4. TinyLlama's prefill is so fast (<15ms) that it doesn't
   block decode iterations. The fixed 53ms disagg overhead compounds under load.

3. **2D vs 1D:** At C=16, disagg-2D (740ms) beats disagg-1D (811ms) — the second
   decode GPU provides ~10% improvement, limited by the shared prefill bottleneck.

### 2.4 Experiment 3: Prefill Isolation

**Question:** Does disaggregation protect latency-sensitive requests from heavy prefills?

**Method:** Mixed workload per trial: 1 heavy request (1000-token prompt, 50 max tokens)
+ 5 light requests (10-token prompt, 20 max tokens), all launched simultaneously via
ThreadPoolExecutor. 10 trials per config. Two configs: baseline, disagg-2D.

| Config | Light Median | Heavy Median |
|--------|-------------|-------------|
| Baseline | 238 ms | 540 ms |
| Disagg-2D | 409 ms | 766 ms |

**Isolation ratio: 0.58** (BL light / DG light). Ratio < 1 means disagg makes light
requests *worse*.

**Why disagg hurts here:** With a single prefill GPU, all requests (heavy and light)
still queue through the same prefill. Disagg adds overhead to every request without
eliminating the prefill bottleneck. Isolation would require either multiple prefill GPUs
or a model large enough that decode blocking is the dominant latency source.

### 2.5 Experiment 4: Fault Tolerance

**Question:** How does the system behave when components fail?

This is the most architecturally significant experiment. Unlike the performance results
(which are TinyLlama-specific), fault tolerance behavior is protocol-level and
generalizes across model sizes.

#### 4a: Decode Pod Failure

| Phase | Status | Latency | Observation |
|-------|--------|---------|-------------|
| Pre-flight (D1) | 200 | 185 ms | Healthy |
| Pre-flight (D2) | 200 | 179 ms | Healthy |
| D2 during D1 death | 200 | 156 ms | **Unaffected** |
| D1 while dead | timeout | 10 s | Pod gone |
| D1 post-recovery | 200 | 990 ms | Cold start |

- **Recovery time: 130s** (pod scheduling + model load + vLLM init)
- Decode pods are independently resilient — killing D1 has no effect on D2
- New pod received new `engine_id`, handshake re-established automatically

#### 4b: Prefill Pod Failure

| Phase | Status | Latency | Observation |
|-------|--------|---------|-------------|
| Pre-flight | 200 | 150 ms | Disagg working |
| During (prefill dead) | timeout | 10 s | **Complete pipeline failure** |
| Post-recovery (D1) | 200 | 925 ms | New prefill, new engine_id |
| Post-recovery (D2) | 200 | 901 ms | Both decoders reconnect |

- **Recovery time: 167s**
- **Prefill is a single point of failure.** When it dies, all disagg requests fail.
- New prefill got new pod IP and engine_id. Both decode pods discovered and handshaked
  with the new prefill automatically via ZMQ side-channel (port 5600).

#### 4d: Graceful Degradation (3 GPU -> 2 GPU -> 3 GPU)

| Phase | Latency | Observation |
|-------|---------|-------------|
| 2-GPU (5 requests) | 86-173 ms | D1 handles all traffic, no degradation |
| D2 restored | 930 ms | Cold start after scale-up (156s) |

**Fault Tolerance Summary:**

1. **NIXL handshake re-establishment is automatic.** New engine_id, new pod IP — the
   ZMQ side-channel handles discovery without manual intervention.

2. **Prefill is a SPOF** in single-prefill topologies. Production deployments need
   prefill redundancy.

3. **Recovery time is dominated by model loading** (130-167s for TinyLlama on T4).
   Scales with model size and inversely with GPU memory bandwidth.

4. **Cold start penalty: ~900-990ms** for the first request after recovery (5x normal).

### 2.6 Experiment 5: Sequence Length Sweep

**Question:** Does the NIXL transfer cost scale with sequence length? At what length
does transfer cost exceed prefill compute (the crossover point)?

**Method:** Same 4-path decomposition as exp 1b, repeated at 7 prompt lengths
(10, 50, 100, 250, 500, 1000, 2000 tokens). 30 runs per config per length.

| Length | T_sidecar | T_transfer | T_overhead | A (prefill) |
|--------|-----------|------------|------------|-------------|
| 10 | 33 ms | 29 ms | 61 ms | 202 ms |
| 50 | 33 ms | 31 ms | 69 ms | 204 ms |
| 100 | 31 ms | 31 ms | 68 ms | 206 ms |
| 250 | 32 ms | 31 ms | 68 ms | 209 ms |
| 500 | 31 ms | 31 ms | 65 ms | 211 ms |
| 1000 | 31 ms | 35 ms | 66 ms | 213 ms |
| 2000 | — | — | — | (all failed) |

*All values are medians over n=26-30 successful runs per config.*

**Findings:**

1. **Transfer cost is flat at ~30ms** across the entire 10-1000 token range. Pearson
   r(seq_len, T_transfer) = 0.925. The correlation is positive but weak relative to
   the 100x increase in sequence length — this is protocol/infrastructure overhead,
   not bandwidth-limited transfer.

2. **No crossover observed.** T_transfer < T_prefill at all measured lengths. The
   crossover where transfer cost exceeds prefill compute would occur beyond 1000 tokens,
   but TinyLlama's 2048 context window limits testing.

3. **2000-token runs failed** with HTTP 400. `build_prompt()` produces ~1.3x the target
   token count (systematic tokenizer behavior), so 2000 target → ~2600 actual tokens
   exceeds the 2048 context window.

4. **Sidecar overhead is stable** at 31-33ms across all lengths, confirming it's
   independent of payload size.

### 2.7 Experiment 6: Saturation Profiling

**Question:** At what request rate does each deployment topology saturate?

**Method:** Open-loop, rate-controlled Poisson arrivals at 8 QPS levels
(1, 2, 4, 8, 12, 16, 24, 32), 30 seconds per level, 3 configs
(BASELINE, DISAGG-1D, DISAGG-2D). 8,910 total requests.

| Config | QPS | p50 | p90 | p99 | Error % |
|--------|-----|-----|-----|-----|---------|
| BASELINE | 1 | 204 ms | 208 ms | 210 ms | 0% |
| BASELINE | 8 | 233 ms | 238 ms | 243 ms | 0% |
| BASELINE | 16 | 236 ms | 241 ms | 245 ms | 0% |
| BASELINE | 24 | 253 ms | 258 ms | 260 ms | 0% |
| BASELINE | 32 | 258 ms | 263 ms | 275 ms | 0% |
| DISAGG-2D | 1 | 294 ms | 306 ms | 306 ms | 0% |
| DISAGG-2D | 8 | 267 ms | 282 ms | 284 ms | 0% |
| DISAGG-2D | 16 | 318 ms | 324 ms | 339 ms | 0% |
| DISAGG-2D | 24 | 177,316 ms | 208,614 ms | 213,492 ms | 1% |
| DISAGG-2D | 32 | 210,606 ms | 281,525 ms | 296,871 ms | 3% |

**Findings:**

1. **Baseline is healthy through QPS=32.** Latency increases gradually from 204ms to
   258ms with zero errors. The step from 208ms (QPS=4) to 233ms (QPS=8) is vLLM's
   continuous batching engaging.

2. **Disagg collapses at QPS=24.** Both DISAGG-1D and DISAGG-2D show a catastrophic
   cliff — p50 jumps from ~320ms (QPS=16) to 165,000-177,000ms (QPS=24). Errors include
   broken pipes and SSL timeouts.

3. **Depart delay confirms server saturation.** At QPS=24, median depart delay is
   563-586 seconds, meaning requests queue for ~10 minutes before even being sent.
   This is server-side saturation (the sidecar/routing path), not test harness limitation.

4. **DISAGG-2D does not outperform DISAGG-1D.** The second decode GPU provides no
   meaningful throughput benefit, confirming the single prefill GPU is the bottleneck.

### 2.8 Experiment 7: Mixed Workload

**Question:** Under realistic conditions, is disaggregation worth it?

**Method:** Poisson arrivals at QPS=4 for 60 seconds. 80% short prompts (10 tokens,
20 max output) and 20% long prompts (500 tokens, 50 max output). Streaming responses
with per-token timestamps for ITL measurement. Two configs: BASELINE, DISAGG-2D.
229 requests per config (458 total).

| Config | Class | n | TTFT p50 | TTFT p99 | ITL mean | ITL p99 |
|--------|-------|---|----------|----------|----------|---------|
| BASELINE | short | 184 | 30.8 ms | 39.6 ms | 10.5 ms | 11.3 ms |
| BASELINE | long | 45 | 33.1 ms | 41.7 ms | 10.3 ms | 11.6 ms |
| DISAGG-2D | short | 184 | 78.0 ms | 163.8 ms | 9.7 ms | 11.0 ms |
| DISAGG-2D | long | 45 | 85.2 ms | 147.0 ms | 10.0 ms | 12.7 ms |

**The three essential numbers:**

1. **TTFT overhead: +47ms (+153%).** Disagg adds 47ms per short request (78ms vs 31ms).
   This is consistent with the 53ms overhead floor from exp 1b.

2. **ITL stability: 0.97x.** Disagg ITL p99 is 11.0ms vs baseline 11.3ms — a 0.3ms
   difference that is noise, not signal. At QPS=4, neither topology's decode path is
   stressed enough for isolation to matter.

3. **Goodput: 1.00x.** Both configs serve 3.8 req/s. At this QPS, both topologies handle
   the load without dropping requests.

**Verdict: NOT JUSTIFIED at this operating point.** Disaggregation adds significant TTFT
overhead with no measurable improvement in decode quality or throughput. This is expected
for TinyLlama at low QPS — the value proposition requires either (a) a model where prefill
dominates latency, or (b) higher QPS where prefill contention degrades monolithic decode.

---

## 3. Overhead Budget

The disaggregation overhead decomposes into two measured, model-independent components:

```
Client request
  |
  +-- T_sidecar    (20 ms) -- TLS termination + routing logic (C - B)
  |
  +-- T_prefill_rt (33 ms) -- prefill round-trip including KV transfer (D - C)
  |
  = T_overhead     (53 ms) -- total cost of disaggregation (D - A)
```

| Component | Measured | How derived |
|-----------|----------|-------------|
| T_sidecar | 20 ms | Sidecar-only path minus direct decode |
| T_prefill_rt | 33 ms | Full disagg path minus sidecar-only |
| **T_overhead** | **53 ms** | Full disagg path minus direct baseline |

These are infrastructure costs. On a larger model where prefill takes seconds,
this 53ms is noise. On TinyLlama (prefill <15ms), it dominates.

---

## 4. Limitations

1. **TinyLlama is too small to demonstrate disagg benefits.** Every performance
   comparison shows baseline winning. The value is in mechanism validation and
   fault tolerance data.

2. **No RDMA.** NIXL over TCP on cluster CNI. Production uses InfiniBand/RoCE.

3. **No EPP.** Manual routing via headers, not the Endpoint Picker (inference scheduler).

4. **Single prefill GPU.** Cannot test prefill isolation or prefill redundancy.

5. **Shared cluster.** Other tenants may affect results.

6. **n=20-30 per config.** Sufficient for medians but variance is high in some configs
   (D2 std up to 52.8ms). Higher-n runs would tighten confidence intervals.

7. **Non-streaming measurements (Exp 1, 1b, 2).** TTFT values are actually
   time-to-first-response-byte, which equals total latency for non-streaming requests.
   The overhead *comparisons* remain valid (same measurement applied to all configs).
   Exp 3 and 7 use streaming for true TTFT/ITL.

8. **build_prompt tokenizes at ~1.3x target.** The tokenizer produces more tokens
   than requested (e.g., target=2000 → actual=~2600). This is systematic (consistent
   across all lengths), not a data quality issue. It caused 2000-token runs to exceed
   the 2048 context window.

9. **Exp 7 at QPS=4 is below saturation.** The mixed workload verdict would likely
   differ at higher QPS where prefill contention affects monolithic decode. QPS=4
   is well within both topologies' capacity.

10. **No GPU utilization data.** nvidia-smi is not available in the test-client pod.
    GPU utilization during experiments is not captured.

---

## 5. Predictions for Larger Models

These are hypotheses, not findings. Included to guide future work.

| Property | TinyLlama (measured) | 8B+ (predicted) | Basis |
|----------|---------------------|------------------|-------|
| Disagg overhead | 53-69 ms | ~25-30 ms (with RDMA) | Sidecar + routing is model-independent |
| Prefill compute | <15 ms at 1K tokens | ~200 ms+ at 1K tokens | Scales with params^2 x seq_length |
| Throughput crossover | Not observed | Depends on CB efficiency | Hypothesis: prefill >> overhead, but unverified |
| Fault recovery | 130-167 s | Longer (larger checkpoint) | Model loading dominates |
| NIXL handshake | Automatic | Same (protocol-level) | Independent of model size |

---

## 6. Related Work

- [llm-d](https://github.com/llm-d/llm-d) — the llm-d project
- [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark) — official benchmark
  framework (Helm-based, uses inference-perf). For production CI/CD benchmarking.
  This toolkit fills a different niche: lightweight, portable diagnostics for
  validating individual deployments.
- [llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) — GPU-free
  vLLM simulator. Use with `SIM=1` mode for toolkit validation without hardware.
- [vLLM](https://github.com/vllm-project/vllm) — the inference engine

