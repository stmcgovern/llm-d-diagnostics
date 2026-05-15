# Sweep 1: KV Head Ratio and Disaggregation Overhead

## Thesis

Disaggregated inference transfers KV cache from prefill to decode over the
network. The transfer time is:

    T_transfer(model, L) = alpha + beta(model) * L

where L is the prompt length in tokens, and:

- **alpha** (protocol overhead): NIXL handshake, memory registration,
  connection setup. Model-independent. A property of the NIC and protocol
  stack. Estimated ~17ms on T4 (from TinyLlama NIXL measurement).

- **beta(model)** = KV_bytes_per_token / effective_bandwidth: proportional
  to `n_kv_heads`. This is where GQA vs MHA shows up.

KV cache per token:

    KV_bytes = 2 * n_layers * n_kv_heads * d_head * 2(fp16)

Models with grouped-query attention (GQA) transfer fewer KV heads and
should see lower beta. At short prompts, alpha dominates and all models
look the same. At long prompts, beta * L dominates and the gap between
GQA and MHA models grows linearly.

## Models (ordered by KV cache per token)

| Model             | Params | Q:KV ratio | KV heads | KV/tok  | Attention |
|-------------------|--------|------------|----------|---------|-----------|
| Qwen2.5 0.5B     | 0.5B   | 16:1       | 2        | 12 KB   | GQA       |
| Qwen2.5 1.5B     | 1.5B   | 16:1       | 2        | 14 KB   | GQA       |
| TinyLlama 1.1B   | 1.1B   | 8:1        | 4        | 22 KB   | GQA       |
| Qwen2.5 3B       | 3.0B   | 16:1       | 2        | 36 KB   | GQA       |
| StableLM 1.6B    | 1.6B   | 1:1        | 32       | 192 KB  | MHA       |
| SmolLM2 1.7B     | 1.7B   | 1:1        | 32       | 192 KB  | MHA       |
| Phi-3 3.8B       | 3.8B   | 1:1        | 32       | 384 KB  | MHA       |

The KV cache spans a 32x range (12 KB to 384 KB per token).

## Fermi predictions

Assumptions: alpha = 17ms, effective NIC bandwidth = 2.0 GB/s (T4, TCP).

| Model             | T @ L=10 | T @ L=100 | T @ L=1000 | T @ L=2048 |
|-------------------|----------|-----------|------------|------------|
| Qwen2.5 0.5B     | 17 ms    | 18 ms     | 23 ms      | 30 ms      |
| Qwen2.5 1.5B     | 17 ms    | 18 ms     | 24 ms      | 32 ms      |
| TinyLlama 1.1B   | 17 ms    | 18 ms     | 28 ms      | 40 ms      |
| Qwen2.5 3B       | 17 ms    | 19 ms     | 35 ms      | 55 ms      |
| StableLM 1.6B    | 18 ms    | 27 ms     | 115 ms     | 218 ms     |
| SmolLM2 1.7B     | 18 ms    | 27 ms     | 115 ms     | 218 ms     |
| Phi-3 3.8B       | 19 ms    | 37 ms     | 214 ms     | 420 ms     |

At L=10, all models are within 17-19ms (protocol-dominated, as predicted).
At L=1000, Phi-3 (MHA) is ~10x slower than Qwen 0.5B (GQA).

## Falsifiable predictions

1. **alpha is constant across models** (CV < 30%). If alpha varies
   significantly, the protocol overhead is model-dependent (e.g., NIXL
   memory registration scales with KV structure). This would refute the
   "hardware-only" claim about protocol overhead.

2. **beta is proportional to KV_bytes_per_token**. The proportionality
   constant (1/eff_bw) should be the same for all models. If it varies,
   effective bandwidth depends on transfer size (small transfers hit a
   latency floor, large transfers hit bandwidth ceiling).

3. **R-squared > 0.95 for all models**. If the linear fit is poor,
   something nonlinear is happening: TCP congestion, memory fragmentation,
   CUDA synchronization interference, or vLLM-internal scheduling delays.

4. **StableLM and SmolLM2 should be nearly identical**. Both have 24
   layers, 32 KV heads, d_head=64 (192 KB/tok). Same KV structure,
   similar parameter count. If they diverge, there is a model-specific
   factor we are not accounting for.

## Expected advisor verdicts

For a deployment planner asking "should I disaggregate this model on T4?":

| Model             | KV/tok  | Disagg verdict  | Reasoning                              |
|-------------------|---------|-----------------|----------------------------------------|
| Qwen2.5 0.5B     | 12 KB   | BEST CANDIDATE  | KV transfer negligible at all lengths   |
| Qwen2.5 1.5B     | 14 KB   | BEST CANDIDATE  | Same GQA advantage                     |
| TinyLlama 1.1B   | 22 KB   | GOOD            | Low KV, modest overhead growth          |
| Qwen2.5 3B       | 36 KB   | GOOD            | GQA keeps KV small despite 3B params   |
| StableLM 1.6B    | 192 KB  | MARGINAL        | MHA, 16x more KV than Qwen             |
| SmolLM2 1.7B     | 192 KB  | MARGINAL        | MHA, same penalty as StableLM          |
| Phi-3 3.8B       | 384 KB  | CAUTION         | MHA + large model, transfer dominates  |

The punchline: GQA models get disaggregation almost for free. MHA models
pay a tax proportional to sequence length. A 3B GQA model (Qwen) transfers
less KV cache than a 1.6B MHA model (StableLM).

## What could go wrong

- **VRAM pressure on Phi-3**: model weights (~7.6 GB) + KV cache at
  L=2048 (~768 MB) = ~8.4 GB on a 16 GB T4 (53% utilization). Fits, but
  leaves limited headroom. gpu_memory_utilization=0.85 should be safe.

- **OLMoE anomaly**: The existing baselines show OLMoE has nixl_ms=267ms
  at *short* prompts, 15x higher than TinyLlama. MoE architectures may
  violate the linear model entirely (sparse expert activation changes KV
  structure). OLMoE is in the sweep config but may need special handling.
  **Update**: removed from sweep 1 — will test separately.

- **Model download time**: Each model needs to download to the PVC.
  Estimated 5-10 min per model at typical cluster bandwidth. Total sweep
  time: ~7 models x (10 min deploy + 15 min experiments + 2 min undeploy)
  = ~3 hours.

- **Non-linearity at large transfers**: At L=2048, Phi-3 transfers ~800 MB.
  If TCP windows or RDMA buffers saturate, we may see sub-linear scaling
  (bandwidth drops at large transfer sizes). This would show up as
  curvature in the T_transfer vs L plot and R-squared < 0.95.

## Hardware

- GPU: NVIDIA T4 (16 GB GDDR6, ~300 GB/s memory bandwidth)
- Network: 25 GbE TCP (theoretical ~3.1 GB/s, effective ~2-2.5 GB/s)
- Topology: 2 prefill + 3 decode pods (5x T4)
- NIXL: TCP transport (UCX_TLS=^cuda_ipc), no RDMA

## Experiments per model

| Experiment  | Internal ID | What it measures                        |
|-------------|-------------|-----------------------------------------|
| decompose   | exp1b       | Sidecar vs NIXL latency breakdown       |
| latency     | exp1        | Baseline vs disagg TTFT sweep           |
| throughput  | exp2        | Throughput under concurrent load         |
| seqlen      | exp5        | T_transfer vs sequence length (the key) |

exp5 is the critical experiment: it sweeps sequence length and measures
T_transfer = D_config - C_config (disaggregated minus sidecar-only).
The linear regression on this data yields alpha and beta per model.

## Run command

```bash
./toolkit/run.sh sweeps/kv-ratio.json sweep
```

## Analysis

```bash
python3 advisor/kv_sweep.py clusters/_sweep/*/data/
```

Expected output: a table showing alpha, beta, effective bandwidth, and
R-squared per model, confirming (or refuting) the linear transfer model.
