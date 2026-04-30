# Profiling Tools

PyTorch ecosystem profiling wrappers for investigating vLLM/llm-d
behavior. Use these when the diagnostics toolkit shows unexpected
results and you need to understand *why*.

## When to use what

| Symptom | Tool |
|---------|------|
| Slow inference, unknown cause | `torch_profiler.py` — generate profiling wrapper or nsys commands |
| Collective hang or timeout | `nccl_flight_recorder.py` — see what each rank was doing |
| Slow compilation | `tlparse_runner.py` — parse torch.compile traces |
| Need to enable debug logging | `env_presets.py` — apply named env var sets |

## env_presets.py

Named sets of environment variables for common debugging scenarios.
Prints `oc set env` commands — does not modify deployments directly.

```bash
# List all presets
python3 profiling/env_presets.py list

# See what a preset sets
python3 profiling/env_presets.py show flight-recorder

# Get oc commands to apply
python3 profiling/env_presets.py apply nccl-debug vllm-decode -n mynamespace
```

Available presets:

| Preset | What it enables |
|--------|----------------|
| `nccl-debug` | NCCL verbose logging (topology, transport, ring selection) |
| `nccl-trace` | NCCL timing per collective (less noisy) |
| `flight-recorder` | NCCL flight recorder ring buffer |
| `torch-trace` | torch.compile trace logs (for tlparse) |
| `nixl-verbose` | NIXL/UCX transport debug |
| `memory-debug` | PyTorch CUDA memory snapshot (expandable segments) |
| `cuda-memcheck` | Disable caching allocator for cuda-memcheck (very slow) |
| `all-debug` | Everything (very noisy) |

## torch_profiler.py

`torch.profiler` only profiles the process it runs in. You cannot copy a
script into a pod and profile the vLLM server from a separate process —
you'd get an empty trace. This tool provides two approaches:

### Option 1: Startup wrapper (in-process profiling)

Generate a Python wrapper that starts vLLM with profiling injected:

```bash
# Generate wrapper
python3 profiling/torch_profiler.py wrapper --duration 30 -o /tmp/profile_wrapper.py

# Copy to pod
oc cp /tmp/profile_wrapper.py NS/POD:/tmp/

# Restart vLLM with profiling (modify deployment command or exec)
# The wrapper waits for model load (default: 60s), then captures 30s of GPU activity
```

The wrapper produces Chrome JSON traces viewable in
[Perfetto](https://ui.perfetto.dev).

### Option 2: Nsight Systems (external profiling)

Profile GPU kernels from outside the process using `nsys`:

```bash
# Print the nsys commands
python3 profiling/torch_profiler.py nsys-cmd --pod my-pod -n NS --duration 10

# This prints:
#   oc exec POD -n NS -- nsys profile --attach-pid=$VLLM_PID ...
```

Requires `nsys` installed in the container image (available in NVIDIA
NGC vLLM images).

### Option 3: Kineto (via TORCH_TRACE)

```bash
python3 profiling/torch_profiler.py kineto-env --deployment vllm-decode -n NS
```

Prints `oc set env` commands for `TORCH_TRACE` — the standard way to
enable PyTorch's Kineto tracing engine. There are no user-facing
`KINETO_*` env vars. For richer profiling (record_shapes, memory,
stacks), use the wrapper approach.

## nccl_flight_recorder.py

The NCCL flight recorder is a ring buffer of collective operations that
PyTorch maintains in-process. On timeout or crash, it dumps to a pickle
file showing what each rank was doing.

```bash
# See how to enable it
python3 profiling/nccl_flight_recorder.py setup

# After a hang/crash, copy dump from pod
oc cp NS/POD:/tmp/nccl_trace_dump_*.pkl ./

# Parse the dump
python3 profiling/nccl_flight_recorder.py parse ./nccl_trace_dump_rank0.pkl

# Parse all dumps in a directory
python3 profiling/nccl_flight_recorder.py parse ./dumps/
```

You can trigger a dump manually without waiting for a timeout:
```bash
# vLLM is typically PID 1 in the container
oc exec POD -n NS -c vllm -- kill -USR2 1

# If not PID 1, find it first:
oc exec POD -n NS -c vllm -- ps aux | grep vllm
```

## tlparse_runner.py

Parses `TORCH_TRACE` logs into browsable HTML showing compilation steps,
graph breaks, FX graphs, and inductor output. Requires `pip install tlparse`.

```bash
# Enable trace collection (via env_presets or manually)
python3 profiling/env_presets.py apply torch-trace vllm-decode -n NS

# After some requests, collect traces
oc cp NS/POD:/tmp/torch_trace/ ./torch_traces/

# Parse into HTML
python3 profiling/tlparse_runner.py parse ./torch_traces/

# Open the report
open ./torch_traces/tlparse_output/index.html
```

## Dependencies

All tools run on your workstation with Python stdlib only. The generated
profiling wrapper requires `torch` inside the pod (already available in
vLLM images).

| Tool | Runs on | Needs |
|------|---------|-------|
| env_presets.py | Workstation | Nothing |
| torch_profiler.py | Workstation | Nothing (generates code; wrapper needs torch in-pod) |
| nccl_flight_recorder.py | Workstation | Nothing (uses pickle, stdlib) |
| tlparse_runner.py | Workstation | `pip install tlparse` |
