# Profiling Tools

PyTorch ecosystem profiling wrappers for vLLM/llm-d deployments. These
complement the diagnostics toolkit — the toolkit measures *what* the
performance is, profiling answers *why*.

## Tools

### env_presets.py — Debug environment presets

Named sets of environment variables for common debugging scenarios.

```bash
# List available presets
python3 profiling/env_presets.py list

# Show vars for a preset
python3 profiling/env_presets.py show flight-recorder

# Print oc commands to apply to a deployment
python3 profiling/env_presets.py apply nccl-debug vllm-decode -n mynamespace
```

Presets: `nccl-debug`, `nccl-trace`, `flight-recorder`, `torch-trace`,
`nixl-verbose`, `memory-debug`, `cuda-memcheck`, `all-debug`.

### torch_profiler.py — GPU profiling instrumentation

`torch.profiler` only profiles the process it runs in — you can't attach
from outside. This tool generates instrumentation that runs *inside* the
vLLM process:

```bash
# Generate a startup wrapper that profiles after model load
python3 profiling/torch_profiler.py wrapper --duration 30 -o /tmp/profile_wrapper.py

# Or print nsys commands for external GPU profiling (no code changes)
python3 profiling/torch_profiler.py nsys-cmd --pod my-pod -n mynamespace --duration 10
```

The wrapper approach injects `torch.profiler.profile()` into the vLLM
process via a startup script. The nsys approach uses Nsight Systems to
profile GPU activity from outside (requires nsys in the container image).

### nccl_flight_recorder.py — NCCL collective debugging

Configure the NCCL flight recorder (ring buffer of collective ops) and
parse its dump files. Essential for diagnosing hangs in tensor-parallel
deployments.

```bash
# Show how to enable
python3 profiling/nccl_flight_recorder.py setup

# Parse a dump
python3 profiling/nccl_flight_recorder.py parse /tmp/nccl_trace_rank0.pkl
```

### tlparse_runner.py — torch.compile trace analysis

Parses `TORCH_TRACE` logs into browsable HTML showing compilation steps,
graph breaks, and inductor output.

```bash
# Install tlparse
pip install tlparse

# Collect traces from pod and parse
python3 profiling/tlparse_runner.py collect my-pod -n mynamespace
python3 profiling/tlparse_runner.py parse ./torch_traces/
```

## Dependencies

| Tool | Dependency | Where it runs |
|------|-----------|---------------|
| env_presets.py | None (stdlib) | Workstation |
| torch_profiler.py | None (generates code) | Workstation; generated wrapper needs torch (in-pod) |
| nccl_flight_recorder.py | None (stdlib: pickle) | Workstation |
| tlparse_runner.py | tlparse (`pip install`) | Workstation |
