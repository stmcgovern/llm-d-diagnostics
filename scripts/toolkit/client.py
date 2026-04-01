"""
llm-d Diagnostics Toolkit — HTTP client and shared utilities.

Zero external dependencies. Uses http.client for precise TTFT measurement.

All configuration via environment variables:
    SIM             Set to 1 for inference-sim mode (HTTP, fake model)
    MODEL           Model name (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    NS              Kubernetes namespace (default: default)
    BASELINE_URL    Prefill direct URL (default: http://vllm-prefill-svc:8100/v1/completions)
    DISAGG_D1_URL   Decode-1 sidecar URL (default: https://vllm-decode-svc:8000/v1/completions)
    DISAGG_D2_URL   Decode-2 sidecar URL (default: https://vllm-decode-2-svc:8000/v1/completions)
    DECODE_DIRECT_URL  Decode bypass sidecar (default: http://vllm-decode-direct-svc:8001/v1/completions)
    PREFILL_HOST    Prefill host:port for routing header
    DATA_DIR        Output directory (default: data)
    WARMUP          Warmup requests (default: 3)
    RUNS            Measured runs per config (default: 20)
    MAX_TOKENS      Max completion tokens (default: 20)

Sim mode (SIM=1):
    Uses inference-sim (llm-d-inference-sim) as a GPU-free test target.
    Switches decode URLs to HTTP (sidecar runs --secure-proxy=false).
    Deploy with: oc apply -f manifests/sim/
"""

import http.client
import json
import os
import ssl
import time
import csv
import sys
import threading
from dataclasses import dataclass, field
from urllib.parse import urlparse


# ── Configuration ────────────────────────────────────────────────────────────

def env(name, default):
    return os.environ.get(name, default)


SIM = env("SIM", "") == "1"
NS = env("NS", "default")

if SIM:
    MODEL = env("MODEL", "sim-model")
    BASELINE_URL = env("BASELINE_URL", "http://vllm-prefill-svc:8100/v1/completions")
    DISAGG_D1_URL = env("DISAGG_D1_URL", "http://vllm-decode-svc:8000/v1/completions")
    DISAGG_D2_URL = env("DISAGG_D2_URL", "http://vllm-decode-2-svc:8000/v1/completions")
    DECODE_DIRECT_URL = env("DECODE_DIRECT_URL", "http://vllm-decode-direct-svc:8001/v1/completions")
else:
    MODEL = env("MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    BASELINE_URL = env("BASELINE_URL", "http://vllm-prefill-svc:8100/v1/completions")
    DISAGG_D1_URL = env("DISAGG_D1_URL", "https://vllm-decode-svc:8000/v1/completions")
    DISAGG_D2_URL = env("DISAGG_D2_URL", "https://vllm-decode-2-svc:8000/v1/completions")
    DECODE_DIRECT_URL = env("DECODE_DIRECT_URL", "http://vllm-decode-direct-svc:8001/v1/completions")

PREFILL_HOST = env("PREFILL_HOST", f"vllm-prefill-svc.{NS}.svc.cluster.local:8100")
PREFILL_HEADER = f"x-prefiller-host-port: {PREFILL_HOST}"
DATA_DIR = env("DATA_DIR", "data")
WARMUP = int(env("WARMUP", "3"))
RUNS = int(env("RUNS", "20"))
MAX_TOKENS = int(env("MAX_TOKENS", "20"))


# ── Prompt builder ───────────────────────────────────────────────────────────

BASE_SENTENCE = "The quick brown fox jumps over the lazy dog again"


def build_prompt(target_tokens):
    """Build a prompt targeting approximately `target_tokens` tokens.
    Each repetition of BASE_SENTENCE is ~10 tokens."""
    reps = max(1, target_tokens // 10)
    return " ".join([BASE_SENTENCE] * reps)


# ── HTTP client with precise timing ──────────────────────────────────────────

@dataclass
class RequestResult:
    ttft_ms: float        # time to first token (streaming) or first byte (non-streaming)
    total_ms: float       # total request time
    status: int           # HTTP status code
    prompt_tokens: int    # from response usage
    completion_tokens: int  # from response usage
    body: dict = field(default_factory=dict)  # full response JSON
    error: str = ""       # error message if request failed
    token_times: list = field(default_factory=list)  # per-token timestamps (s from start)


def send_request(url, prompt, max_tokens=MAX_TOKENS, extra_headers=None):
    """Send a completion request with precise TTFT measurement.

    Uses http.client directly (not urllib/requests) for precise timing.
    Note: for non-streaming requests, TTFT ≈ total_ms because vLLM
    buffers the complete response before sending. The timing is still
    valid for comparing latency across configs (overhead cancels).

    Args:
        url: Full URL (http:// or https://)
        prompt: Prompt string
        max_tokens: Maximum completion tokens
        extra_headers: Dict of additional headers (e.g., prefill routing)

    Returns:
        RequestResult with timing, status, and parsed response.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    use_tls = parsed.scheme == "https"

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
    })

    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    try:
        start = time.monotonic()

        if use_tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=30)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=30)

        conn.request("POST", path, body=payload, headers=headers)
        response = conn.getresponse()

        # Read first byte — this is the true TTFT
        first_byte = response.read(1)
        ttft = time.monotonic() - start

        # Read the rest
        rest = response.read()
        total = time.monotonic() - start

        body_bytes = first_byte + rest
        conn.close()

        # Parse response
        try:
            body = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            body = {"raw": body_bytes.decode("utf-8", errors="replace")}

        usage = body.get("usage", {})

        return RequestResult(
            ttft_ms=round(ttft * 1000, 2),
            total_ms=round(total * 1000, 2),
            status=response.status,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            body=body,
        )

    except Exception as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            ttft_ms=0,
            total_ms=round(elapsed * 1000, 2),
            status=0,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(e),
        )


def send_disagg(url, prompt, max_tokens=MAX_TOKENS):
    """Send a disaggregated request (with prefill routing header)."""
    return send_request(
        url, prompt, max_tokens,
        extra_headers={"x-prefiller-host-port": PREFILL_HOST},
    )


def send_streaming(url, prompt, max_tokens=MAX_TOKENS, extra_headers=None):
    """Send a streaming completion request with true TTFT measurement.

    Uses SSE (server-sent events) to measure the actual time to first
    generated token, and records per-token arrival times for ITL.

    Returns:
        RequestResult with accurate ttft_ms, per-token token_times,
        and completion_tokens counted from the stream.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    use_tls = parsed.scheme == "https"

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    })

    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    try:
        start = time.monotonic()

        if use_tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=30)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=30)

        conn.request("POST", path, body=payload, headers=headers)
        response = conn.getresponse()

        status = response.status
        token_times = []
        prompt_tokens = 0
        last_chunk = {}

        # Parse SSE stream line by line
        while True:
            line = response.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text", ""):
                        token_times.append(time.monotonic() - start)
                    # Capture usage from last chunk (vLLM includes it there)
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                    last_chunk = chunk
                except (json.JSONDecodeError, ValueError):
                    pass

        total = time.monotonic() - start
        conn.close()

        ct = len(token_times)
        ttft = token_times[0] if token_times else total

        return RequestResult(
            ttft_ms=round(ttft * 1000, 2),
            total_ms=round(total * 1000, 2),
            status=status,
            prompt_tokens=prompt_tokens,
            completion_tokens=ct,
            body=last_chunk,
            token_times=token_times,
        )

    except Exception as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            ttft_ms=0,
            total_ms=round(elapsed * 1000, 2),
            status=0,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(e),
        )


# ── CSV output ───────────────────────────────────────────────────────────────

class CSVWriter:
    """Thread-safe CSV writer."""

    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._lock = threading.Lock()
        self._file = open(filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def write(self, row):
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def close(self):
        self._file.close()


# ── Progress output ──────────────────────────────────────────────────────────

def write_run_info(experiment, extra=None):
    """Write or update DATA_DIR/run-info.json with toolkit config and timestamps.

    Each experiment call adds its entry under the experiment key. This makes
    every dataset self-describing: you can always tell what config, model, and
    environment produced the data.

    Args:
        experiment: Experiment identifier (e.g., "exp1", "exp4").
        extra: Optional dict of additional metadata (e.g., cluster info from oc).
    """
    import platform
    filepath = os.path.join(DATA_DIR, "run-info.json")

    # Load existing if present (multiple experiments append to same file)
    existing = {}
    if os.path.exists(filepath):
        try:
            with open(filepath) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

    # Toolkit-level config (shared across all experiments)
    existing["toolkit"] = {
        "model": MODEL,
        "namespace": NS,
        "sim_mode": SIM,
        "baseline_url": BASELINE_URL,
        "disagg_d1_url": DISAGG_D1_URL,
        "disagg_d2_url": DISAGG_D2_URL,
        "prefill_host": PREFILL_HOST,
        "warmup": WARMUP,
        "runs": RUNS,
        "max_tokens": MAX_TOKENS,
        "data_dir": DATA_DIR,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    # Per-experiment entry
    entry = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        entry.update(extra)

    if "experiments" not in existing:
        existing["experiments"] = {}
    existing["experiments"][experiment] = entry

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")


def progress(msg, end="\n"):
    """Print progress to stderr so stdout stays clean for piping."""
    print(msg, end=end, file=sys.stderr, flush=True)


def dot():
    """Print a progress dot."""
    print(".", end="", file=sys.stderr, flush=True)


def print_config():
    """Print current configuration."""
    if SIM:
        progress(f"  Mode:     SIM (inference-sim, no GPU)")
    progress(f"  Model:    {MODEL}")
    progress(f"  NS:       {NS}")
    progress(f"  Baseline: {BASELINE_URL}")
    progress(f"  Decode-1: {DISAGG_D1_URL}")
    progress(f"  Decode-2: {DISAGG_D2_URL}")
    progress(f"  Warmup:   {WARMUP}  Runs: {RUNS}  Max tokens: {MAX_TOKENS}")
    progress(f"  Data dir: {DATA_DIR}")
    progress("")
