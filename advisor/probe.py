"""
Synthetic disaggregated request prober.

Sends a real disagg request through the full P/D pipeline (client -> sidecar
-> prefill -> NIXL KV transfer -> decode -> response) and tracks TTFT
against a rolling baseline for anomaly detection.

Uses ``toolkit/client.py``'s ``send_streaming`` for true TTFT measurement
instead of the legacy ``oc exec``-based approach.
"""

import os
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "toolkit"))
from client import send_streaming  # noqa: E402


@dataclass
class ProbeResult:
    timestamp: float
    ttft_ms: float
    status: int
    healthy: bool
    baseline_ms: float
    ratio: float  # ttft / baseline
    error: str = ""


class DisaggProber:
    """Sends synthetic disagg requests and tracks TTFT against baseline."""

    def __init__(self, namespace: str, model: str, window_size: int = 20):
        self.namespace = namespace
        self.model = model
        self.window_size = window_size
        self._baseline_window: deque[float] = deque(maxlen=window_size)
        self._baseline_ms: float = 0.0
        self.disagg_url = "https://vllm-decode-svc:8000/v1/completions"
        self.prefill_host = f"vllm-prefill-svc.{namespace}.svc.cluster.local:8100"

        os.environ.setdefault("MODEL", model)
        os.environ.setdefault("NS", namespace)

    def probe(self) -> ProbeResult:
        """Send one synthetic disagg request and evaluate health."""
        t = time.time()
        try:
            r = send_streaming(
                self.disagg_url, "Hello", max_tokens=5,
                extra_headers={"x-prefiller-host-port": self.prefill_host},
            )
            ttft = r.ttft_ms
            status = r.status
            error = r.error
        except Exception as e:
            ttft = 0
            status = 0
            error = str(e)[:200]

        if status == 200 and ttft > 0:
            self._baseline_window.append(ttft)
            if len(self._baseline_window) >= 3:
                self._baseline_ms = statistics.median(self._baseline_window)

        baseline = self._baseline_ms or ttft or 1
        ratio = ttft / baseline if baseline > 0 else 0

        healthy = status == 200 and ratio < 3.0

        return ProbeResult(
            timestamp=t, ttft_ms=ttft, status=status,
            healthy=healthy, baseline_ms=round(baseline, 1),
            ratio=round(ratio, 2), error=error,
        )

    def warmup(self, n: int = 5):
        """Send warmup probes to establish baseline."""
        for _ in range(n):
            self.probe()
