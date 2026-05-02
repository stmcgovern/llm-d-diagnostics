"""
Continuous NIXL health checker for disaggregated inference.

Runs 10 checks every interval, combining passive monitoring (metrics scraping,
pod inspection) with active probing (synthetic disagg requests). Alerts on
NIXL failures, version mismatches, KV pressure, and latency degradation
BEFORE users notice.

Part of the llm-d-diagnostics advisory layer.  Uses ``_cluster.py`` for pod
discovery and ``probe.py`` (which itself uses ``toolkit/client.py``) for
synthetic requests.
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from ._cluster import oc_safe, get_pods_full as get_pods, scrape_pod_metrics
    from .probe import DisaggProber, ProbeResult
except ImportError:
    from _cluster import oc_safe, get_pods_full as get_pods, scrape_pod_metrics
    from probe import DisaggProber, ProbeResult


@dataclass
class HealthCheck:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "ok"  # "ok" | "warning" | "critical"


@dataclass
class HealthSnapshot:
    timestamp: float
    checks: list[HealthCheck] = field(default_factory=list)
    overall: str = "HEALTHY"  # "HEALTHY" | "DEGRADED" | "UNHEALTHY"
    probe: Optional[ProbeResult] = None


class HealthMonitor:
    """Continuous health monitor for P/D disaggregated inference."""

    def __init__(self, namespace: str, model: str, interval_s: float = 30.0):
        self.namespace = namespace
        self.model = model
        self.interval_s = interval_s
        self.prober = DisaggProber(namespace, model)
        self.snapshots: list[HealthSnapshot] = []
        self._prev_nixl_fails: dict[str, float] = {}
        self._prev_kv_expired: dict[str, float] = {}
        self._transfer_duration_history: dict[str, list] = {}

    def run(self, duration_s: float = 0):
        """Run health monitoring loop. duration_s=0 runs forever."""
        print(f"\n{'='*60}")
        print(f"  DISAGG HEALTH MONITOR")
        print(f"{'='*60}")
        print(f"  Namespace: {self.namespace}")
        print(f"  Model:     {self.model}")
        print(f"  Interval:  {self.interval_s}s")
        print(f"  {'Running for ' + str(int(duration_s)) + 's' if duration_s else 'Running until Ctrl+C'}")
        print()

        print(f"  Warming up probe baseline...", end=" ", flush=True)
        self.prober.warmup(5)
        print(f"done (baseline: {self.prober._baseline_ms:.0f}ms)")
        print()

        t_start = time.time()
        try:
            while True:
                if duration_s and (time.time() - t_start) > duration_s:
                    break
                snap = self._check_all()
                self.snapshots.append(snap)
                self._print_snapshot(snap)
                time.sleep(self.interval_s)
        except KeyboardInterrupt:
            print("\n  Interrupted.")

        self._print_summary()

    def _check_all(self) -> HealthSnapshot:
        snap = HealthSnapshot(timestamp=time.time())

        pods = get_pods(self.namespace, "app.kubernetes.io/part-of=vllm-disagg")
        prefill_pods = [p for p in pods if p["labels"].get("app") == "vllm-prefill"]
        decode_pods = [p for p in pods if "decode" in p["labels"].get("app", "")]

        snap.checks.append(self._check_pod_health(prefill_pods, decode_pods))
        snap.checks.append(self._check_version_compat(pods))
        snap.checks.append(self._check_kv_roles(pods))
        snap.checks.append(self._check_nixl_failures(prefill_pods + decode_pods))
        snap.checks.append(self._check_kv_pressure(prefill_pods + decode_pods))
        snap.checks.append(self._check_queue_balance(prefill_pods, decode_pods))
        snap.checks.append(self._check_kv_expiration(prefill_pods + decode_pods))
        snap.checks.append(self._check_transfer_duration(prefill_pods + decode_pods))

        probe = self.prober.probe()
        snap.probe = probe
        snap.checks.append(self._check_probe(probe))
        snap.checks.append(self._check_nixl_config(prefill_pods))

        critical = sum(1 for c in snap.checks if c.severity == "critical")
        warning = sum(1 for c in snap.checks if c.severity == "warning")
        if critical:
            snap.overall = "UNHEALTHY"
        elif warning:
            snap.overall = "DEGRADED"
        else:
            snap.overall = "HEALTHY"

        return snap

    def _check_pod_health(self, prefill, decode) -> HealthCheck:
        not_ready = [p["name"] for p in prefill + decode if not p["ready"]]
        if not prefill:
            return HealthCheck("pod_health", False, "No prefill pods found", "critical")
        if not decode:
            return HealthCheck("pod_health", False, "No decode pods found", "critical")
        if not_ready:
            return HealthCheck("pod_health", False, f"Not ready: {', '.join(not_ready)}", "critical")
        return HealthCheck("pod_health", True, f"{len(prefill)}P + {len(decode)}D all ready")

    def _check_version_compat(self, pods) -> HealthCheck:
        images = set(p["image"] for p in pods)
        if len(images) > 1:
            return HealthCheck("version_compat", False,
                               f"Mixed images: {images}", "warning")
        return HealthCheck("version_compat", True, f"All pods: {images.pop() if images else 'N/A'}")

    def _check_kv_roles(self, pods) -> HealthCheck:
        producers = [p for p in pods if "kv_producer" in p["args"] or "kv_both" in p["args"]]
        consumers = [p for p in pods if "kv_consumer" in p["args"] or "kv_both" in p["args"]]
        if not producers:
            return HealthCheck("kv_roles", False, "No KV producer found", "critical")
        return HealthCheck("kv_roles", True, f"{len(producers)} producer(s), {len(consumers)} consumer(s)")

    def _check_nixl_failures(self, pods) -> HealthCheck:
        for p in pods:
            body = scrape_pod_metrics(p, self.namespace)
            if not body:
                continue
            for line in body.split("\n"):
                if line.startswith("vllm:nixl_num_failed_transfers"):
                    val = float(line.split()[-1])
                    prev = self._prev_nixl_fails.get(p["name"], 0)
                    self._prev_nixl_fails[p["name"]] = val
                    if val > prev:
                        return HealthCheck("nixl_failures", False,
                                           f"{p['name']}: {int(val - prev)} new NIXL failures", "critical")
        return HealthCheck("nixl_failures", True, "No NIXL transfer failures")

    def _check_kv_pressure(self, pods) -> HealthCheck:
        for p in pods:
            body = scrape_pod_metrics(p, self.namespace)
            if not body:
                continue
            for line in body.split("\n"):
                if line.startswith("vllm:kv_cache_usage_perc"):
                    val = float(line.split()[-1])
                    if val > 0.9:
                        return HealthCheck("kv_pressure", False,
                                           f"{p['name']}: KV cache at {val*100:.0f}%", "warning")
        return HealthCheck("kv_pressure", True, "KV cache pressure normal")

    def _check_queue_balance(self, prefill, decode) -> HealthCheck:
        """Check queue depth balance across decode pods."""
        if len(decode) < 2:
            return HealthCheck("queue_balance", True, f"{len(prefill)}P + {len(decode)}D")

        queue_depths = {}
        for p in decode:
            body = scrape_pod_metrics(p, self.namespace)
            if not body:
                continue
            for line in body.split("\n"):
                if line.startswith("vllm:num_requests_waiting"):
                    try:
                        queue_depths[p["name"]] = float(line.split()[-1])
                    except (ValueError, IndexError):
                        pass

        if len(queue_depths) < 2:
            return HealthCheck("queue_balance", True,
                               f"{len(prefill)}P + {len(decode)}D (queue metrics unavailable)")

        avg = sum(queue_depths.values()) / len(queue_depths)
        max_pod = max(queue_depths, key=queue_depths.get)
        max_val = queue_depths[max_pod]
        if avg > 0 and max_val > avg * 2:
            return HealthCheck("queue_balance", False,
                               f"{max_pod} has {max_val:.0f} waiting (avg {avg:.0f})", "warning")
        return HealthCheck("queue_balance", True,
                           f"{len(decode)} decode pods balanced (avg queue: {avg:.0f})")

    def _check_kv_expiration(self, pods) -> HealthCheck:
        """Check for KV block expiration (stranded transfers). vLLM PR #32340."""
        for p in pods:
            body = scrape_pod_metrics(p, self.namespace)
            if not body:
                continue
            for line in body.split("\n"):
                if "nixl_num_kv_expired_reqs" in line and not line.startswith("#"):
                    try:
                        val = float(line.split()[-1])
                        prev = self._prev_kv_expired.get(p["name"], 0)
                        self._prev_kv_expired[p["name"]] = val
                        if val > prev:
                            delta = int(val - prev)
                            return HealthCheck("kv_expiration", False,
                                               f"{p['name']}: {delta} new KV expired reqs (total {int(val)}). "
                                               "Increase VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "critical")
                    except (ValueError, IndexError):
                        pass
        return HealthCheck("kv_expiration", True, "No KV block expirations")

    def _check_transfer_duration(self, pods) -> HealthCheck:
        """Track NIXL transfer duration for network degradation detection."""
        for p in pods:
            body = scrape_pod_metrics(p, self.namespace)
            if not body:
                continue
            for line in body.split("\n"):
                if "nixl_transfer_duration_seconds" in line and not line.startswith("#") and "sum" in line:
                    try:
                        val = float(line.split()[-1])
                        hist = self._transfer_duration_history.setdefault(p["name"], [])
                        hist.append(val)
                        if len(hist) > 20:
                            hist.pop(0)
                        if len(hist) >= 5:
                            recent = hist[-3:]
                            older = hist[:-3]
                            recent_avg = sum(recent) / len(recent)
                            older_avg = sum(older) / len(older) if older else recent_avg
                            if older_avg > 0 and recent_avg > older_avg * 2:
                                return HealthCheck("transfer_duration", False,
                                                   f"{p['name']}: transfer duration trending up "
                                                   f"({recent_avg:.3f}s vs {older_avg:.3f}s avg)",
                                                   "warning")
                    except (ValueError, IndexError):
                        pass
        return HealthCheck("transfer_duration", True, "NIXL transfer duration stable")

    def _check_probe(self, probe: ProbeResult) -> HealthCheck:
        if not probe.healthy:
            if probe.status != 200:
                return HealthCheck("synthetic_probe", False,
                                   f"Probe failed: {probe.error[:100]}", "critical")
            return HealthCheck("synthetic_probe", False,
                               f"TTFT {probe.ttft_ms:.0f}ms ({probe.ratio:.1f}x baseline {probe.baseline_ms:.0f}ms)",
                               "warning")
        return HealthCheck("synthetic_probe", True,
                           f"TTFT {probe.ttft_ms:.0f}ms ({probe.ratio:.1f}x baseline)")

    def _check_nixl_config(self, prefill_pods) -> HealthCheck:
        """Check that prefill pod has NIXL side-channel configured and IP matches."""
        for p in prefill_pods:
            out, _ = oc_safe("get", "pod", p["name"], "-n", self.namespace,
                             "-o", "jsonpath={.spec.containers[0].env}", timeout=10)
            if "VLLM_NIXL_SIDE_CHANNEL_HOST" not in out:
                out2, _ = oc_safe("exec", p["name"], "-n", self.namespace, "--",
                                  "printenv", "VLLM_NIXL_SIDE_CHANNEL_HOST", timeout=10)
                if not out2.strip():
                    return HealthCheck("nixl_config", False,
                                       f"{p['name']}: VLLM_NIXL_SIDE_CHANNEL_HOST not set", "warning")
            pod_ip = p.get("ip", "")
            if pod_ip and pod_ip not in out:
                out2, _ = oc_safe("exec", p["name"], "-n", self.namespace, "--",
                                  "printenv", "VLLM_NIXL_SIDE_CHANNEL_HOST", timeout=10)
                if out2.strip() and pod_ip not in out2:
                    return HealthCheck("nixl_config", False,
                                       f"{p['name']}: side-channel host ({out2.strip()}) != pod IP ({pod_ip})",
                                       "warning")
        if not prefill_pods:
            return HealthCheck("nixl_config", False, "No prefill pods to check", "warning")
        return HealthCheck("nixl_config", True, "NIXL side-channel configured")

    def _print_snapshot(self, snap: HealthSnapshot):
        ts = time.strftime("%H:%M:%S", time.localtime(snap.timestamp))
        status = {"HEALTHY": "OK", "DEGRADED": "WARN", "UNHEALTHY": "CRIT"}[snap.overall]

        if snap.overall == "HEALTHY":
            probe_info = f"probe={snap.probe.ttft_ms:.0f}ms" if snap.probe else ""
            print(f"  [{ts}] {status}: {probe_info}")
        else:
            print(f"\n  [{ts}] *** {status} ***")
            for c in snap.checks:
                if not c.passed:
                    sev = c.severity.upper()
                    print(f"    [{sev}] {c.name}: {c.detail}")
            if snap.probe and not snap.probe.healthy:
                print(f"    Probe: {snap.probe.ttft_ms:.0f}ms (baseline {snap.probe.baseline_ms:.0f}ms, {snap.probe.ratio:.1f}x)")
            print()

    def _print_summary(self):
        total = len(self.snapshots)
        healthy = sum(1 for s in self.snapshots if s.overall == "HEALTHY")
        degraded = sum(1 for s in self.snapshots if s.overall == "DEGRADED")
        unhealthy = sum(1 for s in self.snapshots if s.overall == "UNHEALTHY")
        print(f"\n{'='*60}")
        print(f"  HEALTH SUMMARY")
        print(f"{'='*60}")
        print(f"  Snapshots: {total}")
        if total:
            print(f"  Healthy:   {healthy} ({healthy/total*100:.0f}%)")
        print(f"  Degraded:  {degraded}")
        print(f"  Unhealthy: {unhealthy}")
        if unhealthy:
            print(f"  ALERT: {unhealthy} unhealthy snapshot(s) detected!")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuous health monitoring for disaggregated inference")
    parser.add_argument("--namespace", "-n", required=True, help="Kubernetes namespace")
    parser.add_argument("--model", "-m", required=True, help="Model name")
    parser.add_argument("--interval", type=float, default=30.0, help="Check interval in seconds (default: 30)")
    parser.add_argument("--duration", type=float, default=0, help="Run duration in seconds (0 = forever)")
    args = parser.parse_args()

    monitor = HealthMonitor(args.namespace, args.model, interval_s=args.interval)
    monitor.run(duration_s=args.duration)
