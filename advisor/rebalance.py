"""
P/D ratio optimizer for running disaggregated inference clusters.

Monitors your live cluster and recommends prefill/decode ratio adjustments
based on KV cache utilization, queue depth, and measured throughput.
Unlike static manifest generation, this observes actual production load
and tells you when to add/remove replicas.

Decision logic derived from exp6 empirical data.

Part of the llm-d-diagnostics advisory layer.
"""

import argparse
import time
from dataclasses import dataclass, field

try:
    from ._cluster import get_pods_full as get_pods, scrape_pod_metrics
except ImportError:
    from _cluster import get_pods_full as get_pods, scrape_pod_metrics


@dataclass
class PodMetrics:
    name: str
    role: str  # "prefill" | "decode"
    kv_cache_pct: float = 0
    requests_waiting: int = 0
    requests_running: int = 0
    throughput_rps: float = 0  # not yet scraped; reserved for future use


@dataclass
class RebalanceResult:
    prefill_count: int = 0
    decode_count: int = 0
    recommendation: str = ""
    confidence: str = "low"
    reasoning: list = field(default_factory=list)
    metrics: list = field(default_factory=list)


# Thresholds from exp6 analysis
DECODE_KV_HIGH = 0.80
DECODE_KV_CRITICAL = 0.90
PREFILL_QUEUE_HIGH = 5
PREFILL_QUEUE_CRITICAL = 10


def _parse_pod_metrics(pod, namespace) -> PodMetrics:
    """Scrape and parse Prometheus metrics from a single pod into PodMetrics."""
    pm = PodMetrics(
        name=pod["name"],
        role="prefill" if "prefill" in pod.get("role", pod["name"]) else "decode",
    )

    body = scrape_pod_metrics(pod, namespace)

    for line in body.split("\n"):
        if line.startswith("#"):
            continue
        if "gpu_cache_usage" in line or "kv_cache_usage_perc" in line:
            try:
                pm.kv_cache_pct = float(line.split()[-1])
            except (ValueError, IndexError):
                pass
        elif "num_requests_waiting" in line:
            try:
                pm.requests_waiting = int(float(line.split()[-1]))
            except (ValueError, IndexError):
                pass
        elif "num_requests_running" in line:
            try:
                pm.requests_running = int(float(line.split()[-1]))
            except (ValueError, IndexError):
                pass

    return pm


def rebalance(namespace: str) -> RebalanceResult:
    """Analyze the running cluster and recommend P/D ratio changes."""
    pods = get_pods(namespace, "app.kubernetes.io/part-of=vllm-disagg")
    prefill_pods = [p for p in pods if "prefill" in p.get("role", "")]
    decode_pods = [p for p in pods if "decode" in p.get("role", "")]

    result = RebalanceResult(
        prefill_count=len(prefill_pods),
        decode_count=len(decode_pods),
    )

    if not prefill_pods or not decode_pods:
        result.recommendation = "NO DISAGG DEPLOYMENT FOUND"
        result.reasoning.append(f"Found {len(prefill_pods)} prefill, {len(decode_pods)} decode pods")
        return result

    all_metrics = []
    for p in prefill_pods + decode_pods:
        pm = _parse_pod_metrics(p, namespace)
        all_metrics.append(pm)
        result.metrics.append(pm)

    prefill_metrics = [m for m in all_metrics if m.role == "prefill"]
    decode_metrics = [m for m in all_metrics if m.role == "decode"]

    avg_prefill_queue = sum(m.requests_waiting for m in prefill_metrics) / max(len(prefill_metrics), 1)
    avg_decode_kv = sum(m.kv_cache_pct for m in decode_metrics) / max(len(decode_metrics), 1)
    max_decode_kv = max((m.kv_cache_pct for m in decode_metrics), default=0)
    total_decode_waiting = sum(m.requests_waiting for m in decode_metrics)

    result.reasoning.append(f"Current topology: {len(prefill_pods)}P + {len(decode_pods)}D")
    result.reasoning.append(f"Prefill avg queue: {avg_prefill_queue:.1f} requests")
    result.reasoning.append(f"Decode avg KV util: {avg_decode_kv*100:.0f}%, max: {max_decode_kv*100:.0f}%")
    result.reasoning.append(f"Decode total waiting: {total_decode_waiting}")

    if max_decode_kv >= DECODE_KV_CRITICAL:
        result.recommendation = "ADD DECODE REPLICA (CRITICAL)"
        result.confidence = "high"
        result.reasoning.append(
            f"Decode KV cache at {max_decode_kv*100:.0f}% -- near OOM. "
            "Add a decode replica to distribute KV cache pressure.")
    elif avg_decode_kv >= DECODE_KV_HIGH and avg_prefill_queue < PREFILL_QUEUE_HIGH:
        result.recommendation = "ADD DECODE REPLICA"
        result.confidence = "high"
        result.reasoning.append(
            f"Decode KV util ({avg_decode_kv*100:.0f}%) is high while prefill queue is low ({avg_prefill_queue:.0f}). "
            "Decode is the bottleneck -- add a decode replica.")
    elif avg_prefill_queue >= PREFILL_QUEUE_CRITICAL:
        result.recommendation = "ADD PREFILL REPLICA"
        result.confidence = "high"
        result.reasoning.append(
            f"Prefill queue ({avg_prefill_queue:.0f}) is critically deep. "
            "Prefill is the bottleneck -- add a prefill replica.")
    elif avg_prefill_queue >= PREFILL_QUEUE_HIGH and avg_decode_kv < 0.50:
        result.recommendation = "ADD PREFILL REPLICA"
        result.confidence = "medium"
        result.reasoning.append(
            f"Prefill queue ({avg_prefill_queue:.0f}) is growing while decode KV ({avg_decode_kv*100:.0f}%) has headroom. "
            "Consider adding a prefill replica.")
    elif avg_decode_kv < 0.30 and len(decode_pods) > 1 and avg_prefill_queue < 2:
        result.recommendation = "REMOVE DECODE REPLICA (SAVE COST)"
        result.confidence = "medium"
        result.reasoning.append(
            f"Decode KV util ({avg_decode_kv*100:.0f}%) is very low with {len(decode_pods)} decode replicas. "
            "You can save GPU cost by removing one.")
    else:
        result.recommendation = "BALANCED -- NO CHANGE NEEDED"
        result.confidence = "high"
        result.reasoning.append("Current P/D ratio is well-balanced for the observed load.")

    return result


def rebalance_watch(namespace: str, interval_s: float = 15, duration_s: float = 300):
    """Watch mode: scrape metrics over time and predict when scaling is needed."""
    print(f"\n{'='*60}")
    print(f"  PREDICTIVE P/D SCALING (watch mode)")
    print(f"{'='*60}")
    print(f"  Namespace: {namespace}")
    print(f"  Interval: {interval_s}s, Duration: {duration_s}s")
    print()

    kv_history = []
    t_start = time.time()

    try:
        while time.time() - t_start < duration_s:
            pods = get_pods(namespace, "app.kubernetes.io/part-of=vllm-disagg")
            decode_pods = [p for p in pods if "decode" in p.get("role", "")]

            kv_vals = []
            q_vals = []
            for p in decode_pods:
                pm = _parse_pod_metrics(p, namespace)
                kv_vals.append(pm.kv_cache_pct)
                q_vals.append(pm.requests_waiting)

            now = time.time()
            avg_kv = sum(kv_vals) / max(len(kv_vals), 1)
            avg_q = sum(q_vals) / max(len(q_vals), 1)
            kv_history.append((now, avg_kv))

            trend_msg = ""
            if len(kv_history) >= 3:
                window = [(t, v) for t, v in kv_history if t > now - 300]
                if len(window) >= 3:
                    n = len(window)
                    t_vals = [t - window[0][0] for t, _ in window]
                    v_vals = [v for _, v in window]
                    t_mean = sum(t_vals) / n
                    v_mean = sum(v_vals) / n
                    num = sum((t - t_mean) * (v - v_mean) for t, v in zip(t_vals, v_vals))
                    den = sum((t - t_mean) ** 2 for t in t_vals)
                    slope = num / den if den > 0 else 0

                    slope_per_min = slope * 60 * 100
                    if slope > 0 and avg_kv < DECODE_KV_HIGH:
                        time_to_threshold = (DECODE_KV_HIGH - avg_kv) / slope if slope > 0 else float('inf')
                        if time_to_threshold < 600:
                            trend_msg = (f"KV trending +{slope_per_min:.1f}%/min. "
                                        f"Hits {DECODE_KV_HIGH*100:.0f}% in {time_to_threshold/60:.0f} min. "
                                        "RECOMMEND: scale decode NOW")
                        else:
                            trend_msg = f"KV trending +{slope_per_min:.1f}%/min (stable)"
                    elif slope < 0:
                        trend_msg = f"KV trending {slope_per_min:.1f}%/min (decreasing)"
                    else:
                        trend_msg = "KV stable"

            ts = time.strftime("%H:%M:%S")
            print(f"  [{ts}] KV={avg_kv*100:.0f}% queue={avg_q:.0f} {trend_msg}")
            time.sleep(interval_s)

    except KeyboardInterrupt:
        print("\n  Interrupted.")

    if kv_history:
        kv_vals = [v for _, v in kv_history]
        print(f"\n  Summary: KV range {min(kv_vals)*100:.0f}-{max(kv_vals)*100:.0f}%, "
              f"{len(kv_history)} samples over {(kv_history[-1][0]-kv_history[0][0]):.0f}s")


def print_rebalance(r: RebalanceResult):
    print(f"\n{'='*60}")
    print(f"  P/D RATIO ANALYSIS")
    print(f"{'='*60}")
    print(f"  Current: {r.prefill_count}P + {r.decode_count}D")
    print()
    for pm in r.metrics:
        print(f"  {pm.role:7s} {pm.name}: KV={pm.kv_cache_pct*100:.0f}%, "
              f"waiting={pm.requests_waiting}, running={pm.requests_running}")
    print()
    print(f"  RECOMMENDATION: {r.recommendation}")
    print(f"  Confidence: {r.confidence}")
    for reason in r.reasoning:
        print(f"    - {reason}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P/D ratio optimizer for disaggregated inference")
    parser.add_argument("--namespace", "-n", required=True, help="Kubernetes namespace")
    parser.add_argument("--watch", action="store_true", help="Watch mode with trend prediction")
    parser.add_argument("--interval", type=float, default=15, help="Watch interval in seconds (default: 15)")
    parser.add_argument("--duration", type=float, default=300, help="Watch duration in seconds (default: 300)")
    args = parser.parse_args()

    if args.watch:
        rebalance_watch(args.namespace, interval_s=args.interval, duration_s=args.duration)
    else:
        result = rebalance(args.namespace)
        print_rebalance(result)
