"""Tests for advisor/rebalance.py — P/D ratio decision logic.

Tests the rebalance recommendation engine against known scenarios
derived from exp6 empirical thresholds.  All cluster interaction
(get_pods, metrics scraping) is mocked.
"""

import unittest
from unittest.mock import patch

from rebalance import (
    DECODE_KV_CRITICAL,
    DECODE_KV_HIGH,
    PREFILL_QUEUE_CRITICAL,
    PREFILL_QUEUE_HIGH,
    PodMetrics,
    RebalanceResult,
    _parse_pod_metrics,
    print_rebalance,
    rebalance,
)

# ── Fixtures ─────────────────────────────────────────────────────────────

def _pod(name, ip="10.0.0.1"):
    app = "vllm-prefill" if "prefill" in name else "vllm-decode"
    return {
        "name": name, "ip": ip, "ready": True,
        "labels": {"app": app},
        "role": app,
        "image": "vllm/vllm-openai:v0.18.1",
        "args": "[]",
    }


VLLM_METRICS = """\
# HELP vllm:kv_cache_usage_perc KV cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc {KV_PCT}
# HELP vllm:num_requests_waiting requests waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting {WAITING}
# HELP vllm:num_requests_running requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running {RUNNING}
"""


def _metrics_text(kv_pct=0.5, waiting=0, running=1):
    return VLLM_METRICS.format(KV_PCT=kv_pct, WAITING=waiting, RUNNING=running)


# ── Scrape parsing ───────────────────────────────────────────────────────

class TestParsePodMetrics(unittest.TestCase):

    @patch("rebalance.scrape_pod_metrics")
    def test_parses_metrics(self, mock_scrape):
        mock_scrape.return_value = _metrics_text(kv_pct=0.72, waiting=3, running=5)
        pm = _parse_pod_metrics(_pod("vllm-decode-0"), "ns")
        self.assertEqual(pm.role, "decode")
        self.assertAlmostEqual(pm.kv_cache_pct, 0.72)
        self.assertEqual(pm.requests_waiting, 3)
        self.assertEqual(pm.requests_running, 5)

    @patch("rebalance.scrape_pod_metrics")
    def test_prefill_role(self, mock_scrape):
        mock_scrape.return_value = _metrics_text()
        pm = _parse_pod_metrics(_pod("vllm-prefill-0"), "ns")
        self.assertEqual(pm.role, "prefill")


# ── Rebalance decision logic ────────────────────────────────────────────

class TestRebalanceDecisions(unittest.TestCase):
    """Test the full rebalance() function with mocked cluster."""

    def _run_rebalance(self, prefill_pods, decode_pods, scrape_fn):
        all_pods = prefill_pods + decode_pods
        with patch("rebalance.get_pods", return_value=all_pods), \
             patch("rebalance.scrape_pod_metrics", side_effect=scrape_fn):
            return rebalance("test-ns")

    def test_balanced_cluster(self):
        r = self._run_rebalance(
            [_pod("vllm-prefill-0")], [_pod("vllm-decode-0")],
            lambda pod, ns: _metrics_text(kv_pct=0.4, waiting=1, running=2),
        )
        self.assertIn("BALANCED", r.recommendation)
        self.assertEqual(r.confidence, "high")

    def test_critical_kv_pressure(self):
        r = self._run_rebalance(
            [_pod("vllm-prefill-0")], [_pod("vllm-decode-0")],
            lambda pod, ns: _metrics_text(kv_pct=0.95, waiting=0, running=5),
        )
        self.assertIn("ADD DECODE", r.recommendation)
        self.assertIn("CRITICAL", r.recommendation)
        self.assertEqual(r.confidence, "high")

    def test_high_kv_low_queue(self):
        r = self._run_rebalance(
            [_pod("vllm-prefill-0")], [_pod("vllm-decode-0")],
            lambda pod, ns: _metrics_text(kv_pct=0.85, waiting=1, running=5),
        )
        self.assertIn("ADD DECODE", r.recommendation)

    def test_critical_prefill_queue(self):
        """When prefill queue is critically deep, recommend adding prefill."""
        def scrape_fn(pod, ns):
            if "prefill" in pod["name"]:
                return _metrics_text(kv_pct=0.1, waiting=15, running=1)
            return _metrics_text(kv_pct=0.3, waiting=0, running=2)
        r = self._run_rebalance(
            [_pod("vllm-prefill-0")], [_pod("vllm-decode-0")], scrape_fn,
        )
        self.assertIn("PREFILL", r.recommendation)

    def test_low_kv_remove_decode(self):
        """When KV utilization is very low and multiple decode pods exist."""
        r = self._run_rebalance(
            [_pod("vllm-prefill-0")],
            [_pod("vllm-decode-0"), _pod("vllm-decode-1")],
            lambda pod, ns: _metrics_text(kv_pct=0.1, waiting=0, running=0),
        )
        self.assertIn("REMOVE", r.recommendation)
        self.assertIn("SAVE COST", r.recommendation)

    def test_no_pods(self):
        with patch("rebalance.get_pods", return_value=[]):
            r = rebalance("ns")
        self.assertIn("NO DISAGG", r.recommendation)

    def test_prefill_only(self):
        with patch("rebalance.get_pods", return_value=[_pod("vllm-prefill-0")]):
            r = rebalance("ns")
        self.assertIn("NO DISAGG", r.recommendation)


# ── Thresholds ───────────────────────────────────────────────────────────

class TestThresholds(unittest.TestCase):
    """Verify threshold constants match exp6 analysis values."""

    def test_kv_thresholds(self):
        self.assertAlmostEqual(DECODE_KV_HIGH, 0.80)
        self.assertAlmostEqual(DECODE_KV_CRITICAL, 0.90)

    def test_queue_thresholds(self):
        self.assertEqual(PREFILL_QUEUE_HIGH, 5)
        self.assertEqual(PREFILL_QUEUE_CRITICAL, 10)


# ── Output ───────────────────────────────────────────────────────────────

class TestPrintRebalance(unittest.TestCase):

    def test_prints_without_error(self):
        r = RebalanceResult(
            prefill_count=1, decode_count=2,
            recommendation="BALANCED", confidence="high",
            reasoning=["test"],
            metrics=[PodMetrics("d-0", "decode", kv_cache_pct=0.4, requests_waiting=1)],
        )
        print_rebalance(r)


if __name__ == "__main__":
    unittest.main()
