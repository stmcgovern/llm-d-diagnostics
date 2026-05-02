"""Tests for advisor/rebalance.py — P/D ratio decision logic.

Tests the rebalance recommendation engine against known scenarios
derived from exp6 empirical thresholds.  All cluster interaction
(get_pods, metrics scraping) is mocked.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rebalance import (
    DECODE_KV_CRITICAL,
    DECODE_KV_HIGH,
    PREFILL_QUEUE_CRITICAL,
    PREFILL_QUEUE_HIGH,
    PodMetrics,
    RebalanceResult,
    _scrape_pod_metrics,
    print_rebalance,
    rebalance,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

def _pod(name, ip="10.0.0.1"):
    role = "vllm-prefill" if "prefill" in name else "vllm-decode"
    return {
        "name": name, "ip": ip, "ready": True,
        "labels": {"app": role},
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


def _mock_metrics(kv_pct=0.5, waiting=0, running=1):
    body = VLLM_METRICS.format(KV_PCT=kv_pct, WAITING=waiting, RUNNING=running)
    mock_resp = MagicMock()
    mock_resp.read.return_value = body.encode()
    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = mock_resp
    return mock_conn


# ── Scrape parsing ───────────────────────────────────────────────────────

class TestScrapePodMetrics(unittest.TestCase):

    @patch("rebalance.http.client.HTTPConnection")
    def test_parses_metrics(self, mock_http):
        mock_http.return_value = _mock_metrics(kv_pct=0.72, waiting=3, running=5)
        pm = _scrape_pod_metrics(_pod("vllm-decode-0"), "ns")
        self.assertEqual(pm.role, "decode")
        self.assertAlmostEqual(pm.kv_cache_pct, 0.72)
        self.assertEqual(pm.requests_waiting, 3)
        self.assertEqual(pm.requests_running, 5)

    @patch("rebalance.http.client.HTTPConnection")
    def test_prefill_role(self, mock_http):
        mock_http.return_value = _mock_metrics()
        pm = _scrape_pod_metrics(_pod("vllm-prefill-0"), "ns")
        self.assertEqual(pm.role, "prefill")


# ── Rebalance decision logic ────────────────────────────────────────────

class TestRebalanceDecisions(unittest.TestCase):
    """Test the full rebalance() function with mocked cluster."""

    def _run_rebalance(self, prefill_pods, decode_pods, mock_http):
        all_pods = prefill_pods + decode_pods
        with patch("rebalance.get_pods", return_value=all_pods), \
             patch("rebalance.http.client.HTTPConnection", mock_http):
            return rebalance("test-ns")

    def test_balanced_cluster(self):
        mock_http = lambda *a, **kw: _mock_metrics(kv_pct=0.4, waiting=1, running=2)
        r = self._run_rebalance(
            [_pod("vllm-prefill-0")], [_pod("vllm-decode-0")], mock_http,
        )
        self.assertIn("BALANCED", r.recommendation)
        self.assertEqual(r.confidence, "high")

    def test_critical_kv_pressure(self):
        mock_http = lambda *a, **kw: _mock_metrics(kv_pct=0.95, waiting=0, running=5)
        r = self._run_rebalance(
            [_pod("vllm-prefill-0")], [_pod("vllm-decode-0")], mock_http,
        )
        self.assertIn("ADD DECODE", r.recommendation)
        self.assertIn("CRITICAL", r.recommendation)
        self.assertEqual(r.confidence, "high")

    def test_high_kv_low_queue(self):
        mock_http = lambda *a, **kw: _mock_metrics(kv_pct=0.85, waiting=1, running=5)
        r = self._run_rebalance(
            [_pod("vllm-prefill-0")], [_pod("vllm-decode-0")], mock_http,
        )
        self.assertIn("ADD DECODE", r.recommendation)

    def test_critical_prefill_queue(self):
        """When prefill queue is critically deep, recommend adding prefill."""
        call_count = [0]
        def mock_http(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:  # prefill pod
                return _mock_metrics(kv_pct=0.1, waiting=15, running=1)
            else:  # decode pod
                return _mock_metrics(kv_pct=0.3, waiting=0, running=2)
        r = self._run_rebalance(
            [_pod("vllm-prefill-0")], [_pod("vllm-decode-0")], mock_http,
        )
        self.assertIn("PREFILL", r.recommendation)

    def test_low_kv_remove_decode(self):
        """When KV utilization is very low and multiple decode pods exist."""
        mock_http = lambda *a, **kw: _mock_metrics(kv_pct=0.1, waiting=0, running=0)
        r = self._run_rebalance(
            [_pod("vllm-prefill-0")],
            [_pod("vllm-decode-0"), _pod("vllm-decode-1")],
            mock_http,
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
