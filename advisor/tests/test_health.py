"""Tests for advisor/health.py — continuous NIXL health checker.

All cluster interaction (pods, metrics, probes) is mocked.  Tests
exercise the 10 individual _check_* methods, overall status rollup,
delta tracking for failures/expirations, and duration trend detection.
"""

import time
import unittest
from unittest.mock import patch, MagicMock

from health import HealthMonitor, HealthCheck, HealthSnapshot
from probe import ProbeResult


# ── Test fixtures ────────────────────────────────────────────────────────

def _pod(name, role="vllm-decode", ready=True, ip="10.0.0.1",
         image="vllm/vllm-openai:v0.18.1", args="[]"):
    return {
        "name": name, "ip": ip, "ready": ready,
        "labels": {"app": role},
        "role": role,
        "image": image,
        "args": str(args),
    }


PREFILL = _pod("vllm-prefill-0", role="vllm-prefill", ip="10.0.0.1",
               args='["--kv-transfer-config", \'{"kv_role":"kv_producer"}\']')
DECODE_1 = _pod("vllm-decode-0", role="vllm-decode", ip="10.0.0.2",
                args='["--kv-transfer-config", \'{"kv_role":"kv_consumer"}\']')
DECODE_2 = _pod("vllm-decode-1", role="vllm-decode", ip="10.0.0.3",
                args='["--kv-transfer-config", \'{"kv_role":"kv_consumer"}\']')


def _make_monitor():
    with patch("health.DisaggProber"):
        m = HealthMonitor("test-ns", "test-model")
    return m


def _healthy_probe(ttft=50.0, baseline=45.0):
    return ProbeResult(
        timestamp=time.time(), ttft_ms=ttft, status=200,
        healthy=True, baseline_ms=baseline, ratio=ttft / baseline,
    )


def _unhealthy_probe(ttft=200.0, baseline=45.0, status=200, error=""):
    return ProbeResult(
        timestamp=time.time(), ttft_ms=ttft, status=status,
        healthy=False, baseline_ms=baseline, ratio=ttft / baseline,
        error=error,
    )


# ── Pod health ───────────────────────────────────────────────────────────

class TestCheckPodHealth(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    def test_all_healthy(self):
        c = self.mon._check_pod_health([PREFILL], [DECODE_1, DECODE_2])
        self.assertTrue(c.passed)
        self.assertEqual(c.severity, "ok")
        self.assertIn("2D", c.detail)

    def test_no_prefill(self):
        c = self.mon._check_pod_health([], [DECODE_1])
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "critical")
        self.assertIn("prefill", c.detail.lower())

    def test_no_decode(self):
        c = self.mon._check_pod_health([PREFILL], [])
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "critical")
        self.assertIn("decode", c.detail.lower())

    def test_not_ready(self):
        bad = _pod("vllm-decode-bad", ready=False)
        c = self.mon._check_pod_health([PREFILL], [bad])
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "critical")
        self.assertIn("vllm-decode-bad", c.detail)


# ── Version compat ──────────────────────────────────────────────────────

class TestCheckVersionCompat(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    def test_matching_images(self):
        c = self.mon._check_version_compat([PREFILL, DECODE_1])
        self.assertTrue(c.passed)

    def test_mixed_images(self):
        old = _pod("d-old", image="vllm/vllm-openai:v0.17.0")
        c = self.mon._check_version_compat([PREFILL, old])
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "warning")

    def test_empty_pods(self):
        c = self.mon._check_version_compat([])
        self.assertTrue(c.passed)
        self.assertIn("N/A", c.detail)


# ── KV roles ─────────────────────────────────────────────────────────────

class TestCheckKvRoles(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    def test_producer_and_consumer(self):
        c = self.mon._check_kv_roles([PREFILL, DECODE_1])
        self.assertTrue(c.passed)

    def test_no_producer(self):
        bare = _pod("p-bare", role="vllm-prefill", args="[]")
        c = self.mon._check_kv_roles([bare, DECODE_1])
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "critical")

    def test_kv_both_counts_as_producer(self):
        both = _pod("p-both", role="vllm-prefill",
                     args='["--kv-transfer-config", \'{"kv_role":"kv_both"}\']')
        c = self.mon._check_kv_roles([both])
        self.assertTrue(c.passed)


# ── NIXL failures (delta tracking) ──────────────────────────────────────

class TestCheckNixlFailures(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    def test_no_failures(self):
        m = {"vllm-decode-0": "vllm:nixl_num_failed_transfers 0\n"}
        c = self.mon._check_nixl_failures([DECODE_1], m)
        self.assertTrue(c.passed)

    def test_new_failures_detected(self):
        m = {"vllm-decode-0": "vllm:nixl_num_failed_transfers 5\n"}
        self.mon._prev_nixl_fails["vllm-decode-0"] = 3
        c = self.mon._check_nixl_failures([DECODE_1], m)
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "critical")
        self.assertIn("2", c.detail)

    def test_no_increase_passes(self):
        m = {"vllm-decode-0": "vllm:nixl_num_failed_transfers 5\n"}
        self.mon._prev_nixl_fails["vllm-decode-0"] = 5
        c = self.mon._check_nixl_failures([DECODE_1], m)
        self.assertTrue(c.passed)

    def test_empty_metrics(self):
        m = {"vllm-decode-0": ""}
        c = self.mon._check_nixl_failures([DECODE_1], m)
        self.assertTrue(c.passed)

    def test_malformed_metric_skipped(self):
        m = {"vllm-decode-0": "vllm:nixl_num_failed_transfers not_a_number\n"}
        c = self.mon._check_nixl_failures([DECODE_1], m)
        self.assertTrue(c.passed)


# ── KV pressure ──────────────────────────────────────────────────────────

class TestCheckKvPressure(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    def test_normal(self):
        m = {"vllm-decode-0": "vllm:kv_cache_usage_perc 0.45\n"}
        c = self.mon._check_kv_pressure([DECODE_1], m)
        self.assertTrue(c.passed)

    def test_high_pressure(self):
        m = {"vllm-decode-0": "vllm:kv_cache_usage_perc 0.95\n"}
        c = self.mon._check_kv_pressure([DECODE_1], m)
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "warning")
        self.assertIn("95%", c.detail)

    def test_threshold_boundary(self):
        m = {"vllm-decode-0": "vllm:kv_cache_usage_perc 0.90\n"}
        c = self.mon._check_kv_pressure([DECODE_1], m)
        self.assertTrue(c.passed)

    def test_malformed_metric_skipped(self):
        m = {"vllm-decode-0": "vllm:kv_cache_usage_perc not_a_number\n"}
        c = self.mon._check_kv_pressure([DECODE_1], m)
        self.assertTrue(c.passed)


# ── Queue balance ────────────────────────────────────────────────────────

class TestCheckQueueBalance(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    def test_balanced(self):
        m = {"vllm-decode-0": "vllm:num_requests_waiting 3\n",
             "vllm-decode-1": "vllm:num_requests_waiting 3\n"}
        c = self.mon._check_queue_balance([PREFILL], [DECODE_1, DECODE_2], m)
        self.assertTrue(c.passed)

    def test_imbalanced_3_pods(self):
        decode_3 = _pod("vllm-decode-2", ip="10.0.0.4",
                        args='["--kv-transfer-config", \'{"kv_role":"kv_consumer"}\']')
        m = {"vllm-decode-0": "vllm:num_requests_waiting 30\n",
             "vllm-decode-1": "vllm:num_requests_waiting 1\n",
             "vllm-decode-2": "vllm:num_requests_waiting 1\n"}
        c = self.mon._check_queue_balance([PREFILL], [DECODE_1, DECODE_2, decode_3], m)
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "warning")

    def test_imbalanced_2_pods(self):
        """B5 fix: imbalance detection must work with just 2 decode pods."""
        m = {"vllm-decode-0": "vllm:num_requests_waiting 20\n",
             "vllm-decode-1": "vllm:num_requests_waiting 0\n"}
        c = self.mon._check_queue_balance([PREFILL], [DECODE_1, DECODE_2], m)
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "warning")

    def test_low_queues_not_flagged(self):
        """Low absolute values shouldn't trigger even if ratio is high."""
        m = {"vllm-decode-0": "vllm:num_requests_waiting 3\n",
             "vllm-decode-1": "vllm:num_requests_waiting 0\n"}
        c = self.mon._check_queue_balance([PREFILL], [DECODE_1, DECODE_2], m)
        self.assertTrue(c.passed)

    def test_single_decode_skipped(self):
        c = self.mon._check_queue_balance([PREFILL], [DECODE_1], {})
        self.assertTrue(c.passed)


# ── KV expiration (delta tracking, _total suffix) ───────────────────────

class TestCheckKvExpiration(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    def test_no_expirations(self):
        m = {"vllm-decode-0": "vllm:nixl_num_kv_expired_reqs_total 0\n"}
        c = self.mon._check_kv_expiration([DECODE_1], m)
        self.assertTrue(c.passed)

    def test_new_expirations(self):
        m = {"vllm-decode-0": "vllm:nixl_num_kv_expired_reqs_total 7\n"}
        self.mon._prev_kv_expired["vllm-decode-0"] = 3
        c = self.mon._check_kv_expiration([DECODE_1], m)
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "critical")
        self.assertIn("4", c.detail)
        self.assertIn("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", c.detail)

    def test_no_increase_passes(self):
        m = {"vllm-decode-0": "vllm:nixl_num_kv_expired_reqs_total 5\n"}
        self.mon._prev_kv_expired["vllm-decode-0"] = 5
        c = self.mon._check_kv_expiration([DECODE_1], m)
        self.assertTrue(c.passed)

    def test_total_suffix_required(self):
        """Metric without _total suffix should not match."""
        m = {"vllm-decode-0": "vllm:nixl_num_kv_expired_reqs 10\n"}
        c = self.mon._check_kv_expiration([DECODE_1], m)
        self.assertTrue(c.passed)

    def test_created_metric_ignored(self):
        """_created epoch timestamp must not be parsed as expired count."""
        m = {"vllm-decode-0": (
            "vllm:nixl_num_kv_expired_reqs_total 0\n"
            "vllm:nixl_num_kv_expired_reqs_created 1.78e+09\n"
        )}
        c = self.mon._check_kv_expiration([DECODE_1], m)
        self.assertTrue(c.passed)


# ── Transfer duration (trend detection) ─────────────────────────────────

class TestCheckTransferDuration(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    def test_stable(self):
        m = {"vllm-decode-0": "nixl_transfer_duration_seconds_sum 0.1\n"}
        self.mon._transfer_duration_history["vllm-decode-0"] = [0.1, 0.1, 0.1, 0.1]
        c = self.mon._check_transfer_duration([DECODE_1], m)
        self.assertTrue(c.passed)

    def test_trending_up(self):
        m = {"vllm-decode-0": "nixl_transfer_duration_seconds_sum 5.0\n"}
        self.mon._transfer_duration_history["vllm-decode-0"] = [0.1, 0.1, 0.1, 0.1]
        c = self.mon._check_transfer_duration([DECODE_1], m)
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "warning")

    def test_insufficient_history_passes(self):
        m = {"vllm-decode-0": "nixl_transfer_duration_seconds_sum 5.0\n"}
        c = self.mon._check_transfer_duration([DECODE_1], m)
        self.assertTrue(c.passed)

    def test_history_capped_at_20(self):
        m = {"vllm-decode-0": "nixl_transfer_duration_seconds_sum 0.1\n"}
        self.mon._transfer_duration_history["vllm-decode-0"] = [0.1] * 20
        self.mon._check_transfer_duration([DECODE_1], m)
        self.assertEqual(len(self.mon._transfer_duration_history["vllm-decode-0"]), 20)


# ── Synthetic probe ─────────────────────────────────────────────────────

class TestCheckProbe(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    def test_healthy_probe(self):
        c = self.mon._check_probe(_healthy_probe())
        self.assertTrue(c.passed)

    def test_ttft_spike(self):
        c = self.mon._check_probe(_unhealthy_probe(ttft=200.0, baseline=45.0))
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "warning")
        self.assertIn("200", c.detail)

    def test_error_probe(self):
        c = self.mon._check_probe(_unhealthy_probe(status=500, error="connection refused"))
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "critical")
        self.assertIn("connection refused", c.detail)


# ── NIXL config ──────────────────────────────────────────────────────────

class TestCheckNixlConfig(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    @patch("health.oc_safe")
    def test_configured(self, mock_oc):
        mock_oc.return_value = ('VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.1', "")
        c = self.mon._check_nixl_config([PREFILL])
        self.assertTrue(c.passed)

    @patch("health.oc_safe")
    def test_missing_env(self, mock_oc):
        mock_oc.return_value = ("", "")
        c = self.mon._check_nixl_config([PREFILL])
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "warning")
        self.assertIn("VLLM_NIXL_SIDE_CHANNEL_HOST", c.detail)

    def test_no_prefill_pods(self):
        c = self.mon._check_nixl_config([])
        self.assertFalse(c.passed)
        self.assertEqual(c.severity, "warning")


# ── Overall rollup (_check_all) ──────────────────────────────────────────

class TestCheckAll(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    @patch("health.scrape_pod_metrics")
    @patch("health.get_pods")
    @patch("health.oc_safe")
    def test_healthy_cluster(self, mock_oc, mock_get_pods, mock_scrape):
        mock_get_pods.return_value = [PREFILL, DECODE_1]
        mock_scrape.return_value = (
            "vllm:nixl_num_failed_transfers 0\n"
            "vllm:kv_cache_usage_perc 0.4\n"
            "vllm:nixl_num_kv_expired_reqs_total 0\n"
        )
        mock_oc.return_value = ('VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.1', "")
        self.mon.prober.probe = MagicMock(return_value=_healthy_probe())

        snap = self.mon._check_all()
        self.assertEqual(snap.overall, "HEALTHY")
        self.assertEqual(len(snap.checks), 10)

    @patch("health.scrape_pod_metrics")
    @patch("health.get_pods")
    @patch("health.oc_safe")
    def test_unhealthy_on_critical(self, mock_oc, mock_get_pods, mock_scrape):
        mock_get_pods.return_value = [DECODE_1]  # no prefill
        mock_scrape.return_value = ""
        mock_oc.return_value = ("", "")
        self.mon.prober.probe = MagicMock(return_value=_healthy_probe())

        snap = self.mon._check_all()
        self.assertEqual(snap.overall, "UNHEALTHY")

    @patch("health.scrape_pod_metrics")
    @patch("health.get_pods")
    @patch("health.oc_safe")
    def test_degraded_on_warning(self, mock_oc, mock_get_pods, mock_scrape):
        old_decode = _pod("vllm-decode-old", image="vllm/vllm-openai:v0.17.0",
                          args='["--kv-transfer-config", \'{"kv_role":"kv_consumer"}\']')
        mock_get_pods.return_value = [PREFILL, old_decode]
        mock_scrape.return_value = (
            "vllm:nixl_num_failed_transfers 0\n"
            "vllm:kv_cache_usage_perc 0.4\n"
            "vllm:nixl_num_kv_expired_reqs_total 0\n"
        )
        mock_oc.return_value = ('VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.1', "")
        self.mon.prober.probe = MagicMock(return_value=_healthy_probe())

        snap = self.mon._check_all()
        self.assertEqual(snap.overall, "DEGRADED")


# ── Duration loop ────────────────────────────────────────────────────────

class TestRunLoop(unittest.TestCase):

    @patch("health.scrape_pod_metrics")
    @patch("health.get_pods")
    @patch("health.oc_safe")
    @patch("health.time.sleep")
    def test_at_least_one_snapshot(self, mock_sleep, mock_oc, mock_get_pods, mock_scrape):
        """Bug 4: loop must produce at least one snapshot before exiting."""
        mock_get_pods.return_value = [PREFILL, DECODE_1]
        mock_scrape.return_value = (
            "vllm:nixl_num_failed_transfers 0\n"
            "vllm:kv_cache_usage_perc 0.4\n"
            "vllm:nixl_num_kv_expired_reqs_total 0\n"
        )
        mock_oc.return_value = ('VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.1', "")

        mon = _make_monitor()
        mon.prober.warmup = MagicMock()
        mon.prober._baseline_ms = 50.0
        mon.prober.probe = MagicMock(return_value=_healthy_probe())

        mon.run(duration_s=0.001)
        self.assertGreaterEqual(len(mon.snapshots), 1)


# ── Print functions ──────────────────────────────────────────────────────

class TestPrintFunctions(unittest.TestCase):

    def setUp(self):
        self.mon = _make_monitor()

    def test_print_snapshot_healthy(self):
        snap = HealthSnapshot(timestamp=time.time(), overall="HEALTHY",
                              checks=[HealthCheck("test", True)],
                              probe=_healthy_probe())
        self.mon._print_snapshot(snap)

    def test_print_snapshot_unhealthy(self):
        snap = HealthSnapshot(timestamp=time.time(), overall="UNHEALTHY",
                              checks=[HealthCheck("test", False, "bad", "critical")],
                              probe=_unhealthy_probe(status=500, error="fail"))
        self.mon._print_snapshot(snap)

    def test_print_summary_no_snapshots(self):
        self.mon._print_summary()

    def test_print_summary_with_snapshots(self):
        self.mon.snapshots = [
            HealthSnapshot(time.time(), overall="HEALTHY"),
            HealthSnapshot(time.time(), overall="DEGRADED"),
            HealthSnapshot(time.time(), overall="UNHEALTHY"),
        ]
        self.mon._print_summary()


if __name__ == "__main__":
    unittest.main()
