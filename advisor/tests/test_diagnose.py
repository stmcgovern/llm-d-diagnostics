"""Tests for advisor/diagnose.py — diagnostic checks with mocked cluster data.

Every _check_* function takes pod dicts as input, so we can test all
decision logic without a real cluster.  HTTP metrics scraping is mocked
at the http.client level.
"""

import unittest
from unittest.mock import patch, MagicMock

from diagnose import (
    Issue,
    _check_image_mismatch,
    _check_kv_cache_pressure,
    _check_kv_expiration,
    _check_kv_roles,
    _check_nixl_config,
    _check_nixl_failures,
    _check_pod_health,
    _check_prefill_spof,
    _check_stale_kv_timeout,
    _check_transfer_duration,
    diagnose,
    print_diagnosis,
)


# ── Test fixtures ────────────────────────────────────────────────────────

def _pod(name, app_label, ready=True, image="vllm/vllm-openai:v0.18.1",
         ip="10.0.0.1", args="", role=None):
    labels = {"app": app_label}
    return {
        "name": name, "ip": ip, "ready": ready,
        "labels": labels,
        "role": role if role is not None else app_label,
        "image": image,
        "args": str(args),
    }


PREFILL = _pod("vllm-prefill-0", "vllm-prefill", ip="10.0.0.1",
               args='["--kv-transfer-config", \'{"kv_role":"kv_producer"}\']')
DECODE_1 = _pod("vllm-decode-0", "vllm-decode", ip="10.0.0.2",
                args='["--kv-transfer-config", \'{"kv_role":"kv_consumer"}\']')
DECODE_2 = _pod("vllm-decode-1", "vllm-decode", ip="10.0.0.3",
                args='["--kv-transfer-config", \'{"kv_role":"kv_consumer"}\']')


# ── Pod health ───────────────────────────────────────────────────────────

class TestCheckPodHealth(unittest.TestCase):

    def test_all_healthy(self):
        issues = _check_pod_health([PREFILL], [DECODE_1, DECODE_2], "ns")
        self.assertEqual(len(issues), 0)

    def test_not_ready_pod(self):
        bad = _pod("vllm-decode-bad", "vllm-decode", ready=False)
        issues = _check_pod_health([PREFILL], [bad], "ns")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "critical")
        self.assertIn("not ready", issues[0].title.lower())

    def test_no_prefill(self):
        issues = _check_pod_health([], [DECODE_1], "ns")
        self.assertTrue(any("prefill" in i.title.lower() for i in issues))

    def test_no_decode(self):
        issues = _check_pod_health([PREFILL], [], "ns")
        self.assertTrue(any("decode" in i.title.lower() for i in issues))

    def test_multiple_not_ready(self):
        bad1 = _pod("p-bad", "vllm-prefill", ready=False)
        bad2 = _pod("d-bad", "vllm-decode", ready=False)
        issues = _check_pod_health([bad1], [bad2], "ns")
        not_ready_issues = [i for i in issues if "not ready" in i.title.lower()]
        self.assertEqual(len(not_ready_issues), 2)


# ── Image mismatch ───────────────────────────────────────────────────────

class TestCheckImageMismatch(unittest.TestCase):

    def test_matching_images(self):
        pods = [PREFILL, DECODE_1]
        issues = _check_image_mismatch(pods, "ns")
        self.assertEqual(len(issues), 0)

    def test_mismatched_images(self):
        old = _pod("vllm-decode-old", "vllm-decode", image="vllm/vllm-openai:v0.17.0")
        pods = [PREFILL, old]
        issues = _check_image_mismatch(pods, "ns")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "critical")
        self.assertTrue(issues[0].auto_fixable)


# ── Prefill SPOF ─────────────────────────────────────────────────────────

class TestCheckPrefillSpof(unittest.TestCase):

    def test_single_prefill_warns(self):
        issues = _check_prefill_spof([PREFILL], "ns")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "warning")
        self.assertIn("SPOF", issues[0].title)

    def test_two_prefill_ok(self):
        p2 = _pod("vllm-prefill-1", "vllm-prefill")
        issues = _check_prefill_spof([PREFILL, p2], "ns")
        self.assertEqual(len(issues), 0)

    def test_zero_prefill_ok(self):
        issues = _check_prefill_spof([], "ns")
        self.assertEqual(len(issues), 0)


# ── KV roles ─────────────────────────────────────────────────────────────

class TestCheckKvRoles(unittest.TestCase):

    def test_producer_and_consumer(self):
        issues = _check_kv_roles([PREFILL, DECODE_1], "ns")
        self.assertEqual(len(issues), 0)

    def test_no_producer(self):
        no_role = _pod("p", "vllm-prefill", args="[]")
        d_no_role = _pod("d", "vllm-decode", args="[]")
        issues = _check_kv_roles([no_role, d_no_role], "ns")
        self.assertEqual(len(issues), 1)
        self.assertIn("producer", issues[0].title.lower())

    def test_kv_both_counts(self):
        both = _pod("p", "vllm-prefill", args='["--kv-transfer-config", \'{"kv_role":"kv_both"}\']')
        issues = _check_kv_roles([both], "ns")
        self.assertEqual(len(issues), 0)


# ── NIXL failures (metrics scraping) ─────────────────────────────────────

class TestCheckNixlFailures(unittest.TestCase):

    @patch("diagnose.scrape_pod_metrics")
    def test_no_failures(self, mock_scrape):
        mock_scrape.return_value = "vllm:nixl_num_failed_transfers 0\n"
        issues = _check_nixl_failures([PREFILL, DECODE_1], "ns")
        self.assertEqual(len(issues), 0)

    @patch("diagnose.scrape_pod_metrics")
    def test_failures_detected(self, mock_scrape):
        mock_scrape.return_value = "vllm:nixl_num_failed_transfers 5\n"
        issues = _check_nixl_failures([DECODE_1], "ns")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "critical")
        self.assertIn("5", issues[0].evidence)

    @patch("diagnose.scrape_pod_metrics")
    def test_empty_metrics(self, mock_scrape):
        mock_scrape.return_value = ""
        issues = _check_nixl_failures([DECODE_1], "ns")
        self.assertEqual(len(issues), 0)


# ── KV cache pressure ───────────────────────────────────────────────────

class TestCheckKvCachePressure(unittest.TestCase):

    @patch("diagnose.scrape_pod_metrics")
    def test_normal_usage(self, mock_scrape):
        mock_scrape.return_value = "vllm:kv_cache_usage_perc 0.45\n"
        issues = _check_kv_cache_pressure([DECODE_1], "test-ns")
        self.assertEqual(len(issues), 0)

    @patch("diagnose.scrape_pod_metrics")
    def test_high_pressure(self, mock_scrape):
        mock_scrape.return_value = "vllm:kv_cache_usage_perc 0.95\n"
        issues = _check_kv_cache_pressure([DECODE_1], "test-ns")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "warning")
        self.assertIn("95%", issues[0].evidence)

    @patch("diagnose.scrape_pod_metrics")
    def test_fix_uses_correct_namespace(self, mock_scrape):
        """Regression: fix command must use the ns parameter, not $(oc project -q)."""
        mock_scrape.return_value = "vllm:kv_cache_usage_perc 0.95\n"
        pod = _pod("vllm-prefill-0", "vllm-prefill", ip="10.0.0.1")
        issues = _check_kv_cache_pressure([pod], "my-namespace")
        self.assertEqual(len(issues), 1)
        self.assertNotIn("$(oc project", issues[0].fix)
        self.assertIn("my-namespace", issues[0].fix)


# ── Transfer duration ────────────────────────────────────────────────────

class TestCheckTransferDuration(unittest.TestCase):

    @patch("diagnose.scrape_pod_metrics")
    def test_normal_duration(self, mock_scrape):
        mock_scrape.return_value = (
            "nixl_transfer_duration_seconds_sum 0.1\n"
            "nixl_transfer_duration_seconds_count 100\n"
        )
        issues = _check_transfer_duration([DECODE_1], "test-ns")
        self.assertEqual(len(issues), 0)

    @patch("diagnose.scrape_pod_metrics")
    def test_high_duration(self, mock_scrape):
        mock_scrape.return_value = (
            "nixl_transfer_duration_seconds_sum 100\n"
            "nixl_transfer_duration_seconds_count 100\n"
        )
        issues = _check_transfer_duration([DECODE_1], "test-ns")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "warning")

    @patch("diagnose.scrape_pod_metrics")
    def test_fix_includes_namespace(self, mock_scrape):
        mock_scrape.return_value = (
            "nixl_transfer_duration_seconds_sum 100\n"
            "nixl_transfer_duration_seconds_count 100\n"
        )
        issues = _check_transfer_duration([DECODE_1], "prod-ns")
        self.assertIn("prod-ns", issues[0].fix)


# ── KV expiration (_total suffix fix, Bug 1) ────────────────────────────

class TestCheckKvExpiration(unittest.TestCase):

    @patch("diagnose.scrape_pod_metrics")
    def test_no_expirations(self, mock_scrape):
        mock_scrape.return_value = "vllm:nixl_num_kv_expired_reqs_total 0\n"
        issues = _check_kv_expiration([DECODE_1], "ns")
        self.assertEqual(len(issues), 0)

    @patch("diagnose.scrape_pod_metrics")
    def test_expirations_detected(self, mock_scrape):
        mock_scrape.return_value = "vllm:nixl_num_kv_expired_reqs_total 5\n"
        issues = _check_kv_expiration([DECODE_1], "ns")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "critical")
        self.assertIn("5", issues[0].evidence)

    @patch("diagnose.scrape_pod_metrics")
    def test_total_suffix_required(self, mock_scrape):
        """Bug 1: metric without _total suffix must not match."""
        mock_scrape.return_value = "vllm:nixl_num_kv_expired_reqs 10\n"
        issues = _check_kv_expiration([DECODE_1], "ns")
        self.assertEqual(len(issues), 0)

    @patch("diagnose.scrape_pod_metrics")
    def test_created_epoch_ignored(self, mock_scrape):
        """_created is an epoch timestamp (~1.78e9), must not trigger."""
        mock_scrape.return_value = (
            "vllm:nixl_num_kv_expired_reqs_total 0\n"
            "vllm:nixl_num_kv_expired_reqs_created 1.78e+09\n"
        )
        issues = _check_kv_expiration([DECODE_1], "ns")
        self.assertEqual(len(issues), 0)


# ── Stale KV timeout ────────────────────────────────────────────────────

class TestCheckStaleKvTimeout(unittest.TestCase):

    @patch("diagnose.oc_safe")
    def test_missing_env_no_issue(self, mock_oc):
        mock_oc.return_value = ("", "")
        issues = _check_stale_kv_timeout([PREFILL], "ns")
        self.assertEqual(len(issues), 0)

    @patch("diagnose.oc_safe")
    def test_high_timeout_warns(self, mock_oc):
        mock_oc.return_value = ("480\n", "")
        issues = _check_stale_kv_timeout([PREFILL], "ns")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "info")
        self.assertIn("480", issues[0].evidence)

    @patch("diagnose.oc_safe")
    def test_reasonable_timeout_ok(self, mock_oc):
        mock_oc.return_value = ("120\n", "")
        issues = _check_stale_kv_timeout([PREFILL], "ns")
        self.assertEqual(len(issues), 0)


# ── NIXL config ──────────────────────────────────────────────────────────

class TestCheckNixlConfig(unittest.TestCase):

    @patch("diagnose.oc_safe")
    def test_configured(self, mock_oc):
        mock_oc.return_value = ('VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.1', "")
        issues = _check_nixl_config([PREFILL], "ns")
        self.assertEqual(len(issues), 0)

    @patch("diagnose.oc_safe")
    def test_missing(self, mock_oc):
        mock_oc.return_value = ("SOME_OTHER_ENV=yes", "")
        issues = _check_nixl_config([PREFILL], "ns")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "warning")

    def test_no_prefill_no_issue(self):
        issues = _check_nixl_config([], "ns")
        self.assertEqual(len(issues), 0)


# ── Orchestration ────────────────────────────────────────────────────────

class TestDiagnoseOrchestration(unittest.TestCase):

    @patch("diagnose.oc_safe")
    @patch("diagnose.scrape_pod_metrics")
    @patch("diagnose.get_pods")
    def test_all_checks_run(self, mock_get_pods, mock_scrape, mock_oc):
        mock_get_pods.return_value = [PREFILL, DECODE_1]
        mock_scrape.return_value = (
            "vllm:nixl_num_failed_transfers 0\n"
            "vllm:kv_cache_usage_perc 0.4\n"
            "vllm:nixl_num_kv_expired_reqs_total 0\n"
        )
        mock_oc.return_value = ('VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.1', "")
        issues = diagnose("test-ns")
        critical = [i for i in issues if i.severity == "critical"]
        self.assertEqual(len(critical), 0, f"Unexpected criticals: {[i.title for i in critical]}")

    @patch("diagnose.get_pods")
    def test_no_pods_returns_critical(self, mock_get_pods):
        mock_get_pods.return_value = []
        issues = diagnose("empty-ns")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "critical")
        self.assertIn("No P/D", issues[0].title)


# ── Print diagnosis ──────────────────────────────────────────────────────

class TestPrintDiagnosis(unittest.TestCase):

    def test_no_issues(self):
        print_diagnosis([])

    def test_with_issues(self):
        issues = [
            Issue("critical", "Test", "ev", "cause", "http://ref", "fix cmd"),
            Issue("warning", "Warn", "ev2", "", "", "fix2"),
            Issue("info", "Info", "ev3", "cause3", "", "fix3"),
        ]
        print_diagnosis(issues)


if __name__ == "__main__":
    unittest.main()
