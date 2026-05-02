"""Tests for advisor/probe.py — synthetic request prober with mocked HTTP.

Tests the DisaggProber's baseline tracking, health evaluation, and
anomaly detection logic without making real HTTP requests.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "toolkit"))

from probe import DisaggProber, ProbeResult


def _mock_result(ttft_ms=100.0, status=200, error=""):
    """Create a mock RequestResult matching toolkit/client.py's interface."""
    r = MagicMock()
    r.ttft_ms = ttft_ms
    r.status = status
    r.error = error
    return r


class TestProbeResult(unittest.TestCase):

    def test_fields(self):
        pr = ProbeResult(
            timestamp=1000.0, ttft_ms=50.0, status=200,
            healthy=True, baseline_ms=45.0, ratio=1.11,
        )
        self.assertTrue(pr.healthy)
        self.assertEqual(pr.error, "")


class TestDisaggProberHealth(unittest.TestCase):

    @patch("probe.send_streaming")
    def test_healthy_when_ttft_near_baseline(self, mock_send):
        mock_send.return_value = _mock_result(ttft_ms=100.0, status=200)
        prober = DisaggProber("ns", "model")

        for _ in range(5):
            result = prober.probe()

        self.assertTrue(result.healthy)
        self.assertEqual(result.status, 200)
        self.assertAlmostEqual(result.ratio, 1.0, places=1)

    @patch("probe.send_streaming")
    def test_unhealthy_when_ttft_spikes(self, mock_send):
        mock_send.return_value = _mock_result(ttft_ms=100.0, status=200)
        prober = DisaggProber("ns", "model")

        for _ in range(5):
            prober.probe()

        mock_send.return_value = _mock_result(ttft_ms=500.0, status=200)
        result = prober.probe()
        self.assertFalse(result.healthy)
        self.assertGreater(result.ratio, 3.0)

    @patch("probe.send_streaming")
    def test_unhealthy_on_error(self, mock_send):
        mock_send.side_effect = ConnectionError("refused")
        prober = DisaggProber("ns", "model")
        result = prober.probe()

        self.assertFalse(result.healthy)
        self.assertEqual(result.status, 0)
        self.assertIn("refused", result.error)

    @patch("probe.send_streaming")
    def test_unhealthy_on_non_200(self, mock_send):
        mock_send.return_value = _mock_result(ttft_ms=100.0, status=503)
        prober = DisaggProber("ns", "model")
        result = prober.probe()

        self.assertFalse(result.healthy)
        self.assertEqual(result.status, 503)


class TestDisaggProberBaseline(unittest.TestCase):

    @patch("probe.send_streaming")
    def test_baseline_established_after_3_probes(self, mock_send):
        mock_send.return_value = _mock_result(ttft_ms=100.0, status=200)
        prober = DisaggProber("ns", "model")

        self.assertEqual(prober._baseline_ms, 0.0)
        prober.probe()
        prober.probe()
        self.assertEqual(prober._baseline_ms, 0.0)  # not yet established
        prober.probe()
        self.assertGreater(prober._baseline_ms, 0)

    @patch("probe.send_streaming")
    def test_baseline_window_size(self, mock_send):
        mock_send.return_value = _mock_result(ttft_ms=100.0, status=200)
        prober = DisaggProber("ns", "model", window_size=5)

        for _ in range(10):
            prober.probe()

        self.assertEqual(len(prober._baseline_window), 5)

    @patch("probe.send_streaming")
    def test_failed_probes_dont_update_baseline(self, mock_send):
        mock_send.return_value = _mock_result(ttft_ms=100.0, status=200)
        prober = DisaggProber("ns", "model")
        for _ in range(5):
            prober.probe()

        baseline_before = prober._baseline_ms
        mock_send.return_value = _mock_result(ttft_ms=0, status=500)
        prober.probe()

        self.assertEqual(prober._baseline_ms, baseline_before)


class TestDisaggProberWarmup(unittest.TestCase):

    @patch("probe.send_streaming")
    def test_warmup_sends_n_probes(self, mock_send):
        mock_send.return_value = _mock_result(ttft_ms=80.0, status=200)
        prober = DisaggProber("ns", "model")
        prober.warmup(n=7)
        self.assertEqual(mock_send.call_count, 7)


if __name__ == "__main__":
    unittest.main()
