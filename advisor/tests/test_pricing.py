"""Tests for advisor/pricing.py — pricing API with live fetch + static fallback."""

import json
import unittest
from unittest.mock import MagicMock, patch

from pricing import (
    _STATIC_PRICING,
    GPU_NIC_BW_GBPS,
    GPU_TFLOPS_FP16,
    GPU_VRAM_GB,
    PRICING,
    cost_per_1k_requests,
    get_cheapest,
    get_price,
    pricing_source,
    refresh_pricing,
)


class TestGetPrice(unittest.TestCase):

    def test_known_provider_and_gpu(self):
        price = get_price("aws", "t4")
        self.assertEqual(price, 0.53)

    def test_unknown_provider(self):
        self.assertEqual(get_price("nonexistent", "t4"), 0.0)

    def test_unknown_gpu(self):
        self.assertEqual(get_price("aws", "nonexistent"), 0.0)

    def test_case_insensitive(self):
        self.assertEqual(get_price("AWS", "T4"), get_price("aws", "t4"))


class TestGetCheapest(unittest.TestCase):

    def test_t4_cheapest(self):
        provider, price = get_cheapest("t4")
        self.assertGreater(price, 0)
        self.assertLess(price, 1.0)
        self.assertIn(provider, PRICING)

    def test_unknown_gpu_returns_sentinel(self):
        provider, price = get_cheapest("nonexistent_gpu_xyz")
        self.assertEqual(provider, "")
        self.assertEqual(price, 999.0)

    def test_h100_has_cheapest(self):
        provider, price = get_cheapest("h100")
        self.assertGreater(price, 0)
        self.assertNotEqual(provider, "")


class TestCostPer1kRequests(unittest.TestCase):

    def test_normal_case(self):
        cost = cost_per_1k_requests(1.0, 1, 1.0)
        self.assertGreater(cost, 0)
        self.assertLess(cost, 1.0)

    def test_zero_throughput(self):
        self.assertEqual(cost_per_1k_requests(0.0, 1, 1.0), float('inf'))

    def test_scaling(self):
        cost_1gpu = cost_per_1k_requests(1.0, 1, 1.0)
        cost_2gpu = cost_per_1k_requests(1.0, 2, 1.0)
        self.assertAlmostEqual(cost_2gpu, cost_1gpu * 2, places=3)


class TestPricingTables(unittest.TestCase):

    def test_all_providers_have_at_least_one_gpu(self):
        for provider, gpus in PRICING.items():
            self.assertGreater(len(gpus), 0, f"{provider} has no GPUs")

    def test_vram_table_completeness(self):
        for gpu in ["t4", "h100", "a100_80"]:
            self.assertIn(gpu, GPU_VRAM_GB)

    def test_tflops_table_completeness(self):
        for gpu in ["t4", "h100", "a100_80"]:
            self.assertIn(gpu, GPU_TFLOPS_FP16)

    def test_nic_bw_table_completeness(self):
        for gpu in ["t4", "h100", "a100_80", "h200"]:
            self.assertIn(gpu, GPU_NIC_BW_GBPS)
            self.assertGreater(GPU_NIC_BW_GBPS[gpu], 0)

    def test_nic_bw_increases_with_tier(self):
        self.assertLess(GPU_NIC_BW_GBPS["t4"], GPU_NIC_BW_GBPS["h100"])

    def test_static_matches_initial(self):
        for provider, gpus in _STATIC_PRICING.items():
            for gpu, price in gpus.items():
                self.assertEqual(PRICING[provider][gpu], price)


class TestRefreshPricing(unittest.TestCase):

    @patch("pricing.urllib.request.urlopen")
    def test_successful_refresh(self, mock_urlopen):
        live_data = [
            {"gpu": "NVIDIA T4", "provider": "Amazon Web Services", "price": 0.42},
            {"gpu": "NVIDIA H100", "provider": "Lambda Labs", "price": 1.79},
        ]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(live_data).encode()
        mock_urlopen.return_value = mock_resp

        import pricing as _pm
        old_source = _pm._pricing_source

        ok = refresh_pricing()
        self.assertTrue(ok)
        self.assertEqual(PRICING["aws"]["t4"], 0.42)
        self.assertEqual(PRICING["lambda"]["h100"], 1.79)
        self.assertIn("live", pricing_source())

        PRICING["aws"]["t4"] = _STATIC_PRICING["aws"]["t4"]
        PRICING["lambda"]["h100"] = _STATIC_PRICING["lambda"]["h100"]
        _pm._pricing_source = old_source

    @patch("pricing.urllib.request.urlopen")
    def test_failed_fetch_keeps_static(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionError("no internet")
        original_t4 = PRICING["aws"]["t4"]

        ok = refresh_pricing()
        self.assertFalse(ok)
        self.assertEqual(PRICING["aws"]["t4"], original_t4)

    @patch("pricing.urllib.request.urlopen")
    def test_empty_response_keeps_static(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"[]"
        mock_urlopen.return_value = mock_resp

        ok = refresh_pricing()
        self.assertFalse(ok)

    @patch("pricing.urllib.request.urlopen")
    def test_bad_json_keeps_static(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_urlopen.return_value = mock_resp

        ok = refresh_pricing()
        self.assertFalse(ok)

    @patch("pricing.urllib.request.urlopen")
    def test_unknown_gpu_ignored(self, mock_urlopen):
        live_data = [
            {"gpu": "NVIDIA Z9000", "provider": "AWS", "price": 99.99},
        ]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(live_data).encode()
        mock_urlopen.return_value = mock_resp

        ok = refresh_pricing()
        self.assertFalse(ok)


class TestPricingSource(unittest.TestCase):

    def test_default_is_static(self):
        self.assertIn("static", pricing_source())


if __name__ == "__main__":
    unittest.main()
