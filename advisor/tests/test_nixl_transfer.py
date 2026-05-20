"""Tests for advisor/nixl_transfer.py — NIXL transfer time regression."""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_toolkit = os.path.join(os.path.dirname(__file__), "..", "..", "toolkit")
sys.path.insert(0, _toolkit)

from nixl_transfer import _linreg, _stats, analyze_nixl_transfer, load_exp5b  # noqa: E402

from schemas import Exp5bRow, fields_for  # noqa: E402


def _write_exp5b_csv(path, rows):
    fieldnames = fields_for(Exp5bRow)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _make_row(**overrides):
    row = {f: "0" for f in fields_for(Exp5bRow)}
    row["experiment"] = "exp5b"
    row.update(overrides)
    return row


class TestLinreg(unittest.TestCase):

    def test_perfect_linear(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [5.0, 7.0, 9.0, 11.0]
        intercept, slope, r_sq = _linreg(xs, ys)
        self.assertAlmostEqual(slope, 2.0, places=5)
        self.assertAlmostEqual(intercept, 3.0, places=5)
        self.assertAlmostEqual(r_sq, 1.0, places=5)

    def test_fewer_than_two(self):
        self.assertEqual(_linreg([1], [2]), (0, 0, 0))

    def test_constant_x(self):
        _, slope, _ = _linreg([3, 3, 3], [1, 2, 3])
        self.assertEqual(slope, 0)


class TestStats(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(_stats([]), (0, 0))

    def test_single(self):
        self.assertEqual(_stats([42.0]), (42.0, 0))

    def test_mean(self):
        mean, _ = _stats([10.0, 20.0, 30.0])
        self.assertAlmostEqual(mean, 20.0)


class TestLoadExp5b(unittest.TestCase):

    def test_valid_rows(self):
        d = tempfile.mkdtemp()
        rows = []
        for i in range(5):
            rows.append(_make_row(
                status_code="200",
                prompt_tokens_target="100",
                prompt_tokens_actual="103",
                ttft_ms=str(500 + i),
                total_ms=str(510 + i),
                run=str(i + 1),
                nixl_bytes_delta=str(6291456),
                nixl_xfer_time_delta_ms=str(22.4 + i),
                nixl_transfers_delta="1",
            ))
        _write_exp5b_csv(os.path.join(d, "exp5b-results.csv"), rows)
        result = load_exp5b(d)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0]["nixl_bytes"], 6291456.0)
        self.assertEqual(result[0]["nixl_transfers"], 1)

    def test_non_200_filtered(self):
        d = tempfile.mkdtemp()
        rows = [
            _make_row(status_code="500", prompt_tokens_target="100",
                      prompt_tokens_actual="103", ttft_ms="500",
                      nixl_bytes_delta="6291456",
                      nixl_xfer_time_delta_ms="22.4",
                      nixl_transfers_delta="1"),
        ]
        _write_exp5b_csv(os.path.join(d, "exp5b-results.csv"), rows)
        result = load_exp5b(d)
        self.assertEqual(len(result), 0)

    def test_missing_nixl_columns_filtered(self):
        d = tempfile.mkdtemp()
        rows = [
            _make_row(status_code="200", prompt_tokens_target="100",
                      prompt_tokens_actual="103", ttft_ms="500",
                      nixl_bytes_delta="", nixl_xfer_time_delta_ms="",
                      nixl_transfers_delta=""),
        ]
        _write_exp5b_csv(os.path.join(d, "exp5b-results.csv"), rows)
        result = load_exp5b(d)
        self.assertEqual(len(result), 0)

    def test_missing_file(self):
        d = tempfile.mkdtemp()
        result = load_exp5b(d)
        self.assertIsNone(result)


class TestAnalyzeNixlTransfer(unittest.TestCase):

    def _make_linear_data(self, d, n=10, transfers=1):
        """Create exp5b CSV with known linear relationship: time = 5 + bytes/1e6."""
        rows = []
        for i in range(n):
            nbytes = 1_000_000 * (i + 1)
            time_ms = 5.0 + nbytes / 1e6
            rows.append(_make_row(
                status_code="200",
                prompt_tokens_target=str(100 * (i + 1)),
                prompt_tokens_actual=str(100 * (i + 1) + 3),
                ttft_ms=str(time_ms + 100),
                total_ms=str(time_ms + 200),
                run=str(i + 1),
                nixl_bytes_delta=str(nbytes),
                nixl_xfer_time_delta_ms=str(time_ms),
                nixl_transfers_delta=str(transfers),
            ))
        _write_exp5b_csv(os.path.join(d, "exp5b-results.csv"), rows)

    def test_known_linear(self):
        d = tempfile.mkdtemp()
        self._make_linear_data(d)
        result = analyze_nixl_transfer(d)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["protocol_ms"], 5.0, places=1)
        self.assertAlmostEqual(result["eff_bw_gbs"], 1.0, places=2)
        self.assertAlmostEqual(result["slope_ms_per_byte"], 1e-6, places=10)
        self.assertGreater(result["r_sq"], 0.99)
        self.assertEqual(result["n_points"], 10)

    def test_no_single_transfers_fatal(self):
        d = tempfile.mkdtemp()
        self._make_linear_data(d, n=5, transfers=2)
        result = analyze_nixl_transfer(d)
        self.assertIsNotNone(result)
        self.assertIn("fatal", result["gate_checks"])

    def test_mixed_transfers_uses_only_singles(self):
        d = tempfile.mkdtemp()
        rows = []
        for i in range(5):
            nbytes = 1_000_000 * (i + 1)
            time_ms = 5.0 + nbytes / 1e6
            rows.append(_make_row(
                status_code="200",
                prompt_tokens_target=str(100 * (i + 1)),
                prompt_tokens_actual=str(100 * (i + 1) + 3),
                ttft_ms=str(time_ms + 100),
                total_ms=str(time_ms + 200),
                run=str(i + 1),
                nixl_bytes_delta=str(nbytes),
                nixl_xfer_time_delta_ms=str(time_ms),
                nixl_transfers_delta="1",
            ))
        for i in range(3):
            rows.append(_make_row(
                status_code="200",
                prompt_tokens_target="100",
                prompt_tokens_actual="103",
                ttft_ms="999",
                total_ms="1099",
                run=str(10 + i),
                nixl_bytes_delta="9999999",
                nixl_xfer_time_delta_ms="999",
                nixl_transfers_delta="3",
            ))
        _write_exp5b_csv(os.path.join(d, "exp5b-results.csv"), rows)
        result = analyze_nixl_transfer(d)
        self.assertEqual(result["n_points"], 5)
        self.assertEqual(result["gate_checks"]["multi_transfer_rows"], 3)

    def test_with_kv_bytes_per_token(self):
        d = tempfile.mkdtemp()
        kv_per_tok = 1000
        rows = []
        for i in range(5):
            tokens = 100 * (i + 1)
            nbytes = kv_per_tok * tokens
            time_ms = 5.0 + nbytes / 1e6
            rows.append(_make_row(
                status_code="200",
                prompt_tokens_target=str(tokens),
                prompt_tokens_actual=str(tokens),
                ttft_ms=str(time_ms + 100),
                total_ms=str(time_ms + 200),
                run=str(i + 1),
                nixl_bytes_delta=str(nbytes),
                nixl_xfer_time_delta_ms=str(time_ms),
                nixl_transfers_delta="1",
            ))
        _write_exp5b_csv(os.path.join(d, "exp5b-results.csv"), rows)
        result = analyze_nixl_transfer(d, kv_bytes_per_token=kv_per_tok)
        for _L, summary in result["by_seq_len"].items():
            self.assertIn("cache_miss_ratio", summary)
            self.assertAlmostEqual(summary["cache_miss_ratio"], 1.0, places=1)

    def test_empty_data(self):
        d = tempfile.mkdtemp()
        result = analyze_nixl_transfer(d)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
