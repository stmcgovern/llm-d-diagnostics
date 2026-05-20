"""Tests for advisor/kv_sweep.py — KV head ratio sweep analysis."""

import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_toolkit = os.path.join(os.path.dirname(__file__), "..", "..", "toolkit")
sys.path.insert(0, _toolkit)

from kv_sweep import _kv_bytes, _linreg, _load_exp5, _stats, analyze  # noqa: E402

from schemas import Exp5Row, fields_for  # noqa: E402


def _write_exp5_csv(path, rows):
    fieldnames = fields_for(Exp5Row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _make_row(**overrides):
    row = {f: "0" for f in fields_for(Exp5Row)}
    row.update(overrides)
    return row


class TestKvBytes(unittest.TestCase):

    def test_known_architecture(self):
        arch = {"n_layers": 22, "n_kv_heads": 4, "d_head": 64}
        self.assertEqual(_kv_bytes(arch), 2 * 22 * 4 * 64 * 2)

    def test_double_kv_heads_doubles_bytes(self):
        base = {"n_layers": 22, "n_kv_heads": 4, "d_head": 64}
        doubled = {"n_layers": 22, "n_kv_heads": 8, "d_head": 64}
        self.assertEqual(_kv_bytes(doubled), 2 * _kv_bytes(base))


class TestLinreg(unittest.TestCase):

    def test_perfect_linear(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [3.0, 5.0, 7.0, 9.0, 11.0]
        intercept, slope, r_sq = _linreg(xs, ys)
        self.assertAlmostEqual(slope, 2.0, places=5)
        self.assertAlmostEqual(intercept, 1.0, places=5)
        self.assertAlmostEqual(r_sq, 1.0, places=5)

    def test_fewer_than_two_points(self):
        self.assertEqual(_linreg([1], [2]), (0, 0, 0))
        self.assertEqual(_linreg([], []), (0, 0, 0))

    def test_constant_x(self):
        _intercept, slope, _r_sq = _linreg([5, 5, 5], [1, 2, 3])
        self.assertEqual(slope, 0)

    def test_constant_y(self):
        intercept, _slope, r_sq = _linreg([1, 2, 3], [5, 5, 5])
        self.assertEqual(r_sq, 0)
        self.assertAlmostEqual(intercept, 5.0)


class TestStats(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(_stats([]), (0, 0))

    def test_single(self):
        self.assertEqual(_stats([7.0]), (7.0, 0))

    def test_known_values(self):
        mean, sd = _stats([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertAlmostEqual(mean, 5.0, places=5)
        self.assertAlmostEqual(sd, 2.138, places=2)


class TestLoadExp5(unittest.TestCase):

    def test_valid_cd_pairs(self):
        d = tempfile.mkdtemp()
        rows = []
        for run in range(1, 4):
            rows.append(_make_row(
                config="C-sidecar-only", status_code="200",
                prompt_tokens_target="100", run=str(run),
                ttft_ms=str(500 + run), total_ms=str(510 + run),
            ))
            rows.append(_make_row(
                config="D-disaggregated", status_code="200",
                prompt_tokens_target="100", run=str(run),
                ttft_ms=str(600 + run), total_ms=str(610 + run),
            ))
        _write_exp5_csv(os.path.join(d, "exp5-results.csv"), rows)
        transfer_by_len, _model = _load_exp5(d)
        self.assertIsNotNone(transfer_by_len)
        self.assertIn(100, transfer_by_len)
        self.assertEqual(len(transfer_by_len[100]), 3)
        for v in transfer_by_len[100]:
            self.assertAlmostEqual(v, 100.0, places=5)

    def test_non_200_excluded(self):
        d = tempfile.mkdtemp()
        rows = [
            _make_row(config="C-sidecar-only", status_code="500",
                      prompt_tokens_target="100", run="1",
                      ttft_ms="500", total_ms="510"),
            _make_row(config="D-disaggregated", status_code="200",
                      prompt_tokens_target="100", run="1",
                      ttft_ms="600", total_ms="610"),
        ]
        _write_exp5_csv(os.path.join(d, "exp5-results.csv"), rows)
        transfer_by_len, _ = _load_exp5(d)
        self.assertEqual(transfer_by_len, {})

    def test_missing_partner_excluded(self):
        d = tempfile.mkdtemp()
        rows = [
            _make_row(config="C-sidecar-only", status_code="200",
                      prompt_tokens_target="100", run="1",
                      ttft_ms="500", total_ms="510"),
        ]
        _write_exp5_csv(os.path.join(d, "exp5-results.csv"), rows)
        transfer_by_len, _ = _load_exp5(d)
        self.assertEqual(transfer_by_len, {})

    def test_missing_csv(self):
        d = tempfile.mkdtemp()
        transfer_by_len, model = _load_exp5(d)
        self.assertIsNone(transfer_by_len)
        self.assertIsNone(model)

    def test_model_from_run_info(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "run-info.json"), "w") as f:
            json.dump({"toolkit": {"model": "test/model"}}, f)
        rows = [
            _make_row(config="C-sidecar-only", status_code="200",
                      prompt_tokens_target="100", run="1",
                      ttft_ms="500", total_ms="510"),
            _make_row(config="D-disaggregated", status_code="200",
                      prompt_tokens_target="100", run="1",
                      ttft_ms="600", total_ms="610"),
        ]
        _write_exp5_csv(os.path.join(d, "exp5-results.csv"), rows)
        _, model = _load_exp5(d)
        self.assertEqual(model, "test/model")


class TestAnalyze(unittest.TestCase):

    def test_known_model(self):
        d = tempfile.mkdtemp()
        model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        with open(os.path.join(d, "run-info.json"), "w") as f:
            json.dump({"toolkit": {"model": model}}, f)
        rows = []
        for seq_len in [100, 500, 1000]:
            for run in range(1, 4):
                c_ttft = 500 + seq_len * 0.1
                d_ttft = c_ttft + seq_len * 0.05
                rows.append(_make_row(
                    config="C-sidecar-only", status_code="200",
                    prompt_tokens_target=str(seq_len), run=str(run),
                    ttft_ms=str(c_ttft), total_ms=str(c_ttft + 10),
                ))
                rows.append(_make_row(
                    config="D-disaggregated", status_code="200",
                    prompt_tokens_target=str(seq_len), run=str(run),
                    ttft_ms=str(d_ttft), total_ms=str(d_ttft + 10),
                ))
        _write_exp5_csv(os.path.join(d, "exp5-results.csv"), rows)
        results = analyze([d])
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["model"], model)
        self.assertAlmostEqual(r["protocol_ms"], 0.0, delta=1.0)
        self.assertAlmostEqual(r["slope_ms_per_tok"], 0.05, places=3)
        self.assertAlmostEqual(r["r_sq"], 1.0, places=3)
        self.assertGreater(r["eff_bw_gbs"], 0)

    def test_unknown_model_skipped(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "run-info.json"), "w") as f:
            json.dump({"toolkit": {"model": "unknown/model-xyz"}}, f)
        rows = []
        for run in range(1, 4):
            rows.append(_make_row(
                config="C-sidecar-only", status_code="200",
                prompt_tokens_target="100", run=str(run),
                ttft_ms="500", total_ms="510",
            ))
            rows.append(_make_row(
                config="D-disaggregated", status_code="200",
                prompt_tokens_target="100", run=str(run),
                ttft_ms="600", total_ms="610",
            ))
        _write_exp5_csv(os.path.join(d, "exp5-results.csv"), rows)
        results = analyze([d])
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
