"""Tests for advisor/validate.py — validation against experiment data."""

import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "toolkit"))

from validate import (
    _grade,
    compare,
    load_exp_baselines,
    load_run_info,
)


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


EXP11_FIELDS = [
    "experiment", "config", "prompt_tokens_target", "max_tokens",
    "concurrency", "run", "ttft_ms", "total_ms", "status_code",
    "prompt_tokens_actual", "completion_tokens", "target", "error",
]

EXP14_FIELDS = [
    "experiment", "config", "pod", "prompt_tokens_target",
    "concurrency", "run", "ttft_ms", "total_ms", "status_code",
    "prompt_tokens_actual", "completion_tokens", "error",
]


def _make_exp11_rows(config, concurrency, prompt_tokens, ttft_values):
    rows = []
    for i, ttft in enumerate(ttft_values):
        rows.append({
            "experiment": "exp11", "config": config,
            "prompt_tokens_target": prompt_tokens, "max_tokens": 20,
            "concurrency": concurrency, "run": i + 1,
            "ttft_ms": ttft, "total_ms": ttft + 5,
            "status_code": 200, "prompt_tokens_actual": prompt_tokens + 10,
            "completion_tokens": 20, "target": "d1", "error": "",
        })
    return rows


def _make_exp14_rows(config, prompt_tokens, ttft_values, pod="pod-1"):
    rows = []
    for i, ttft in enumerate(ttft_values):
        rows.append({
            "experiment": "exp14", "config": config,
            "pod": pod, "prompt_tokens_target": prompt_tokens,
            "concurrency": 8, "run": i + 1,
            "ttft_ms": ttft, "total_ms": ttft + 5,
            "status_code": 200, "prompt_tokens_actual": prompt_tokens + 10,
            "completion_tokens": 20, "error": "",
        })
    return rows


class TestGrade(unittest.TestCase):

    def test_good(self):
        self.assertEqual(_grade(10), "GOOD")
        self.assertEqual(_grade(-15), "GOOD")

    def test_fair(self):
        self.assertEqual(_grade(30), "FAIR")
        self.assertEqual(_grade(-45), "FAIR")

    def test_poor(self):
        self.assertEqual(_grade(60), "POOR")
        self.assertEqual(_grade(-90), "POOR")

    def test_wrong(self):
        self.assertEqual(_grade(150), "WRONG")
        self.assertEqual(_grade(-200), "WRONG")


class TestLoadRunInfo(unittest.TestCase):

    def test_loads_model(self):
        with tempfile.TemporaryDirectory() as d:
            info = {"toolkit": {"model": "test/model"}, "experiments": {}}
            with open(os.path.join(d, "run-info.json"), "w") as f:
                json.dump(info, f)
            result = load_run_info(d)
            self.assertEqual(result["toolkit"]["model"], "test/model")

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            result = load_run_info(d)
            self.assertEqual(result, {})


class TestLoadExpBaselines(unittest.TestCase):

    def test_extracts_exp11_medians(self):
        with tempfile.TemporaryDirectory() as d:
            rows = (_make_exp11_rows("BASELINE", 1, 100, [100, 110, 105, 108, 103])
                    + _make_exp11_rows("DISAGG-1D", 1, 100, [200, 210, 205, 208, 203]))
            _write_csv(os.path.join(d, "exp11-results.csv"), EXP11_FIELDS, rows)

            result = load_exp_baselines(d)
            bl = result["exp11"][("BASELINE", 1, 100)]
            self.assertGreater(bl["n"], 0)
            self.assertAlmostEqual(bl["median"], 105.0, places=0)

            dg = result["exp11"][("DISAGG-1D", 1, 100)]
            self.assertAlmostEqual(dg["median"], 205.0, places=0)

    def test_extracts_exp14_decomposition(self):
        with tempfile.TemporaryDirectory() as d:
            rows = (_make_exp14_rows("B-decode-direct", 100, [500, 510, 505])
                    + _make_exp14_rows("C-sidecar-only", 100, [520, 530, 525])
                    + _make_exp14_rows("D-disaggregated", 100, [700, 710, 705]))
            _write_csv(os.path.join(d, "exp14-results.csv"), EXP14_FIELDS, rows)

            result = load_exp_baselines(d)
            b = result["exp14"][("B-decode-direct", 100)]
            c = result["exp14"][("C-sidecar-only", 100)]
            d_val = result["exp14"][("D-disaggregated", 100)]

            sidecar = c["median"] - b["median"]
            nixl = d_val["median"] - c["median"]
            self.assertGreater(sidecar, 0)
            self.assertGreater(nixl, 0)

    def test_filters_errors(self):
        with tempfile.TemporaryDirectory() as d:
            good = _make_exp11_rows("BASELINE", 1, 100, [100, 110, 105])
            bad = _make_exp11_rows("BASELINE", 1, 100, [9999])
            bad[0]["status_code"] = 500
            _write_csv(os.path.join(d, "exp11-results.csv"), EXP11_FIELDS, good + bad)

            result = load_exp_baselines(d)
            bl = result["exp11"][("BASELINE", 1, 100)]
            self.assertEqual(bl["n"], 3)

    def test_missing_data(self):
        with tempfile.TemporaryDirectory() as d:
            result = load_exp_baselines(d)
            self.assertEqual(result["exp11"], {})
            self.assertEqual(result["exp14"], {})


class TestCompare(unittest.TestCase):

    def test_good_prediction(self):
        predictions = {100: {
            "mono_ttft_ms": 105, "disagg_ttft_ms": 200,
            "nixl_ms": 50, "sidecar_ms": 12,
        }}
        measurements = {
            "exp11": {
                ("BASELINE", 1, 100): {"median": 100, "n": 10},
                ("DISAGG-1D", 1, 100): {"median": 195, "n": 10},
            },
            "exp14": {},
        }
        results = compare(predictions, measurements)
        mono_r = next(r for r in results if r.metric == "mono_ttft")
        self.assertEqual(mono_r.grade, "GOOD")
        self.assertAlmostEqual(mono_r.error_pct, 5.0, places=0)

    def test_wrong_prediction(self):
        predictions = {100: {
            "mono_ttft_ms": 174, "disagg_ttft_ms": 338,
            "nixl_ms": 152, "sidecar_ms": 12,
        }}
        measurements = {
            "exp11": {
                ("BASELINE", 1, 100): {"median": 770, "n": 24},
            },
            "exp14": {},
        }
        results = compare(predictions, measurements)
        mono_r = next(r for r in results if r.metric == "mono_ttft")
        self.assertEqual(mono_r.grade, "POOR")

    def test_sidecar_comparison(self):
        predictions = {100: {
            "mono_ttft_ms": 770, "disagg_ttft_ms": 1022,
            "nixl_ms": 152, "sidecar_ms": 12,
        }}
        measurements = {
            "exp11": {},
            "exp14": {
                ("B-decode-direct", 100): {"median": 500, "n": 24},
                ("C-sidecar-only", 100): {"median": 503, "n": 24},
            },
        }
        results = compare(predictions, measurements)
        sidecar_r = next(r for r in results if r.metric == "sidecar_ms")
        self.assertEqual(sidecar_r.measured, 3.0)
        self.assertEqual(sidecar_r.predicted, 12)


if __name__ == "__main__":
    unittest.main()
