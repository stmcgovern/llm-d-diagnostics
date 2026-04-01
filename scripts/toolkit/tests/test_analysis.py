"""Tests for analysis functions: pearson_r, crossover detection, saturation detection.

Validates the statistical computations added for exp5 and exp6 analysis
against known synthetic datasets.
"""

import os
import sys
import tempfile
import unittest

# Import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyze import pearson_r, stats, analyze_exp5, analyze_exp6


class TestPearsonR(unittest.TestCase):
    """Test Pearson correlation coefficient."""

    def test_perfect_positive(self):
        r = pearson_r([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        self.assertAlmostEqual(r, 1.0, places=5)

    def test_perfect_negative(self):
        r = pearson_r([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        self.assertAlmostEqual(r, -1.0, places=5)

    def test_uncorrelated(self):
        # Symmetric pattern: no linear correlation
        r = pearson_r([1, 2, 3, 4, 5], [1, 3, 1, 3, 1])
        self.assertAlmostEqual(r, 0.0, places=1)

    def test_too_few_points(self):
        r = pearson_r([1, 2], [3, 4])
        self.assertEqual(r, 0.0)

    def test_constant_y(self):
        r = pearson_r([1, 2, 3, 4], [5, 5, 5, 5])
        self.assertEqual(r, 0.0)


class TestExp5Analysis(unittest.TestCase):
    """Test exp5 crossover analysis with synthetic data."""

    def _write_csv(self, tmpdir, rows):
        """Write synthetic exp5-results.csv."""
        import csv
        path = os.path.join(tmpdir, "exp5-results.csv")
        fields = ["experiment", "config", "prompt_tokens_target", "run",
                   "ttft_ms", "total_ms", "status_code",
                   "prompt_tokens_actual", "completion_tokens", "error"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def _make_rows(self, length_data):
        """Generate synthetic rows from {prompt_len: {config: [ttft values]}}."""
        rows = []
        for pt, configs in length_data.items():
            for cfg, values in configs.items():
                for i, v in enumerate(values):
                    rows.append({
                        "experiment": "exp5", "config": cfg,
                        "prompt_tokens_target": pt, "run": i + 1,
                        "ttft_ms": v, "total_ms": v * 1.1,
                        "status_code": 200, "prompt_tokens_actual": pt,
                        "completion_tokens": 20, "error": "",
                    })
        return rows

    def test_crossover_detected(self):
        """T_transfer starts below T_prefill and crosses above."""
        import io
        from contextlib import redirect_stdout

        # At pt=100: A=50ms, D-C=10ms (transfer < prefill)
        # At pt=1000: A=100ms, D-C=200ms (transfer > prefill)
        data = {
            100: {
                "A-prefill-direct":  [50.0] * 10,
                "B-decode-direct":   [30.0] * 10,
                "C-sidecar-only":    [35.0] * 10,
                "D-disaggregated":   [45.0] * 10,  # D-C = 10ms < A = 50ms
            },
            1000: {
                "A-prefill-direct":  [100.0] * 10,
                "B-decode-direct":   [30.0] * 10,
                "C-sidecar-only":    [35.0] * 10,
                "D-disaggregated":   [235.0] * 10,  # D-C = 200ms > A = 100ms
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_csv(tmpdir, self._make_rows(data))
            output = io.StringIO()
            with redirect_stdout(output):
                analyze_exp5(tmpdir)

            text = output.getvalue()
            self.assertIn("CROSSOVER", text)
            self.assertIn("100", text)
            self.assertIn("1000", text)

    def test_no_crossover_compute_dominated(self):
        """T_transfer < T_prefill everywhere."""
        import io
        from contextlib import redirect_stdout

        data = {
            100: {
                "A-prefill-direct":  [50.0] * 10,
                "B-decode-direct":   [30.0] * 10,
                "C-sidecar-only":    [35.0] * 10,
                "D-disaggregated":   [40.0] * 10,  # D-C = 5ms < A = 50ms
            },
            1000: {
                "A-prefill-direct":  [200.0] * 10,
                "B-decode-direct":   [30.0] * 10,
                "C-sidecar-only":    [35.0] * 10,
                "D-disaggregated":   [50.0] * 10,  # D-C = 15ms < A = 200ms
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_csv(tmpdir, self._make_rows(data))
            output = io.StringIO()
            with redirect_stdout(output):
                analyze_exp5(tmpdir)

            text = output.getvalue()
            self.assertIn("compute-dominated", text)
            self.assertNotIn("CROSSOVER", text)


class TestExp6Analysis(unittest.TestCase):
    """Test exp6 saturation analysis with synthetic data."""

    def _write_csv(self, tmpdir, rows):
        """Write synthetic exp6-results.csv."""
        import csv
        path = os.path.join(tmpdir, "exp6-results.csv")
        fields = ["experiment", "config", "qps_target", "seq",
                   "depart_delay_ms", "ttft_ms", "total_ms",
                   "status_code", "completion_tokens", "error"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def _make_rows(self, config_qps_data):
        """Generate rows from {config: {qps: [ttft values]}}."""
        rows = []
        for cfg, qps_data in config_qps_data.items():
            for qps, values in qps_data.items():
                for i, v in enumerate(values):
                    rows.append({
                        "experiment": "exp6", "config": cfg,
                        "qps_target": qps, "seq": i + 1,
                        "depart_delay_ms": 0.5,
                        "ttft_ms": v, "total_ms": v * 1.1,
                        "status_code": 200, "completion_tokens": 20,
                        "error": "",
                    })
        return rows

    def test_saturation_detected(self):
        """p99 exceeds 2x baseline p50 at high QPS."""
        import io
        from contextlib import redirect_stdout

        # BASELINE: p50=100ms at QPS=1, p99=250ms at QPS=8 (> 2*100=200ms threshold)
        data = {
            "BASELINE": {
                1: [100.0] * 30,
                8: [150.0] * 28 + [250.0, 260.0],  # p99 ≈ 260ms > 200ms
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_csv(tmpdir, self._make_rows(data))
            output = io.StringIO()
            with redirect_stdout(output):
                analyze_exp6(tmpdir)

            text = output.getvalue()
            self.assertIn("saturates at QPS=8", text)

    def test_depart_delay_warning(self):
        """High depart delays trigger a warning."""
        import io
        from contextlib import redirect_stdout

        rows = []
        for i in range(30):
            rows.append({
                "experiment": "exp6", "config": "BASELINE",
                "qps_target": 32, "seq": i + 1,
                "depart_delay_ms": 200.0,  # way too high
                "ttft_ms": 100.0, "total_ms": 110.0,
                "status_code": 200, "completion_tokens": 20,
                "error": "",
            })
        # Need a baseline QPS level too
        for i in range(30):
            rows.append({
                "experiment": "exp6", "config": "BASELINE",
                "qps_target": 1, "seq": i + 1,
                "depart_delay_ms": 0.5,
                "ttft_ms": 100.0, "total_ms": 110.0,
                "status_code": 200, "completion_tokens": 20,
                "error": "",
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_csv(tmpdir, rows)
            output = io.StringIO()
            with redirect_stdout(output):
                analyze_exp6(tmpdir)

            text = output.getvalue()
            self.assertIn("harness saturation", text.lower()
                          if "harness" in text.lower()
                          else text)  # check for the warning

    def test_no_saturation(self):
        """System stays healthy at all QPS levels."""
        import io
        from contextlib import redirect_stdout

        data = {
            "BASELINE": {
                1: [100.0] * 30,
                8: [105.0] * 30,  # p99 ≈ 105ms < 200ms threshold
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_csv(tmpdir, self._make_rows(data))
            output = io.StringIO()
            with redirect_stdout(output):
                analyze_exp6(tmpdir)

            text = output.getvalue()
            self.assertIn("no saturation", text)


if __name__ == "__main__":
    unittest.main()
