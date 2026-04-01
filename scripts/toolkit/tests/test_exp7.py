"""Tests for exp7 mixed workload utilities and analysis.

Tests Poisson interval generation, workload assignment, ITL computation,
and the analysis verdict logic against synthetic data.
"""

import csv
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp7_mixed_workload import poisson_intervals, pick_workload, compute_itl
from analyze import analyze_exp7


class TestPoissonIntervals(unittest.TestCase):

    def test_rate_proportional(self):
        """Higher rate should produce more arrivals."""
        low = poisson_intervals(1.0, 10.0, seed=1)
        high = poisson_intervals(10.0, 10.0, seed=1)
        self.assertGreater(len(high), len(low))

    def test_expected_count(self):
        """Count should be approximately rate * duration."""
        arrivals = poisson_intervals(5.0, 100.0, seed=42)
        expected = 5.0 * 100.0
        # Within 20% of expected (law of large numbers)
        self.assertGreater(len(arrivals), expected * 0.8)
        self.assertLess(len(arrivals), expected * 1.2)

    def test_all_within_duration(self):
        """All arrival times should be < duration."""
        arrivals = poisson_intervals(10.0, 5.0, seed=7)
        for t in arrivals:
            self.assertLess(t, 5.0)

    def test_monotonically_increasing(self):
        """Arrival times should be sorted."""
        arrivals = poisson_intervals(5.0, 20.0, seed=3)
        for i in range(len(arrivals) - 1):
            self.assertLess(arrivals[i], arrivals[i + 1])

    def test_deterministic_with_seed(self):
        """Same seed should produce same arrivals."""
        a1 = poisson_intervals(5.0, 10.0, seed=42)
        a2 = poisson_intervals(5.0, 10.0, seed=42)
        self.assertEqual(a1, a2)

    def test_zero_duration(self):
        arrivals = poisson_intervals(5.0, 0.0, seed=1)
        self.assertEqual(len(arrivals), 0)


class TestPickWorkload(unittest.TestCase):

    def test_20pct_long(self):
        """With 20% long, every 5th request should be long."""
        results = [pick_workload(i, 20) for i in range(1, 101)]
        long_count = results.count("long")
        # 20 out of 100 should be long
        self.assertEqual(long_count, 20)

    def test_0pct_long(self):
        """0% long = all short."""
        results = [pick_workload(i, 0) for i in range(1, 101)]
        self.assertTrue(all(r == "short" for r in results))

    def test_deterministic(self):
        """Same seq should always give same workload class."""
        for seq in range(1, 50):
            a = pick_workload(seq, 20)
            b = pick_workload(seq, 20)
            self.assertEqual(a, b)


class TestComputeITL(unittest.TestCase):

    def test_uniform_tokens(self):
        """Evenly spaced tokens should give constant ITL."""
        # 10 tokens, 10ms apart (0.01s)
        times = [i * 0.01 for i in range(10)]
        mean, p99 = compute_itl(times)
        self.assertAlmostEqual(mean, 10.0, places=0)
        self.assertAlmostEqual(p99, 10.0, places=0)

    def test_single_token(self):
        """Single token = no gaps = 0."""
        mean, p99 = compute_itl([0.5])
        self.assertEqual(mean, 0.0)
        self.assertEqual(p99, 0.0)

    def test_empty(self):
        mean, p99 = compute_itl([])
        self.assertEqual(mean, 0.0)

    def test_variable_gaps(self):
        """Variable spacing should give non-zero ITL."""
        times = [0.0, 0.01, 0.03, 0.06, 0.10]  # gaps: 10, 20, 30, 40ms
        mean, p99 = compute_itl(times)
        self.assertAlmostEqual(mean, 25.0, places=0)
        self.assertGreater(p99, mean)


class TestExp7Analysis(unittest.TestCase):

    def _write_csv(self, tmpdir, rows):
        path = os.path.join(tmpdir, "exp7-results.csv")
        fields = ["experiment", "config", "seq", "workload_class",
                   "prompt_tokens_target", "max_tokens",
                   "ttft_ms", "total_ms", "itl_mean_ms", "itl_p99_ms",
                   "status_code", "completion_tokens",
                   "scheduled_at_s", "depart_delay_ms", "error"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def _make_rows(self, config, workload_class, n, ttft, itl_mean, itl_p99,
                   start_seq=1):
        rows = []
        for i in range(n):
            rows.append({
                "experiment": "exp7", "config": config,
                "seq": start_seq + i, "workload_class": workload_class,
                "prompt_tokens_target": 10 if workload_class == "short" else 500,
                "max_tokens": 20 if workload_class == "short" else 50,
                "ttft_ms": ttft, "total_ms": ttft * 1.5,
                "itl_mean_ms": itl_mean, "itl_p99_ms": itl_p99,
                "status_code": 200, "completion_tokens": 15,
                "scheduled_at_s": i * 0.25, "depart_delay_ms": 1.0,
                "error": "",
            })
        return rows

    def test_verdict_beneficial(self):
        """Disagg wins on ITL and throughput → BENEFICIAL."""
        rows = []
        # BASELINE: high ITL, moderate TTFT
        rows += self._make_rows("BASELINE", "short", 50, ttft=50, itl_mean=30, itl_p99=80)
        rows += self._make_rows("BASELINE", "long", 10, ttft=200, itl_mean=30, itl_p99=80,
                                start_seq=51)
        # DISAGG-2D: low ITL, slightly higher TTFT
        rows += self._make_rows("DISAGG-2D", "short", 50, ttft=55, itl_mean=15, itl_p99=25)
        rows += self._make_rows("DISAGG-2D", "long", 10, ttft=220, itl_mean=15, itl_p99=25,
                                start_seq=51)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_csv(tmpdir, rows)
            output = io.StringIO()
            with redirect_stdout(output):
                analyze_exp7(tmpdir)
            text = output.getvalue()
            self.assertIn("BENEFICIAL", text)

    def test_verdict_not_justified(self):
        """Disagg has high overhead and same ITL → NOT justified."""
        rows = []
        # BASELINE: good
        rows += self._make_rows("BASELINE", "short", 50, ttft=50, itl_mean=15, itl_p99=25)
        rows += self._make_rows("BASELINE", "long", 10, ttft=100, itl_mean=15, itl_p99=25,
                                start_seq=51)
        # DISAGG-2D: much worse TTFT, same ITL
        rows += self._make_rows("DISAGG-2D", "short", 50, ttft=100, itl_mean=15, itl_p99=25)
        rows += self._make_rows("DISAGG-2D", "long", 10, ttft=200, itl_mean=15, itl_p99=25,
                                start_seq=51)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_csv(tmpdir, rows)
            output = io.StringIO()
            with redirect_stdout(output):
                analyze_exp7(tmpdir)
            text = output.getvalue()
            self.assertIn("NOT justified", text)


if __name__ == "__main__":
    unittest.main()
