"""Tests for statistical functions in analyze.py.

Validates the t-distribution table, CI computation, percentile calculation,
and data validation against known values.
"""

import os
import sys
import unittest

# Import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyze import (
    _T95,
    _t95,
    compare_error_rates,
    mann_whitney_u,
    safe_float,
    safe_int,
    sample_adequacy,
    stats,
    validate_completeness,
)


class TestTDistributionTable(unittest.TestCase):
    """Verify t-distribution critical values against known reference values.

    Reference: standard two-tailed t-table at alpha=0.05.
    """

    REFERENCE_T95 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980,
    }

    def test_table_values_match_reference(self):
        for df, expected in self.REFERENCE_T95.items():
            self.assertAlmostEqual(
                _T95[df], expected, places=3,
                msg=f"t95 for df={df}: expected {expected}, got {_T95[df]}")

    def test_exact_lookup(self):
        self.assertEqual(_t95(10), 2.228)
        self.assertEqual(_t95(20), 2.086)

    def test_interpolation(self):
        # df=12 should interpolate between df=10 (2.228) and df=15 (2.131)
        t12 = _t95(12)
        self.assertGreater(t12, 2.131)
        self.assertLess(t12, 2.228)
        # Linear interpolation: 2.228 + (12-10)/(15-10) * (2.131-2.228)
        expected = 2.228 + (2/5) * (2.131 - 2.228)
        self.assertAlmostEqual(t12, expected, places=3)

    def test_large_df_uses_normal(self):
        self.assertEqual(_t95(500), 1.96)
        self.assertEqual(_t95(1000), 1.96)

    def test_df_1_is_largest(self):
        """df=1 should have the largest critical value."""
        self.assertEqual(_t95(1), max(_T95.values()))


class TestStats(unittest.TestCase):
    """Test the stats() function against known values."""

    def test_empty(self):
        s = stats([])
        self.assertEqual(s["n"], 0)

    def test_single_value(self):
        s = stats([5.0])
        self.assertEqual(s["n"], 1)
        self.assertAlmostEqual(s["median"], 5.0, places=1)
        self.assertAlmostEqual(s["mean"], 5.0, places=1)
        # With n=1, std and CI are undefined (reported as inf)
        self.assertEqual(s["std"], float('inf'))
        self.assertEqual(s["ci95_lo"], float('-inf'))
        self.assertEqual(s["ci95_hi"], float('inf'))

    def test_known_dataset(self):
        """Test against hand-calculated values.

        Dataset: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Mean = 5.5, Median = 5.5
        Variance (sample) = 9.1667, Std = 3.0277
        """
        data = list(range(1, 11))
        s = stats([float(x) for x in data])

        self.assertEqual(s["n"], 10)
        self.assertAlmostEqual(s["mean"], 5.5, places=1)
        self.assertAlmostEqual(s["median"], 5.5, places=1)
        self.assertAlmostEqual(s["std"], 3.0, places=0)
        self.assertEqual(s["min"], 1.0)
        self.assertEqual(s["max"], 10.0)

    def test_ci_contains_mean(self):
        """95% CI should contain the sample mean."""
        data = [100.0, 102.0, 98.0, 101.0, 99.0]
        s = stats(data)
        self.assertLessEqual(s["ci95_lo"], s["mean"])
        self.assertGreaterEqual(s["ci95_hi"], s["mean"])

    def test_ci_width_decreases_with_n(self):
        """Larger samples should produce narrower CIs (for same std)."""
        import random
        random.seed(42)
        small = stats([random.gauss(100, 10) for _ in range(10)])
        large = stats([random.gauss(100, 10) for _ in range(100)])
        small_width = small["ci95_hi"] - small["ci95_lo"]
        large_width = large["ci95_hi"] - large["ci95_lo"]
        self.assertGreater(small_width, large_width)

    def test_ci_known_value(self):
        """Verify CI against hand calculation.

        Dataset: [10, 20, 30]
        Mean = 20, Std = 10, n = 3
        df = 2, t95 = 4.303
        CI half-width = 4.303 * 10 / sqrt(3) = 24.84
        CI = [20 - 24.84, 20 + 24.84] = [-4.8, 44.8]
        """
        s = stats([10.0, 20.0, 30.0])
        self.assertAlmostEqual(s["mean"], 20.0, places=1)
        self.assertAlmostEqual(s["std"], 10.0, places=1)
        # CI half-width: t(2) * 10 / sqrt(3) = 4.303 * 5.7735 = 24.84
        self.assertAlmostEqual(s["ci95_lo"], -4.8, places=0)
        self.assertAlmostEqual(s["ci95_hi"], 44.8, places=0)

    def test_median_odd_n(self):
        s = stats([1.0, 3.0, 5.0])
        self.assertAlmostEqual(s["median"], 3.0, places=1)

    def test_median_even_n(self):
        s = stats([1.0, 3.0, 5.0, 7.0])
        self.assertAlmostEqual(s["median"], 4.0, places=1)

    def test_percentiles(self):
        # 100 evenly spaced values: percentiles should be close to the value
        data = [float(i) for i in range(101)]
        s = stats(data)
        self.assertAlmostEqual(s["p10"], 10.0, places=0)
        self.assertAlmostEqual(s["p25"], 25.0, places=0)
        self.assertAlmostEqual(s["p75"], 75.0, places=0)
        self.assertAlmostEqual(s["p90"], 90.0, places=0)

    def test_iqr(self):
        data = [float(i) for i in range(101)]
        s = stats(data)
        self.assertAlmostEqual(s["iqr"], 50.0, places=0)


class TestValidateCompleteness(unittest.TestCase):

    def test_complete_data(self):
        rows = [
            {"config": "A", "status_code": "200"},
            {"config": "A", "status_code": "200"},
            {"config": "B", "status_code": "200"},
            {"config": "B", "status_code": "200"},
        ]
        warnings = validate_completeness(rows, ["config"])
        self.assertEqual(warnings, [])

    def test_missing_runs(self):
        rows = [
            {"config": "A", "status_code": "200"},
            {"config": "A", "status_code": "200"},
            {"config": "A", "status_code": "200"},
            {"config": "B", "status_code": "200"},
            {"config": "B", "status_code": "200"},
            {"config": "B", "status_code": "200"},
            {"config": "C", "status_code": "200"},  # only 1 run for C
        ]
        warnings = validate_completeness(rows, ["config"])
        # Mode is 3 (both A and B have 3), so C with 1 should be flagged
        self.assertTrue(any("config=C" in w for w in warnings))

    def test_high_error_rate(self):
        rows = [
            {"config": "A", "status_code": "200"},
            {"config": "A", "status_code": "500"},
            {"config": "A", "status_code": "500"},
        ]
        warnings = validate_completeness(rows, ["config"])
        self.assertTrue(any("error rate" in w.lower() for w in warnings))

    def test_empty_data(self):
        warnings = validate_completeness([], ["config"])
        self.assertTrue(any("no data" in w.lower() for w in warnings))

    def test_multi_field_grouping(self):
        rows = [
            {"config": "A", "concurrency": "1"}, {"config": "A", "concurrency": "1"},
            {"config": "A", "concurrency": "2"}, {"config": "A", "concurrency": "2"},
            {"config": "B", "concurrency": "1"}, {"config": "B", "concurrency": "1"},
            # missing B/concurrency=2 entirely
        ]
        warnings = validate_completeness(rows, ["config", "concurrency"])
        self.assertTrue(len(warnings) > 0)


class TestStatsOutliersAndCV(unittest.TestCase):
    """Verify new stats() fields: outliers, cv."""

    def test_no_outliers_uniform(self):
        s = stats([float(i) for i in range(20)])
        self.assertEqual(s["outliers"], 0)

    def test_outlier_detected(self):
        # 19 values near 100, one extreme value at 1000
        data = [100.0] * 19 + [1000.0]
        s = stats(data)
        self.assertGreater(s["outliers"], 0)

    def test_cv_precise_measurement(self):
        # Very tight distribution: CV should be low
        data = [100.0, 100.1, 99.9, 100.0, 100.2]
        s = stats(data)
        self.assertLess(s["cv"], 0.01)

    def test_cv_noisy_measurement(self):
        # Wide distribution: CV should be high
        data = [10.0, 50.0, 100.0, 200.0, 500.0]
        s = stats(data)
        self.assertGreater(s["cv"], 0.5)

    def test_cv_single_value(self):
        s = stats([42.0])
        self.assertEqual(s["cv"], float('inf'))


class TestMannWhitneyU(unittest.TestCase):
    """Verify Mann-Whitney U rank-sum test."""

    def test_identical_samples(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = mann_whitney_u(x, x)
        self.assertIsNotNone(result)
        _, _, p, _ = result
        self.assertGreater(p, 0.05)  # not significant

    def test_clearly_different_samples(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        y = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        result = mann_whitney_u(x, y)
        self.assertIsNotNone(result)
        _, _, p, r_eff = result
        self.assertLess(p, 0.01)  # highly significant
        self.assertGreater(r_eff, 0.3)  # large effect

    def test_too_few_samples(self):
        result = mann_whitney_u([1.0, 2.0], [3.0, 4.0])
        self.assertIsNone(result)

    def test_effect_size_range(self):
        x = list(range(20))
        y = [v + 0.5 for v in range(20)]  # slight shift
        result = mann_whitney_u(x, y)
        self.assertIsNotNone(result)
        _, _, _, r_eff = result
        self.assertGreaterEqual(r_eff, 0.0)
        self.assertLessEqual(r_eff, 1.0)


class TestSampleAdequacy(unittest.TestCase):
    """Verify sample adequacy warnings."""

    def test_n_3_warns(self):
        warnings = sample_adequacy(3)
        self.assertTrue(any("too few" in w for w in warnings))

    def test_n_50_warns_p99(self):
        warnings = sample_adequacy(50)
        self.assertTrue(any("p99" in w for w in warnings))

    def test_n_200_no_p99_warning(self):
        warnings = sample_adequacy(200)
        self.assertFalse(any("p99" in w for w in warnings))

    def test_context_prefix(self):
        warnings = sample_adequacy(3, context="BASELINE/50tok")
        self.assertTrue(any("BASELINE/50tok" in w for w in warnings))


class TestCompareErrorRates(unittest.TestCase):
    """Verify error rate comparison and survivorship bias detection."""

    def test_equal_rates_no_warning(self):
        rows = [
            {"config": "A", "status_code": "200"},
            {"config": "A", "status_code": "200"},
            {"config": "B", "status_code": "200"},
            {"config": "B", "status_code": "200"},
        ]
        rates, warning = compare_error_rates(rows)
        self.assertAlmostEqual(rates["A"], 0.0)
        self.assertAlmostEqual(rates["B"], 0.0)
        self.assertIsNone(warning)

    def test_unequal_rates_warns(self):
        rows = [
            {"config": "A", "status_code": "200"},
            {"config": "A", "status_code": "200"},
            {"config": "A", "status_code": "200"},
            {"config": "B", "status_code": "200"},
            {"config": "B", "status_code": "500"},
            {"config": "B", "status_code": "500"},
        ]
        rates, warning = compare_error_rates(rows)
        self.assertAlmostEqual(rates["A"], 0.0)
        self.assertAlmostEqual(rates["B"], 2 / 3, places=2)
        self.assertIsNotNone(warning)
        self.assertIn("survivorship", warning.lower())


class TestSafeConversions(unittest.TestCase):

    def test_safe_float_valid(self):
        self.assertEqual(safe_float("3.14"), 3.14)

    def test_safe_float_invalid(self):
        self.assertEqual(safe_float("abc"), 0.0)
        self.assertEqual(safe_float(""), 0.0)
        self.assertEqual(safe_float(None), 0.0)

    def test_safe_float_custom_default(self):
        self.assertEqual(safe_float("abc", -1.0), -1.0)

    def test_safe_int_valid(self):
        self.assertEqual(safe_int("42"), 42)

    def test_safe_int_invalid(self):
        self.assertEqual(safe_int("abc"), 0)
        self.assertEqual(safe_int(None), 0)


if __name__ == "__main__":
    unittest.main()
