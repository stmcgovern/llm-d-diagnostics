"""Tests for scaling_model.py: linear regression, KV cache math, prefill FLOPs."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scaling_model import (
    GPUS,
    MODELS,
    kv_bytes_per_token,
    linreg,
    linreg_ci,
    prefill_flops_per_token,
)


class TestLinreg(unittest.TestCase):

    def test_perfect_line(self):
        xs = [1, 2, 3, 4, 5]
        ys = [2, 4, 6, 8, 10]
        intercept, slope = linreg(xs, ys)
        self.assertAlmostEqual(slope, 2.0, places=10)
        self.assertAlmostEqual(intercept, 0.0, places=10)

    def test_with_offset(self):
        xs = [0, 1, 2, 3]
        ys = [5, 7, 9, 11]
        intercept, slope = linreg(xs, ys)
        self.assertAlmostEqual(slope, 2.0, places=10)
        self.assertAlmostEqual(intercept, 5.0, places=10)

    def test_single_point(self):
        _intercept, slope = linreg([1], [5])
        self.assertEqual(slope, 0)

    def test_empty(self):
        intercept, slope = linreg([], [])
        self.assertEqual(slope, 0)
        self.assertEqual(intercept, 0)

    def test_constant_y(self):
        xs = [1, 2, 3, 4]
        ys = [5, 5, 5, 5]
        intercept, slope = linreg(xs, ys)
        self.assertAlmostEqual(slope, 0.0, places=10)
        self.assertAlmostEqual(intercept, 5.0, places=10)

    def test_negative_slope(self):
        xs = [0, 1, 2, 3]
        ys = [10, 7, 4, 1]
        intercept, slope = linreg(xs, ys)
        self.assertAlmostEqual(slope, -3.0, places=10)
        self.assertAlmostEqual(intercept, 10.0, places=10)

    def test_noisy_data(self):
        xs = [1, 2, 3, 4, 5]
        ys = [2.1, 3.9, 6.2, 7.8, 10.1]
        intercept, slope = linreg(xs, ys)
        self.assertAlmostEqual(slope, 2.0, delta=0.2)
        self.assertAlmostEqual(intercept, 0.0, delta=0.5)


class TestLinregCi(unittest.TestCase):

    def test_perfect_line_r_squared(self):
        xs = [1, 2, 3, 4, 5]
        ys = [2, 4, 6, 8, 10]
        result = linreg_ci(xs, ys)
        slope = result[1]
        r_sq = result[6]
        self.assertAlmostEqual(slope, 2.0, places=5)
        self.assertAlmostEqual(r_sq, 1.0, places=5)

    def test_ci_contains_true_value(self):
        xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ys = [2.1 * x + 0.5 + (0.1 * ((-1) ** x)) for x in xs]
        result = linreg_ci(xs, ys)
        slope_lo, slope_hi = result[2], result[3]
        self.assertLess(slope_lo, 2.1)
        self.assertGreater(slope_hi, 2.1)

    def test_too_few_points(self):
        result = linreg_ci([1, 2], [3, 4])
        self.assertEqual(result[7], float('inf'))

    def test_se_slope_positive(self):
        xs = [1, 2, 3, 4, 5]
        ys = [2.1, 3.9, 6.2, 7.8, 10.1]
        result = linreg_ci(xs, ys)
        se_slope = result[7]
        self.assertGreater(se_slope, 0)

    def test_wider_ci_with_noise(self):
        xs = [1, 2, 3, 4, 5]
        ys_clean = [2, 4, 6, 8, 10]
        ys_noisy = [1, 5, 4, 9, 11]

        result_clean = linreg_ci(xs, ys_clean)
        result_noisy = linreg_ci(xs, ys_noisy)

        ci_width_clean = result_clean[3] - result_clean[2]
        ci_width_noisy = result_noisy[3] - result_noisy[2]
        self.assertGreater(ci_width_noisy, ci_width_clean)


class TestKvBytesPerToken(unittest.TestCase):

    def test_tinyllama(self):
        info = MODELS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
        bpt = kv_bytes_per_token(info)
        per_layer = 2 * 4 * 64 * 2
        expected = per_layer * 22
        self.assertEqual(bpt, expected)

    def test_phi3_mha(self):
        info = MODELS["microsoft/Phi-3-mini-4k-instruct"]
        bpt = kv_bytes_per_token(info)
        per_layer = 2 * 32 * 96 * 2
        expected = per_layer * 32
        self.assertEqual(bpt, expected)

    def test_gqa_smaller_than_mha(self):
        tinyllama = kv_bytes_per_token(MODELS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
        phi3 = kv_bytes_per_token(MODELS["microsoft/Phi-3-mini-4k-instruct"])
        self.assertLess(tinyllama, phi3)

    def test_proportional_to_kv_heads(self):
        base = {"n_kv_heads": 8, "d_head": 128, "dtype_bytes": 2, "n_layers": 32}
        double = {**base, "n_kv_heads": 16}
        self.assertEqual(kv_bytes_per_token(double), 2 * kv_bytes_per_token(base))


class TestPrefillFlopsPerToken(unittest.TestCase):

    def test_scales_with_params(self):
        small = MODELS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
        large = MODELS["meta-llama/Llama-3.1-8B-Instruct"]
        ratio = prefill_flops_per_token(large) / prefill_flops_per_token(small)
        param_ratio = large["params_b"] / small["params_b"]
        self.assertAlmostEqual(ratio, param_ratio, places=1)

    def test_value_magnitude(self):
        info = MODELS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
        flops = prefill_flops_per_token(info)
        self.assertAlmostEqual(flops, 2.2e9, delta=0.1e9)


class TestGpuDatabase(unittest.TestCase):

    def test_all_gpus_have_required_fields(self):
        for name, spec in GPUS.items():
            with self.subTest(gpu=name):
                self.assertIn("hbm_bw_gbs", spec)
                self.assertIn("fp16_tflops", spec)
                self.assertIn("memory_gb", spec)

    def test_h200_faster_than_t4(self):
        self.assertGreater(GPUS["H200"]["hbm_bw_gbs"], GPUS["T4"]["hbm_bw_gbs"])
        self.assertGreater(GPUS["H200"]["fp16_tflops"], GPUS["T4"]["fp16_tflops"])


class TestModelDatabase(unittest.TestCase):

    def test_all_models_have_required_fields(self):
        required = ["params_b", "n_layers", "n_kv_heads", "n_q_heads", "d_head",
                     "d_model", "dtype_bytes"]
        for name, info in MODELS.items():
            with self.subTest(model=name):
                for field in required:
                    self.assertIn(field, info, f"{name} missing '{field}'")

    def test_kv_heads_le_q_heads(self):
        for name, info in MODELS.items():
            with self.subTest(model=name):
                self.assertLessEqual(info["n_kv_heads"], info["n_q_heads"])


if __name__ == "__main__":
    unittest.main()
