"""Tests for advisor/plan.py — capacity planning, no cluster needed."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plan import (
    KV_BYTES_PER_TOKEN,
    MEASURED_BASELINES,
    MOE_NIXL_CORRECTION,
    NIXL_PROTOCOL_MS,
    CapacityPlan,
    ModelProfile,
    _baseline_ttft,
    _check_vram_feasibility,
    _estimate_kv_bytes,
    _estimate_nixl_ms,
    _find_nearest_baselines,
    _generate_recommendation,
    _validate_profile,
    plan_capacity,
    save_plan,
)


class TestPlanCapacityMeasured(unittest.TestCase):
    """Test plan_capacity when a measured baseline exists."""

    def test_tinyllama_t4_measured(self):
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="t4",
        )
        self.assertEqual(plan.confidence, "measured")
        self.assertEqual(plan.mono_est_ttft_ms, 73)
        self.assertEqual(plan.disagg_est_ttft_ms, 109)
        self.assertGreater(plan.mono_total_gpus, 0)
        self.assertGreater(plan.disagg_total_gpus, 0)

    def test_olmoe_t4_measured(self):
        plan = plan_capacity(
            "allenai/OLMoE-1B-7B-0924-Instruct",
            target_throughput=0.5, target_ttft_ms=5000, gpu_type="t4",
        )
        self.assertEqual(plan.confidence, "measured")
        self.assertEqual(plan.mono_est_ttft_ms, 2999)

    def test_recommendation_is_set(self):
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="t4",
        )
        self.assertIn("MONOLITHIC", plan.recommendation)

    def test_cost_is_positive(self):
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="t4",
        )
        self.assertGreater(plan.mono_cost_per_hr, 0)
        self.assertGreater(plan.disagg_cost_per_hr, 0)


class TestPlanCapacityExtrapolated(unittest.TestCase):
    """Test plan_capacity when no measured baseline exists (extrapolation)."""

    def test_extrapolation_h100(self):
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="h100",
        )
        self.assertIn(plan.confidence, ["interpolated", "extrapolated", "low"])
        self.assertGreater(plan.mono_est_ttft_ms, 0)
        self.assertGreater(plan.disagg_est_ttft_ms, 0)

    def test_unknown_model_does_not_crash(self):
        plan = plan_capacity(
            "nonexistent/fake-model-999B",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="t4",
        )
        self.assertIn(plan.confidence, ["interpolated", "extrapolated", "low"])
        self.assertGreater(len(plan.reasoning), 0)


class TestFindNearestBaselines(unittest.TestCase):

    def test_finds_neighbors_for_2b(self):
        nearest = _find_nearest_baselines(2.0, "t4", is_moe=False)
        self.assertGreaterEqual(len(nearest), 1)
        self.assertLessEqual(len(nearest), 2)

    def test_brackets_target_in_range(self):
        """Target between measured points should return bracket pair."""
        nearest = _find_nearest_baselines(2.0, "t4", is_moe=False)
        self.assertEqual(len(nearest), 2)
        lo_b = nearest[0][0]["params_b"]
        hi_b = nearest[1][0]["params_b"]
        self.assertLessEqual(lo_b, 2.0)
        self.assertGreaterEqual(hi_b, 2.0)

    def test_below_range_returns_two_nearest(self):
        nearest = _find_nearest_baselines(0.1, "t4", is_moe=False)
        self.assertEqual(len(nearest), 2)

    def test_moe_filter(self):
        nearest = _find_nearest_baselines(7.0, "t4", is_moe=True)
        self.assertEqual(len(nearest), 1)
        self.assertTrue(nearest[0][0]["is_moe"])

    def test_no_match_wrong_gpu(self):
        nearest = _find_nearest_baselines(1.0, "nonexistent_gpu", is_moe=False)
        self.assertEqual(len(nearest), 0)


class TestGenerateRecommendation(unittest.TestCase):

    def test_pre_experiment_monolithic(self):
        """Without experiment data, always recommends MONOLITHIC (c=1 estimate)."""
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            mono_total_gpus=2, mono_est_throughput=1.0, mono_est_ttft_ms=100,
            disagg_total_gpus=4, disagg_est_throughput=1.0, disagg_est_ttft_ms=150,
        )
        _generate_recommendation(plan)
        self.assertIn("MONOLITHIC", plan.recommendation)
        self.assertIn("c=1", plan.recommendation)

    def test_experiment_disagg_requires_p90(self):
        """Even with fewer disagg GPUs, without experiment data cannot recommend DISAGGREGATE."""
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            mono_total_gpus=9, mono_est_throughput=1.0, mono_est_ttft_ms=112,
            disagg_total_gpus=7, disagg_est_throughput=1.0, disagg_est_ttft_ms=161,
        )
        _generate_recommendation(plan)
        self.assertNotEqual(plan.recommendation, "DISAGGREGATE")


class TestSavePlan(unittest.TestCase):

    def test_saves_valid_json(self):
        plan = CapacityPlan(
            model="test/model", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
        )
        plan.recommendation = "MONOLITHIC"
        plan.reasoning = ["test reason"]

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub" / "plan.json"
            save_plan(plan, path)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
            self.assertEqual(data["model"], "test/model")
            self.assertEqual(data["recommendation"], "MONOLITHIC")
            self.assertIn("mono", data)
            self.assertIn("disagg", data)


class TestMeasuredBaselines(unittest.TestCase):

    def test_all_baselines_have_required_keys(self):
        required = {"params_b", "is_moe", "mono_ttft_ms", "disagg_ttft_ms",
                     "mono_throughput", "disagg_throughput"}
        rate_fields = {"mono_ttft_base_ms", "mono_ttft_rate",
                       "disagg_ttft_base_ms", "disagg_ttft_rate", "ref_seq_len"}
        for key, data in MEASURED_BASELINES.items():
            for rk in required:
                self.assertIn(rk, data, f"{key} missing {rk}")
            has_any_rate = rate_fields & set(data.keys())
            if has_any_rate:
                for rf in rate_fields:
                    self.assertIn(rf, data,
                                  f"{key} has partial rate fields — missing {rf}")

    def test_eight_models(self):
        self.assertEqual(len(MEASURED_BASELINES), 9)


# ── Seq-len-aware TTFT ────────────────────────────────────────────────────

class TestBaselineTtft(unittest.TestCase):

    def test_with_rates(self):
        baseline = {"mono_ttft_ms": 770, "mono_ttft_base_ms": 560,
                     "mono_ttft_rate": 1.70}
        self.assertAlmostEqual(_baseline_ttft(baseline, 100), 730.0)
        self.assertAlmostEqual(_baseline_ttft(baseline, 1000), 2260.0)

    def test_without_rates(self):
        baseline = {"mono_ttft_ms": 73}
        self.assertEqual(_baseline_ttft(baseline, 100), 73)
        self.assertEqual(_baseline_ttft(baseline, 1000), 73)

    def test_disagg_field(self):
        baseline = {"disagg_ttft_ms": 1022, "disagg_ttft_base_ms": 636,
                     "disagg_ttft_rate": 3.25}
        self.assertAlmostEqual(_baseline_ttft(baseline, 100, "disagg"), 961.0)


class TestSeqLenScaling(unittest.TestCase):

    def test_phi3_mini_scales_with_seq_len(self):
        plan_short = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=50,
        )
        plan_long = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=1000,
        )
        self.assertLess(plan_short.mono_est_ttft_ms, 700)
        self.assertGreater(plan_long.mono_est_ttft_ms, 2000)
        self.assertLess(plan_short.disagg_est_ttft_ms, 900)
        self.assertGreater(plan_long.disagg_est_ttft_ms, 3000)

    def test_constant_baseline_unchanged_by_seq_len(self):
        plan_50 = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500,
            gpu_type="t4", seq_len=50,
        )
        plan_1000 = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500,
            gpu_type="t4", seq_len=1000,
        )
        self.assertEqual(plan_50.mono_est_ttft_ms, 73)
        self.assertEqual(plan_1000.mono_est_ttft_ms, 73)
        self.assertEqual(plan_50.disagg_est_ttft_ms, 109)
        self.assertEqual(plan_1000.disagg_est_ttft_ms, 109)

    def test_seq_len_noted_in_reasoning(self):
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=500,
        )
        reasoning = " ".join(plan.reasoning)
        self.assertIn("500 tokens", reasoning)

    def test_domain_warning_without_rates(self):
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500,
            gpu_type="t4", seq_len=500,
        )
        reasoning = " ".join(plan.reasoning)
        self.assertIn("may diverge", reasoning)

    def test_no_warning_near_ref_seq_len(self):
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500,
            gpu_type="t4", seq_len=100,
        )
        reasoning = " ".join(plan.reasoning)
        self.assertNotIn("may diverge", reasoning)


# ── KV bytes estimation ──────────────────────────────────────────────────

class TestEstimateKvBytes(unittest.TestCase):

    def test_known_model_lookup(self):
        p = ModelProfile(model_id="microsoft/Phi-3.5-mini-instruct")
        result = _estimate_kv_bytes(p)
        self.assertEqual(result, KV_BYTES_PER_TOKEN["microsoft/Phi-3.5-mini-instruct"])
        self.assertEqual(result, 2 * 32 * 32 * 96 * 2)

    def test_computed_from_profile(self):
        p = ModelProfile(
            model_id="unknown/model",
            num_layers=40, num_kv_heads=8, head_dim=128, torch_dtype="bfloat16",
        )
        result = _estimate_kv_bytes(p)
        self.assertEqual(result, 2 * 40 * 8 * 128 * 2)

    def test_fallback_to_reference(self):
        p = ModelProfile(model_id="unknown/bare-model")
        result = _estimate_kv_bytes(p)
        self.assertEqual(result, KV_BYTES_PER_TOKEN["microsoft/Phi-3.5-mini-instruct"])


# ── NIXL transfer estimation ────────────────────────────────────────────

class TestEstimateNixlMs(unittest.TestCase):

    def test_zero_seq_len_returns_protocol_only(self):
        ms = _estimate_nixl_ms(393_216, seq_len=0)
        self.assertAlmostEqual(ms, NIXL_PROTOCOL_MS)

    def test_seq_len_increases_transfer_time(self):
        ms_short = _estimate_nixl_ms(393_216, seq_len=10)
        ms_long = _estimate_nixl_ms(393_216, seq_len=1000)
        self.assertGreater(ms_long, ms_short)

    def test_data_term_scales_linearly_with_seq_len(self):
        ms_128 = _estimate_nixl_ms(393_216, seq_len=128)
        ms_256 = _estimate_nixl_ms(393_216, seq_len=256)
        data_128 = ms_128 - NIXL_PROTOCOL_MS
        data_256 = ms_256 - NIXL_PROTOCOL_MS
        self.assertAlmostEqual(data_256 / data_128, 2.0, places=3)

    def test_data_term_scales_with_kv_bytes(self):
        ms_small = _estimate_nixl_ms(14_336, seq_len=1000)
        ms_large = _estimate_nixl_ms(393_216, seq_len=1000)
        data_small = ms_small - NIXL_PROTOCOL_MS
        data_large = ms_large - NIXL_PROTOCOL_MS
        self.assertAlmostEqual(data_large / data_small, 393_216 / 14_336, places=1)

    def test_moe_correction_on_data_term(self):
        """MOE correction scales data term only, not protocol overhead."""
        ms_base = _estimate_nixl_ms(65_536, seq_len=100)
        ms_moe = _estimate_nixl_ms(65_536, seq_len=100, is_moe=True)
        data_base = ms_base - NIXL_PROTOCOL_MS
        data_moe = ms_moe - NIXL_PROTOCOL_MS
        self.assertAlmostEqual(data_moe / data_base, MOE_NIXL_CORRECTION, places=2)

    def test_h100_faster_nic(self):
        ms_t4 = _estimate_nixl_ms(393_216, seq_len=1000, gpu_type="t4")
        ms_h100 = _estimate_nixl_ms(393_216, seq_len=1000, gpu_type="h100")
        self.assertLess(ms_h100, ms_t4)
        data_t4 = ms_t4 - NIXL_PROTOCOL_MS
        data_h100 = ms_h100 - NIXL_PROTOCOL_MS
        self.assertAlmostEqual(data_t4 / data_h100, 400 / 25, places=0)

    def test_protocol_dominates_at_short_seq(self):
        """For small-KV models at L=1, protocol overhead > data term."""
        ms = _estimate_nixl_ms(14_336, seq_len=1)
        data_ms = ms - NIXL_PROTOCOL_MS
        self.assertGreater(NIXL_PROTOCOL_MS, data_ms)

    def test_block_alignment(self):
        """L=15 and L=16 both round to 16-token block; L=17 rounds to 32."""
        ms_15 = _estimate_nixl_ms(393_216, seq_len=15)
        ms_16 = _estimate_nixl_ms(393_216, seq_len=16)
        ms_17 = _estimate_nixl_ms(393_216, seq_len=17)
        self.assertEqual(ms_15, ms_16)
        self.assertGreater(ms_17, ms_16)

    def test_gqa_visible_at_long_seq(self):
        """At L=1000, high-KV model should be much slower than low-KV model."""
        ms_qwen = _estimate_nixl_ms(14_336, seq_len=1000)
        ms_phi3 = _estimate_nixl_ms(393_216, seq_len=1000)
        self.assertGreater(ms_phi3 - ms_qwen, 50)


# ── Recommendation branches ─────────────────────────────────────────────

class TestGenerateRecommendationBranches(unittest.TestCase):

    def test_pre_experiment_always_mono(self):
        """Without experiment data, recommendation is always MONOLITHIC (c=1)."""
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=200,
            mono_total_gpus=2, mono_est_throughput=1.0, mono_est_ttft_ms=250,
            disagg_total_gpus=3, disagg_est_throughput=1.0, disagg_est_ttft_ms=180,
        )
        _generate_recommendation(plan)
        self.assertIn("MONOLITHIC", plan.recommendation)
        self.assertIn("c=1", plan.recommendation)

    def test_pre_experiment_prescribes_experiments(self):
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            mono_total_gpus=3, mono_est_throughput=1.0, mono_est_ttft_ms=200,
            disagg_total_gpus=3, disagg_est_throughput=1.0, disagg_est_ttft_ms=210,
        )
        _generate_recommendation(plan)
        reasoning = " ".join(plan.reasoning)
        self.assertIn("run experiments", reasoning.lower())

    def test_experiment_no_crossover(self):
        """With experiment data but no crossover → MONOLITHIC."""
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            confidence="experiment",
        )
        _generate_recommendation(plan)
        self.assertEqual(plan.recommendation, "MONOLITHIC")

    def test_experiment_p90_crossover(self):
        """With p90 crossover and p50 crossover → DISAGGREGATE."""
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            confidence="experiment",
            crossovers=[{
                "seq_len": 1000, "concurrency": 8, "config": "DISAGG-1D",
                "p50_delta_pct": -10.0, "p90_delta_pct": -5.0,
                "p50_cross": True, "p90_cross": True,
                "mono_cv": 0.02, "disagg_cv": 0.05,
                "alpha_mono": 5.67, "alpha_disagg": 2.95,
                "contention_ratio": 1.92, "threshold": 1.72,
                "contention_ratio_p90": 1.80, "threshold_p90": 1.74,
            }],
        )
        _generate_recommendation(plan)
        self.assertEqual(plan.recommendation, "DISAGGREGATE")
        reasoning = " ".join(plan.reasoning)
        self.assertIn("p50 AND p90", reasoning)
        self.assertIn("R=1.92", reasoning)
        self.assertIn("T=1.72", reasoning)

    def test_experiment_p90_only_crossover(self):
        """p90 crosses but p50 does not — still DISAGGREGATE, correct text."""
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            confidence="experiment",
            crossovers=[{
                "seq_len": 1000, "concurrency": 8, "config": "DISAGG-1D",
                "p50_delta_pct": 3.0, "p90_delta_pct": -8.0,
                "p50_cross": False, "p90_cross": True,
                "mono_cv": 0.15, "disagg_cv": 0.04,
                "alpha_mono": 4.0, "alpha_disagg": 4.2,
                "contention_ratio": 0.95, "threshold": 1.50,
                "contention_ratio_p90": 1.80, "threshold_p90": 1.60,
            }],
        )
        _generate_recommendation(plan)
        self.assertEqual(plan.recommendation, "DISAGGREGATE")
        reasoning = " ".join(plan.reasoning)
        self.assertNotIn("p50 AND p90", reasoning)
        self.assertIn("R_p90=1.80", reasoning)
        self.assertIn("T_p90=1.60", reasoning)

    def test_experiment_p50_only_crossover(self):
        """With p50 crossover but not p90 → MONOLITHIC with R/T explanation."""
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            confidence="experiment",
            crossovers=[{
                "seq_len": 1000, "concurrency": 8, "config": "DISAGG-1D",
                "p50_delta_pct": -10.0, "p90_delta_pct": 14.0,
                "p50_cross": True, "p90_cross": False,
                "mono_cv": 0.02, "disagg_cv": 0.39,
                "alpha_mono": 5.67, "alpha_disagg": 2.95,
                "contention_ratio": 1.92, "threshold": 1.72,
                "contention_ratio_p90": 1.53, "threshold_p90": 1.74,
            }],
        )
        _generate_recommendation(plan)
        self.assertEqual(plan.recommendation, "MONOLITHIC")
        reasoning = " ".join(plan.reasoning)
        self.assertIn("R=1.92", reasoning)
        self.assertIn("T=1.72", reasoning)
        self.assertIn("R=1.53", reasoning)
        self.assertIn("T=1.74", reasoning)
        self.assertIn("variance", reasoning.lower())


# ── BW scaling ───────────────────────────────────────────────────────────

class TestBwScaling(unittest.TestCase):

    def test_h100_faster_than_t4(self):
        plan_t4 = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="t4",
        )
        plan_h100 = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="h100",
        )
        self.assertLess(plan_h100.mono_est_ttft_ms, plan_t4.mono_est_ttft_ms)

    def test_nixl_uses_nic_not_hbm(self):
        """NIXL transfers go over NIC, not HBM. Verify different scaling ratios."""
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="h100",
        )
        reasoning = " ".join(plan.reasoning)
        self.assertIn("NIC scaling", reasoning)
        self.assertIn("memory BW", reasoning)

    def test_extrapolation_outside_range(self):
        """Model outside measured range falls back to proportional scaling."""
        plan = plan_capacity(
            "nonexistent/fake-20B-model",
            target_throughput=1.0, target_ttft_ms=2000, gpu_type="t4",
        )
        self.assertGreater(plan.mono_est_ttft_ms, 0)
        self.assertIn(plan.confidence, ["interpolated", "low"])


# ── TP correction ──────────────────────────────────────────────────────

class TestTpCorrection(unittest.TestCase):

    def test_interpolation_uses_2b_bracket(self):
        """Target 2.0B should interpolate between 1.7B and 3.0B baselines."""
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="t4",
        )
        reasoning = " ".join(plan.reasoning)
        self.assertNotIn("TP=", reasoning)


# ── Validation gates ──────────────────────────────────────────────────

class TestValidateProfile(unittest.TestCase):

    def test_warning_on_missing_params(self):
        profile = ModelProfile(model_id="nonexistent/model")
        issues = _validate_profile(profile)
        self.assertTrue(any("WARNING" in i for i in issues))
        self.assertTrue(any("params" in i for i in issues))

    def test_warning_on_missing_kv_heads(self):
        profile = ModelProfile(model_id="x", num_params=1_000_000_000)
        issues = _validate_profile(profile)
        self.assertTrue(any("KV head" in i for i in issues))

    def test_no_warning_on_complete_profile(self):
        profile = ModelProfile(
            model_id="x", num_params=1_000_000_000, num_kv_heads=32,
        )
        issues = _validate_profile(profile)
        self.assertEqual(len(issues), 0)

    @patch("plan.fetch_model_profile")
    def test_profile_warning_in_plan(self, mock_fetch):
        mock_fetch.return_value = ModelProfile(model_id="fake/empty-model")
        plan = plan_capacity("fake/empty-model", target_throughput=1.0, gpu_type="t4")
        reasoning = " ".join(plan.reasoning)
        self.assertIn("WARNING", reasoning)


class TestVramFeasibility(unittest.TestCase):

    def test_vram_overflow_warns(self):
        plan = CapacityPlan(
            model="big/model", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
        )
        _check_vram_feasibility(
            weight_gb=30.0, kv_bytes_per_token=393_216,
            seq_len=128, gpus_per_instance=1, gpu_vram=16, plan=plan,
        )
        reasoning = " ".join(plan.reasoning)
        self.assertIn("VRAM WARNING", reasoning)

    def test_vram_ok_when_model_fits(self):
        plan = CapacityPlan(
            model="small/model", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
        )
        _check_vram_feasibility(
            weight_gb=3.0, kv_bytes_per_token=14_336,
            seq_len=128, gpus_per_instance=1, gpu_vram=16, plan=plan,
        )
        reasoning = " ".join(plan.reasoning)
        self.assertNotIn("VRAM WARNING", reasoning)

    def test_pp_reduces_per_gpu_weight(self):
        """With PP=2, TP=8 (16 GPUs), 130 GB model fits on T4."""
        plan = CapacityPlan(
            model="big/model", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
        )
        _check_vram_feasibility(
            weight_gb=130.0, kv_bytes_per_token=327_680,
            seq_len=128, gpus_per_instance=16, gpu_vram=16, plan=plan,
        )
        reasoning = " ".join(plan.reasoning)
        self.assertNotIn("VRAM WARNING", reasoning)

    def test_skipped_when_weight_zero(self):
        """VRAM check is skipped when profile fetch failed (weight_gb=0)."""
        from unittest.mock import patch as _p
        with _p("plan.fetch_model_profile") as mock:
            mock.return_value = ModelProfile(model_id="fake/unknown")
            plan = plan_capacity("fake/unknown", target_throughput=1.0, gpu_type="t4")
            self.assertNotIn("VRAM WARNING", " ".join(plan.reasoning))
            self.assertIn("WARNING", " ".join(plan.reasoning))


class TestExtrapolationConfidence(unittest.TestCase):

    @patch("plan.fetch_model_profile")
    def test_confidence_degrades_at_5x(self, mock_fetch):
        mock_fetch.return_value = ModelProfile(
            model_id="fake/50B-model", num_params=50_000_000_000,
            num_kv_heads=32, head_dim=128, num_layers=80,
            torch_dtype="bfloat16",
        )
        plan = plan_capacity(
            "fake/50B-model", target_throughput=1.0,
            target_ttft_ms=5000, gpu_type="t4",
        )
        self.assertEqual(plan.confidence, "low")
        reasoning = " ".join(plan.reasoning)
        self.assertIn("LOW CONFIDENCE", reasoning)

    def test_small_extrapolation_not_low(self):
        """2B model on T4 should interpolate, not trigger low confidence."""
        plan = plan_capacity(
            "Qwen/Qwen2.5-1.5B-Instruct",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="t4",
        )
        self.assertEqual(plan.confidence, "measured")


# ── Prefill capacity ──────────────────────────────────────────────────

class TestPrefillCapacity(unittest.TestCase):

    def test_slow_prefill_gets_more_pods(self):
        """OLMoE (mono_ttft=2999ms) needs 3 prefill pods at target=1.0 req/s."""
        plan = plan_capacity(
            "allenai/OLMoE-1B-7B-0924-Instruct",
            target_throughput=1.0, target_ttft_ms=5000, gpu_type="t4",
        )
        prefill_capacity = plan.disagg_prefill_gpus * (1000 / 2999)
        self.assertGreaterEqual(prefill_capacity, 1.0)

    def test_fast_prefill_needs_one_pod(self):
        """TinyLlama (mono_ttft=73ms) needs only 1 prefill pod at target=1.0."""
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="t4",
        )
        self.assertEqual(plan.disagg_prefill_gpus, 1)

    def test_high_throughput_scales_prefill(self):
        """At target=10 req/s, Phi-3.5 (173ms prefill) needs 2 prefill pods."""
        plan = plan_capacity(
            "microsoft/Phi-3.5-mini-instruct",
            target_throughput=10.0, target_ttft_ms=500, gpu_type="t4",
        )
        self.assertGreater(plan.disagg_prefill_gpus, 1)

    def test_extrapolation_disagg_bias_noted(self):
        """Extrapolation adds reasoning note about disagg using more GPUs."""
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="h100",
        )
        reasoning = " ".join(plan.reasoning)
        self.assertIn("measured disagg throughput", reasoning)


# ── TP throughput scaling ────────────────────────────────────────────────

class TestTpThroughputScaling(unittest.TestCase):

    @patch("plan.fetch_model_profile")
    def test_tp_reduces_gpu_count(self, mock_fetch):
        """Model needing TP=3 should have per-instance throughput scaled by TP."""
        mock_fetch.return_value = ModelProfile(
            model_id="fake/16B", num_params=16_000_000_000,
            num_kv_heads=32, head_dim=128, num_layers=40,
            torch_dtype="bfloat16",
        )
        plan = plan_capacity(
            "fake/16B", target_throughput=1.0,
            target_ttft_ms=5000, gpu_type="t4",
        )
        tp = plan.mono_gpus_per_instance
        self.assertGreater(tp, 1)
        self.assertLess(plan.mono_total_gpus, tp * 50)

    @patch("plan.fetch_model_profile")
    def test_tp1_throughput_unchanged(self, mock_fetch):
        """TP=1 should not change throughput vs baseline scaling."""
        mock_fetch.return_value = ModelProfile(
            model_id="fake/2B", num_params=2_000_000_000,
            num_kv_heads=16, head_dim=64, num_layers=24,
            torch_dtype="bfloat16",
        )
        plan = plan_capacity(
            "fake/2B", target_throughput=1.0,
            target_ttft_ms=5000, gpu_type="t4",
        )
        self.assertEqual(plan.mono_gpus_per_instance, 1)


# ── Confidence labels ───────────────────────────────────────────────────

class TestConfidenceLabels(unittest.TestCase):

    def test_interpolated_on_same_gpu(self):
        """In-range model on same GPU type → 'interpolated'."""
        plan = plan_capacity(
            "nonexistent/model", target_throughput=1.0, gpu_type="t4",
        )
        self.assertEqual(plan.confidence, "interpolated")

    @patch("plan.fetch_model_profile")
    def test_extrapolated_cross_gpu(self, mock_fetch):
        """T4 baselines used for H100 → 'extrapolated'."""
        mock_fetch.return_value = ModelProfile(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            num_params=1_100_000_000, num_kv_heads=4, head_dim=64,
            num_layers=22, torch_dtype="float16",
        )
        plan = plan_capacity(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target_throughput=1.0, target_ttft_ms=500, gpu_type="h100",
        )
        self.assertEqual(plan.confidence, "extrapolated")

    @patch("plan.fetch_model_profile")
    def test_extrapolated_outside_range(self, mock_fetch):
        """Model outside measured range on same GPU → 'extrapolated'."""
        mock_fetch.return_value = ModelProfile(
            model_id="fake/5B", num_params=5_000_000_000,
            num_kv_heads=32, head_dim=128, num_layers=40,
            torch_dtype="bfloat16",
        )
        plan = plan_capacity(
            "fake/5B", target_throughput=1.0,
            target_ttft_ms=5000, gpu_type="t4",
        )
        self.assertEqual(plan.confidence, "extrapolated")

    def test_run_experiments_always_present(self):
        """All extrapolated plans include 'Run experiments' reasoning."""
        plan = plan_capacity(
            "nonexistent/model", target_throughput=1.0, gpu_type="t4",
        )
        reasoning = " ".join(plan.reasoning)
        self.assertIn("Run experiments for empirical validation", reasoning)


# ── Experiment-aware planning ──────────────────────────────────────────

EXP11_FIELDS = [
    "experiment", "config", "prompt_tokens_target", "max_tokens",
    "concurrency", "run", "ttft_ms", "total_ms", "status_code",
    "prompt_tokens_actual", "completion_tokens", "target", "error",
]


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


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


class TestPlanFromExperiments(unittest.TestCase):

    def _make_data_dir(self, rows):
        d = tempfile.mkdtemp()
        _write_csv(os.path.join(d, "exp11-results.csv"), EXP11_FIELDS, rows)
        return d

    def test_strong_crossover(self):
        """Disagg wins at BOTH p50 AND p90 → DISAGGREGATE with T/R values."""
        rows = (
            _make_exp11_rows("BASELINE", 1, 1000, [2300] * 24)
            + _make_exp11_rows("DISAGG-1D", 1, 1000, [2800] * 24)
            + _make_exp11_rows("BASELINE", 8, 1000, [13000 + i * 10 for i in range(24)])
            + _make_exp11_rows("DISAGG-1D", 8, 1000, [9000 + i * 10 for i in range(24)])
        )
        d = self._make_data_dir(rows)
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=1000, data_dir=d,
        )
        self.assertEqual(plan.confidence, "experiment")
        self.assertTrue(any(c["p50_cross"] and c["p90_cross"] for c in plan.crossovers))
        self.assertEqual(plan.recommendation, "DISAGGREGATE")

        cross = plan.crossovers[0]
        self.assertAlmostEqual(cross["threshold"], 2800 / 2300, places=1)
        self.assertGreater(cross["alpha_mono"], cross["alpha_disagg"])
        self.assertGreater(cross["contention_ratio"], cross["threshold"])

    def test_weak_crossover(self):
        """Disagg wins at p50 but loses at p90 → MONOLITHIC."""
        mono_c8 = [13000 + i * 10 for i in range(24)]
        disagg_c8 = ([10000 + i * 5 for i in range(12)]
                     + [15000 + i * 5 for i in range(12)])
        rows = (
            _make_exp11_rows("BASELINE", 1, 1000, [2300] * 24)
            + _make_exp11_rows("DISAGG-1D", 1, 1000, [2800] * 24)
            + _make_exp11_rows("BASELINE", 8, 1000, mono_c8)
            + _make_exp11_rows("DISAGG-1D", 8, 1000, disagg_c8)
        )
        d = self._make_data_dir(rows)
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=1000, data_dir=d,
        )
        self.assertEqual(plan.confidence, "experiment")
        p50_only = [c for c in plan.crossovers
                    if c["p50_cross"] and not c["p90_cross"]]
        self.assertTrue(len(p50_only) > 0)
        self.assertEqual(plan.recommendation, "MONOLITHIC")
        reasoning = " ".join(plan.reasoning)
        self.assertIn("p90", reasoning)
        self.assertIn("variance", reasoning.lower())

    def test_no_crossover(self):
        """Mono wins everywhere → MONOLITHIC, no crossovers."""
        rows = (
            _make_exp11_rows("BASELINE", 1, 500, [1200] * 24)
            + _make_exp11_rows("DISAGG-1D", 1, 500, [1800] * 24)
            + _make_exp11_rows("BASELINE", 8, 500, [5000] * 24)
            + _make_exp11_rows("DISAGG-1D", 8, 500, [7000] * 24)
        )
        d = self._make_data_dir(rows)
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=500, data_dir=d,
        )
        self.assertEqual(plan.confidence, "experiment")
        self.assertEqual(plan.recommendation, "MONOLITHIC")
        p50_crosses = [c for c in plan.crossovers if c["p50_cross"]]
        self.assertEqual(len(p50_crosses), 0)

    def test_c1_excluded_from_crossover_scan(self):
        """Even if disagg beats mono at c=1 (noise), no crossover recorded."""
        rows = (
            _make_exp11_rows("BASELINE", 1, 500, [1500] * 24)
            + _make_exp11_rows("DISAGG-1D", 1, 500, [1400] * 24)
        )
        d = self._make_data_dir(rows)
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=500, data_dir=d,
        )
        self.assertEqual(len(plan.crossovers), 0)
        self.assertEqual(plan.recommendation, "MONOLITHIC")

    def test_c1_only_data_warns(self):
        """With only c=1 data, warn that crossovers require concurrency."""
        rows = (
            _make_exp11_rows("BASELINE", 1, 500, [1200] * 24)
            + _make_exp11_rows("DISAGG-1D", 1, 500, [1800] * 24)
        )
        d = self._make_data_dir(rows)
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=500, data_dir=d,
        )
        reasoning = " ".join(plan.reasoning)
        self.assertIn("c=1", reasoning.lower())
        self.assertIn("concurrent", reasoning.lower())


class TestOverheadThresholds(unittest.TestCase):

    def test_thresholds_computed_per_seq_len(self):
        """T(s) = disagg_c1 / mono_c1 computed for each measured seq_len."""
        rows = (
            _make_exp11_rows("BASELINE", 1, 100, [700] * 24)
            + _make_exp11_rows("DISAGG-1D", 1, 100, [1000] * 24)
            + _make_exp11_rows("BASELINE", 1, 500, [1300] * 24)
            + _make_exp11_rows("DISAGG-1D", 1, 500, [1800] * 24)
        )
        d = tempfile.mkdtemp()
        _write_csv(os.path.join(d, "exp11-results.csv"), EXP11_FIELDS, rows)
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=100, data_dir=d,
        )
        self.assertEqual(len(plan.overhead_thresholds), 2)
        t100 = next(t for t in plan.overhead_thresholds if t["seq_len"] == 100)
        t500 = next(t for t in plan.overhead_thresholds if t["seq_len"] == 500)
        self.assertAlmostEqual(t100["threshold"], 1000 / 700, places=1)
        self.assertAlmostEqual(t500["threshold"], 1800 / 1300, places=1)
        self.assertEqual(t100["overhead_pct"], round((1000 / 700 - 1) * 100))
        self.assertEqual(t500["overhead_pct"], round((1800 / 1300 - 1) * 100))

    def test_contention_ratio_exceeds_threshold_at_crossover(self):
        """At crossover points, R > T (by definition)."""
        rows = (
            _make_exp11_rows("BASELINE", 1, 1000, [2000] * 24)
            + _make_exp11_rows("DISAGG-1D", 1, 1000, [3000] * 24)
            + _make_exp11_rows("BASELINE", 8, 1000, [14000] * 24)
            + _make_exp11_rows("DISAGG-1D", 8, 1000, [10000] * 24)
        )
        d = tempfile.mkdtemp()
        _write_csv(os.path.join(d, "exp11-results.csv"), EXP11_FIELDS, rows)
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=1000, data_dir=d,
        )
        cross = plan.crossovers[0]
        self.assertAlmostEqual(cross["alpha_mono"], 14000 / 2000, places=1)
        self.assertAlmostEqual(cross["alpha_disagg"], 10000 / 3000, places=1)
        self.assertGreater(cross["contention_ratio"], cross["threshold"])


class TestPreExperimentHonesty(unittest.TestCase):

    def test_no_data_dir_gives_c1_estimate(self):
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=500,
        )
        self.assertIn("c=1", plan.recommendation)

    def test_no_data_dir_prescribes_experiments(self):
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=500,
        )
        reasoning = " ".join(plan.reasoning)
        self.assertIn("run experiments", reasoning.lower())

    def test_no_data_dir_reports_contention_threshold(self):
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=500,
        )
        reasoning = " ".join(plan.reasoning)
        self.assertIn("contention advantage", reasoning.lower())

    def test_never_recommends_disagg_without_data(self):
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=1000,
        )
        self.assertNotEqual(plan.recommendation, "DISAGGREGATE")


class TestNearestSeqLen(unittest.TestCase):

    def test_uses_nearest_when_exact_unavailable(self):
        rows = (
            _make_exp11_rows("BASELINE", 1, 100, [700] * 24)
            + _make_exp11_rows("DISAGG-1D", 1, 100, [1000] * 24)
            + _make_exp11_rows("BASELINE", 1, 500, [1300] * 24)
            + _make_exp11_rows("DISAGG-1D", 1, 500, [1800] * 24)
        )
        d = tempfile.mkdtemp()
        _write_csv(os.path.join(d, "exp11-results.csv"), EXP11_FIELDS, rows)
        plan = plan_capacity(
            "microsoft/Phi-3-mini-4k-instruct",
            target_throughput=1.0, target_ttft_ms=99999,
            gpu_type="t4", seq_len=300, data_dir=d,
        )
        self.assertEqual(plan.confidence, "experiment")
        reasoning = " ".join(plan.reasoning)
        self.assertIn("nearest", reasoning.lower())
        self.assertTrue(
            plan.mono_est_ttft_ms in (700, 1300),
            f"Expected TTFT from nearest measured seq_len, got {plan.mono_est_ttft_ms}")


if __name__ == "__main__":
    unittest.main()
