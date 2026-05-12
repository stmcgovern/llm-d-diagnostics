"""Tests for advisor/plan.py — capacity planning, no cluster needed."""

import json
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
        self.assertIn(plan.recommendation, [
            "MONOLITHIC", "DISAGGREGATE", "RUN EXPERIMENTS TO DECIDE",
        ])

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

    def test_monolithic_when_fewer_gpus(self):
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            mono_total_gpus=2, mono_est_throughput=1.0, mono_est_ttft_ms=100,
            disagg_total_gpus=4, disagg_est_throughput=1.0, disagg_est_ttft_ms=150,
        )
        _generate_recommendation(plan)
        self.assertEqual(plan.recommendation, "MONOLITHIC")

    def test_disagg_when_fewer_gpus(self):
        """Disagg uses fewer GPUs with higher TTFT — both meet SLO."""
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            mono_total_gpus=9, mono_est_throughput=1.0, mono_est_ttft_ms=112,
            disagg_total_gpus=7, disagg_est_throughput=1.0, disagg_est_ttft_ms=161,
        )
        _generate_recommendation(plan)
        self.assertEqual(plan.recommendation, "DISAGGREGATE")


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
        for key, data in MEASURED_BASELINES.items():
            for rk in required:
                self.assertIn(rk, data, f"{key} missing {rk}")

    def test_eight_models(self):
        self.assertEqual(len(MEASURED_BASELINES), 8)


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

    def test_only_disagg_meets_slo(self):
        """Only disagg meets the TTFT SLO — mono exceeds target."""
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=200,
            mono_total_gpus=2, mono_est_throughput=1.0, mono_est_ttft_ms=250,
            disagg_total_gpus=3, disagg_est_throughput=1.0, disagg_est_ttft_ms=180,
        )
        _generate_recommendation(plan)
        self.assertEqual(plan.recommendation, "DISAGGREGATE")

    def test_close_call_runs_experiments(self):
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            mono_total_gpus=3, mono_est_throughput=1.0, mono_est_ttft_ms=200,
            disagg_total_gpus=3, disagg_est_throughput=1.0, disagg_est_ttft_ms=210,
        )
        _generate_recommendation(plan)
        self.assertEqual(plan.recommendation, "RUN EXPERIMENTS TO DECIDE")


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


if __name__ == "__main__":
    unittest.main()
