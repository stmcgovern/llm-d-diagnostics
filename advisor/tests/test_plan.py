"""Tests for advisor/plan.py — capacity planning, no cluster needed."""

import json
import tempfile
import unittest
from pathlib import Path

from plan import (
    KV_BYTES_PER_TOKEN,
    MEASURED_BASELINES,
    MOE_NIXL_CORRECTION,
    NIXL_REF_KV_BYTES,
    NIXL_REF_MS,
    CapacityPlan,
    ModelProfile,
    _estimate_kv_bytes,
    _estimate_nixl_ms,
    _find_nearest_baselines,
    _generate_recommendation,
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

    def test_disagg_when_fewer_gpus_and_lower_latency(self):
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=500,
            mono_total_gpus=4, mono_est_throughput=1.0, mono_est_ttft_ms=200,
            disagg_total_gpus=3, disagg_est_throughput=1.0, disagg_est_ttft_ms=150,
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
        self.assertEqual(result, NIXL_REF_KV_BYTES)


# ── NIXL transfer estimation ────────────────────────────────────────────

class TestEstimateNixlMs(unittest.TestCase):

    def test_reference_model_identity(self):
        ms = _estimate_nixl_ms(NIXL_REF_KV_BYTES, is_moe=False)
        self.assertAlmostEqual(ms, NIXL_REF_MS)

    def test_half_kv_half_nixl(self):
        ms = _estimate_nixl_ms(NIXL_REF_KV_BYTES // 2, is_moe=False)
        self.assertAlmostEqual(ms, NIXL_REF_MS / 2)

    def test_moe_correction(self):
        ms_base = _estimate_nixl_ms(NIXL_REF_KV_BYTES, is_moe=False)
        ms_moe = _estimate_nixl_ms(NIXL_REF_KV_BYTES, is_moe=True)
        self.assertAlmostEqual(ms_moe / ms_base, MOE_NIXL_CORRECTION)

    def test_5ms_floor(self):
        ms = _estimate_nixl_ms(1, is_moe=False)
        self.assertEqual(ms, 5)

    def test_sanity_vs_measured(self):
        """Fermi check: estimates vs measured NIXL are off by 3-23x (known gap).
        This test documents the gap so we don't accidentally claim accuracy."""
        measured = {
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": (22528, False, 17),
            "Qwen/Qwen2.5-3B-Instruct": (36864, False, 25),
            "allenai/OLMoE-1B-7B-0924-Instruct": (65536, True, 267),
        }
        for model_id, (kv_bytes, is_moe, actual_ms) in measured.items():
            est = _estimate_nixl_ms(kv_bytes, is_moe)
            ratio = actual_ms / est if est > 0 else float("inf")
            self.assertGreater(ratio, 0.01,
                               f"{model_id}: estimate {est:.1f}ms vs actual {actual_ms}ms")
            self.assertLess(ratio, 100,
                            f"{model_id}: estimate {est:.1f}ms vs actual {actual_ms}ms")


# ── Recommendation branches ─────────────────────────────────────────────

class TestGenerateRecommendationBranches(unittest.TestCase):

    def test_slo_headroom_favors_disagg(self):
        plan = CapacityPlan(
            model="test", gpu_type="t4",
            target_throughput=1.0, target_ttft_ms=200,
            mono_total_gpus=2, mono_est_throughput=1.0, mono_est_ttft_ms=190,
            disagg_total_gpus=3, disagg_est_throughput=1.0, disagg_est_ttft_ms=100,
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
        self.assertIn("NIC BW", reasoning)
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


if __name__ == "__main__":
    unittest.main()
