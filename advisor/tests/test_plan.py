"""Tests for advisor/plan.py — capacity planning, no cluster needed."""

import json
import tempfile
import unittest
from pathlib import Path

from plan import (
    MEASURED_BASELINES,
    CapacityPlan,
    ModelProfile,
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

    def test_moe_filter(self):
        nearest = _find_nearest_baselines(7.0, "t4", is_moe=True)
        self.assertEqual(len(nearest), 1)
        self.assertTrue(nearest[0]["is_moe"])

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


if __name__ == "__main__":
    unittest.main()
