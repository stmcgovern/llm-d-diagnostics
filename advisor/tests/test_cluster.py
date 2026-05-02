"""Tests for advisor/_cluster.py — oc wrapper, mocked subprocess."""

import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _cluster import oc, oc_safe, get_pods_full


class TestOc(unittest.TestCase):

    @patch("_cluster.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="output-text\n", stderr="")
        result = oc("get", "pods")
        self.assertEqual(result, "output-text")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args, ["oc", "get", "pods"])

    @patch("_cluster.subprocess.run")
    def test_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error msg")
        with self.assertRaises(RuntimeError) as ctx:
            oc("get", "pods")
        self.assertIn("error msg", str(ctx.exception))

    @patch("_cluster.subprocess.run")
    def test_timeout_passed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        oc("get", "pods", timeout=120)
        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["timeout"], 120)


class TestOcSafe(unittest.TestCase):

    @patch("_cluster.subprocess.run")
    def test_returns_tuple(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="out", stderr="err")
        stdout, stderr = oc_safe("get", "pods")
        self.assertEqual(stdout, "out")
        self.assertEqual(stderr, "err")

    @patch("_cluster.subprocess.run")
    def test_does_not_raise_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=127, stdout="", stderr="not found")
        stdout, stderr = oc_safe("nonexistent")
        self.assertEqual(stderr, "not found")


SAMPLE_PODS_JSON = json.dumps({
    "items": [
        {
            "metadata": {
                "name": "vllm-prefill-abc",
                "labels": {"app": "vllm-prefill", "app.kubernetes.io/part-of": "vllm-disagg"},
            },
            "spec": {
                "containers": [{
                    "image": "vllm/vllm-openai:v0.18.1",
                    "args": ["--kv-transfer-config", '{"kv_role":"kv_producer"}'],
                }],
            },
            "status": {
                "podIP": "10.0.0.1",
                "conditions": [{"type": "Ready", "status": "True"}],
            },
        },
        {
            "metadata": {
                "name": "vllm-decode-xyz",
                "labels": {"app": "vllm-decode"},
            },
            "spec": {
                "containers": [{
                    "image": "vllm/vllm-openai:v0.18.1",
                    "args": [],
                }],
            },
            "status": {
                "podIP": "10.0.0.2",
                "conditions": [{"type": "Ready", "status": "False"}],
            },
        },
    ]
})


class TestGetPodsFull(unittest.TestCase):

    @patch("_cluster.oc")
    def test_parses_pods(self, mock_oc):
        mock_oc.return_value = SAMPLE_PODS_JSON
        pods = get_pods_full("test-ns", label="app.kubernetes.io/part-of=vllm-disagg")
        self.assertEqual(len(pods), 2)

        prefill = pods[0]
        self.assertEqual(prefill["name"], "vllm-prefill-abc")
        self.assertEqual(prefill["ip"], "10.0.0.1")
        self.assertTrue(prefill["ready"])
        self.assertEqual(prefill["labels"]["app"], "vllm-prefill")
        self.assertIn("vllm-openai", prefill["image"])

        decode = pods[1]
        self.assertEqual(decode["name"], "vllm-decode-xyz")
        self.assertFalse(decode["ready"])

    @patch("_cluster.oc")
    def test_label_passed_to_oc(self, mock_oc):
        mock_oc.return_value = '{"items": []}'
        get_pods_full("ns", label="app=test")
        args = mock_oc.call_args[0]
        self.assertIn("-l", args)
        self.assertIn("app=test", args)

    @patch("_cluster.oc")
    def test_no_label(self, mock_oc):
        mock_oc.return_value = '{"items": []}'
        get_pods_full("ns")
        args = mock_oc.call_args[0]
        self.assertNotIn("-l", args)


if __name__ == "__main__":
    unittest.main()
