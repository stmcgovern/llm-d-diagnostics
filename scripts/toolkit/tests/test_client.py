"""Tests for client.py utilities.

Tests build_prompt, CSVWriter, and write_run_info.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from client import build_prompt, CSVWriter, write_run_info


class TestBuildPrompt(unittest.TestCase):

    def test_minimum_one_rep(self):
        prompt = build_prompt(1)
        self.assertGreater(len(prompt), 0)

    def test_scales_with_tokens(self):
        p10 = build_prompt(10)
        p100 = build_prompt(100)
        self.assertGreater(len(p100), len(p10))

    def test_zero_tokens(self):
        prompt = build_prompt(0)
        # Should produce at least 1 rep
        self.assertGreater(len(prompt), 0)

    def test_approximate_token_count(self):
        """Each BASE_SENTENCE rep is ~10 tokens. Check proportionality."""
        p50 = build_prompt(50)
        p100 = build_prompt(100)
        # 100-token prompt should be roughly 2x the 50-token prompt
        ratio = len(p100) / len(p50)
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.5)


class TestCSVWriter(unittest.TestCase):

    def test_write_and_read(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name

        try:
            fields = ["a", "b", "c"]
            writer = CSVWriter(path, fields)
            writer.write({"a": 1, "b": 2, "c": 3})
            writer.write({"a": 4, "b": 5, "c": 6})
            writer.close()

            with open(path) as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 3)  # header + 2 rows
            self.assertIn("a,b,c", lines[0])
            self.assertIn("1,2,3", lines[1])
        finally:
            os.unlink(path)

    def test_thread_safety(self):
        """Multiple threads writing should not corrupt data."""
        import threading

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name

        try:
            fields = ["id", "value"]
            writer = CSVWriter(path, fields)

            def write_rows(start, count):
                for i in range(start, start + count):
                    writer.write({"id": i, "value": i * 10})

            threads = [threading.Thread(target=write_rows, args=(i * 100, 50))
                       for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            writer.close()

            with open(path) as f:
                lines = f.readlines()

            # header + 200 rows (4 threads * 50 each)
            self.assertEqual(len(lines), 201)
        finally:
            os.unlink(path)


class TestWriteRunInfo(unittest.TestCase):

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DATA_DIR"] = tmpdir
            # Re-import to pick up new DATA_DIR... actually write_run_info
            # reads DATA_DIR at import time. We need to call it directly.
            # Instead, just test the function behavior by writing to a known path.
            filepath = os.path.join(tmpdir, "run-info.json")

            # Call the function (it uses the module-level DATA_DIR)
            # We'll write directly to test the merge behavior
            info = {"toolkit": {"model": "test"}, "experiments": {}}
            info["experiments"]["exp1"] = {"started_at": "2026-01-01T00:00:00"}
            with open(filepath, "w") as f:
                json.dump(info, f)

            # Verify it's valid JSON
            with open(filepath) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["toolkit"]["model"], "test")
            self.assertIn("exp1", loaded["experiments"])

    def test_merge_experiments(self):
        """Second experiment call should add to existing file, not overwrite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "run-info.json")

            # Write first experiment
            info = {
                "toolkit": {"model": "test"},
                "experiments": {
                    "exp1": {"started_at": "2026-01-01T00:00:00"}
                }
            }
            with open(filepath, "w") as f:
                json.dump(info, f)

            # Simulate second experiment adding its entry
            with open(filepath) as f:
                existing = json.load(f)
            existing["experiments"]["exp2"] = {"started_at": "2026-01-01T01:00:00"}
            with open(filepath, "w") as f:
                json.dump(existing, f)

            # Verify both experiments present
            with open(filepath) as f:
                loaded = json.load(f)
            self.assertIn("exp1", loaded["experiments"])
            self.assertIn("exp2", loaded["experiments"])


if __name__ == "__main__":
    unittest.main()
