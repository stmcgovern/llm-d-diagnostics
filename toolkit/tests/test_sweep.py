"""Tests for sweep.py: env.sh parsing, model slug, CSV validation, completeness checks."""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sweep import (
    EXPERIMENT_CSV,
    model_complete,
    model_slug,
    parse_env_sh,
    validate_csv,
    write_env_sh,
)


class TestParseEnvSh(unittest.TestCase):

    def test_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write('export MODEL="microsoft/Phi-3-mini-4k-instruct"\n')
            f.write('export NS="llm-d"\n')
            f.write('export MAX_MODEL_LEN="4096"\n')
            path = f.name

        try:
            env = parse_env_sh(path)
            self.assertEqual(env["MODEL"], "microsoft/Phi-3-mini-4k-instruct")
            self.assertEqual(env["NS"], "llm-d")
            self.assertEqual(env["MAX_MODEL_LEN"], "4096")
        finally:
            os.unlink(path)

    def test_single_quotes(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("export MODEL='some-model'\n")
            path = f.name

        try:
            env = parse_env_sh(path)
            self.assertEqual(env["MODEL"], "some-model")
        finally:
            os.unlink(path)

    def test_no_quotes(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("export NS=llm-d\n")
            path = f.name

        try:
            env = parse_env_sh(path)
            self.assertEqual(env["NS"], "llm-d")
        finally:
            os.unlink(path)

    def test_skips_comments_and_blanks(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\n")
            f.write("# This is a comment\n")
            f.write("\n")
            f.write('export KEY="value"\n')
            path = f.name

        try:
            env = parse_env_sh(path)
            self.assertEqual(len(env), 1)
            self.assertEqual(env["KEY"], "value")
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            path = f.name

        try:
            env = parse_env_sh(path)
            self.assertEqual(env, {})
        finally:
            os.unlink(path)


class TestModelSlug(unittest.TestCase):

    def test_strips_org(self):
        self.assertEqual(model_slug("microsoft/Phi-3-mini-4k-instruct"),
                         "phi-3-mini-4k-instruct")

    def test_no_org(self):
        self.assertEqual(model_slug("gpt2"), "gpt2")

    def test_lowercase(self):
        self.assertEqual(model_slug("Org/ModelName"), "modelname")


class TestModelComplete(unittest.TestCase):

    def _make_csv(self, tmpdir, filename, rows=2):
        path = os.path.join(tmpdir, filename)
        with open(path, "w") as f:
            f.write("header\n")
            for i in range(rows - 1):
                f.write(f"row{i}\n")
        return path

    def test_all_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_csv(tmpdir, "exp1-results.csv")
            self._make_csv(tmpdir, "exp1b-results.csv")
            self.assertTrue(model_complete(tmpdir, ["latency", "decompose"]))

    def test_missing_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_csv(tmpdir, "exp1-results.csv")
            self.assertFalse(model_complete(tmpdir, ["latency", "decompose"]))

    def test_empty_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_csv(tmpdir, "exp1-results.csv", rows=1)
            self.assertFalse(model_complete(tmpdir, ["latency"]))

    def test_unknown_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(model_complete(tmpdir, ["nonexistent"]))

    def test_empty_experiments_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertTrue(model_complete(tmpdir, []))


class TestValidateCsv(unittest.TestCase):

    def test_valid_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,value\n")
            f.write("a,1\n")
            path = f.name

        try:
            ok, msg = validate_csv(path)
            self.assertTrue(ok)
            self.assertEqual(msg, "")
        finally:
            os.unlink(path)

    def test_missing_file(self):
        ok, msg = validate_csv("/nonexistent/path.csv")
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_empty_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("header1,header2\n")
            path = f.name

        try:
            ok, msg = validate_csv(path)
            self.assertFalse(ok)
            self.assertIn("empty", msg)
        finally:
            os.unlink(path)

    def test_low_success_rate(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["status_code", "value"])
            writer.writeheader()
            for i in range(10):
                writer.writerow({"status_code": "200" if i < 2 else "500", "value": str(i)})
            path = f.name

        try:
            ok, msg = validate_csv(path)
            self.assertFalse(ok)
            self.assertIn("200", msg)
        finally:
            os.unlink(path)

    def test_high_success_rate(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["status_code", "value"])
            writer.writeheader()
            for i in range(10):
                writer.writerow({"status_code": "200", "value": str(i)})
            path = f.name

        try:
            ok, _msg = validate_csv(path)
            self.assertTrue(ok)
        finally:
            os.unlink(path)


class TestWriteEnvSh(unittest.TestCase):

    def test_writes_sorted_exports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "env.sh")
            write_env_sh(path, {"B": "2", "A": "1"}, {"C": "3"})

            with open(path) as f:
                content = f.read()

            self.assertIn("#!/bin/bash", content)
            lines = [l for l in content.strip().split("\n") if l.startswith("export")]
            keys = [l.split("=")[0].replace("export ", "") for l in lines]
            self.assertEqual(keys, sorted(keys))

    def test_overrides_take_precedence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "env.sh")
            write_env_sh(path, {"MODEL": "old"}, {"MODEL": "new"})

            env = parse_env_sh(path)
            self.assertEqual(env["MODEL"], "new")

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "env.sh")
            write_env_sh(path, {"KEY": "val"}, {})
            self.assertTrue(os.path.exists(path))


class TestExperimentCsvMapping(unittest.TestCase):

    def test_all_experiments_have_csv_names(self):
        expected = [
            "latency", "decompose", "throughput", "isolation", "seqlen",
            "saturation", "mixed", "prefix-cache", "kv-eviction",
            "tput-seqlen", "tput-outlen", "tput-sat", "overhead-load",
        ]
        for exp in expected:
            self.assertIn(exp, EXPERIMENT_CSV, f"Missing EXPERIMENT_CSV entry for '{exp}'")

    def test_csv_names_are_unique(self):
        values = list(EXPERIMENT_CSV.values())
        self.assertEqual(len(values), len(set(values)))


if __name__ == "__main__":
    unittest.main()
