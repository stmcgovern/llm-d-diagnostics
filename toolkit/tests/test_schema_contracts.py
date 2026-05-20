"""Schema contract tests: verify consumers can read schema-conformant rows.

Each test writes a CSV using fields derived from the TypedDict schema,
passes it through the real consumer function, and asserts no KeyError.
If a schema field is renamed without updating the consumer, these tests
catch it — the round-trip tests in test_schemas.py do not.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_advisor = os.path.join(_root, "advisor")
if _advisor not in sys.path:
    sys.path.insert(0, _advisor)

from schemas import (  # noqa: E402
    Exp1bRow,
    Exp1Row,
    Exp5bRow,
    Exp5Row,
    Exp11Row,
    Exp14Row,
    fields_for,
)


def _make_typed_row(row_type, **overrides):
    """Create a row with all schema fields set to '0', then apply overrides."""
    row = {f: "0" for f in fields_for(row_type)}
    row.update(overrides)
    return row


def _write_typed_csv(path, row_type, rows):
    fieldnames = fields_for(row_type)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


class TestSchemaContracts(unittest.TestCase):
    """Verify each (schema, consumer) pair works without KeyError."""

    def test_exp11_calibrate(self):
        from calibrate import extract_baselines

        d = tempfile.mkdtemp()
        with open(os.path.join(d, "run-info.json"), "w") as f:
            json.dump({"toolkit": {"model": "test/m", "gpu_type": "t4"}}, f)

        rows = []
        for cfg in ["BASELINE", "DISAGG-1D"]:
            for i in range(5):
                rows.append(_make_typed_row(
                    Exp11Row, experiment="exp11", config=cfg,
                    status_code="200", concurrency="1",
                    prompt_tokens_target="100", ttft_ms=str(500 + i),
                    total_ms=str(510 + i), run=str(i + 1),
                    completion_tokens="20",
                ))
        _write_typed_csv(os.path.join(d, "exp11-results.csv"), Exp11Row, rows)
        result = extract_baselines(d, "t4")
        self.assertIsNotNone(result)

    def test_exp14_calibrate(self):
        from calibrate import extract_baselines

        d = tempfile.mkdtemp()
        with open(os.path.join(d, "run-info.json"), "w") as f:
            json.dump({"toolkit": {"model": "test/m", "gpu_type": "t4"}}, f)

        exp11_rows = []
        for cfg in ["BASELINE", "DISAGG-1D"]:
            for i in range(5):
                exp11_rows.append(_make_typed_row(
                    Exp11Row, experiment="exp11", config=cfg,
                    status_code="200", concurrency="1",
                    prompt_tokens_target="100", ttft_ms=str(500 + i),
                    total_ms=str(510 + i), run=str(i + 1),
                    completion_tokens="20",
                ))
        _write_typed_csv(
            os.path.join(d, "exp11-results.csv"), Exp11Row, exp11_rows)

        exp14_rows = []
        for cfg, base in [("C-sidecar", 300), ("D-disagg", 320)]:
            for i in range(5):
                exp14_rows.append(_make_typed_row(
                    Exp14Row, experiment="exp14", config=cfg,
                    status_code="200", concurrency="8",
                    prompt_tokens_target="100", ttft_ms=str(base + i),
                    total_ms=str(base + 5 + i), run=str(i + 1),
                    completion_tokens="20", pod="p0",
                ))
        _write_typed_csv(
            os.path.join(d, "exp14-results.csv"), Exp14Row, exp14_rows)

        result = extract_baselines(d, "t4")
        self.assertIsNotNone(result)
        self.assertIn("nixl_ms", result["baselines"])

    def test_exp11_validate(self):
        from validate import load_exp_baselines

        d = tempfile.mkdtemp()
        rows = []
        for cfg in ["BASELINE", "DISAGG-1D"]:
            for i in range(5):
                rows.append(_make_typed_row(
                    Exp11Row, experiment="exp11", config=cfg,
                    status_code="200", concurrency="1",
                    prompt_tokens_target="100", ttft_ms=str(500 + i),
                    total_ms=str(510 + i), run=str(i + 1),
                    completion_tokens="20",
                ))
        _write_typed_csv(os.path.join(d, "exp11-results.csv"), Exp11Row, rows)
        result = load_exp_baselines(d)
        self.assertIn("exp11", result)
        self.assertGreater(len(result["exp11"]), 0)

    def test_exp14_validate(self):
        from validate import load_exp_baselines

        d = tempfile.mkdtemp()
        rows = []
        for cfg in ["C-sidecar", "D-disagg"]:
            for i in range(5):
                rows.append(_make_typed_row(
                    Exp14Row, experiment="exp14", config=cfg,
                    status_code="200", prompt_tokens_target="100",
                    ttft_ms=str(300 + i), total_ms=str(310 + i),
                    run=str(i + 1), pod="p0", concurrency="8",
                    completion_tokens="20",
                ))
        _write_typed_csv(os.path.join(d, "exp14-results.csv"), Exp14Row, rows)
        result = load_exp_baselines(d)
        self.assertIn("exp14", result)
        self.assertGreater(len(result["exp14"]), 0)

    def test_exp5_kv_sweep(self):
        from kv_sweep import _load_exp5

        d = tempfile.mkdtemp()
        rows = []
        for cfg in ["C-sidecar-only", "D-disaggregated"]:
            for i in range(3):
                rows.append(_make_typed_row(
                    Exp5Row, experiment="exp5", config=cfg,
                    status_code="200", prompt_tokens_target="100",
                    ttft_ms=str(500 + i * 10), total_ms=str(510 + i * 10),
                    run=str(i + 1), pod="p0", completion_tokens="20",
                ))
        _write_typed_csv(os.path.join(d, "exp5-results.csv"), Exp5Row, rows)
        transfer_by_len, _model = _load_exp5(d)
        self.assertIsNotNone(transfer_by_len)
        self.assertIn(100, transfer_by_len)

    def test_exp5b_nixl_transfer(self):
        from nixl_transfer import load_exp5b

        d = tempfile.mkdtemp()
        rows = []
        for i in range(5):
            rows.append(_make_typed_row(
                Exp5bRow, experiment="exp5b", status_code="200",
                prompt_tokens_target="100", prompt_tokens_actual="103",
                ttft_ms=str(500 + i), total_ms=str(510 + i),
                run=str(i + 1), pod="p0",
                nixl_bytes_delta=str(6291456),
                nixl_xfer_time_delta_ms=str(22.4),
                nixl_transfers_delta="1",
            ))
        _write_typed_csv(os.path.join(d, "exp5b-results.csv"), Exp5bRow, rows)
        result = load_exp5b(d)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)

    def test_exp1_analyze(self):
        from analyze import analyze_exp1

        d = tempfile.mkdtemp()
        rows = []
        for cfg in ["BASELINE", "DISAGG-D1"]:
            for i in range(5):
                rows.append(_make_typed_row(
                    Exp1Row, experiment="exp1", config=cfg,
                    status_code="200", prompt_tokens_target="100",
                    ttft_ms=str(500 + i), total_ms=str(510 + i),
                    run=str(i + 1), pod="p0", max_tokens="20",
                    completion_tokens="20",
                ))
        _write_typed_csv(os.path.join(d, "exp1-results.csv"), Exp1Row, rows)
        analyze_exp1(d)

    def test_exp1b_analyze(self):
        from analyze import analyze_exp1b

        d = tempfile.mkdtemp()
        rows = []
        for cfg in ["A-prefill-direct", "B-decode-direct",
                     "C-sidecar-only", "D-disaggregated"]:
            for i in range(5):
                rows.append(_make_typed_row(
                    Exp1bRow, experiment="exp1b", config=cfg,
                    status_code="200", ttft_ms=str(500 + i),
                    total_ms=str(510 + i), run=str(i + 1),
                    pod="p0", completion_tokens="20",
                ))
        _write_typed_csv(os.path.join(d, "exp1b-results.csv"), Exp1bRow, rows)
        analyze_exp1b(d)

    def test_exp11_analyze(self):
        from analyze import analyze_exp11

        d = tempfile.mkdtemp()
        rows = []
        for cfg in ["BASELINE", "DISAGG-1D"]:
            for i in range(5):
                rows.append(_make_typed_row(
                    Exp11Row, experiment="exp11", config=cfg,
                    status_code="200", concurrency="1",
                    prompt_tokens_target="100", ttft_ms=str(500 + i),
                    total_ms=str(510 + i), run=str(i + 1),
                    completion_tokens="20",
                ))
        _write_typed_csv(
            os.path.join(d, "exp11-results.csv"), Exp11Row, rows)
        analyze_exp11(d)

    def test_exp14_analyze(self):
        from analyze import analyze_exp14

        d = tempfile.mkdtemp()
        rows = []
        for cfg in ["C-sidecar", "D-disagg"]:
            for i in range(5):
                rows.append(_make_typed_row(
                    Exp14Row, experiment="exp14", config=cfg,
                    status_code="200", concurrency="8",
                    prompt_tokens_target="100", ttft_ms=str(300 + i),
                    total_ms=str(310 + i), run=str(i + 1),
                    pod="p0", completion_tokens="20",
                ))
        _write_typed_csv(
            os.path.join(d, "exp14-results.csv"), Exp14Row, rows)
        analyze_exp14(d)


if __name__ == "__main__":
    unittest.main()
