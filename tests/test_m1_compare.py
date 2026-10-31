"""对拍报告本身也需要可信：失败不能写成通过，非有限值不能污染 JSON。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from bench import compare_m1
from bench.m1_common import compare_tensors


class ComparisonTests(unittest.TestCase):
    def test_stats_and_ranking_disagreement(self):
        reference = torch.tensor([[[1.0, 2.0, 3.0]]])
        actual = torch.tensor([[[1.0, 3.5, 3.0]]])
        result = compare_tensors(actual, reference, atol=0.01, rtol=0.01, logits=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failing_elements"], 1)
        self.assertEqual(result["max_abs"], 1.5)
        self.assertEqual(result["worst_flat_index"], 1)
        self.assertEqual(result["top1_agreement"], 0)
        self.assertEqual(result["first_top1_mismatch"]["reference_top2_gap"], 1)

    def test_nonfinite_and_shape_failure_are_json_serializable(self):
        for actual in (torch.full((1, 1, 3), float("nan")), torch.ones(1, 1, 2)):
            result = compare_tensors(actual, torch.ones(1, 1, 3), atol=0.01, rtol=0.01)
            self.assertFalse(result["passed"])
            json.dumps(result, allow_nan=False)

    def test_failure_and_interrupt_preserve_partial_results(self):
        for exception, expected_code, status in ((RuntimeError("injected"), 1, "error"),
                                                   (KeyboardInterrupt(), 130, "interrupted")):
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / "report.json")
                def fail(args, report):
                    report["cases"].append({"step": 0, "passed": False})
                    raise exception
                with patch.object(compare_m1, "parse_args", return_value=SimpleNamespace(output=path)), \
                     patch.object(compare_m1, "environment_info", return_value={}), \
                     patch.object(compare_m1, "compare", side_effect=fail):
                    self.assertEqual(compare_m1.main(), expected_code)
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                self.assertEqual(data["status"], status)
                self.assertFalse(data["passed"])
                self.assertEqual(len(data["cases"]), 1)


if __name__ == "__main__":
    unittest.main()
