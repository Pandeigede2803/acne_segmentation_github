"""Tests for the counting-head section of the final evaluation."""

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_final_experiment import count_metrics  # noqa: E402


class CountMetricsTests(unittest.TestCase):
    def test_returns_none_without_count_columns(self):
        rows = [{"true_label_index": 0, "pred_label_index": 0}]
        self.assertIsNone(count_metrics(rows, [0]))

    def test_mae_and_bias_per_grade(self):
        rows = [
            {"true_label_index": 0, "pred_label_index": 0, "count_error": "-2.0"},
            {"true_label_index": 0, "pred_label_index": 0, "count_error": "0.0"},
            {"true_label_index": 3, "pred_label_index": 2, "count_error": "-8.0"},
        ]
        metrics = count_metrics(rows, [0, 0, 3])
        self.assertEqual(metrics["n"], 3)
        self.assertAlmostEqual(metrics["count_mae"], 10.0 / 3)
        self.assertAlmostEqual(metrics["count_bias"], -10.0 / 3)
        self.assertAlmostEqual(metrics["by_true_grade"]["3"]["count_bias"], -8.0)

    def test_grade_from_count_is_reported_when_complete(self):
        rows = [
            {"true_label_index": 0, "pred_label_index": 1, "count_error": "1.0",
             "pred_count_grade": 0},
            {"true_label_index": 3, "pred_label_index": 2, "count_error": "-5.0",
             "pred_count_grade": 3},
        ]
        metrics = count_metrics(rows, [0, 3])
        self.assertIn("grade_from_count", metrics)
        self.assertAlmostEqual(metrics["grade_from_count"]["accuracy"], 1.0)

    def test_grade_from_count_skipped_when_incomplete(self):
        rows = [
            {"true_label_index": 0, "pred_label_index": 0, "count_error": "0.0",
             "pred_count_grade": 0},
            {"true_label_index": 3, "pred_label_index": 2, "count_error": "-5.0",
             "pred_count_grade": ""},
        ]
        metrics = count_metrics(rows, [0, 3])
        self.assertNotIn("grade_from_count", metrics)


if __name__ == "__main__":
    unittest.main()
