"""Tests for applying the NNEW ground truth to a Step 42 source manifest."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.apply_ground_truth_labels import (apply_labels, load_ground_truth,
                                               read_csv)


MANIFEST_FIELDS = ["filename", "label_name", "label_index", "n_circles", "split"]
TRUTH_FIELDS = ["file_name", "fold", "grade", "count_median"]


class ApplyTests(unittest.TestCase):
    def _write(self, path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_patches_label_and_adds_count(self):
        rows = [
            {"filename": "levle1_151.jpg", "label_name": "levle1", "label_index": "1",
             "n_circles": "6", "split": "train"},
            {"filename": "levle2_5.jpg", "label_name": "levle2", "label_index": "2",
             "n_circles": "25", "split": "train"},
        ]
        truth = {
            "levle1_151.jpg": {"grade": 0, "count": 4.0},
            "levle2_5.jpg": {"grade": 2, "count": 22.0},
        }
        patched, report, fields = apply_labels(rows, MANIFEST_FIELDS, truth,
                                               "filename", allow_missing=False)
        self.assertIn("nnew_count", fields)
        by_name = {row["filename"]: row for row in patched}
        self.assertEqual(by_name["levle1_151.jpg"]["label_name"], "levle0")
        self.assertEqual(by_name["levle1_151.jpg"]["label_index"], "0")
        self.assertEqual(by_name["levle1_151.jpg"]["nnew_count"], "4.0")
        self.assertEqual(by_name["levle2_5.jpg"]["label_name"], "levle2")
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["filename"], "levle1_151.jpg")

    def test_missing_file_is_an_error_unless_allowed(self):
        rows = [{"filename": "unknown.jpg", "label_name": "levle0",
                 "label_index": "0", "n_circles": "1", "split": "train"}]
        truth = {"levle0_1.jpg": {"grade": 0, "count": 3.0}}
        with self.assertRaises(ValueError):
            apply_labels(rows, MANIFEST_FIELDS, truth, "filename", allow_missing=False)
        patched, _, _ = apply_labels(rows, MANIFEST_FIELDS, truth, "filename",
                                     allow_missing=True)
        self.assertEqual(patched[0]["label_name"], "levle0")
        self.assertNotIn("nnew_count", patched[0])

    def test_load_ground_truth_rejects_conflicting_grades(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.csv"
            self._write(path, TRUTH_FIELDS, [
                {"file_name": "a.jpg", "fold": "0", "grade": "0", "count_median": "4"},
                {"file_name": "a.jpg", "fold": "1", "grade": "1", "count_median": "4"},
            ])
            with self.assertRaises(ValueError):
                load_ground_truth(path)

    def test_read_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            self._write(path, MANIFEST_FIELDS, [
                {"filename": "a.jpg", "label_name": "levle0", "label_index": "0",
                 "n_circles": "3", "split": "train"},
            ])
            rows, fields = read_csv(path)
        self.assertEqual(fields, MANIFEST_FIELDS)
        self.assertEqual(rows[0]["filename"], "a.jpg")


if __name__ == "__main__":
    unittest.main()
