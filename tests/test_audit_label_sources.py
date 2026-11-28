"""Tests for the label-source audit helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_label_sources import (load_circle_counts, load_fold_rows,
                                         prefix_label, read_nnew, union_rows)


class ReadNNEWTests(unittest.TestCase):
    def test_reads_three_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "NNEW_trainval_0.txt"
            path.write_text("levle1_151.jpg 0 4\nlevle2_5.jpg 1 8\n", encoding="utf-8")
            rows = read_nnew(path)
        self.assertEqual(rows, [("levle1_151.jpg", 0, 4), ("levle2_5.jpg", 1, 8)])


class PrefixTests(unittest.TestCase):
    def test_parses_level_prefix(self):
        self.assertEqual(prefix_label("levle0_1.jpg"), 0)
        self.assertEqual(prefix_label("levle3_90.jpg"), 3)

    def test_returns_none_for_unknown_pattern(self):
        self.assertIsNone(prefix_label("acne_1.jpg"))


class CircleCountTests(unittest.TestCase):
    def test_counts_annotations_per_image(self):
        payload = {
            "images": [
                {"id": 0, "file_name": "a.jpg"},
                {"id": 1, "file_name": "b.jpg"},
            ],
            "annotations": [
                {"image_id": 0}, {"image_id": 0}, {"image_id": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            counts = load_circle_counts(path)
        self.assertEqual(counts, {"a.jpg": 2, "b.jpg": 1})


class FoldTests(unittest.TestCase):
    def _write_fold(self, directory: Path, fold: int, rows: str) -> None:
        self._write_folds(directory, {fold: rows})

    def _write_folds(self, directory: Path, rows_by_fold: dict[int, str]) -> None:
        for index in range(5):
            (directory / f"NNEW_trainval_{index}.txt").write_text(
                rows_by_fold.get(index, ""), encoding="utf-8")
            (directory / f"NNEW_test_{index}.txt").write_text("", encoding="utf-8")

    def test_union_allows_count_variation_but_not_label_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_folds(directory, {0: "levle3_10.jpg 3 53\n",
                                          1: "levle3_10.jpg 3 65\n"})
            merged = union_rows(load_fold_rows(directory))
        self.assertEqual(merged["levle3_10.jpg"][0], 3)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_folds(directory, {0: "levle3_10.jpg 3 53\n",
                                          1: "levle3_10.jpg 2 53\n"})
            with self.assertRaises(ValueError):
                union_rows(load_fold_rows(directory))

    def test_duplicate_within_fold_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_fold(directory, 0, "levle1_1.jpg 1 8\nlevle1_1.jpg 1 8\n")
            with self.assertRaises(ValueError):
                load_fold_rows(directory)


if __name__ == "__main__":
    unittest.main()
