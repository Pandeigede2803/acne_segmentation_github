"""Unit tests for the train/test-only stratified manifest splitter."""

import unittest
from collections import Counter

from scripts.split_train_test_manifest import assign_splits


def sample_rows():
    rows = []
    for label in range(4):
        for index in range(25 if label < 3 else 5):
            rows.append({"filename": f"c{label}_{index}.jpg", "label_index": str(label),
                         "split": "val", "image_path": f"/img/{label}_{index}.jpg"})
    return rows


class SplitTrainTestTests(unittest.TestCase):
    def test_60_40_has_no_validation_and_keeps_all_rows(self):
        rows = sample_rows()
        output = assign_splits(rows, 0.6, seed=42)
        self.assertEqual(len(output), len(rows))
        self.assertEqual({row["split"] for row in output}, {"train", "test"})
        self.assertNotIn("val", {row["split"] for row in output})
        self.assertEqual([row["filename"] for row in output], [row["filename"] for row in rows])
        train = Counter(int(row["label_index"]) for row in output if row["split"] == "train")
        test = Counter(int(row["label_index"]) for row in output if row["split"] == "test")
        self.assertEqual(train, {0: 15, 1: 15, 2: 15, 3: 3})
        self.assertEqual(test, {0: 10, 1: 10, 2: 10, 3: 2})

    def test_70_30_and_deterministic(self):
        first = assign_splits(sample_rows(), 0.7, seed=7)
        second = assign_splits(sample_rows(), 0.7, seed=7)
        self.assertEqual(first, second)
        train = Counter(int(row["label_index"]) for row in first if row["split"] == "train")
        self.assertEqual(train[3], 3)

    def test_no_class_is_empty_in_either_split(self):
        rows = sample_rows()
        for ratio in (0.5, 0.6, 0.7, 0.9):
            output = assign_splits(rows, ratio, seed=1)
            labels = {(row["split"], int(row["label_index"])) for row in output}
            for label in range(4):
                self.assertIn(("train", label), labels)
                self.assertIn(("test", label), labels)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            assign_splits(sample_rows(), 1.0)
        with self.assertRaises(ValueError):
            assign_splits([{"filename": "a", "label_index": "0"},
                           {"filename": "a", "label_index": "1"}], 0.6)
        rows = [{"filename": f"c0_{i}", "label_index": "0"} for i in range(10)]
        with self.assertRaises(ValueError):
            assign_splits(rows, 0.6)


if __name__ == "__main__":
    unittest.main()
