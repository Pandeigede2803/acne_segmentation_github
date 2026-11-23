"""Tests for the separate skin-filter preprocessing helpers."""

import tempfile
import unittest
from pathlib import Path

from scripts.face_skin_utils import expand_box
from scripts.preprocess_skin_filter import file_hash, select_examples


def row(filename, label):
    return {"filename": filename, "label_index": str(label), "split": "train",
            "image_path": f"/img/{filename}", "preprocessed_path": f"/img/{filename}",
            "refined_mask": f"/mask/{filename}.png"}


class SkinFilterTests(unittest.TestCase):
    def test_examples_cover_each_label(self):
        rows = [row(f"c{label}_{index}.jpg", label) for label in range(4) for index in range(3)]
        chosen = select_examples(rows, 4)
        self.assertEqual({int(item["label_index"]) for item in chosen}, {0, 1, 2, 3})
        self.assertEqual(len(chosen), 4)

    def test_examples_limit_and_fill(self):
        rows = [row("zero.jpg", 0), row("three_a.jpg", 3), row("three_b.jpg", 3),
                row("one.jpg", 1)]
        chosen = select_examples(rows, 4)
        self.assertEqual(len(chosen), 4)
        self.assertIn("three_a.jpg", {item["filename"] for item in chosen})

    def test_expand_box_clamps_to_image(self):
        self.assertEqual(expand_box((10, 10, 20, 20), 1.1, 100, 100), (9, 9, 22, 22))
        self.assertEqual(expand_box((0, 0, 100, 100), 1.1, 100, 100), (0, 0, 100, 100))

    def test_file_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file"
            path.write_bytes(b"a")
            first = file_hash(path)
            path.write_bytes(b"b")
            self.assertNotEqual(first, file_hash(path))


if __name__ == "__main__":
    unittest.main()
