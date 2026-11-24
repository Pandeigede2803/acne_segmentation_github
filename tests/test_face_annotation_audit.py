"""Geometry/summary tests for the face vs annotation coverage audit."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_face_annotation_coverage import (
    expand_box,
    load_circles,
    percentage,
    select_examples,
    summarize,
)


def record(filename, label, coverage, face=1, circles=3, face_pct=40.0, skin_area=50.0):
    return {
        "filename": filename, "label_name": f"levle{label}", "label_index": label,
        "split": "train", "image_path": f"/img/{filename}", "image_width": 320,
        "image_height": 320, "face_detected": face, "face_x": 10, "face_y": 10,
        "face_w": 100, "face_h": 100, "face_area_pct": face_pct, "n_circles": circles,
        "annotation_area_px": 500, "annotation_area_pct": 0.5,
        "annotation_inside_face_px": 0, "annotation_inside_face_pct": coverage,
        "annotation_inside_skin_px": 0, "annotation_inside_skin_pct": coverage,
        "annotation_inside_union_pct": coverage, "circles_center_inside_union": 1,
        "circles_center_inside_skin": 1,
        "skin_area_pct": skin_area, "skin_mask_usable": int(skin_area >= 40.0),
    }


class FaceAuditTests(unittest.TestCase):
    def test_skin_mask_is_full_frame_and_reports_low_area(self):
        source = Path(__file__).resolve().parents[1] / "scripts" / "audit_face_annotation_coverage.py"
        code = source.read_text(encoding="utf-8")
        self.assertIn("skin_map(bgr, None", code)
        self.assertIn("region = skin", code)
        self.assertIn("SKIN_MIN_AREA_PCT = 40.0", code)

    def test_examples_rank_skin_coverage_not_haar_union(self):
        union_worst = record("union-worst.jpg", 0, 10.0)
        skin_worst = record("skin-worst.jpg", 0, 90.0)
        union_worst["annotation_inside_skin_pct"] = 80.0
        skin_worst["annotation_inside_skin_pct"] = 5.0
        chosen = select_examples([union_worst, skin_worst], limit=1)
        self.assertEqual(chosen[0]["filename"], "skin-worst.jpg")

    def test_percentage_and_expand_box(self):
        self.assertEqual(percentage(50, 100), 50.0)
        self.assertIsNone(percentage(1, 0))
        self.assertEqual(expand_box((10, 10, 20, 20), 1.1, 100, 100), (9, 9, 22, 22))
        self.assertEqual(expand_box((0, 0, 100, 100), 1.1, 100, 100), (0, 0, 100, 100))

    def test_load_circles_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.json"
            path.write_text(json.dumps({
                "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80}],
                "annotations": [{"image_id": 1, "coordinates": [10, 20], "radius": 3}],
            }))
            circles, sizes = load_circles(path)
        self.assertEqual(circles["a.jpg"], [(10.0, 20.0, 3.0)])
        self.assertEqual(sizes["a.jpg"], (100, 80))

    def test_examples_include_every_label_and_worst_case(self):
        records = [
            record("best0.jpg", 0, 99.0), record("worst0.jpg", 0, 40.0),
            record("one1.jpg", 1, 70.0), record("two2.jpg", 2, 60.0),
            record("three3.jpg", 3, 30.0), record("extra.jpg", 2, 10.0),
            record("ignored.jpg", 1, 5.0, face=0),
        ]
        chosen = select_examples(records, limit=6)
        self.assertIn("worst0.jpg", {row["filename"] for row in chosen})
        self.assertIn("three3.jpg", {row["filename"] for row in chosen})
        self.assertIn("extra.jpg", {row["filename"] for row in chosen})
        self.assertNotIn("ignored.jpg", {row["filename"] for row in chosen})
        self.assertEqual({row["label_index"] for row in chosen}, {0, 1, 2, 3})

    def test_very_severe_example_kept_when_face_detection_fails(self):
        records = [record("a0.jpg", 0, 90.0), record("a1.jpg", 1, 80.0),
                   record("a2.jpg", 2, 70.0), record("severe.jpg", 3, 20.0, face=0)]
        chosen = select_examples(records, limit=6)
        self.assertIn("severe.jpg", {row["filename"] for row in chosen})

    def test_summary_counts_and_means(self):
        records = [record("a.jpg", 0, 100.0), record("b.jpg", 1, 50.0),
                   record("c.jpg", 3, 0.0, skin_area=20.0), record("d.jpg", 2, None, face=0, circles=0)]
        summary = summarize(records)
        self.assertEqual(summary["n_images"], 4)
        self.assertEqual(summary["face_detected"], 3)
        self.assertEqual(summary["annotated_images"], 3)
        self.assertEqual(summary["mean_inside_union_pct"], 50.0)
        self.assertEqual(summary["mean_skin_area_pct"], 40.0)
        self.assertEqual(summary["skin_masks_below_40pct"], 1)
        self.assertEqual(summary["images_zero_skin_coverage"], 1)
        self.assertEqual(summary["images_below_50pct_skin"], 1)
        self.assertEqual(summary["images_below_50pct"], 1)
        self.assertEqual(summary["images_zero_coverage"], 1)
        self.assertEqual(summary["images_fully_inside_union"], 1)
        self.assertEqual(summary["images_below_90pct"], 2)
        self.assertEqual(summary["per_label"]["3"]["mean_inside_union_pct"], 0.0)
        self.assertEqual(summary["worst_union_coverage"][0]["filename"], "c.jpg")


if __name__ == "__main__":
    unittest.main()
