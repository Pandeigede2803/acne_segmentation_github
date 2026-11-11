"""Dataset audit tests with synthetic fixtures; no download or ML dependencies."""

import csv
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "audit_acne04v2", Path(__file__).resolve().parents[1] / "scripts/audit_acne04v2.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class DatasetAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.images = cls.root / "images"
        cls.images.mkdir()
        cls.asset = cls.root / "mask.png"
        cls.asset.touch()
        images = []
        cls.rows = []
        for label, count in AUDIT.EXPECTED_CLASSES.items():
            for i in range(count):
                name = f"{label}_{i}.jpg"
                path = cls.images / name
                path.touch()
                images.append({"id": len(images), "file_name": name})
                cls.rows.append({
                    "filename": name, "label_name": label, "label_index": label[-1],
                    "split": "val" if i % 5 == 0 else "test" if i % 5 == 1 else "train",
                    "image_path": str(path), "preprocessed_path": str(path),
                    "refined_mask": str(cls.asset), "circle_mask": str(cls.asset),
                    "resize_mode": "letterbox", "preprocessing_mode": "none", "image_size": "320",
                })
        annotations = [{"image_id": i % 1204} for i in range(32443)]
        counts = Counter(a["image_id"] for a in annotations)
        for i, row in enumerate(cls.rows):
            row["n_circles"] = str(counts[i])
        cls.annotations = cls.root / "annotations.json"
        cls.annotations.write_text(json.dumps({"images": images, "annotations": annotations}))
        cls.manifest = cls.root / "manifest.csv"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        # Only fixtures bypass the pinned hash; production requires exact official bytes.
        self.hash_patch = patch.object(AUDIT, "OFFICIAL_SHA256", AUDIT.sha256(self.annotations))
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.write_manifest(self.rows)

    def write_manifest(self, rows, path=None):
        with (path or self.manifest).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_source_excludes_extra_images(self):
        extra = self.images / "not_v2.jpg"
        extra.touch()
        self.addCleanup(extra.unlink)
        report = AUDIT.audit_source(self.images, self.annotations)
        self.assertEqual(report["eligible_images"], 1204)
        self.assertEqual(report["source_image_files"], 1205)
        self.assertEqual(report["excluded_source_files"], 1)
        self.assertEqual(report["class_counts_from_filename"], AUDIT.EXPECTED_CLASSES)

    def test_unofficial_json_is_rejected(self):
        wrong = self.root / "wrong.json"
        wrong.write_text('{}')
        with self.assertRaisesRegex(ValueError, "bukan JSON v2 resmi"):
            AUDIT.audit_source(self.images, wrong)

    def test_missing_and_duplicate_source_rejected(self):
        image = self.images / self.rows[0]["filename"]
        moved = self.root / "moved.jpg"
        image.rename(moved)
        try:
            with self.assertRaisesRegex(ValueError, "missing="):
                AUDIT.audit_source(self.images, self.annotations)
        finally:
            moved.rename(image)
        duplicate_dir = self.images / "duplicate"
        duplicate_dir.mkdir(exist_ok=True)
        duplicate = duplicate_dir / image.name
        duplicate.touch()
        self.addCleanup(duplicate.unlink)
        with self.assertRaisesRegex(ValueError, "duplicate="):
            AUDIT.audit_source(self.images, self.annotations)

    def test_valid_manifest(self):
        report = AUDIT.audit_manifest(self.manifest, self.annotations, self.images)
        self.assertEqual(sum(report["split_counts"].values()), 1204)
        self.assertEqual(report["manifest_sha256"], AUDIT.sha256(self.manifest))

    def test_wrong_manifest_membership_rejected(self):
        self.write_manifest(self.rows + [dict(self.rows[0])])
        with self.assertRaisesRegex(ValueError, "1.204 nama unik"):
            AUDIT.audit_manifest(self.manifest, self.annotations, self.images)

    def test_wrong_annotation_count_and_preprocessing_rejected(self):
        for field, value, message in [("n_circles", "1", "Jumlah lesi"),
                                      ("preprocessing_mode", "legacy", "Preprocessing"),
                                      ("label_index", "3", "Label")]:
            rows = [dict(row) for row in self.rows]
            rows[0][field] = value
            self.write_manifest(rows)
            with self.assertRaisesRegex(ValueError, message):
                AUDIT.audit_manifest(self.manifest, self.annotations, self.images)

    def test_augmented_train_only(self):
        rows = [dict(row) for row in self.rows]
        counts = Counter(r["label_index"] for r in rows if r["split"] == "train")
        for label in counts:
            source = next(r for r in rows if r["split"] == "train" and r["label_index"] == label)
            for i in range(max(counts.values()) - counts[label]):
                rows.append(dict(source, filename=f"synthetic_{label}_{i}.jpg"))
        path = self.root / "augmented.csv"
        self.write_manifest(rows, path)
        report = AUDIT.audit_augmented_manifest(self.manifest, path)
        self.assertEqual(len(set(report["train_counts_after"].values())), 1)
        rows[0]["split"] = "train"
        self.write_manifest(rows, path)
        with self.assertRaises(ValueError):
            AUDIT.audit_augmented_manifest(self.manifest, path)


if __name__ == "__main__":
    unittest.main()
