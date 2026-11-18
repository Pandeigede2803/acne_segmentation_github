"""Small local fixtures for the read-only ablation validation/test evaluation."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_ablation_validation import (
    EXPECTED_TEST_IMAGES,
    TEST_RESULT_FIELDS,
    VARIANTS,
    evaluate,
    prepare_test,
    write_test_results,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AblationValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "train_val_manifest.csv"
        fields = ["filename", "split", "label_index", "image_path", "preprocessed_path", "mask_path"]
        self.fields = fields
        self.image = self.root / "images" / "face.jpg"
        self.image.parent.mkdir()
        self.image.write_bytes(b"photo")
        self.mask = self.root / "masks" / "face.png"
        self.mask.parent.mkdir()
        self.mask.write_bytes(b"mask")
        self.train_val_rows = [
            {"filename": "train_0.jpg", "split": "train", "label_index": "0",
             "image_path": str(self.image), "preprocessed_path": str(self.image), "mask_path": str(self.mask)},
            {"filename": "val_1.jpg", "split": "val", "label_index": "1",
             "image_path": str(self.image), "preprocessed_path": str(self.image), "mask_path": str(self.mask)},
        ]
        self.write_csv(self.manifest, self.train_val_rows)

        self.augmented = self.root / "refined_manifest_offline_augmented_100pct.csv"
        self.test_rows = [
            {"filename": f"test_{i}.jpg", "split": "test", "label_index": str(i % 4),
             "image_path": str(self.image), "preprocessed_path": str(self.image), "mask_path": str(self.mask)}
            for i in range(EXPECTED_TEST_IMAGES)
        ]
        self.augmented_rows = self.train_val_rows + self.test_rows
        self.write_csv(self.augmented, self.augmented_rows)
        self.trainer = self.root / "source" / "train_proposal_hybrid_resnet50_malds.py"
        self.provenance = self.root / "ablation_provenance.json"
        self.write_provenance()
        for i, variant in enumerate(VARIANTS):
            path = self.root / variant
            path.mkdir()
            (path / "metrics.json").write_text(json.dumps({
                "config": {"variant": variant, "manifest": str(self.manifest),
                           "skip_test_eval": True, "weighted_sampler": False,
                           "class_weighting": "balanced", "image_size": 320, "batch_size": 16,
                           "input_normalization": "imagenet",
                           "resolved_input_normalization": "imagenet"},
                "test_metrics": None, "best_val_loss": 0.5,
                "history": [
                    {"epoch": 1, "val_loss": 1.0, "train_acc": 0.8,
                     "val_acc": 0.99, "val_kappa": 0.9, "val_dice": 0.8, "val_iou": 0.7},
                    {"epoch": 2, "val_loss": 0.5, "train_acc": 0.7,
                     "val_acc": i / 10, "val_kappa": 0.3, "val_dice": 0.4, "val_iou": 0.2},
                ],
            }))
            (path / "best_model.pt").write_bytes(f"checkpoint-{variant}".encode())

    def write_csv(self, path, rows):
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_provenance(self):
        self.provenance.write_text(json.dumps({
            "variants": list(VARIANTS), "skip_test_eval": True,
            "manifest_sha256": sha256(self.manifest),
            "step11": {
                "command": ["python", "train.py", "--manifest", str(self.augmented)],
                "sha256": {str(self.augmented): sha256(self.augmented),
                           str(self.trainer): "a" * 64},
            },
        }))

    def test_uses_lowest_validation_loss_and_proposed_from_ablation(self):
        rows = evaluate(self.root)
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0]["best_epoch"], 2)
        self.assertAlmostEqual(rows[0]["delta_val_accuracy_vs_proposed"], -0.6)
        self.assertEqual(rows[-1]["delta_val_accuracy_vs_proposed"], 0)

    def test_missing_run_rejected(self):
        (self.root / "proposed" / "metrics.json").unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate(self.root)

    def test_test_split_rejected(self):
        self.manifest.write_text("split\ntrain\nval\ntest\n", encoding="utf-8")
        self.write_provenance()
        with self.assertRaises(ValueError):
            evaluate(self.root)

    def test_wrong_variant_or_test_metrics_rejected(self):
        path = self.root / "baseline" / "metrics.json"
        data = json.loads(path.read_text())
        data["config"]["variant"] = "proposed"
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            evaluate(self.root)
        data["config"]["variant"] = "baseline"
        data["test_metrics"] = {"acc": 1}
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            evaluate(self.root)

    def test_different_seed_rejected(self):
        path = self.root / "no_attention" / "metrics.json"
        data = json.loads(path.read_text())
        data["config"]["seed"] = 99
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            evaluate(self.root)

    def test_test_plan_uses_seven_checkpoints_and_shared_split(self):
        plan, rows = prepare_test(self.root)
        self.assertEqual(plan["run"], self.root.name)
        self.assertEqual(plan["split"], "test")
        self.assertEqual(plan["total_images"], EXPECTED_TEST_IMAGES)
        self.assertEqual(len(rows), EXPECTED_TEST_IMAGES)
        self.assertEqual([item["variant"] for item in plan["variants"]], list(VARIANTS))
        self.assertEqual(plan["variants"][0]["best_epoch"], 2)
        self.assertEqual(plan["image_size"], 320)
        self.assertEqual(plan["input_normalization"], "imagenet")

    def test_wrong_test_manifest_hash_rejected(self):
        other = self.root / "other" / "refined_manifest_offline_augmented_100pct.csv"
        other.parent.mkdir()
        self.write_csv(other, self.augmented_rows[:-1] + [
            dict(self.augmented_rows[-1], filename="test_replacement.jpg")])
        with self.assertRaises(ValueError):
            prepare_test(self.root, other)

    def test_wrong_test_count_rejected(self):
        self.augmented_rows = self.augmented_rows[:-1]
        self.write_csv(self.augmented, self.augmented_rows)
        self.write_provenance()
        with self.assertRaises(ValueError):
            prepare_test(self.root)

    def test_train_val_not_part_of_test_manifest_rejected(self):
        self.train_val_rows[0]["label_index"] = "2"
        self.write_csv(self.manifest, self.train_val_rows)
        self.write_provenance()
        with self.assertRaises(ValueError):
            prepare_test(self.root)

    def test_manifest_used_mismatch_rejected(self):
        other = self.root / "other_manifest.csv"
        self.write_csv(other, self.train_val_rows[:1])
        path = self.root / "baseline" / "metrics.json"
        data = json.loads(path.read_text())
        data["config"]["manifest_used"] = str(other)
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            prepare_test(self.root)

    def test_manifest_used_matches_training_manifest(self):
        for variant in VARIANTS:
            path = self.root / variant / "metrics.json"
            data = json.loads(path.read_text())
            data["config"]["manifest_used"] = str(self.manifest)
            path.write_text(json.dumps(data))
        plan, _ = prepare_test(self.root)
        self.assertEqual(plan["total_images"], EXPECTED_TEST_IMAGES)

    def test_results_csv_columns_are_exact(self):
        output = self.root / "ablation_test_results.csv"
        write_test_results(output, [{"variant": "proposed", "acc": 0.7, "kappa": 0.5,
                                     "macro_f1": 0.6, "dice": 0.3, "iou": 0.2, "run": "ignored"}])
        with output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames), TEST_RESULT_FIELDS)
            row = next(reader)
        self.assertEqual(row["variant"], "proposed")
        self.assertEqual(row["acc"], "0.7")


if __name__ == "__main__":
    unittest.main()
