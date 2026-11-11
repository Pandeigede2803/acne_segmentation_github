"""Strict inference setup checks without PyTorch or image decoding."""

import ast
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inference_guard", ROOT / "scripts/inference_run_guard.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class InferenceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "best_model.pt"
        self.checkpoint.write_bytes(b"fixture checkpoint, never deserialized")
        self.image = self.root / "photo.jpg"
        self.mask = self.root / "mask.png"
        self.image.touch()
        self.mask.touch()
        self.manifest = self.root / "augmented.csv"
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "split", "label_index", "preprocessed_path", "mask_path", "refined_mask"])
            writer.writeheader()
            writer.writerow({"filename": "levle0_1.jpg", "split": "test", "label_index": "0",
                             "preprocessed_path": str(self.image), "mask_path": str(self.mask), "refined_mask": "/unused/missing.png"})
            writer.writerow({"filename": "train.jpg", "split": "train", "label_index": "1",
                             "preprocessed_path": "/unused/train.jpg"})
        self.metrics = {
            "config": {"manifest": str(self.manifest), "variant": "proposed", "encoder": "resnet50",
                       "image_size": 320, "input_normalization": "imagenet", "skip_test_eval": False},
            "best_val_loss": 0.7, "history": [{"epoch": 112, "val_loss": 0.7}, {"epoch": 119, "val_loss": 0.8}],
            "test_metrics": {"acc": 1.0, "kappa": 0.0, "dice": 0.3, "iou": 0.2},
        }
        (self.root / "metrics.json").write_text(json.dumps(self.metrics))
        self.provenance = {
            "command": ["python", "trainer.py", "--manifest", str(self.manifest)],
            "sha256": {str(self.manifest): GUARD.file_hash(self.manifest)},
            "dataset_audit": {"eligible_images": 1204, "annotations_sha256":
                              "ae03fc23fed0925c254dc4c4c9e55b539bd199b236151cb649857378a43539e9",
                              "split_counts": {"test": 1}},
        }
        (self.root / "augmentation_training_provenance.json").write_text(json.dumps(self.provenance))

    def args(self):
        return SimpleNamespace(checkpoint=self.checkpoint, manifest=self.manifest,
                               split="test", image_size=320, threshold=0.5, model_type="proposal",
                               variant="auto", input_normalization="auto")

    def test_resolves_training_inputs_and_best_loss_epoch(self):
        plan = GUARD.resolve_run(self.checkpoint)
        self.assertEqual(plan["manifest"], str(self.manifest))
        self.assertEqual(plan["total_images"], 1)
        self.assertEqual(plan["best_epoch_from_history"], 112)
        self.assertEqual(plan["input_normalization"], "imagenet")
        GUARD.validate_arguments(self.args())

    def test_wrong_manifest_rejected(self):
        args = self.args()
        args.manifest = self.root / "legacy_manifest.csv"
        with self.assertRaisesRegex(ValueError, "bukan manifest"):
            GUARD.validate_arguments(args)

    def test_changed_manifest_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Manifest training hilang/berubah"):
            GUARD.resolve_run(self.checkpoint)

    def test_missing_mask_rejected_not_silently_skipped(self):
        self.mask.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Foto/mask test"):
            GUARD.resolve_run(self.checkpoint)

    def test_overrides_rejected(self):
        for key, value in [("image_size", 512), ("threshold", 0.3), ("split", "all"),
                           ("input_normalization", "none"), ("variant", "baseline")]:
            args = self.args()
            setattr(args, key, value)
            with self.assertRaises(ValueError):
                GUARD.validate_arguments(args)

    def test_test_count_must_match_training_audit(self):
        self.provenance["dataset_audit"]["split_counts"]["test"] = 224
        (self.root / "augmentation_training_provenance.json").write_text(json.dumps(self.provenance))
        with self.assertRaisesRegex(ValueError, "Jumlah test berbeda"):
            GUARD.resolve_run(self.checkpoint)

    def test_normalization_cannot_fallback_silently(self):
        self.metrics["config"]["input_normalization"] = "auto"
        (self.root / "metrics.json").write_text(json.dumps(self.metrics))
        with self.assertRaisesRegex(ValueError, "Normalisasi input"):
            GUARD.resolve_run(self.checkpoint)

    def test_notebook_uses_run_plan_not_runtime_manifest(self):
        cells = json.loads((ROOT / "acne_colab.ipynb").read_text())["cells"]
        heading = next(i for i, c in enumerate(cells) if "## 14. Inference" in "".join(c["source"]))
        code = "".join(cells[heading + 1]["source"])
        ast.parse(code)
        self.assertIn("RUN_TEST_INFERENCE = False", code)
        self.assertIn("inference_guard.resolve_run(CHECKPOINT)", code)
        self.assertIn("'--manifest', plan['manifest']", code)
        self.assertIn("'--strict-run'", code)
        self.assertNotIn("REFINED_MANIFEST", code)
        self.assertIn("check=True", code)
        ast.parse((ROOT / "scripts/infer_testing_images.py").read_text())


if __name__ == "__main__":
    unittest.main()
