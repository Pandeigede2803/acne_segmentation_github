"""Test-set selection and checkpoint locking checks without PyTorch."""

import ast
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("single_test", ROOT / "scripts/evaluate_single_task_test.py")
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


class SingleTaskTestEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.proposed = root / "proposed"
        self.bundle = root / "single"
        self.single = self.bundle / "run"
        self.proposed.mkdir()
        self.single.mkdir(parents=True)
        self.photo = root / "test.jpg"
        self.photo.touch()
        self.manifest = root / "augmented.csv"
        self.train_manifest = self.bundle / "train_val_images.csv"
        self.training_rows = [{"filename": f"{split}_{i}.jpg", "split": split,
                               "label_index": str(i), "image_path": "/unused/raw.jpg",
                               "preprocessed_path": "/unused/train_val.jpg"}
                              for split in ("train", "val") for i in range(4)]
        self.test_rows = [{"filename": f"test_{i}.jpg", "split": "test", "label_index": str(i),
                           "image_path": "/unused/raw.jpg", "preprocessed_path": str(self.photo)} for i in (0, 3)]
        self.write_csv(self.manifest, self.training_rows + self.test_rows)
        self.write_csv(self.train_manifest, self.training_rows)
        for name in ("best_model.pt", "last_checkpoint.pt"):
            (self.single / name).write_bytes(b"fixture; never deserialized")
        self.config = {"variant": "resnet50_single_task", "manifest": str(self.train_manifest),
                       "manifest_sha256": EVALUATION.file_hash(self.train_manifest),
                       "script_sha256": "historical-trainer", "input_normalization": "imagenet",
                       "image_size": 320, "batch_size": 16}
        self.single_metrics = {"config": self.config, "status": "completed", "best_epoch": 77,
                               "best_val_loss": 0.69, "history": [{"epoch": 77, "val_loss": 0.69}, {"epoch": 120, "val_loss": 0.8}]}
        (self.single / "metrics.json").write_text(json.dumps(self.single_metrics))
        (self.single / "protocol.json").write_text(json.dumps(self.config))
        (self.proposed / "metrics.json").write_text(json.dumps({
            "config": {"manifest": str(self.manifest), "variant": "proposed", "image_size": 320,
                       "resolved_input_normalization": "imagenet"},
            "test_metrics": {"acc": 0.5, "kappa": 0.0}}))
        self.source = {"command": ["python", "trainer", "--manifest", str(self.manifest)],
                       "sha256": {str(self.manifest): EVALUATION.file_hash(self.manifest)},
                       "dataset_audit": {"eligible_images": 1204, "annotations_sha256":
                                         "ae03fc23fed0925c254dc4c4c9e55b539bd199b236151cb649857378a43539e9",
                                         "split_counts": {"test": 2}}}
        self.source_path = self.proposed / "augmentation_training_provenance.json"
        self.source_path.write_text(json.dumps(self.source))
        self.bundle_source = {"source_record_sha256": EVALUATION.file_hash(self.source_path),
                              "script_sha256": "historical-trainer",
                              "manifest_sha256": EVALUATION.file_hash(self.train_manifest)}
        (self.bundle / "source_provenance.json").write_text(json.dumps(self.bundle_source))

    def write_csv(self, path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_exact_proposed_test_split_not_validation(self):
        plan, rows, _ = EVALUATION.prepare_test(self.single, self.proposed)
        self.assertEqual(plan["split"], "test")
        self.assertEqual(plan["single_task_best_epoch"], 77)
        self.assertEqual(plan["total_images"], 2)
        self.assertEqual([r["filename"] for r in rows], ["test_0.jpg", "test_3.jpg"])
        self.assertTrue(all(r["split"] == "test" for r in rows))

    def test_changed_manifest_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Manifest proposed berubah"):
            EVALUATION.prepare_test(self.single, self.proposed)

    def test_wrong_source_run_rejected(self):
        self.source["changed"] = True
        self.source_path.write_text(json.dumps(self.source))
        with self.assertRaisesRegex(ValueError, "provenance proposed"):
            EVALUATION.prepare_test(self.single, self.proposed)

    def test_wrong_best_epoch_rejected(self):
        self.single_metrics["best_epoch"] = 120
        (self.single / "metrics.json").write_text(json.dumps(self.single_metrics))
        with self.assertRaisesRegex(ValueError, "validation loss minimum"):
            EVALUATION.prepare_test(self.single, self.proposed)

    def test_missing_test_photo_rejected(self):
        self.photo.unlink()
        with self.assertRaisesRegex(ValueError, "Foto/label test"):
            EVALUATION.prepare_test(self.single, self.proposed)

    def test_scores(self):
        result = EVALUATION.scores([0, 0, 1, 2], [0, 1, 1, 3])
        self.assertEqual(result["acc"], 0.5)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["total_images"], 4)
        self.assertEqual(result["confusion_matrix"][2][3], 1)

    def test_notebook_cell_has_no_training_dependency(self):
        cells = json.loads((ROOT / "acne_colab.ipynb").read_text())["cells"]
        index = next(i for i, c in enumerate(cells) if "## 30. Evaluasi TEST" in "".join(c["source"]))
        code = "".join(cells[index + 1]["source"])
        ast.parse(code)
        self.assertIn("RUN_SINGLE_TASK_TEST = True", code)
        self.assertIn("'--proposed-run', str(TEST_PROPOSED_RUN)", code)
        self.assertNotIn("RUN_SINGLE_TASK =", code)
        self.assertNotIn("SCRIPTS_DIR", code)
        self.assertNotIn("OUTPUT_A", code)
        tree = ast.parse((ROOT / "scripts/evaluate_single_task_test.py").read_text())
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in {"backward", "step", "zero_grad"} for n in ast.walk(tree)))

    def test_model_head_matches_training(self):
        training = ast.parse((ROOT / "scripts/train_single_task_resnet50.py").read_text())
        evaluation = ast.parse((ROOT / "scripts/evaluate_single_task_test.py").read_text())
        def assignments(tree):
            model = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name in {"SingleTaskResNet50", "Model"})
            return {ast.unparse(n.targets[0]): ast.dump(n.value) for n in ast.walk(model)
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)}
        a, b = assignments(training), assignments(evaluation)
        for field in ("self.global_proj", "self.classifier", "self.encoder.fc"):
            self.assertEqual(a[field], b[field])


if __name__ == "__main__":
    unittest.main()
