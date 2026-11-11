"""Image-only control checks, without ML dependencies."""

import ast
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("single_task", ROOT / "scripts/train_single_task_resnet50.py")
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


class SingleTaskTests(unittest.TestCase):
    def test_only_train_val_images_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "photo.jpg"
            image.touch()
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["split", "label_index", "image_path"])
                writer.writeheader()
                for split in ("train", "val"):
                    writer.writerows({"split": split, "label_index": label, "image_path": str(image)} for label in range(4))
                writer.writerow({"split": "test", "label_index": "invalid", "image_path": "/missing/test.jpg"})
            rows = TRAINER.read_train_val_rows(manifest)
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["split"] for r in rows}, {"train", "val"})
            self.assertTrue(all("mask_path" not in r for r in rows))

    def test_missing_class_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "empty.csv"
            path.write_text("split,label_index,image_path\n")
            with self.assertRaisesRegex(ValueError, "keempat kelas"):
                TRAINER.read_train_val_rows(path)

    def test_perfect_classification(self):
        metrics = TRAINER.classification_summary(list(range(4)), list(range(4)))
        self.assertEqual(metrics["acc"], 1)
        self.assertEqual(metrics["kappa"], 1)
        self.assertEqual(metrics["macro_f1"], 1)

    def test_confusion_direction(self):
        metrics = TRAINER.classification_summary([0, 0, 1, 2], [0, 1, 1, 3])
        self.assertEqual(metrics["acc"], 0.5)
        self.assertEqual(metrics["confusion_matrix"][2][3], 1)
        self.assertEqual(metrics["per_class"][0]["recall"], 0.5)

    def test_invalid_metric_inputs(self):
        for actual, predicted in [([], []), ([0], [])]:
            with self.assertRaises(ValueError):
                TRAINER.classification_summary(actual, predicted)

    def test_notebook_step29(self):
        cells = json.loads((ROOT / "acne_colab.ipynb").read_text())["cells"]
        heading = next(i for i, cell in enumerate(cells) if "## 29. Pembanding Single-Task" in "".join(cell["source"]))
        code = "".join(cells[heading + 1]["source"])
        summary = "".join(cells[heading + 2]["source"])
        ast.parse(code)
        ast.parse(summary)
        self.assertIn("RUN_SINGLE_TASK = False", code)
        self.assertIn("isin(['train', 'val'])", code)
        self.assertIn("v2_audit.audit_augmented_manifest", code)
        self.assertIn("source_protocol['sha256']", code)
        self.assertIn("command.append('--resume')", code)
        command = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "command" for t in n.targets))
        self.assertNotIn("--weighted-sampler", [n.value for n in command.value.elts if isinstance(n, ast.Constant)])
        self.assertNotIn("command.append('--weighted-sampler')", code)
        self.assertNotIn("test_metrics", summary)

    def test_image_dataset_has_no_mask_access(self):
        tree = ast.parse((ROOT / "scripts/train_single_task_resnet50.py").read_text())
        dataset = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ImageDataset")
        literals = [n.value for n in ast.walk(dataset) if isinstance(n, ast.Constant)]
        for forbidden in ("mask", "mask_path", "refined_mask", "morph_target"):
            self.assertNotIn(forbidden, literals)


if __name__ == "__main__":
    unittest.main()
