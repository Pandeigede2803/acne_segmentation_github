"""Static checks for the train/test-only split + 1000/class augmentation step."""

import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "acne_colab.ipynb"
TRAINER = Path(__file__).resolve().parents[1] / "scripts" / "train_proposal_hybrid_resnet50_malds.py"


class SplitAugNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        index = next(i for i, cell in enumerate(cells)
                     if cell["cell_type"] == "markdown"
                     and "## 31." in "".join(cell["source"]))
        cls.markdown = "".join(cells[index]["source"])
        cls.code = "".join(cells[index + 1]["source"])
        ast.parse(cls.code)

    def test_default_off_and_both_splits(self):
        self.assertIn("RUN_SPLIT_AUG = False", self.code)
        self.assertIn("'60_40': 0.60", self.code)
        self.assertIn("'70_30': 0.70", self.code)
        self.assertIn("AUG_PER_CLASS_TARGET = 1000", self.code)

    def test_no_validation_and_checkpoint_resume(self):
        self.assertIn("--no-validation", self.code)
        self.assertIn("--resume", self.code)
        self.assertIn("last_checkpoint", self.markdown)
        self.assertNotIn("--skip-test-eval", self.code)

    def test_absolute_augmentation_and_manifest_guards(self):
        self.assertIn("--target-per-class", self.code)
        self.assertIn("manifest_split.csv", self.code)
        self.assertIn("manifest_aug.csv", self.code)
        self.assertIn("Baris test berubah", self.code)
        self.assertIn("split_aug_results.csv", self.code)

    def test_markdown_has_icons(self):
        self.assertIn("🎯", self.markdown)
        self.assertIn("🛡️", self.markdown)
        self.assertIn("📤", self.markdown)

    def test_trainer_supports_no_validation(self):
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn('"--no-validation"', source)
        self.assertIn("has_validation = len(val_loader.dataset) > 0", source)
        self.assertIn("final_epoch_no_validation", source)
        self.assertIn("if not has_validation:", source)


if __name__ == "__main__":
    unittest.main()
