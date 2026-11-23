"""Static checks for the separate skin-filter preprocessing step (Step 33)."""

import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "acne_colab.ipynb"


class SkinFilterNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        index = next(i for i, cell in enumerate(cells)
                     if cell["cell_type"] == "markdown"
                     and "".join(cell["source"]).startswith("---\n## 33. "))
        cls.markdown = "".join(cells[index]["source"])
        cls.code = "".join(cells[index + 1]["source"])
        ast.parse(cls.code)

    def test_separate_and_not_training(self):
        self.assertIn("RUN_SKIN_FILTER = False", self.code)
        self.assertNotIn("--no-validation", self.code)
        self.assertNotIn("--resume", self.code)
        self.assertNotIn("train_proposal_hybrid_resnet50_malds.py", self.code)
        self.assertIn("TANPA TRAINING", self.markdown)

    def test_reads_step_31_preprocessing_only(self):
        self.assertIn("SKIN_FILTER_INPUT_MANIFEST", self.code)
        self.assertIn("preprocess_{EXPERIMENT_IMAGE_SIZE}", self.code)
        self.assertIn("Jalankan Step 31 dulu", self.code)
        self.assertIn("experiments_512_skin", self.code)

    def test_filter_options_and_provenance(self):
        self.assertIn("SKIN_FILTER_FILL", self.code)
        self.assertIn("SKIN_FILTER_CLIP_MASKS", self.code)
        self.assertIn("preprocess_skin_filter.py", self.code)
        self.assertIn("face_skin_utils.py", self.code)
        self.assertIn("skin_filter_provenance.json", self.code)
        self.assertIn("variant': 'skinfilter'", self.code)
        self.assertIn("split_aug_preparation.csv", self.code)

    def test_training_is_delegated_to_step_31b(self):
        self.assertIn("TRAINING_VARIANT='skinfilter'", self.code)
        self.assertIn("Step 31b", self.markdown)


if __name__ == "__main__":
    unittest.main()
