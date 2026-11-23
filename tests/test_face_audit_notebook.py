"""Static checks for the Step 32 face/annotation audit cell."""

import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "acne_colab.ipynb"


class FaceAuditNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        index = next(i for i, cell in enumerate(cells)
                     if cell["cell_type"] == "markdown"
                     and "## 32." in "".join(cell["source"]))
        cls.markdown = "".join(cells[index]["source"])
        cls.code = "".join(cells[index + 1]["source"])
        ast.parse(cls.code)

    def test_default_off_and_three_step_scope(self):
        self.assertIn("RUN_FACE_AUDIT = False", self.code)
        self.assertIn("Deteksi wajah", self.markdown)
        self.assertIn("peta kulit", self.markdown)
        self.assertIn("sebelum vs sesudah", self.markdown)

    def test_examples_and_outputs(self):
        self.assertIn("--examples', '6", self.code)
        self.assertIn("face_annotation_audit.csv", self.code)
        self.assertIn("face_annotation_summary.json", self.code)
        self.assertIn("visual_examples", self.code)
        self.assertIn("audit_face_annotation_coverage.py", self.code)
        self.assertIn("face_skin_utils.py", self.code)

    def test_fast_resumable_audit(self):
        self.assertIn("FACE_AUDIT_DETECT_MAX_SIZE = 512", self.code)
        self.assertIn("FACE_AUDIT_CHECKPOINT_EVERY", self.code)
        self.assertIn("--detect-max-size", self.code)
        self.assertIn("--checkpoint-every", self.code)
        self.assertIn("Checkpoint parsial", self.markdown)

    def test_uses_original_manifest_and_no_training(self):
        self.assertIn("recover_source_run", self.code)
        self.assertIn("original_manifest", self.code)
        self.assertNotIn("run_epoch", self.code)
        self.assertNotIn("train_proposal_hybrid_resnet50_malds.py", self.code)


if __name__ == "__main__":
    unittest.main()
