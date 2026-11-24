"""Static checks for the Step 32 face/annotation audit cell."""

import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "acne_colab.ipynb"


class FaceAuditNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = cls.nb["cells"]
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
        self.assertIn("fetch_script('face_skin_utils.py', force=True)", self.code)

    def test_skin_only_metrics_and_branch_source(self):
        self.assertIn("acne_segmentation_github/hermes/scripts", self.code)
        self.assertIn("'mean_skin_area_pct'", self.code)
        self.assertIn("'skin_masks_below_40pct'", self.code)
        self.assertIn("summary['worst_skin_coverage']", self.code)
        self.assertIn("summary['images_zero_skin_coverage']", self.code)
        self.assertIn("peta kulit penuh-kanvas", self.markdown)

    def test_fast_resumable_audit(self):
        self.assertIn("FACE_AUDIT_DETECT_MAX_SIZE = 512", self.code)
        self.assertIn("FACE_AUDIT_CHECKPOINT_EVERY", self.code)
        self.assertIn("--detect-max-size", self.code)
        self.assertIn("--checkpoint-every", self.code)
        self.assertIn("Checkpoint parsial", self.markdown)

    def test_public_manifest_without_oauth(self):
        self.assertIn("USE_PUBLIC_MANIFEST = True", self.code)
        self.assertIn("PUBLIC_MANIFEST_ID", self.code)
        self.assertIn("gdown.download", self.code)
        self.assertIn("LOCAL_OUTPUT_ROOT", self.code)
        self.assertIn("Mode tanpa OAuth", self.markdown)

    def test_service_account_drive_api_mode(self):
        setup = ''.join(self.nb['cells'][4]['source'])
        self.assertIn("DRIVE_ACCESS_MODE = 'mount'", setup)
        self.assertIn("GDRIVE_SERVICE_ACCOUNT_JSON", setup)
        self.assertIn("baliweb-gsc@concrete-crow-424203-p0.iam.gserviceaccount.com", setup)
        self.assertIn("upload_folder(DRIVE_API, FACE_AUDIT_OUTPUT", self.code)

    def test_repairs_incomplete_opencv_install(self):
        self.assertIn("hasattr(cv2, \\\"CascadeClassifier\\\")", self.code)
        self.assertIn("opencv-contrib-python-headless==4.11.0.86", self.code)
        self.assertIn("OpenCV Haar siap", self.code)

    def test_uses_original_manifest_and_no_training(self):
        self.assertIn("recover_source_run", self.code)
        self.assertIn("original_manifest", self.code)
        self.assertNotIn("run_epoch", self.code)
        self.assertNotIn("train_proposal_hybrid_resnet50_malds.py", self.code)


if __name__ == "__main__":
    unittest.main()
