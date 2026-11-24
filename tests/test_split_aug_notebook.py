"""Static checks for the preprocessing-only Step 31 and training Step 31b."""

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

        def heading(prefix):
            return next(i for i, cell in enumerate(cells)
                        if cell["cell_type"] == "markdown"
                        and "".join(cell["source"]).startswith(f"---\n## {prefix} "))

        cls.prep_index = heading("31.")
        cls.train_index = heading("31b.")
        cls.prep_markdown = "".join(cells[cls.prep_index]["source"])
        cls.prep_code = "".join(cells[cls.prep_index + 1]["source"])
        cls.train_markdown = "".join(cells[cls.train_index]["source"])
        cls.train_code = "".join(cells[cls.train_index + 1]["source"])
        ast.parse(cls.prep_code)
        ast.parse(cls.train_code)

    def test_step_31_is_preprocessing_only(self):
        self.assertIn("RUN_SPLIT_AUG = False", self.prep_code)
        self.assertIn("TIDAK ada training di step ini", self.prep_code)
        self.assertNotIn("--no-validation", self.prep_code)
        self.assertNotIn("--resume", self.prep_code)
        self.assertNotIn("train_proposal_hybrid_resnet50_malds.py", self.prep_code)
        self.assertIn("split_aug_preparation.csv", self.prep_code)

    def test_preprocessing_builds_512_before_split(self):
        self.assertIn("EXPERIMENT_IMAGE_SIZE = 512", self.prep_code)
        self.assertIn("REBUILD_PREPROCESSING = True", self.prep_code)
        self.assertIn("--target-size", self.prep_code)
        self.assertIn("--preprocessing-mode", self.prep_code)
        self.assertIn("preprocess_{EXPERIMENT_IMAGE_SIZE}", self.prep_code)

    def test_two_splits_and_absolute_augmentation(self):
        self.assertIn("'60_40': 0.60", self.prep_code)
        self.assertIn("'70_30': 0.70", self.prep_code)
        self.assertIn("AUG_PER_CLASS_TARGET = 1000", self.prep_code)
        self.assertIn("--target-per-class", self.prep_code)
        self.assertIn("Baris test berubah", self.prep_code)

    def test_fresh_output_path_and_no_old_mixing(self):
        self.assertIn("SPLIT_AUG_OUTPUT_ROOT", self.prep_code)
        self.assertIn("/output/experiments_512", self.prep_code)
        self.assertIn("is_relative_to", self.prep_code)
        self.assertIn("split_aug_preparation.csv", self.prep_code)

    def test_service_account_stages_and_syncs_files(self):
        self.assertIn("DRIVE_ACCESS_MODE == 'service_account'", self.prep_code)
        self.assertIn("download_folder(DRIVE_API, source_id", self.prep_code)
        self.assertIn("upload_folder(DRIVE_API, SPLIT_AUG_BASE", self.prep_code)

    def test_step_31b_trains_only_prepared_data(self):
        self.assertIn("RUN_SPLIT_AUG_TRAINING = False", self.train_code)
        self.assertIn("TRAINING_VARIANT = 'raw'", self.train_code)
        self.assertIn("'skinfilter'", self.train_code)
        self.assertIn("--no-validation", self.train_code)
        self.assertIn("--resume", self.train_code)
        self.assertIn("train_proposal_hybrid_resnet50_malds.py", self.train_code)
        self.assertIn("--image-size', str(EXPERIMENT_IMAGE_SIZE)", self.train_code)
        self.assertIn("EXPERIMENT_BATCH_SIZE = 8", self.train_code)
        self.assertIn("split_aug_results_{TRAINING_RUN_ID}.csv", self.train_code)
        self.assertIn("Jalankan Step 31 dulu", self.train_code)

    def test_step_31b_selects_one_fresh_split(self):
        self.assertIn("TRAINING_SPLIT = '60_40'", self.train_code)
        self.assertIn("'60_40': 0.60", self.train_code)
        self.assertIn("'70_30': 0.70", self.train_code)
        self.assertIn("TRAINING_RUN_ID = 'fresh_01'", self.train_code)
        self.assertIn("run_root / 'training_runs' / TRAINING_RUN_ID", self.train_code)
        self.assertIn("TRAINING_SPLIT not in SPLIT_RATIOS", self.train_code)
        self.assertIn("for setup, train_ratio in SELECTED_SPLITS.items()", self.train_code)
        self.assertIn("with urlopen('https://raw.githubusercontent.com/Pandeigede2803/acne_segmentation_github/main/scripts/train_proposal_hybrid_resnet50_malds.py'", self.train_code)
        self.assertNotIn("if not trainer.is_file()", self.train_code)
        self.assertIn("split_aug_results_{TRAINING_RUN_ID}.csv", self.train_code)
        self.assertIn("Pilih `TRAINING_SPLIT", self.train_markdown)

    def test_step_31b_honors_service_account_drive_mode(self):
        self.assertIn("DRIVE_ACCESS_MODE == 'service_account'", self.train_code)
        self.assertIn("download_folder(DRIVE_API", self.train_code)
        self.assertIn("upload_folder(DRIVE_API, train_dir", self.train_code)
        self.assertIn("upload_file(DRIVE_API, results_path", self.train_code)
        self.assertIn("DRIVE_ACCESS_MODE == 'mount'", self.train_code)
        self.assertIn("Path('/content/drive').is_mount()", self.train_code)
        self.assertNotIn("if str(SPLIT_AUG_SOURCE_RUN).startswith('/content/drive/') and not Path('/content/drive/MyDrive').is_dir():\n        from google.colab import drive", self.train_code)
        self.assertIn("TRAINING_RUN_ID", self.train_code)
        self.assertIn("find_folder(DRIVE_API", self.train_code)

    def test_trainer_supports_no_validation(self):
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn('"--no-validation"', source)
        self.assertIn("has_validation = len(val_loader.dataset) > 0", source)
        self.assertIn("final_epoch_no_validation", source)
        self.assertIn("if not has_validation:", source)


if __name__ == "__main__":
    unittest.main()
