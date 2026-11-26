"""Static checks for the preprocessing-only Step 31 and training Step 31b."""

import ast
import json
import random
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
        cls.report_index = next(i for i, cell in enumerate(cells)
                                if cell["cell_type"] == "markdown"
                                and "".join(cell["source"]).startswith("## 31c."))
        cls.report_markdown = "".join(cells[cls.report_index]["source"])
        cls.report_code = "".join(cells[cls.report_index + 1]["source"])
        cls.low_cost_index = next(i for i, cell in enumerate(cells)
                                  if cell["cell_type"] == "markdown"
                                  and "".join(cell["source"]).startswith("## 31d."))
        cls.low_cost_markdown = "".join(cells[cls.low_cost_index]["source"])
        cls.low_cost_code = "".join(cells[cls.low_cost_index + 1]["source"])
        ast.parse(cls.prep_code)
        ast.parse(cls.train_code)
        ast.parse(cls.report_code)
        ast.parse(cls.low_cost_code)

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

    def test_step_31c_fails_closed_on_mixed_run_artifacts(self):
        self.assertIn("RUN_STEP31B_REPORT = False", self.report_code)
        self.assertIn("len(metric_candidates) != 1", self.report_code)
        self.assertIn("plan_sha256", self.report_code)
        self.assertIn("augmented_manifest_sha256", self.report_code)
        self.assertIn("trainer_sha256", self.report_code)
        self.assertIn("checkpoint.get('history') != history", self.report_code)
        self.assertIn("checkpoint_args.get('manifest')", self.report_code)
        self.assertIn("REPORT_DIR = REPORT_RUN_DIR / 'evaluation_report'", self.report_code)
        self.assertNotIn("best_model.pt", self.report_code)
        self.assertIn("test_predictions.csv", self.report_code)
        self.assertIn("training_curves.png", self.report_code)
        self.assertIn("train_test_comparison.png", self.report_code)
        self.assertIn("test_confusion_matrix.png", self.report_code)

    def test_step_31d_uses_validation_macro_f1_and_keeps_test_out_of_selection(self):
        for expected in ("RUN_LOW_COST_EXPERIMENT = False", "LOW_COST_PHASE1_EPOCHS = 30",
                         "LOW_COST_PHASE2_EPOCHS = 20", "LOW_COST_PATIENCE = 5",
                         "'val_macro_f1'", "if phase2 and stale_phase2 >= LOW_COST_PATIENCE",
                         "selection_locked.json", "test_predictions.csv", "per_class_metrics.csv",
                         "LOW_COST_CHECKPOINT_EVERY = 5"):
            self.assertIn(expected, self.low_cost_code)
        self.assertIn("low_cost_split(original_train", self.low_cost_code)
        self.assertIn("source_test != augmented_test", self.low_cost_code)
        self.assertIn("fixed_splits != output_fixed", self.low_cost_code)
        self.assertIn("set_encoder_trainable(phase2)", self.low_cost_code)
        self.assertIn("terpisah dari `fresh_01`", self.low_cost_markdown)
        self.assertNotIn("scripts/train_proposal_hybrid_resnet50_malds.py", self.low_cost_code.split("trainer_path =", 1)[0])

    def test_step_31d_stratified_split_is_reproducible_and_disjoint(self):
        tree = ast.parse(self.low_cost_code)
        split_function = next(node for node in tree.body
                              if isinstance(node, ast.FunctionDef) and node.name == "low_cost_split")
        module = ast.Module(body=[split_function], type_ignores=[])
        namespace = {"random": random}
        exec(compile(module, "step31d_split", "exec"), namespace)
        rows = [{"filename": f"class{label}_{i}.jpg", "label_index": str(label), "split": "train"}
                for label in range(4) for i in range(10)]
        train, val, grouping = namespace["low_cost_split"](rows, ["filename", "label_index", "split"], 42)
        self.assertEqual((len(train), len(val)), (32, 8))
        self.assertEqual(grouping, "image_level_no_patient_id")
        self.assertFalse({r["filename"] for r in train} & {r["filename"] for r in val})
        self.assertEqual([sum(int(r["label_index"]) == c for r in val) for c in range(4)], [2, 2, 2, 2])
        train2, val2, _ = namespace["low_cost_split"](rows, ["filename", "label_index", "split"], 42)
        self.assertEqual([r["filename"] for r in train], [r["filename"] for r in train2])
        self.assertEqual([r["filename"] for r in val], [r["filename"] for r in val2])

    def test_trainer_supports_no_validation(self):
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn('"--no-validation"', source)
        self.assertIn("has_validation = len(val_loader.dataset) > 0", source)
        self.assertIn("final_epoch_no_validation", source)
        self.assertIn("if not has_validation:", source)


if __name__ == "__main__":
    unittest.main()
