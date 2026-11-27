"""Static checks for Steps 35-40 (mask statistics, morphology, finalization, training)."""

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "acne_colab.ipynb"
EVALUATE = ROOT / "scripts" / "evaluate_mask_variants.py"
FINALIZE = ROOT / "scripts" / "finalize_mask_dataset.py"


class MaskStatisticsScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = EVALUATE.read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_writes_documented_statistics(self):
        for name in ("metrics_per_image.csv", "metrics_per_annotation.csv",
                     "metrics_per_class.csv", "morphology_severity_correlation.csv",
                     "audit_report.md", "comparison_grid.png", "provenance.json"):
            self.assertIn(name, self.source)

    def test_compares_three_variants_and_reports_limitations(self):
        self.assertIn('VARIANT_COLUMNS = {"circle": "circle_mask", "v1": "v1_mask", "v2": "refined_mask"}', self.source)
        self.assertIn("bukan akurasi klinis", self.source)
        self.assertIn("Jangan memilih varian memakai test set", self.source)
        self.assertIn("eta_squared_by_class", self.source)
        self.assertIn("--annotations", self.source)
        self.assertIn('"fallback_annotations": sum(', self.source)
        self.assertIn('"lost_annotations": sum(', self.source)
        self.assertIn('"out_of_bounds_annotations": sum(', self.source)
        self.assertIn("--grids", self.source)

    def test_refuses_to_overwrite_and_supports_self_test(self):
        self.assertIn("Refusing to overwrite", self.source)
        self.assertIn("--self-test", self.source)


class FinalizeMaskScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = FINALIZE.read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_writes_final_manifest_with_provenance(self):
        for name in ("manifest_final.csv", "final_counts.csv", "finalization.json",
                     "finalization_report.md"):
            self.assertIn(name, self.source)
        self.assertIn("--decision-note", self.source)
        self.assertIn("Refusing to overwrite", self.source)
        self.assertIn("no pixel data was modified", self.source)

    def test_validates_variant_and_mask_files(self):
        self.assertIn('VARIANTS = {"circle": "circle_mask", "v1": "v1_mask", "v2": "refined_mask"}', self.source)
        self.assertIn("Mask {args.variant} tidak ditemukan", self.source)
        self.assertIn("Ukuran foto/mask berbeda", self.source)


DOC = ROOT / "docs" / "MASK_VALIDATION_STEP34_41.md"


class MaskPipelineNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]

    def heading(self, prefix):
        return next(i for i, cell in enumerate(self.cells)
                    if cell["cell_type"] == "markdown"
                    and "".join(cell["source"]).startswith(f"---\n## {prefix} "))

    def code(self, prefix):
        return "".join(self.cells[self.heading(prefix) + 1]["source"])

    def markdown(self, prefix):
        return "".join(self.cells[self.heading(prefix)]["source"])

    def test_every_step_writes_pipeline_logs(self):
        config = self.code("34.")
        self.assertIn("def pipeline_log(", config)
        self.assertIn("def run_logged(", config)
        self.assertIn("pipeline_steps.jsonl", config)
        for prefix in ("35.", "36.", "37.", "38.", "39.", "39a.", "40.", "40a.", "41."):
            code = self.code(prefix)
            self.assertIn("pipeline_log(", code, prefix)
        for prefix in ("40.", "41."):
            self.assertIn("run_logged(", self.code(prefix), prefix)

    def test_documentation_covers_goals_and_steps(self):
        self.assertTrue(DOC.is_file(), DOC)
        self.assertTrue((ROOT / "docs" / "AGENT_PROMPT_STEP34_41.md").is_file())
        text = DOC.read_text(encoding="utf-8")
        for expected in ("Goal", "Step 34", "Step 39a", "Step 41",
                         "Test lock", "Augmentasi wajib", "Troubleshooting"):
            self.assertIn(expected, text)
        notebook = "".join("".join(cell["source"]) for cell in self.cells if cell["cell_type"] == "markdown")
        self.assertIn("MASK_VALIDATION_STEP34_41.md", notebook)

    def test_steps_exist_and_parse(self):
        indexes = [self.heading(prefix) for prefix in
                   ("35.", "36.", "37.", "38.", "39.", "39a.", "40.", "40a.", "41.")]
        self.assertEqual(indexes, sorted(indexes))
        for prefix in ("35.", "36.", "37.", "38.", "39.", "39a.", "40.", "40a.", "41."):
            ast.parse(self.code(prefix))

    def test_preprocessing_steps_are_not_training(self):
        for prefix in ("35.", "36.", "37.", "38.", "39.", "39a."):
            code = self.code(prefix)
            self.assertNotIn("--phase1-epochs", code)
            self.assertNotIn("--phase1-epochs", code)
            self.assertNotIn("--phase2-epochs", code)

    def test_step_35_and_36_run_mask_statistics(self):
        for prefix in ("35.", "36."):
            code = self.code(prefix)
            self.assertIn("evaluate_mask_variants.py", code)
            self.assertIn("RUN_", code)
        self.assertIn("--split', 'all'", self.code("35."))
        self.assertIn("step35_stats_pilot_v2", self.code("35."))
        self.assertIn("has_fallback", self.code("35."))
        self.assertIn("--no-overlays", self.code("36."))
        self.assertIn("step36_full_v2", self.code("36."))
        self.assertIn("step36_stats_train", self.code("36."))
        self.assertIn("step36_stats_test", self.code("36."))
        self.assertIn("jangan dipakai memilih varian", self.code("36."))

    def test_step_37_uses_train_statistics_and_marks_association_only(self):
        code = self.code("37.")
        self.assertIn("morphology_by_severity.png", code)
        self.assertIn("FULL_STATS_TRAIN", code)
        self.assertIn("bukan kausalitas", code)

    def test_step_38_displays_disagreement_cases(self):
        code = self.code("38.")
        self.assertIn("delta_iou", code)
        self.assertIn("comparison_grid.png", code)
        self.assertIn("RUN_VISUAL_REVIEW_38 = False", code)

    def test_step_39_requires_explicit_decision(self):
        code = self.code("39.")
        self.assertIn("RUN_FINALIZE_MASK_DATASET = False", code)
        self.assertIn("FINAL_MASK_VARIANT = None", code)
        self.assertIn("FINAL_DECISION_NOTE", code)
        self.assertIn("finalize_mask_dataset.py", code)
        self.assertIn("FINAL_MANIFEST", code)

    def test_step_40_is_guarded_and_capped_at_50_epochs(self):
        code = self.code("40.")
        self.assertIn("RUN_FINAL_TRAINING_40 = False", code)
        self.assertIn("0 < total_epochs <= 50", code)
        self.assertIn("FINAL_TRAIN_PHASE1_EPOCHS + FINAL_TRAIN_PHASE2_EPOCHS", code)
        self.assertIn("--no-validation", code)
        self.assertIn("--skip-test-eval", code)
        self.assertIn("manifest_final.csv", code)
        self.assertIn("Jalankan Step 39 lalu Step 39a dulu", code)
        self.assertIn("fetch_mask_v2_script('train_proposal_hybrid_resnet50_malds.py', force=True)", code)
        self.assertIn("required_trainer_flags", code)
        self.assertIn("Drive belum ter-mount", code)
        self.assertIn("FINAL_TRAIN_MANIFEST", code)
        self.assertIn("Jalankan Step 39 lalu Step 39a dulu", code)
        self.assertIn("Manifest training tidak memiliki validation", code)
        self.assertIn("target {target}/kelas", code)

    def test_step_39a_prepares_validation_and_augmented_train(self):
        code = self.code("39a.")
        self.assertIn("RUN_PREPARE_FINAL_TRAIN_39A = False", code)
        self.assertIn("split_train_validation.py", code)
        self.assertIn("augment_train_pairs_offline.py", code)
        self.assertIn("--target-per-class", code)
        self.assertIn("1000", code)
        self.assertIn("FINAL_TRAIN_MANIFEST", code)
        self.assertIn("TANPA TRAINING", self.markdown("39a."))
        self.assertNotIn("train_proposal_hybrid_resnet50_malds.py", code)

    def test_step_40a_displays_training_table_without_training(self):
        code = self.code("40a.")
        self.assertIn("FINAL_TRAIN_METRICS", code)
        self.assertIn(".get('history', [])", code)
        self.assertIn("metric_columns", code)
        self.assertIn("'val_'", code)
        self.assertIn("describe()", code)
        self.assertIn("checkpoint_selection", code)
        self.assertIn("select_training_run", code)
        self.assertIn("glob('*/metrics.json')", code)
        self.assertIn("history_partial.json", code)
        self.assertIn("last_checkpoint.pt", code)
        self.assertIn("Run terdeteksi", code)
        self.assertIn("TANPA TRAINING", self.markdown("40a."))
        self.assertNotIn("--phase1-epochs", code)
        self.assertNotIn("subprocess.run", code)

    def test_step_41_tests_the_locked_checkpoint(self):
        code = self.code("41.")
        self.assertIn("RUN_FINAL_TEST_41 = False", code)
        self.assertIn("infer_testing_images.py", code)
        self.assertIn("testing_predictions.csv", code)
        self.assertIn("classification_report.csv", code)
        self.assertIn("confusion_matrix.png", code)
        self.assertIn("--split', FINAL_TEST_SPLIT", code)
        self.assertIn("level3", code)
        self.assertIn("select_training_run", code)
        self.assertIn("TRAIN_STATUS", code)
        self.assertIn("train_baseline_dilated_unet.py'", code)
        self.assertIn("for dependency in ('train_proposal_hybrid_resnet50_malds.py'", code)
        self.assertIn("belum selesai", code)
        self.assertIn("TANPA TRAINING", self.markdown("41."))
        self.assertNotIn("--phase1-epochs", code)


if __name__ == "__main__":
    unittest.main()
