"""Tests for Step 42-48 recovery helpers and notebook wiring."""

import ast
import json
import unittest
from pathlib import Path

from scripts import recovery_common as rc

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "acne_colab.ipynb"


class ControlledTargetTests(unittest.TestCase):
    def test_formula_matches_protocol_table(self):
        counts = {0: 240, 1: 295, 2: 67, 3: 11}
        targets = rc.controlled_targets(counts, target=300, cap=7)
        self.assertEqual(targets, {0: 300, 1: 300, 2: 300, 3: 77})
        self.assertEqual(sum(targets.values()), 977)

    def test_no_undersampling_and_cap(self):
        self.assertEqual(rc.controlled_target(500, 300, 7), 500)
        self.assertEqual(rc.controlled_target(10, 300, 7), 70)

    def test_even_allocation_spreads_remainder(self):
        self.assertEqual(rc.even_allocation(10, 3), [4, 3, 3])
        self.assertEqual(rc.even_allocation(0, 3), [0, 0, 0])
        self.assertEqual(rc.max_synthetic_per_source(7), 6)


class GroupSplitTests(unittest.TestCase):
    def test_grouped_split_is_deterministic_and_stratified(self):
        groups = {f"g{index}": index % 4 for index in range(40)}
        first = rc.assign_grouped_splits(groups, 0.6, 0.2, 0.2, seed=7)
        second = rc.assign_grouped_splits(groups, 0.6, 0.2, 0.2, seed=7)
        self.assertEqual(first, second)
        for cls in range(4):
            splits = [first[group] for group, label in groups.items() if label == cls]
            self.assertEqual(set(splits), {"train", "val", "test"})

    def test_ratio_validation(self):
        with self.assertRaises(ValueError):
            rc.assign_grouped_splits({"a": 0, "b": 0, "c": 0}, 0.5, 0.2, 0.2)

    def test_small_class_still_gets_three_splits(self):
        groups = {f"g{index}": 0 for index in range(4)}
        assignment = rc.assign_grouped_splits(groups, 0.6, 0.2, 0.2, seed=42)
        self.assertEqual(sorted(assignment.values()), ["test", "train", "train", "val"])


class DuplicateGroupTests(unittest.TestCase):
    def test_pairs_to_groups_is_transitive(self):
        groups = rc.pairs_to_groups(["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
        self.assertEqual(groups["a"], groups["c"])
        self.assertNotEqual(groups["a"], groups["d"])

    def test_stable_group_ids(self):
        names = rc.stable_group_ids(["z", "a", "z"])
        self.assertEqual(names["a"], "dup_0001")
        self.assertEqual(names["z"], "dup_0002")


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.matrix = [[5, 1, 0, 0], [2, 4, 0, 0], [0, 1, 3, 1], [0, 0, 2, 1]]

    def test_report_and_totals(self):
        report = rc.classification_report(self.matrix)
        self.assertEqual([row["support"] for row in report], [6, 6, 5, 3])
        self.assertAlmostEqual(rc.macro_f1_from_report(report),
                               sum(row["f1"] for row in report) / 4)

    def test_confusion_matrix_roundtrip(self):
        true = [0, 0, 1, 2, 3]
        pred = [0, 1, 1, 2, 2]
        self.assertEqual(rc.confusion_matrix(true, pred), [[1, 1, 0, 0], [0, 1, 0, 0],
                                                           [0, 0, 1, 0], [0, 0, 1, 0]])

    def test_kappa_perfect_and_chance(self):
        self.assertAlmostEqual(rc.cohen_kappa([[5, 0], [0, 5]]), 1.0)
        self.assertAlmostEqual(rc.cohen_kappa([[2, 2], [2, 2]]), 0.0)

    def test_ordinal_metrics(self):
        self.assertAlmostEqual(rc.ordinal_mae([0, 1, 3], [0, 2, 1]), (0 + 1 + 2) / 3)
        self.assertEqual(rc.ordinal_bias([0, 1, 3], [0, 2, 1]),
                         {"underestimated": 1, "overestimated": 1, "exact": 1})

    def test_wilson_interval_bounds(self):
        low, high = rc.wilson_interval(0, 9)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        low, high = rc.wilson_interval(9, 9)
        self.assertEqual(high, 1.0)
        self.assertEqual(rc.wilson_interval(0, 0), (0.0, 0.0))


class RecoveryScriptContractTests(unittest.TestCase):
    EXPECTED = {
        "detect_exact_duplicates.py": [
            "exact_duplicate_groups.csv", "label_mask_conflicts.csv",
            "duplicate_review_grid.png", "quarantine_manifest.csv",
            "clean_original_manifest.csv", "step42_report.md",
        ],
        "detect_near_duplicates.py": [
            "near_duplicate_candidates.csv", "near_duplicate_review.csv",
            "step43_near_duplicate_report.md",
        ],
        "build_grouped_split.py": [
            "split_manifest.csv", "split_counts.csv", "split_integrity.json",
            "split_proposal.md",
        ],
        "audit_augmentation_visual.py": [
            "visual_audit_level3", "visual_audit_level2", "visual_audit_controls",
            "visual_audit_scores.csv", "augmentation_quality_report.md",
        ],
        "build_controlled_augmentation.py": [
            "controlled_aug_manifest.csv", "augmentation_lineage.csv",
            "augmentation_config.json", "controlled_aug_integrity.json",
            "controlled_aug_report.md",
        ],
        "audit_class_weighting.py": [
            "class_weight_audit.md", "experiment_matrix.yaml",
            "training_budget.md", "pretraining_go_no_go.md",
        ],
        "evaluate_final_experiment.py": [
            "test_metrics.json", "classification_report.csv",
            "confusion_matrix.png", "confusion_matrix_normalized.png",
            "test_predictions.csv", "final_research_report.md",
        ],
    }

    def test_scripts_exist_and_emit_documented_artifacts(self):
        for name, artifacts in self.EXPECTED.items():
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            ast.parse(source)
            for artifact in artifacts:
                self.assertIn(artifact, source, f"{name} missing {artifact}")

    def test_counting_head_contract(self):
        trainer = (ROOT / "scripts" / "train_proposal_hybrid_resnet50_malds.py").read_text(encoding="utf-8")
        for expected in ("--count-head", "--count-weight", "count_pred",
                         "count_mae", "n_circles", "nnew_count", "count_head"):
            self.assertIn(expected, trainer)
        inference = (ROOT / "scripts" / "infer_testing_images.py").read_text(encoding="utf-8")
        self.assertIn("has_count_head", inference)
        self.assertIn("count_head=has_count_head", inference)

    def test_trainer_is_self_contained(self):
        source = (ROOT / "scripts" / "train_proposal_hybrid_resnet50_malds.py").read_text(encoding="utf-8")
        self.assertNotIn("from recovery_common import", source)
        self.assertIn("def per_class_report(", source)

    def test_trainer_supports_macro_f1_selection_and_budget(self):
        source = (ROOT / "scripts" / "train_proposal_hybrid_resnet50_malds.py").read_text(encoding="utf-8")
        for expected in ("--checkpoint-metric", "val_macro_f1", "--early-stopping-patience",
                         "--max-optimizer-steps", "epoch_metrics.csv", "optimizer_steps"):
            self.assertIn(expected, source)

    def test_recovery_docs_and_prompt_exist(self):
        for name in ("RECOVERY_STEP42_48.md", "AGENT_PROMPT_STEP42_48.md"):
            self.assertTrue((ROOT / "docs" / name).is_file(), name)

    def test_status_cell_and_auto_skip(self):
        text = "\n".join("".join(cell["source"]) for cell in
                          json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])
        self.assertIn("## 41b.", text)
        self.assertIn("Rekomendasi:", text)
        for phrase in ("✅ Step 42 sudah selesai", "✅ Step 43A sudah selesai",
                       "✅ Step 43B sudah selesai", "✅ Step 44 sudah selesai",
                       "✅ Step 45 sudah selesai", "✅ Step 46 sudah selesai",
                       "✅ Step 47 sudah selesai", "✅ Step 48 sudah selesai"):
            self.assertIn(phrase, text)
        self.assertIn("STEP43_SPLIT_MANIFEST", text)

    def test_recovery_cells_execute_with_run_disabled(self):
        import contextlib
        import io
        import sys
        import types

        fake_pandas = types.ModuleType("pandas")
        fake_pandas.DataFrame = lambda *args, **kwargs: None
        fake_pandas.read_csv = lambda *args, **kwargs: None
        fake_ipython = types.ModuleType("IPython")
        fake_display = types.ModuleType("IPython.display")
        fake_display.display = lambda *args, **kwargs: None
        fake_display.Image = lambda *args, **kwargs: None
        saved = {name: sys.modules.get(name) for name in
                 ("pandas", "IPython", "IPython.display")}
        sys.modules.update({"pandas": fake_pandas, "IPython": fake_ipython,
                            "IPython.display": fake_display})
        try:
            cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
            markers = ("RUN_STEP42", "RUN_STEP43_NEAR", "RUN_STEP43_SPLIT",
                       "RUN_STEP44_VISUAL", "RUN_STEP45_CONTROLLED",
                       "RUN_STEP46_WEIGHTING", "RUN_STEP47_TRAINING", "RUN_STEP48_EVAL")
            with contextlib.redirect_stdout(io.StringIO()):
                for index, cell in enumerate(cells):
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    if not any(marker in source for marker in markers):
                        continue
                    namespace = {"__name__": "__main__"}
                    try:
                        exec(compile(source, f"cell{index}", "exec"), namespace)
                    except Exception as error:  # noqa: BLE001 - surface cell failures
                        self.fail(f"cell {index} failed with RUN=False: {type(error).__name__}: {error}")
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_pipeline_log_calls_do_not_shadow_status(self):
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            try:
                tree = ast.parse("".join(cell["source"]))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name == "pipeline_log":
                    keywords = [kw.arg for kw in node.keywords]
                    self.assertNotIn("status", keywords, f"cell {index}: status shadowing")
                    self.assertNotIn("step", keywords, f"cell {index}: step shadowing")

    def test_notebook_contains_steps_42_to_48(self):
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        text = "\n".join("".join(cell["source"]) for cell in cells)
        for step in range(42, 49):
            if step == 43:
                self.assertIn("## 43A.", text)
                self.assertIn("## 43B.", text)
            else:
                self.assertIn(f"## {step}.", text)
        self.assertIn("pretraining_go_no_go.md", text)
        self.assertIn("val_macro_f1", text)
        # Shared helper must be fetched next to every standalone recovery script.
        self.assertEqual(text.count("fetch_mask_v2_script('recovery_common.py')"), 7)


if __name__ == "__main__":
    unittest.main()
