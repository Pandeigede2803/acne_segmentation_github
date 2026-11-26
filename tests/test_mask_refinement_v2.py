import ast
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "acne_colab.ipynb"
SCRIPT = ROOT / "scripts" / "refine_masks_color_v2.py"


def level_from_filename(filename):
    return Path(filename).stem.split("_")[0]


class MaskRefinementV2ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_script_writes_documented_audit_artifacts(self):
        for expected in (
            "audit_per_image.csv",
            "audit_per_annotation.csv",
            "audit_per_class.csv",
            "comparison_grid.png",
            "audit_report.md",
            "refinement_config.json",
            "provenance.json",
            "refined_manifest_v2.csv",
        ):
            self.assertIn(expected, self.source)

    def test_v1_control_and_v2_annotation_fallback_are_explicit(self):
        self.assertIn("V1_SIGMA, V1_MORPH_KERNEL, V1_MIN_RATIO = 1.2, 5, 0.02", self.source)
        self.assertIn("v1.refine_mask_with_red(", self.source)
        self.assertIn("fallback_reason", self.source)
        self.assertIn("annulus_skin", self.source)
        self.assertIn("Refusing to overwrite existing output folder", self.source)
        self.assertIn("annotation_outside_image", self.source)
        self.assertIn("out_of_bounds_annotations", self.source)

    def test_grid_and_selection_options_exist(self):
        for expected in ("--per-level", "--split", "--levels", "--write-grids",
                         "--grid-samples", "--grid-focus", "--manifest",
                         "--min-valid-pixels"):
            self.assertIn(expected, self.source)

    def test_selection_is_stratified_deterministic_and_split_aware(self):
        function = next(node for node in self.tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "select_images")
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {"random": random, "v1": SimpleNamespace(level_from_filename=level_from_filename)}
        exec(compile(module, "step34_selection", "exec"), namespace)
        select = namespace["select_images"]
        names = [f"levle{level}_{index}.jpg" for level in range(4) for index in range(3)]
        rows = [{"filename": name, "label_name": name.split("_")[0],
                 "split": "train" if not name.endswith("_2.jpg") else "test"}
                for name in names]

        first = select(names, rows, ("levle3",), 1, "train", 42, 0)
        second = select(names, rows, ("levle3",), 1, "train", 42, 0)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["label_name"], "levle3")
        self.assertEqual(first[0]["split"], "train")
        self.assertFalse(first[0]["filename"].endswith("_2.jpg"))

        capped = select(names, rows, (), 2, "", 42, 3)
        self.assertEqual(len(capped), 3)
        with self.assertRaises(ValueError):
            select(names, rows, ("levle3",), 1, "missing", 42, 0)


class MaskRefinementV2NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]

        def heading(prefix):
            return next(i for i, cell in enumerate(cells)
                        if cell["cell_type"] == "markdown"
                        and "".join(cell["source"]).startswith(f"---\n## {prefix} "))

        cls.config_index = heading("34.")
        cls.validate_index = heading("34a.")
        cls.run_index = heading("34b.")
        cls.report_index = heading("34c.")
        cls.markdown = {
            "config": "".join(cells[cls.config_index]["source"]),
            "validate": "".join(cells[cls.validate_index]["source"]),
            "run": "".join(cells[cls.run_index]["source"]),
            "report": "".join(cells[cls.report_index]["source"]),
        }
        cls.code = {
            "config": "".join(cells[cls.config_index + 1]["source"]),
            "validate": "".join(cells[cls.validate_index + 1]["source"]),
            "run": "".join(cells[cls.run_index + 1]["source"]),
            "report": "".join(cells[cls.report_index + 1]["source"]),
        }
        for source in cls.code.values():
            ast.parse(source)
        assert cls.config_index < cls.validate_index < cls.run_index < cls.report_index

    def test_step_34_is_split_into_four_focused_cells(self):
        self.assertIn("Empat sub-step", self.markdown["config"])
        for name in ("config", "validate", "run", "report"):
            self.assertIn("TANPA TRAINING", self.markdown[name].upper())
        self.assertIn("RUN_MASK_REFINEMENT_V2 = False", self.code["run"])
        self.assertIn("subprocess.run(MASK_V2_COMMAND", self.code["run"])
        for name in ("config", "validate", "report"):
            self.assertNotIn("subprocess.run(MASK_V2_COMMAND", self.code[name])
        for name in ("config", "validate", "run", "report"):
            self.assertNotIn("train_proposal_hybrid_resnet50_malds.py", self.code[name])
            self.assertNotIn("--no-validation", self.code[name])
            self.assertNotIn("--resume", self.code[name])

    def test_validation_reads_step_31_and_prepares_command(self):
        self.assertIn("Jalankan Step 31 dulu", self.code["validate"])
        self.assertIn("preprocess_512", self.code["validate"])
        self.assertIn("manifest_split.csv", self.code["validate"])
        self.assertIn("experiments_512_mask_v2", self.code["validate"])
        self.assertIn("MASK_V2_SOURCE_BASE", self.code["validate"])
        self.assertIn("MASK_V2_COMMAND", self.code["validate"])
        self.assertIn("MASK_V2_READY = True", self.code["validate"])
        self.assertIn("Jalankan Step 34 dulu", self.code["validate"])

    def test_report_shows_v1_vs_v2_grid_and_audit(self):
        self.assertIn("comparison_grid.png", self.code["report"])
        self.assertIn("audit_per_class.csv", self.code["report"])
        self.assertIn("display(Image(", self.code["report"])
        self.assertIn("refine_masks_color_v2.py", self.code["validate"])
        self.assertIn("--write-grids", self.code["validate"])
        self.assertIn("--per-level", self.code["validate"])
        self.assertIn("--split", self.code["validate"])
        self.assertIn("Jalankan Step 34b dulu", self.code["report"])

    def test_missing_temporary_annotation_is_recovered_and_verified(self):
        for expected in ("MASK_V2_ANNOTATIONS", "MASK_V2_IMAGE_DIR",
                         "V2_ANNOT_SHA256", "mask_v2_resolve_annotation",
                         "mask_v2_resolve_image_dir", "Acne04-v2_annotations.json"):
            self.assertIn(expected, self.code["config"])
        self.assertIn("SHA-256", self.code["config"])

    def test_small_lesion_defaults_are_audit_friendly(self):
        self.assertIn("MASK_V2_MIN_VALID_PIXELS = 10", self.code["config"])
        self.assertIn("MASK_V2_ANNULUS_FACTOR = 3.0", self.code["config"])
        self.assertIn("MASK_V2_MORPH_KERNEL = 1", self.code["config"])
        self.assertIn("--min-valid-pixels", self.code["validate"])
        self.assertIn("required_v2_flags = ('--per-level', '--min-valid-pixels', '--annulus-radius-factor')", self.code["validate"])

    def test_reuses_existing_output_only_with_matching_provenance(self):
        self.assertIn("provenance.json", self.code["validate"])
        self.assertIn("v2_local_annulus", self.code["validate"])
        self.assertIn("Ganti MASK_V2_RUN_NAME", self.code["validate"])


if __name__ == "__main__":
    unittest.main()
