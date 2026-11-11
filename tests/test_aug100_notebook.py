"""Static notebook protocol checks; no GPU, pandas, or training required."""

import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "acne_colab.ipynb"


class AugmentationNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cells = json.loads(NOTEBOOK.read_text())["cells"]
        cls.training = "".join(cells[28]["source"])
        cls.ablation = "".join(cells[49]["source"])

    def test_cells_are_valid_python(self):
        ast.parse(self.training)
        ast.parse(self.ablation)

    def test_complete_variant_set(self):
        tree = ast.parse(self.ablation)
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ABLATION_VARIANTS"
                    for t in node.targets)
        )
        self.assertEqual(ast.literal_eval(assignment.value), [
            "baseline", "no_attention", "no_dilation", "no_amfm",
            "lds", "no_fusion", "proposed",
        ])

    def test_ablation_command_excludes_test_and_sampler(self):
        assignments = [node for node in ast.walk(ast.parse(self.ablation))
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "cmd"
                               for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        literals = [node.value for node in assignments[0].value.elts
                    if isinstance(node, ast.Constant)]
        self.assertIn("--skip-test-eval", literals)
        self.assertNotIn("--weighted-sampler", literals)
        self.assertIn("--label-distribution", literals)
        self.assertIn("auto", literals)
        self.assertIn("balanced", literals)

    def test_augmented_manifest_and_new_output(self):
        self.assertIn("pd.read_csv(AUGMENTED_MANIFEST", self.ablation)
        self.assertIn("isin(['train', 'val'])", self.ablation)
        self.assertNotIn("FILTERED_MANIFEST", self.ablation)
        self.assertIn("_ablation_aug100_v1", self.ablation)

    def test_default_off_and_provenance_guards(self):
        self.assertIn("RUN_ABLATIONS = False", self.ablation)
        self.assertIn("step11_protocol['sha256'] != current_hashes", self.ablation)
        self.assertIn("json.loads(protocol_path.read_text()) != ablation_protocol",
                      self.ablation)
        self.assertIn("ABLATION_MANIFEST.read_text() != manifest_text", self.ablation)


if __name__ == "__main__":
    unittest.main()
