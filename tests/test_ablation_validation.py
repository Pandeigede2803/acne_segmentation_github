"""Small local fixtures for the read-only ablation validation evaluation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_ablation_validation import VARIANTS, evaluate


class AblationValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "train_val_manifest.csv"
        self.manifest.write_text("split\ntrain\nval\n", encoding="utf-8")
        (self.root / "ablation_provenance.json").write_text(json.dumps({
            "variants": list(VARIANTS), "skip_test_eval": True,
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
        }))
        for i, variant in enumerate(VARIANTS):
            path = self.root / variant
            path.mkdir()
            (path / "metrics.json").write_text(json.dumps({
                "config": {"variant": variant, "manifest": str(self.manifest),
                           "skip_test_eval": True, "weighted_sampler": False,
                           "class_weighting": "balanced"},
                "test_metrics": None, "best_val_loss": 0.5,
                "history": [
                    {"epoch": 1, "val_loss": 1.0, "train_acc": 0.8,
                     "val_acc": 0.99, "val_kappa": 0.9, "val_dice": 0.8, "val_iou": 0.7},
                    {"epoch": 2, "val_loss": 0.5, "train_acc": 0.7,
                     "val_acc": i / 10, "val_kappa": 0.3, "val_dice": 0.4, "val_iou": 0.2},
                ],
            }))

    def test_uses_lowest_validation_loss_and_proposed_from_ablation(self):
        rows = evaluate(self.root)
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0]["best_epoch"], 2)
        self.assertAlmostEqual(rows[0]["delta_val_accuracy_vs_proposed"], -0.6)
        self.assertEqual(rows[-1]["delta_val_accuracy_vs_proposed"], 0)

    def test_missing_run_rejected(self):
        (self.root / "proposed" / "metrics.json").unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate(self.root)

    def test_test_split_rejected(self):
        self.manifest.write_text("split\ntrain\nval\ntest\n", encoding="utf-8")
        provenance = self.root / "ablation_provenance.json"
        data = json.loads(provenance.read_text())
        data["manifest_sha256"] = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        provenance.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            evaluate(self.root)

    def test_wrong_variant_or_test_metrics_rejected(self):
        path = self.root / "baseline" / "metrics.json"
        data = json.loads(path.read_text())
        data["config"]["variant"] = "proposed"
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            evaluate(self.root)
        data["config"]["variant"] = "baseline"
        data["test_metrics"] = {"acc": 1}
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            evaluate(self.root)

    def test_different_seed_rejected(self):
        path = self.root / "no_attention" / "metrics.json"
        data = json.loads(path.read_text())
        data["config"]["seed"] = 99
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            evaluate(self.root)


if __name__ == "__main__":
    unittest.main()
