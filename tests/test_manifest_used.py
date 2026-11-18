"""Every trainer that writes metrics.json must record the manifest it used."""

import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TRAINERS = (
    "train_proposal_hybrid_resnet50_malds.py",
    "train_single_task_resnet50.py",
    "train_layer4_finetune.py",
    "train_smote_classifier_head.py",
    "train_baseline_dilated_unet.py",
    "code_dari_claude/train_hybrid_resnet50_staged.py",
)


class ManifestUsedTests(unittest.TestCase):
    def test_every_metrics_writer_records_manifest_used(self):
        for name in TRAINERS:
            with self.subTest(trainer=name):
                source = (SCRIPTS / name).read_text(encoding="utf-8")
                self.assertIn("manifest_used", source)


if __name__ == "__main__":
    unittest.main()
