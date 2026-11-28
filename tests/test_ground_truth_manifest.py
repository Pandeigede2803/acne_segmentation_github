"""Tests for the deterministic ground-truth manifest builder."""

import unittest

from scripts.build_ground_truth_manifest import (build_manifest_rows,
                                                 build_review_rows,
                                                 sample_audit, unique_files)


def synthetic_folds():
    base = {
        "levle0_1.jpg": (0, 3),
        "levle1_2.jpg": (0, 4),
        "levle2_3.jpg": (2, 25),
    }
    folds = {}
    for fold in range(5):
        rows = dict(base)
        if fold == 4:
            rows["levle0_1.jpg"] = (0, 5)  # count tidak stabil
        folds[fold] = rows
    return folds


CIRCLES = {
    "levle0_1.jpg": 10,
    "levle1_2.jpg": 12,
    "levle2_3.jpg": 40,
}


class ManifestTests(unittest.TestCase):
    def test_nnew_policy_uses_fold_count_and_label(self):
        rows = build_manifest_rows(synthetic_folds(), CIRCLES, "nnew")
        self.assertEqual(len(rows), 15)
        by_key = {(row["file_name"], row["fold"]): row for row in rows}
        self.assertEqual(by_key[("levle0_1.jpg", 4)]["count"], 5)
        self.assertEqual(by_key[("levle0_1.jpg", 4)]["count_median"], 3.0)
        self.assertEqual(by_key[("levle2_3.jpg", 0)]["grade"], 2)

    def test_v2_policy_uses_circle_count_and_grade(self):
        rows = build_manifest_rows(synthetic_folds(), CIRCLES, "v2")
        by_key = {(row["file_name"], row["fold"]): row for row in rows}
        self.assertEqual(by_key[("levle2_3.jpg", 0)]["count"], 40)
        self.assertEqual(by_key[("levle2_3.jpg", 0)]["grade"], 2)

    def test_flags_are_per_file(self):
        rows = build_manifest_rows(synthetic_folds(), CIRCLES, "nnew")
        files = {row["file_name"]: row for row in unique_files(rows)}
        self.assertEqual(files["levle1_2.jpg"]["flag_prefix_mismatch"], 1)
        self.assertEqual(files["levle0_1.jpg"]["flag_count_unstable"], 1)
        self.assertEqual(files["levle2_3.jpg"]["flag_prefix_mismatch"], 0)

    def test_review_includes_prefix_mismatch_for_nnew(self):
        rows = build_manifest_rows(synthetic_folds(), CIRCLES, "nnew")
        review = {row["file_name"] for row in build_review_rows(rows)}
        self.assertIn("levle1_2.jpg", review)

    def test_audit_sample_respects_per_grade_limit_and_seed(self):
        rows = build_manifest_rows(synthetic_folds(), CIRCLES, "nnew")
        first = sample_audit(rows, per_grade=1, seed=42)
        second = sample_audit(rows, per_grade=1, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)  # grade 0 dan grade 2
        self.assertEqual({row["grade"] for row in first}, {0, 2})


if __name__ == "__main__":
    unittest.main()
