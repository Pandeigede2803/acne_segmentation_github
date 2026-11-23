"""Tests for absolute per-class offline augmentation targets."""

import unittest

from scripts.augment_train_pairs_offline import class_targets


class ClassTargetTests(unittest.TestCase):
    def test_absolute_target_fills_every_class(self):
        counts = {0: 282, 1: 347, 2: 79, 3: 13}
        self.assertEqual(class_targets(counts, 0.75, 1000), {0: 1000, 1: 1000, 2: 1000, 3: 1000})

    def test_absolute_target_keeps_larger_class(self):
        self.assertEqual(class_targets({0: 1200, 1: 5}, 1.0, 1000), {0: 1200, 1: 1000})

    def test_ratio_mode_is_unchanged(self):
        self.assertEqual(class_targets({0: 100, 1: 50}, 1.0), {0: 100, 1: 100})
        self.assertEqual(class_targets({0: 100, 1: 50}, 0.75), {0: 100, 1: 75})

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValueError):
            class_targets({0: 10}, 0.75, -1)
        with self.assertRaises(ValueError):
            class_targets({0: 10}, 0)


if __name__ == "__main__":
    unittest.main()
