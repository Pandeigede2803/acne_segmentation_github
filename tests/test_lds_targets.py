"""Tests for the Prokhorov & Kalinin label-distribution targets."""

import unittest

from scripts import lds_targets as lds


class EpsilonScheduleTests(unittest.TestCase):
    def test_schedule_length_and_range(self):
        schedule = lds.piecewise_epsilon()
        self.assertEqual(len(schedule), 65)
        self.assertAlmostEqual(min(schedule), 0.6, places=6)
        self.assertAlmostEqual(max(schedule), 1.0, places=6)

    def test_boundaries_are_one_and_midrange_is_eps_min(self):
        schedule = lds.piecewise_epsilon(eps_min=0.6)
        for boundary in (4, 5, 19, 20, 49, 50):
            self.assertAlmostEqual(schedule[boundary], 1.0, places=6, msg=str(boundary))
        for midpoint in (12, 34, 64):
            self.assertAlmostEqual(schedule[midpoint], 0.6, places=6, msg=str(midpoint))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            lds.piecewise_epsilon(eps_min=-0.1)
        with self.assertRaises(ValueError):
            lds.piecewise_epsilon(eps_min=0.7, eps_max=0.6)
        with self.assertRaises(ValueError):
            lds.piecewise_epsilon(num_counts=64)


class DistributionTests(unittest.TestCase):
    def test_gaussian_sums_to_one(self):
        for count in (1, 12, 30, 60, 65):
            index = lds.count_index(count)
            distribution = lds.gaussian_count_distribution(index)
            self.assertEqual(len(distribution), 65)
            self.assertAlmostEqual(sum(distribution), 1.0, places=10)
            self.assertEqual(max(range(65), key=lambda i: distribution[i]), index)
            if index > 0:
                self.assertGreater(distribution[index], distribution[0])

    def test_count_index_clips(self):
        self.assertEqual(lds.count_index(0), 0)
        self.assertEqual(lds.count_index(1), 0)
        self.assertEqual(lds.count_index(65), 64)
        self.assertEqual(lds.count_index(999), 64)

    def test_smoothed_distribution_sums_to_one(self):
        for count in (1, 5, 6, 20, 21, 50, 51, 64):
            smoothed = lds.smooth_count_distribution(lds.count_index(count))
            self.assertAlmostEqual(sum(smoothed), 1.0, places=9)
            self.assertGreaterEqual(min(smoothed), 0.0)

    def test_smoothing_keeps_the_true_bin_dominant(self):
        index = lds.count_index(30)
        smoothed = lds.smooth_count_distribution(index)
        self.assertEqual(max(range(65), key=lambda i: smoothed[i]), index)

    def test_severity_projections_sum_to_one(self):
        count = 30
        targets = lds.targets_for_count(count)
        self.assertAlmostEqual(sum(targets["severity_13"]), 1.0, places=9)
        self.assertAlmostEqual(sum(targets["severity_4"]), 1.0, places=9)
        self.assertEqual(len(targets["severity_13"]), 13)
        self.assertEqual(len(targets["severity_4"]), 4)

    def test_hayashi_thresholds(self):
        cases = {1: 0, 5: 0, 6: 1, 20: 1, 21: 2, 50: 2, 51: 3, 80: 3}
        for count, grade in cases.items():
            self.assertEqual(lds.count_to_severity_index(count), grade, msg=str(count))


if __name__ == "__main__":
    unittest.main()
