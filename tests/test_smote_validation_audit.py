"""Validation report tests; no PyTorch, image access, or training required."""
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'audit_smote_validation', Path(__file__).resolve().parents[1] / 'scripts/audit_smote_validation.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ValidationSummaryTests(unittest.TestCase):
    def test_confusion_direction_and_class_metrics(self):
        records = [{'true_label': a, 'predicted_label': b}
                   for a, b in [(0, 0), (0, 1), (0, 1), (1, 1), (2, 0)]]
        result = AUDIT.summarize(records, ['0', '1', '2', '3'])
        self.assertEqual(result['confusion_matrix'], [[1, 2, 0, 0], [0, 1, 0, 0],
                                                     [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertAlmostEqual(result['accuracy'], 0.4)
        self.assertEqual(result['confused_pairs'][0], {'true_label': 0, 'predicted_label': 1, 'count': 2})
        first = result['per_class'][0]
        self.assertEqual(first['support'], 3)
        self.assertEqual(first['errors'], 2)
        self.assertAlmostEqual(first['precision'], 0.5)
        self.assertAlmostEqual(first['recall'], 1 / 3)
        self.assertAlmostEqual(first['f1'], 0.4)
        self.assertAlmostEqual(first['error_rate'], 2 / 3)

    def test_missing_class_and_zero_predictions(self):
        result = AUDIT.summarize([{'true_label': 0, 'predicted_label': 1}], ['0', '1', '2'])
        self.assertEqual(result['per_class'][0]['precision'], 0)
        self.assertEqual(result['per_class'][2]['support'], 0)
        self.assertIsNone(result['per_class'][2]['error_rate'])

    def test_perfect_predictions(self):
        result = AUDIT.summarize([{'true_label': i, 'predicted_label': i} for i in range(4)], list('0123'))
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['confused_pairs'], [])
        self.assertTrue(all(row['f1'] == 1 for row in result['per_class']))

    def test_empty_predictions(self):
        result = AUDIT.summarize([], list('0123'))
        self.assertIsNone(result['accuracy'])
        self.assertEqual(result['confused_pairs'], [])


if __name__ == '__main__':
    unittest.main()
