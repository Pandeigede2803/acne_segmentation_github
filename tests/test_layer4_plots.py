"""Plotting protocol tests; synthetic fixtures are temporary, never research results."""
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('layer4_plots', ROOT / 'scripts/plot_layer4_results.py')
PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTS)


def fixture(status='completed', score=0.5):
    best = {'val_loss': 0.8, 'accuracy': score, 'macro_f1': 0.4, 'val_dice': 0.6, 'val_iou': 0.45,
            'confusion_matrix': [[2, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            'recall_per_class': [2 / 3, 1 / 2, 0, 0]}
    return {'status': status, 'best_epoch': 0, 'best_validation': best, 'initial_validation': best,
            'class_names': ['levle0', 'levle1', 'levle2', 'levle3'],
            'history': [{'epoch': 1, 'train_total_loss': 2.0, **best}]}


class PlotProtocolTests(unittest.TestCase):
    def test_partial_excluded_from_aggregate_and_pairs(self):
        runs = {('frozen', 42): fixture(), ('layer4', 42): fixture(score=0.75),
                ('frozen', 43): fixture('running'), ('layer4', 43): fixture()}
        self.assertEqual(len(PLOTS.completed_runs(runs)), 3)
        self.assertEqual(PLOTS.paired_deltas(runs, 'accuracy'), [(42, 0.25)])

    def test_unpaired_runs_do_not_make_deltas(self):
        runs = {('frozen', 42): fixture(), ('layer4', 43): fixture()}
        self.assertEqual(PLOTS.paired_deltas(runs, 'accuracy'), [])

    def test_no_results_produces_no_charts(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            self.assertEqual(PLOTS.plot_results(root, [42]), [])
            self.assertEqual(list(root.iterdir()), [])

    def test_only_requested_seeds_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for seed in [42, 43]:
                run = root / f'frozen_seed{seed}'
                run.mkdir()
                (run / 'metrics.json').write_text(json.dumps(fixture()))
            self.assertEqual(set(PLOTS.load_runs(root, [42])), {('frozen', 42)})

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib not installed')
    def test_rendering_outputs_and_preserves_source_results(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            sources = {}
            for mode, seed, status in [('frozen', 42, 'completed'), ('layer4', 42, 'completed'),
                                       ('layer4', 43, 'running')]:
                run = root / f'{mode}_seed{seed}'
                run.mkdir()
                path = run / 'metrics.json'
                sources[path] = json.dumps(fixture(status))
                path.write_text(sources[path])
            saved = PLOTS.plot_results(root, [42, 43])
            self.assertEqual(len(saved), 11)
            for path in saved:
                self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            for path, original in sources.items():
                self.assertEqual(path.read_text(), original)
            with (root / 'charts/seed_summary.csv').open() as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 10)
            self.assertTrue(all(row['n'] == '1' and row['sample_sd'] == '' for row in summary))
            with (root / 'charts/paired_deltas.csv').open() as handle:
                deltas = list(csv.DictReader(handle))
            self.assertTrue(all(row['seed'] == '42' for row in deltas))


if __name__ == '__main__':
    unittest.main()
