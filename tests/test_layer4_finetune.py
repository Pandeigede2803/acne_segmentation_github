"""Lightweight metric/protocol tests, plus optional PyTorch trainability checks."""
import ast
import contextlib
import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('layer4', ROOT / 'scripts/train_layer4_finetune.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        result = MODULE.classification_metrics([[int(i == j) for j in range(4)] for i in range(4)])
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['macro_f1'], 1)
        self.assertEqual(result['recall_per_class'], [1] * 4)

    def test_missing_classes_and_confusion_direction(self):
        result = MODULE.classification_metrics([[1, 2, 0, 0], [0, 1, 0, 0],
                                                [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(result['accuracy'], 0.4)
        self.assertEqual(result['recall_per_class'], [1 / 3, 1, 0, 0])
        self.assertAlmostEqual(result['macro_f1'], (0.4 + 0.5) / 4)

    def test_empty(self):
        result = MODULE.classification_metrics([[0] * 4 for _ in range(4)])
        self.assertEqual(result['accuracy'], 0)
        self.assertEqual(result['macro_f1'], 0)

    def test_notebook_step28_code_is_valid_and_training_off(self):
        notebook = json.loads((ROOT / 'acne_colab.ipynb').read_text())
        self.assertTrue(''.join(notebook['cells'][-3]['source']).startswith('## 28.'))
        config_source = ''.join(notebook['cells'][-2]['source'])
        self.assertIn('RUN_FT_CONTROL = False', config_source)
        self.assertIn('RUN_FT_LAYER4 = False', config_source)
        for cell in notebook['cells'][-2:]:
            ast.parse(''.join(cell['source']))

    def run_log_fixture(self, root, fail=False):
        notebook = json.loads((ROOT / 'acne_colab.ipynb').read_text())
        source = ''.join(notebook['cells'][-2]['source'])
        source = source.replace('RUN_FT_CONTROL = False', 'RUN_FT_CONTROL = True')
        source = source.replace('FT_DOWNLOAD = True', 'FT_DOWNLOAD = False')
        source = source.replace('FT_RESUME = True', 'FT_RESUME = False')
        source = source.replace('FT_SEEDS = [42, 43, 44]', 'FT_SEEDS = [42]')
        (root / 'manifest.csv').touch()
        (root / 'best_model.pt').touch()
        (root / 'metrics.json').touch()
        fixture = "print('epoch=1 batch=1 loss=0.5', flush=True)\n"
        if fail:
            fixture += "raise RuntimeError('fixture failure')\n"
        (root / 'train_layer4_finetune.py').write_text(fixture)
        env = {'REFINED_MANIFEST': root / 'manifest.csv', 'OUTPUT_A': root,
               'TRAIN_DIR': root, 'SCRIPTS_DIR': root, 'TRAIN_SCRIPT_A': root / 'trainer.py',
               'DEVICE': 'cpu'}
        with contextlib.redirect_stdout(io.StringIO()) as console:
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'Training gagal'):
                    exec(compile(source, '<step28>', 'exec'), env)
            else:
                exec(compile(source, '<step28>', 'exec'), env)
        log = (root / 'layer4_finetune_v1/frozen_seed42.log').read_text()
        self.assertIn('epoch=1 batch=1 loss=0.5', console.getvalue())
        self.assertIn('epoch=1 batch=1 loss=0.5', log)
        return log

    def test_live_log_is_persisted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertIn('Exit code: 0', self.run_log_fixture(root))
            original = (root / 'layer4_finetune_v1/frozen_seed42.log').read_text()
            with self.assertRaises(FileExistsError):
                self.run_log_fixture(root)
            self.assertEqual(original, (root / 'layer4_finetune_v1/frozen_seed42.log').read_text())

    def test_failed_run_preserves_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            log = self.run_log_fixture(Path(folder), fail=True)
            self.assertIn('RuntimeError: fixture failure', log)
            self.assertIn('Exit code: 1', log)

    def test_resume_config_rejects_changes_but_allows_resume_flag(self):
        MODULE.validate_resume({'resume': False, 'seed': 42}, {'resume': True, 'seed': 42})
        for field in ['seed', 'epochs', 'batch_size', 'source_sha256']:
            with self.assertRaisesRegex(ValueError, 'Konfigurasi resume berbeda'):
                MODULE.validate_resume({field: 1}, {field: 2})

    def test_atomic_save_preserves_old_checkpoint_on_failure(self):
        class FailingTorch:
            @staticmethod
            def save(data, path):
                path.write_bytes(b'incomplete')
                raise OSError('disk full')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last_checkpoint.pt'
            path.write_bytes(b'previous good checkpoint')
            with self.assertRaises(OSError):
                MODULE.atomic_torch_save(FailingTorch, {}, path)
            self.assertEqual(path.read_bytes(), b'previous good checkpoint')

    def test_notebook_resume_appends_log_and_passes_resume_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = ''.join(json.loads((ROOT / 'acne_colab.ipynb').read_text())['cells'][-2]['source'])
            source = source.replace('RUN_FT_CONTROL = False', 'RUN_FT_CONTROL = True')
            source = source.replace('FT_DOWNLOAD = True', 'FT_DOWNLOAD = False')
            source = source.replace('FT_SEEDS = [42, 43, 44]', 'FT_SEEDS = [42]')
            for name in ['manifest.csv', 'best_model.pt', 'metrics.json']:
                (root / name).touch()
            (root / 'train_layer4_finetune.py').write_text(
                "import sys\nassert '--resume' in sys.argv\nprint('RESUME epoch 2')\n")
            run = root / 'layer4_finetune_v1/frozen_seed42'
            run.mkdir(parents=True)
            (run / 'last_checkpoint.pt').touch()
            log_path = run.parent / 'frozen_seed42.log'
            log_path.write_text('OLD LOG\n')
            env = {'REFINED_MANIFEST': root / 'manifest.csv', 'OUTPUT_A': root,
                   'TRAIN_DIR': root, 'SCRIPTS_DIR': root, 'TRAIN_SCRIPT_A': root / 'trainer.py',
                   'DEVICE': 'cpu'}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, '<step28>', 'exec'), env)
            self.assertTrue(log_path.read_text().startswith('OLD LOG\n'))
            self.assertIn('RESUME epoch 2', log_path.read_text())

    def test_legacy_trainer_constructor_fallback_is_present(self):
        source = (ROOT / 'scripts/train_layer4_finetune.py').read_text()
        self.assertIn('Trainer legacy terdeteksi', source)
        self.assertIn("requested_variant != 'proposed'", source)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class TrainabilityTests(unittest.TestCase):
    def test_resume_restores_optimizer_dropout_shuffle_and_python_rng(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(42)
        random.seed(42)
        model = self.make_model()
        MODULE.configure_trainable(model, 'layer4')
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        loaders = {'train': DataLoader(TensorDataset(torch.randn(12, 4)), batch_size=3,
                                      shuffle=True, generator=torch.Generator().manual_seed(42))}

        def epoch():
            MODULE.set_training_mode(model)
            for (x,) in loaders['train']:
                optimizer.zero_grad()
                out = model.classifier(model.encoder.layer4(model.encoder.layer3(x)))
                (out.square().mean() * random.random()).backward()
                optimizer.step()

        epoch()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last_checkpoint.pt'
            MODULE.atomic_torch_save(torch, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                             'rng': MODULE.capture_rng(torch, loaders)}, path)
            epoch()
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            saved = torch.load(path, weights_only=True, map_location='cpu')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            # Simulate randomness consumed while re-creating the model/runtime before resume.
            torch.rand(20)
            random.random()
            MODULE.restore_rng(torch, loaders, saved['rng'])
            epoch()
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)

    def make_model(self):
        import torch.nn as nn
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.layer3 = nn.Linear(4, 4)
        model.encoder.layer4 = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        model.decoder = nn.Linear(4, 4)
        model.classifier = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.3))
        return model

    def test_only_layer4_and_classifier_can_change(self):
        model = self.make_model()
        MODULE.configure_trainable(model, 'layer4')
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad,
                             name.startswith(('encoder.layer4.', 'classifier.')))
        MODULE.set_training_mode(model)
        self.assertFalse(model.encoder.layer4[1].training)
        self.assertFalse(model.decoder.training)
        self.assertTrue(model.classifier[1].training)

    def test_frozen_control_only_trains_classifier(self):
        model = self.make_model()
        MODULE.configure_trainable(model, 'frozen')
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith('classifier.'))

    def test_training_step_keeps_frozen_weights_and_bn_statistics(self):
        import torch
        model = self.make_model()
        MODULE.configure_trainable(model, 'layer4')
        MODULE.set_training_mode(model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        x = model.encoder.layer3(torch.randn(8, 4))
        output = model.classifier(model.decoder(model.encoder.layer4(x)))
        output.square().sum().backward()
        opt.step()
        after = model.state_dict()
        for key in ['encoder.layer3.weight', 'decoder.weight',
                    'encoder.layer4.1.running_mean', 'encoder.layer4.1.running_var']:
            self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertFalse(torch.equal(before['encoder.layer4.0.weight'], after['encoder.layer4.0.weight']))


if __name__ == '__main__':
    unittest.main()
