#!/usr/bin/env python3
"""Matched frozen/layer4 experiments on images; validation only, no SMOTE."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def classification_metrics(confusion):
    recalls, f1s = [], []
    for i, row in enumerate(confusion):
        tp = row[i]
        support = sum(row)
        predicted = sum(r[i] for r in confusion)
        recalls.append(tp / support if support else 0.0)
        f1s.append(2 * tp / (support + predicted) if support + predicted else 0.0)
    total = sum(map(sum, confusion))
    return {'accuracy': sum(confusion[i][i] for i in range(4)) / total if total else 0.0,
            'macro_f1': sum(f1s) / 4, 'recall_per_class': recalls,
            'confusion_matrix': confusion}


def configure_trainable(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    if mode == 'layer4':
        for parameter in model.encoder.layer4.parameters():
            parameter.requires_grad_(True)


def set_training_mode(model):
    # Keep all BatchNorm statistics AND frozen modules fixed; only head dropout is active.
    model.eval()
    model.classifier.train()


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(temp, path)


def atomic_torch_save(torch, data, path):
    temp = path.with_suffix('.pt.tmp')
    torch.save(data, temp)
    os.replace(temp, path)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resume(saved, config):
    # Changing settings or source files must start a separate experiment.
    ignored = {'resume'}
    changed = [key for key in set(saved) | set(config)
               if key not in ignored and saved.get(key) != config.get(key)]
    if changed:
        raise ValueError(f'Konfigurasi resume berbeda: {sorted(changed)}. Gunakan folder run baru.')


def capture_rng(torch, loaders):
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loaders': {s: loader.generator.get_state() for s, loader in loaders.items()}}


def restore_rng(torch, loaders, state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Jumlah GPU resume berbeda dari checkpoint')
        torch.cuda.set_rng_state_all(state['cuda'])
    for split, loader in loaders.items():
        loader.generator.set_state(state['loaders'][split])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trainer', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--mode', choices=('frozen', 'layer4'), required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--seg-weight', type=float, default=1.0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true', help='Resume trusted local last_checkpoint.pt at epoch boundary.')
    a = p.parse_args()
    if min(a.epochs, a.patience, a.batch_size) < 1:
        p.error('epochs, patience, batch-size harus positif')
    if any(not math.isfinite(x) or x <= 0 for x in (a.backbone_lr, a.head_lr)):
        p.error('learning rate harus positif dan finite')
    if not math.isfinite(a.seg_weight) or a.seg_weight < 0:
        p.error('seg-weight harus nonnegatif dan finite')
    last_path = a.output_dir / 'last_checkpoint.pt'
    if a.resume and not last_path.is_file():
        raise FileNotFoundError(f'Resume memerlukan {last_path}; run lama tanpa checkpoint tidak bisa di-resume.')
    if not a.resume and a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(f'Gunakan folder output baru: {a.output_dir}')

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(a.trainer.resolve().parent))
    import train_proposal_hybrid_resnet50_malds as tr

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = json.loads(a.metrics.read_text())['config']
    # Colab archives may contain an older trainer whose constructor has no
    # ``variant`` argument. The baseline checkpoint still determines the
    # weights; fall back only for the legacy default/proposed architecture.
    requested_variant = cfg.get('variant', 'proposed')
    try:
        model = tr.ProposalHybridResNet50(pretrained=False, variant=requested_variant)
    except TypeError as error:
        if 'unexpected keyword argument' not in str(error) or requested_variant != 'proposed':
            raise
        print('Trainer legacy terdeteksi: memakai constructor tanpa variant.', flush=True)
        model = tr.ProposalHybridResNet50(pretrained=False)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    # Both arms start from the SAME trained classifier and backbone, not a new random head.
    if model.classifier[2].p != 0.3:
        raise ValueError('Eksperimen mengharuskan dropout 0.3')
    configure_trainable(model, a.mode)
    model.to(a.device)
    rows = tr.load_rows(a.manifest)
    splits = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val')}
    if any(not r for r in splits.values()):
        raise ValueError('Train dan val harus tersedia')
    # Prevent exact path leakage. Patient/source grouping still needs an external audit.
    for key in ('image_path', 'mask_path'):
        paths = [{str(Path(r[key]).resolve()) for r in splits[s]} for s in ('train', 'val')]
        if paths[0] & paths[1]:
            raise ValueError(f'Train/val overlap: {key}')
    ds_args = SimpleNamespace(resolved_input_normalization=cfg.get('resolved_input_normalization', 'imagenet'))
    loaders = {}
    for split in splits:
        dataset = tr.PreprocessedAcneDataset(splits[split], cfg['image_size'], False, ds_args)
        loaders[split] = DataLoader(dataset, batch_size=a.batch_size, shuffle=split == 'train',
                                    num_workers=0, generator=torch.Generator().manual_seed(a.seed))
    groups = [{'params': model.classifier.parameters(), 'lr': a.head_lr}]
    if a.mode == 'layer4':
        groups.append({'params': model.encoder.layer4.parameters(), 'lr': a.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(skip_test=True, smote=False, augmentation=False, class_weighting='none',
                  dropout=0.3, selection_metric='val_loss (unweighted classification CE)',
                  image_size=cfg['image_size'], variant=model.variant,
                  resolved_input_normalization=ds_args.resolved_input_normalization,
                  trainable_parameters=[n for n, v in model.named_parameters() if v.requires_grad])
    config['source_sha256'] = {key: file_digest(getattr(a, key))
                               for key in ('trainer', 'manifest', 'checkpoint', 'metrics')}
    config['resume_format'] = 1
    a.output_dir.mkdir(parents=True, exist_ok=True)
    saved = None
    if a.resume:
        # Only load this script's trusted checkpoint: Python RNG is stored as plain tuples.
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        validate_resume(saved['config'], config)
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
    else:
        write_json(a.output_dir / 'config.json', config)

    def evaluate():
        model.eval()
        loss_sum = dice_sum = iou_sum = 0.0
        confusion = torch.zeros(4, 4, dtype=torch.long)
        predictions = []
        with torch.no_grad():
            for batch in loaders['val']:
                images, labels = batch['image'].to(a.device), batch['label'].to(a.device)
                masks = batch['mask'].to(a.device)
                out = model(images)
                loss_sum += F.cross_entropy(out['class_logits'], labels, reduction='sum').item()
                dice, iou = tr.dice_iou_from_logits(out['seg_logits'], masks)
                dice_sum += dice.item() * len(labels)
                iou_sum += iou.item() * len(labels)
                pred = out['class_logits'].argmax(1).cpu()
                truth = labels.cpu()
                confusion += torch.bincount(truth * 4 + pred, minlength=16).reshape(4, 4)
                predictions.extend(zip(truth.tolist(), pred.tolist()))
        n = len(splits['val'])
        metrics = classification_metrics(confusion.tolist())
        metrics.update(val_loss=loss_sum / n, val_dice=dice_sum / n, val_iou=iou_sum / n)
        if not all(math.isfinite(metrics[k]) for k in ('val_loss', 'val_dice', 'val_iou')):
            raise RuntimeError('Non-finite validation metrics')
        return metrics, predictions

    def export_best():
        atomic_torch_save(torch, {'model_state_dict': best_state, 'config': config,
                                  'epoch': best_epoch, 'validation': best_metrics},
                          a.output_dir / 'best_model.pt')
        csv_path = a.output_dir / 'val_predictions.csv'
        temp = csv_path.with_suffix('.csv.tmp')
        with temp.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'true_label', 'predicted_label'])
            writer.writerows((r['image_path'], y, pred) for r, (y, pred) in zip(splits['val'], best_predictions))
        os.replace(temp, csv_path)

    def persist(epoch, status):
        # The atomic last checkpoint is authoritative; JSON/best exports can be repaired on resume.
        atomic_torch_save(torch, {
            'config': config, 'epoch': epoch, 'status': status,
            'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'rng': capture_rng(torch, loaders), 'initial_validation': initial,
            'history': history, 'stale': stale, 'best_epoch': best_epoch,
            'best_validation': best_metrics, 'best_model_state_dict': best_state,
            'best_predictions': best_predictions}, last_path)
        export_metrics(status)

    def export_metrics(status):
        write_json(a.output_dir / 'metrics.json', {
            'status': status, 'config': config, 'class_names': tr.LABEL_NAMES,
            'initial_validation': initial, 'best_epoch': best_epoch,
            'best_validation': best_metrics, 'history': history})

    if saved is not None:
        initial, history, stale = saved['initial_validation'], saved['history'], saved['stale']
        best_epoch, best_metrics = saved['best_epoch'], saved['best_validation']
        best_state, best_predictions = saved['best_model_state_dict'], saved['best_predictions']
        best_loss = best_metrics['val_loss']
        start_epoch = saved['epoch'] + 1
        restore_rng(torch, loaders, saved['rng'])
        export_best()
        export_metrics(saved['status'])
        if saved['status'] == 'completed':
            print(f'Run sudah completed pada epoch {saved["epoch"]}; training dilewati.', flush=True)
            return
        print(f'RESUME dari epoch {saved["epoch"]}; berikutnya epoch {start_epoch}', flush=True)
        del saved
    else:
        initial, best_predictions = evaluate()
        best_loss, best_epoch, best_metrics = initial['val_loss'], 0, initial
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history, stale, start_epoch = [], 0, 1
        persist(0, 'running')
        export_best()
        print(f'Initial validation: {initial}', flush=True)

    for epoch in range(start_epoch, a.epochs + 1):
        started = time.monotonic()
        set_training_mode(model)
        total_loss = 0.0
        for batch_i, batch in enumerate(loaders['train'], start=1):
            optimizer.zero_grad(set_to_none=True)
            out = model(batch['image'].to(a.device))
            cls_loss = F.cross_entropy(out['class_logits'], batch['label'].to(a.device))
            masks = batch['mask'].to(a.device)
            seg_loss = (F.binary_cross_entropy_with_logits(out['seg_logits'], masks)
                        + tr.dice_loss_from_logits(out['seg_logits'], masks))
            loss = cls_loss + a.seg_weight * seg_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(masks)
            if batch_i == 1 or batch_i % 10 == 0 or batch_i == len(loaders['train']):
                print(f'{a.mode} seed={a.seed} epoch={epoch}/{a.epochs} '
                      f'batch={batch_i}/{len(loaders["train"])} '
                      f'loss={loss.item():.4f} cls={cls_loss.item():.4f} '
                      f'seg={seg_loss.item():.4f}', flush=True)
        metrics, predictions = evaluate()
        history.append({'epoch': epoch, 'train_total_loss': total_loss / len(splits['train']), **metrics})
        if metrics['val_loss'] < best_loss:
            best_loss, best_epoch, best_metrics, stale = metrics['val_loss'], epoch, metrics, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_predictions = predictions
        else:
            stale += 1
        completed = stale >= a.patience or epoch >= a.epochs
        persist(epoch, 'completed' if completed else 'running')
        if best_epoch == epoch:
            export_best()
        print(f'{a.mode} seed={a.seed} epoch={epoch}/{a.epochs} '
              f'train_total_loss={history[-1]["train_total_loss"]:.4f} '
              f'best_epoch={best_epoch} patience={stale}/{a.patience} '
              f'seconds={time.monotonic() - started:.1f}: {metrics}', flush=True)
        if stale >= a.patience:
            print('Early stopping (validation classification loss)', flush=True)
            break
    print(f'COMPLETED: {a.output_dir} | best_epoch={best_epoch}', flush=True)


if __name__ == '__main__':
    main()
