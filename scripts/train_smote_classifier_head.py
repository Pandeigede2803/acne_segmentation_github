#!/usr/bin/env python3
"""SMOTE experiment on frozen hybrid classification features (train split only)."""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def smote_features(x, y, seed=42):
    """Interpolate minority feature vectors to geometric-mean class targets."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counts = Counter(y.tolist())
    max_count = max(counts.values())
    targets = {c: int((n * max_count) ** 0.5 + 0.5) for c, n in counts.items()}
    xs, ys = [x], [y]
    for cls, n in counts.items():
        needed = max(0, targets[cls] - n)
        ids = torch.where(y == cls)[0]
        if needed == 0:
            continue
        if len(ids) < 2:
            raise ValueError(f"SMOTE memerlukan >=2 fitur train untuk kelas {cls}.")
        class_x = x[ids]
        distances = torch.cdist(class_x, class_x)
        distances.fill_diagonal_(float('inf'))
        k = min(5, len(ids) - 1)
        neighbors = distances.topk(k, largest=False).indices
        base = torch.randint(len(ids), (needed,), generator=generator)
        pick = torch.randint(k, (needed,), generator=generator)
        near = neighbors[base, pick]
        alpha = torch.rand((needed, 1), generator=generator)
        synth = class_x[base] + alpha * (class_x[near] - class_x[base])
        xs.append(synth)
        ys.append(torch.full((needed,), cls, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys), targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trainer', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--skip-test', action='store_true', help='Validation-only experiment; do not access test data.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if not 0 <= a.dropout < 1 or a.hidden_dim < 1 or a.epochs < 1 or a.batch_size < 1:
        p.error('dropout harus [0, 1); hidden-dim, epochs, batch-size harus positif.')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(f'Gunakan folder output baru agar hasil lama aman: {a.output_dir}')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"SMOTE run started. Output: {a.output_dir}", flush=True)
    sys.path.insert(0, str(a.trainer.resolve().parent))
    import train_proposal_hybrid_resnet50_malds as tr

    cfg = json.loads(a.metrics.read_text())['config']
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    variant = cfg.get('variant', 'proposed')
    model = tr.ProposalHybridResNet50(pretrained=False, variant=variant)
    model.load_state_dict(ckpt)
    model.to(a.device).eval()
    args = SimpleNamespace(resolved_input_normalization=cfg.get('resolved_input_normalization', 'imagenet'))
    rows = tr.load_rows(a.manifest)
    split_names = ('train', 'val') if a.skip_test else ('train', 'val', 'test')
    splits = {s: [r for r in rows if r['split'] == s] for s in split_names}
    if any(not splits[s] for s in splits):
        raise ValueError(f'Manifest harus memuat split: {split_names}.')

    feats, labels = {}, {}
    with torch.no_grad():
        for split, split_rows in splits.items():
            print(f"Extracting frozen features: {split} ({len(split_rows)} samples)", flush=True)
            ds = tr.PreprocessedAcneDataset(split_rows, cfg['image_size'], False, args)
            loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
            f, y = [], []
            for batch_i, batch in enumerate(loader, start=1):
                out = model(batch['image'].to(a.device))
                f.append(out['class_features'].cpu())
                y.append(batch['label'])
                if batch_i % 10 == 0 or batch_i == len(loader):
                    print(f"  {split}: batch {batch_i}/{len(loader)}", flush=True)
            feats[split], labels[split] = torch.cat(f), torch.cat(y)
            print(f"  {split} feature matrix: {tuple(feats[split].shape)}", flush=True)

    train_x, train_y, targets = smote_features(feats['train'], labels['train'], a.seed)
    print(f"Train samples after SMOTE: {len(labels['train'])} -> {len(train_y)}; targets={targets}", flush=True)
    in_features = train_x.shape[1]
    # Reset head RNG after feature extraction so compared runs share initialization when shapes match.
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    head = nn.Sequential(nn.Linear(in_features, a.hidden_dim), nn.ReLU(), nn.Dropout(a.dropout),
                         nn.Linear(a.hidden_dim, 4)).to(a.device)
    run_config = {'seed': a.seed, 'epochs': a.epochs, 'batch_size': a.batch_size,
                  'dropout': a.dropout, 'hidden_dim': a.hidden_dim, 'skip_test': a.skip_test,
                  'manifest': str(a.manifest.resolve()), 'checkpoint': str(a.checkpoint.resolve()),
                  'baseline_metrics': str(a.metrics.resolve())}
    head_config = {'hidden_dim': a.hidden_dim, 'dropout': a.dropout}
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, history = float('inf'), None, []
    gen = torch.Generator().manual_seed(a.seed)
    def write_json(name, payload):
        temp = a.output_dir / f'.{name}.tmp'
        temp.write_text(json.dumps(payload, indent=2))
        os.replace(temp, a.output_dir / name)

    for epoch in range(1, a.epochs + 1):
        head.train()
        order = torch.randperm(len(train_y), generator=gen)
        for ids in order.split(a.batch_size):
            xb, yb = train_x[ids].to(a.device), train_y[ids].to(a.device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(head(xb), yb)
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            # Evaluate original train features only, without synthetic SMOTE samples.
            train_correct = 0
            train_loss_sum = 0.0
            for xb, yb in zip(feats['train'].split(a.batch_size), labels['train'].split(a.batch_size)):
                train_logits = head(xb.to(a.device))
                train_loss_sum += nn.functional.cross_entropy(
                    train_logits, yb.to(a.device), reduction='sum').item()
                predictions = train_logits.argmax(1).cpu()
                train_correct += (predictions == yb).sum().item()
            train_acc = train_correct / len(labels['train'])
            train_loss = train_loss_sum / len(labels['train'])
            val_logits = head(feats['val'].to(a.device))
            vl = nn.functional.cross_entropy(val_logits, labels['val'].to(a.device)).item()
            vacc = (val_logits.argmax(1).cpu() == labels['val']).float().mean().item()
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_loss': vl, 'val_acc': vacc})
        if vl < best:
            best, best_state = vl, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            torch.save({'classifier_state_dict': best_state, 'variant': variant,
                        'config': run_config, **head_config,
                        'input_features': in_features, 'best_epoch': epoch},
                       a.output_dir / 'smote_classifier_head.pt')
        torch.save({'epoch': epoch, 'classifier_state_dict': head.state_dict(),
                    'config': run_config, **head_config,
                    'optimizer_state_dict': opt.state_dict(), 'best_val_loss': best,
                    'best_epoch': min(history, key=lambda item: item['val_loss'])['epoch'],
                    'history': history}, a.output_dir / 'last_checkpoint.pt')
        write_json('history_partial.json', {
            'method': 'SMOTE on frozen hybrid classification features',
            'config': run_config, 'class_names': tr.LABEL_NAMES,
            'train_counts_original': [int((labels['train'] == i).sum()) for i in range(4)],
            'train_counts_after_smote': [int((train_y == i).sum()) for i in range(4)],
            'target_counts': targets, 'best_val_loss': best, 'history': history,
        })
        print(f"Epoch {epoch:03d}/{a.epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_loss={vl:.4f} | val_acc={vacc:.4f}"
              f" | best={best:.4f}", flush=True)
    head.load_state_dict(best_state); head.eval()
    results = {'method': 'SMOTE on frozen hybrid class features', 'config': run_config,
        'train_counts_original': [int((labels['train'] == i).sum()) for i in range(4)],
        'train_counts_after_smote': [int((train_y == i).sum()) for i in range(4)],
        'target_counts': targets, 'val_best_loss': best, 'class_names': tr.LABEL_NAMES,
        'best_epoch': min(history, key=lambda item: item['val_loss'])['epoch'], 'history': history}
    if a.skip_test:
        write_json('metrics.json', results)
        print(f'Completed validation-only run: best_val_loss={best:.4f}; saved to {a.output_dir}', flush=True)
        return
    with torch.no_grad():
        logits = head(feats['test'].to(a.device)).cpu()
        acc = (logits.argmax(1) == labels['test']).float().mean().item()
        kappa = tr.cohen_kappa_from_logits(logits, labels['test']).item()
    predictions = logits.argmax(1)
    confusion = torch.bincount(labels['test'] * 4 + predictions, minlength=16).reshape(4, 4)
    with (a.output_dir / 'test_class_predictions.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_index', 'true_label', 'predicted_label'])
        writer.writerows((i, int(actual), int(predicted))
                         for i, (actual, predicted) in enumerate(zip(labels['test'], predictions)))
    results.update(test_accuracy=acc, test_kappa=kappa, test_confusion_matrix=confusion.tolist())
    write_json('metrics.json', results)
    print(f"Completed: train features {len(labels['train'])} -> {len(train_y)}; "
          f"test acc={acc:.4f}; kappa={kappa:.4f}; saved to {a.output_dir}", flush=True)

if __name__ == '__main__':
    main()
