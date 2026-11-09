#!/usr/bin/env python3
"""Inspect validation severity errors and example images, without training/test access."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def summarize(records, names):
    matrix = [[0 for _ in names] for _ in names]
    for row in records:
        matrix[row['true_label']][row['predicted_label']] += 1
    classes = []
    for i, name in enumerate(names):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = matrix[i][i] / predicted if predicted else 0.0
        recall = matrix[i][i] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes.append({'class_name': name, 'support': support, 'errors': support - matrix[i][i],
                        'precision': precision, 'recall': recall, 'f1': f1,
                        'error_rate': 1 - recall if support else None})
    pairs = Counter((row['true_label'], row['predicted_label']) for row in records
                    if row['true_label'] != row['predicted_label'])
    return {'confusion_matrix': matrix, 'per_class': classes,
            'confused_pairs': [{'true_label': a, 'predicted_label': b, 'count': n}
                               for (a, b), n in pairs.most_common()],
            'accuracy': sum(matrix[i][i] for i in range(len(names))) / len(records) if records else None}


def write_csv(path, records, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trainer', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='Frozen hybrid backbone checkpoint')
    p.add_argument('--metrics', type=Path, required=True, help='Backbone metrics.json')
    p.add_argument('--head', type=Path, required=True, help='Best SMOTE classifier head checkpoint')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--examples-per-pair', type=int, default=3)
    p.add_argument('--max-examples', type=int, default=24)
    a = p.parse_args()
    if min(a.batch_size, a.examples_per_pair, a.max_examples) < 1:
        p.error('batch-size dan batas contoh harus positif.')
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps

    sys.path.insert(0, str(a.trainer.resolve().parent))
    import train_proposal_hybrid_resnet50_malds as tr

    cfg = json.loads(a.metrics.read_text())['config']
    saved = torch.load(a.head, map_location='cpu', weights_only=True)
    variant = cfg.get('variant', 'proposed')
    if saved.get('variant', variant) != variant:
        raise ValueError('Variant backbone dan head tidak cocok.')
    model = tr.ProposalHybridResNet50(pretrained=False, variant=variant)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    model.to(a.device).eval()
    state = saved['classifier_state_dict']
    # Old checkpoints lack architecture metadata; infer dimensions from their weights.
    hidden, in_features = state['0.weight'].shape
    head = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(),
                         nn.Dropout(saved.get('dropout', 0.3)), nn.Linear(hidden, 4))
    head.load_state_dict(state)
    head.to(a.device).eval()
    rows = [row for row in tr.load_rows(a.manifest) if row['split'] == 'val']
    if not rows:
        raise ValueError('Tidak ada split val dalam manifest.')
    ds = tr.PreprocessedAcneDataset(rows, cfg['image_size'], False, SimpleNamespace(
        resolved_input_normalization=cfg.get('resolved_input_normalization', 'imagenet')))
    records = []
    with torch.inference_mode():
        for batch in DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0):
            features = model(batch['image'].to(a.device))['class_features']
            probabilities = head(features).softmax(1).cpu()
            confidence, predictions = probabilities.max(1)
            for true, predicted, score in zip(batch['label'], predictions, confidence):
                index = len(records)
                records.append({'sample_index': index, 'image_path': rows[index]['image_path'],
                                'true_label': int(true), 'predicted_label': int(predicted),
                                'confidence': float(score)})
    names = tr.LABEL_NAMES
    summary = summarize(records, names)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    summary.update(split='val', class_names=names, num_samples=len(records),
                   manifest=str(a.manifest.resolve()), backbone=str(a.checkpoint.resolve()),
                   head=str(a.head.resolve()))
    (a.output_dir / 'validation_summary.json').write_text(json.dumps(summary, indent=2))
    write_csv(a.output_dir / 'validation_predictions.csv', records, list(records[0]))
    errors = [row for row in records if row['true_label'] != row['predicted_label']]
    write_csv(a.output_dir / 'validation_errors.csv', errors, list(records[0]))
    write_csv(a.output_dir / 'validation_per_class.csv', summary['per_class'], list(summary['per_class'][0]))
    write_csv(a.output_dir / 'validation_confused_pairs.csv', summary['confused_pairs'],
              ['true_label', 'predicted_label', 'count'])
    matrix = summary['confusion_matrix']
    for normalized in (False, True):
        values = [[v / sum(row) if sum(row) else 0.0 for v in row] for row in matrix] if normalized else matrix
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else None)
        fig.colorbar(image, ax=ax)
        threshold = max(max(row) for row in values) / 2
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, f'{value:.1%}' if normalized else str(value), ha='center', va='center',
                        color='white' if value > threshold else 'black')
        ax.set_xticks(range(4), names)
        ax.set_yticks(range(4), names)
        ax.set(xlabel='Predicted severity', ylabel='True severity',
               title='Validation confusion matrix' + (' — row normalized' if normalized else ' — counts'))
        fig.tight_layout()
        fig.savefig(a.output_dir / ('validation_confusion_normalized.png' if normalized else 'validation_confusion.png'), dpi=180)
        plt.close(fig)

    # Sample across confusion pairs, not only the most confident errors overall.
    examples = []
    for pair in summary['confused_pairs']:
        group = [row for row in errors if row['true_label'] == pair['true_label']
                 and row['predicted_label'] == pair['predicted_label']]
        examples.extend(sorted(group, key=lambda row: -row['confidence'])[:a.examples_per_pair])
    examples = examples[:a.max_examples]
    write_csv(a.output_dir / 'validation_example_selection.csv', examples, list(records[0]))
    example_path = a.output_dir / 'validation_error_examples.png'
    if examples:
        columns = min(4, len(examples))
        height = (len(examples) + columns - 1) // columns
        fig, axes = plt.subplots(height, columns, figsize=(4 * columns, 3.6 * height), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, row in zip(axes.flat, examples):
            try:
                with Image.open(row['image_path']) as image:
                    ax.imshow(ImageOps.exif_transpose(image).convert('RGB'))
            except (OSError, ValueError) as exc:
                ax.text(0.5, 0.5, 'Image unavailable', ha='center')
                print(f"Tidak bisa membuka {row['image_path']}: {exc}")
            ax.set_title(f"True: {names[row['true_label']]} → Pred: {names[row['predicted_label']]}\n"
                         f"Confidence: {row['confidence']:.1%} | sample {row['sample_index']}", fontsize=9)
        fig.suptitle('Validation errors — selected examples, not a representative sample')
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(example_path, dpi=150)
        plt.close(fig)
    elif example_path.exists():
        example_path.unlink()  # Remove only stale output produced by this audit.
    print(json.dumps(summary, indent=2))
    print(f'Audit validation selesai tanpa training/test access: {a.output_dir}')


if __name__ == '__main__':
    main()
