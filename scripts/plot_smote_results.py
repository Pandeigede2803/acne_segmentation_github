#!/usr/bin/env python3
"""Plot saved SMOTE results without rerunning training or accessing test images."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_figure(fig, destination):
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {destination}')


def plot_results(run_dir, output_dir=None):
    source = run_dir / 'metrics.json'
    if not source.exists():
        source = run_dir / 'history_partial.json'
    if not source.exists():
        raise FileNotFoundError(f'Belum ada metrics.json/history_partial.json di {run_dir}')
    data = json.loads(source.read_text())
    history = data.get('history', [])
    if not history:
        raise ValueError('Riwayat training masih kosong.')
    output_dir = output_dir or run_dir / 'charts'
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in history for key in row))
    with (output_dir / 'training_history.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    best_epoch = min(history, key=lambda row: row['val_loss'])['epoch']
    for metric, ylabel in [('acc', 'Accuracy'), ('loss', 'Cross-entropy loss')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for split, label in [('train', 'Train (original features)'), ('val', 'Validation')]:
            key = f'{split}_{metric}'
            points = [row for row in history if key in row]
            if points:
                ax.plot([row['epoch'] for row in points], [row[key] for row in points], label=label)
            else:
                print(f'{key} tidak tersedia pada run ini; kurva dilewati.')
        ax.axvline(best_epoch, color='gray', linestyle='--', label=f'Best validation loss: epoch {best_epoch}')
        ax.set(xlabel='Epoch', ylabel=ylabel, title=f'SMOTE classifier head — {ylabel}')
        if metric == 'acc':
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        save_figure(fig, output_dir / f'training_{metric}.png')

    counts = data.get('train_counts_original', [])
    names = data.get('class_names', [f'Class {i}' for i in range(len(counts))])
    if counts and data.get('train_counts_after_smote'):
        fig, ax = plt.subplots(figsize=(8, 5))
        positions = list(range(len(counts)))
        ax.bar([i - 0.2 for i in positions], counts, width=0.4, label='Original')
        ax.bar([i + 0.2 for i in positions], data['train_counts_after_smote'], width=0.4,
               label='After SMOTE (original + synthetic)')
        ax.set_xticks(positions, names)
        ax.set(ylabel='Feature count', title='Train class distribution')
        ax.legend()
        save_figure(fig, output_dir / 'class_distribution.png')

    matrix = data.get('test_confusion_matrix')
    if matrix is None:
        print('Confusion matrix belum tersedia. Run lama/parsial tetap bisa diplot; '
              'untuk confusion matrix, selesaikan training dengan script terbaru.')
        return output_dir
    size = len(matrix)
    if len(names) != size:
        names = [f'Class {i}' for i in range(size)]
    normalized = [[value / sum(row) if sum(row) else 0.0 for value in row] for row in matrix]
    for values, filename, title, fraction in [
        (matrix, 'confusion_matrix_test.png', 'Test confusion matrix — counts', False),
        (normalized, 'confusion_matrix_test_normalized.png', 'Test confusion matrix — row normalized', True),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if fraction else None)
        fig.colorbar(image, ax=ax)
        threshold = max(max(row) for row in values) / 2
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, f'{value:.1%}' if fraction else str(value), ha='center', va='center',
                        color='white' if value > threshold else 'black')
        ax.set_xticks(range(size), names, rotation=30, ha='right')
        ax.set_yticks(range(size), names)
        ax.set(xlabel='Predicted label', ylabel='True label', title=title)
        save_figure(fig, output_dir / filename)
    with (output_dir / 'confusion_matrix_test.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['true_label / predicted_label', *names])
        writer.writerows([name, *row] for name, row in zip(names, matrix))

    report = []
    for i, name in enumerate(names):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = matrix[i][i] / predicted if predicted else 0.0
        recall = matrix[i][i] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report.append(dict(class_name=name, precision=precision, recall=recall, f1=f1, support=support))
    with (output_dir / 'classification_report.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, key in [(-0.25, 'precision'), (0, 'recall'), (0.25, 'f1')]:
        ax.bar([i + offset for i in range(size)], [row[key] for row in report], width=0.25, label=key)
    ax.set_xticks(range(size), names)
    ax.set(ylim=(0, 1), ylabel='Score', title='Test metrics per class (undefined scores set to 0)')
    ax.legend()
    save_figure(fig, output_dir / 'per_class_metrics_test.png')
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    plot_results(args.run_dir, args.output_dir)
