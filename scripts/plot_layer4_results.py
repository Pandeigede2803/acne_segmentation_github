#!/usr/bin/env python3
"""Plot the additional layer4 variant from saved validation metrics only."""
import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = [('val_loss', 'Classification CE (lower is better)'),
           ('accuracy', 'Validation accuracy'), ('macro_f1', 'Validation macro-F1'),
           ('val_dice', 'Validation Dice'), ('val_iou', 'Validation IoU')]


def load_runs(root, seeds):
    runs = {}
    for mode in ('frozen', 'layer4'):
        for seed in seeds:
            source = root / f'{mode}_seed{seed}' / 'metrics.json'
            if source.is_file():
                runs[mode, seed] = json.loads(source.read_text())
    return runs


def completed_runs(runs):
    return {key: data for key, data in runs.items() if data.get('status') == 'completed'}


def paired_deltas(runs, metric):
    complete = completed_runs(runs)
    seeds = sorted({seed for mode, seed in complete if ('frozen', seed) in complete
                    and ('layer4', seed) in complete})
    return [(seed, complete['layer4', seed]['best_validation'][metric]
             - complete['frozen', seed]['best_validation'][metric]) for seed in seeds]


def plot_results(root, seeds):
    runs = load_runs(root, seeds)
    if not runs:
        print('Belum ada metrics.json; chart hasil belum dibuat.')
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    saved = []

    def save(fig, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)
        print(f'Saved: {path}')

    for (mode, seed), data in runs.items():
        out = root / f'{mode}_seed{seed}' / 'charts'
        out.mkdir(parents=True, exist_ok=True)
        history = data.get('history', [])
        initial = data.get('initial_validation')
        points = ([{'epoch': 0, **initial}] if initial else []) + history
        title = f'Additional variant: {mode}, seed={seed} [{data.get("status", "unknown")}]'
        if points:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            panels = [
                ([('train_total_loss', 'Train total: CE + weighted (BCE + Dice loss)')], 'Train objective'),
                ([('val_loss', 'Validation classification CE')], 'Validation objective'),
                ([('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')], 'Validation classification'),
                ([('val_dice', 'Dice'), ('val_iou', 'IoU')], 'Validation segmentation'),
            ]
            for ax, (series, label) in zip(axes.flat, panels):
                for field, legend in series:
                    available = [row for row in points if field in row]
                    if available:
                        ax.plot([r['epoch'] for r in available], [r[field] for r in available],
                                label=legend, marker='.', markersize=3)
                ax.axvline(data['best_epoch'], linestyle='--', color='gray', label=f'Best CE: epoch {data["best_epoch"]}')
                ax.set(xlabel='Epoch (0 = initial checkpoint)', ylabel=label)
                if label.startswith('Validation') and label != 'Validation objective':
                    ax.set_ylim(0, 1)
                ax.legend(fontsize=7)
                ax.grid(alpha=0.2)
            fig.suptitle(title + '\nTrain total loss and validation CE are different objectives', fontsize=11)
            save(fig, out / 'learning_curves.png')
        if history:
            with (out / 'training_history.csv').open('w', newline='') as handle:
                fields = list(dict.fromkeys(key for row in history for key in row))
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(history)

        best = data['best_validation']
        matrix = best['confusion_matrix']
        names = data.get('class_names', ['0', '1', '2', '3'])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        normalized = [[v / sum(row) if sum(row) else 0 for v in row] for row in matrix]
        for ax, values, fraction in zip(axes, [matrix, normalized], [False, True]):
            image = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if fraction else None)
            fig.colorbar(image, ax=ax)
            threshold = max(map(max, values)) / 2
            for i, row in enumerate(values):
                for j, value in enumerate(row):
                    ax.text(j, i, f'{value:.1%}' if fraction else str(value), ha='center', va='center',
                            color='white' if value > threshold else 'black')
            ax.set_xticks(range(4), names, rotation=30, ha='right')
            ax.set_yticks(range(4), names)
            ax.set(xlabel='Predicted class', ylabel='True class',
                   title='Row normalized (empty row = 0)' if fraction else 'Counts')
        fig.suptitle(title + f'\nValidation confusion matrix, best epoch={data["best_epoch"]}')
        save(fig, out / 'confusion_matrix_val.png')

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, best['recall_per_class'], color='#2980b9')
        ax.set(ylim=(0, 1), ylabel='Recall', title=title + '\nBest-checkpoint validation recall (no support = 0)')
        save(fig, out / 'recall_per_class.png')

    complete = completed_runs(runs)
    if not complete:
        print('Run masih parsial: chart per-run tersedia; perbandingan final menunggu completed.')
        return saved
    out = root / 'charts'
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    summary = []
    for ax, (metric, label) in zip(axes.flat, METRICS):
        for i, mode in enumerate(('frozen', 'layer4')):
            values = [data['best_validation'][metric] for (m, seed), data in sorted(complete.items()) if m == mode]
            if not values:
                continue
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else None
            ax.bar(i, mean, color=['#85929e', '#2980b9'][i], alpha=0.7,
                   yerr=sd, capsize=5, label=f'{mode}: n={len(values)}')
            ax.scatter([i + (j - (len(values) - 1) / 2) * 0.07 for j in range(len(values))], values,
                       color='black', s=20, zorder=3)
            summary.append({'mode': mode, 'metric': metric, 'n': len(values), 'mean': mean, 'sample_sd': sd})
        ax.set_xticks([0, 1], ['frozen', 'layer4'])
        ax.set(title=label, ylabel='CE' if metric == 'val_loss' else 'Score (0–1)')
        if metric != 'val_loss':
            ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.2)
        ax.legend(fontsize=7)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, 0.9, 'Completed runs only\nBars: mean; error bars: sample SD\nDots: individual seeds\nSD omitted when n=1\nNot a significance test\nBest checkpoint selected by validation CE', va='top')
    fig.suptitle('Additional layer4 variant — validation comparison (not proposal main method)')
    save(fig, out / 'seed_comparison.png')
    with (out / 'seed_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['mode', 'metric', 'n', 'mean', 'sample_sd'])
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    deltas = []
    for ax, (metric, label) in zip(axes.flat, METRICS):
        pairs = paired_deltas(runs, metric)
        if pairs:
            ax.bar([str(seed) for seed, value in pairs], [value for seed, value in pairs], color='#2980b9')
            deltas.extend({'seed': seed, 'metric': metric, 'delta_layer4_minus_frozen': value} for seed, value in pairs)
        else:
            ax.text(0.5, 0.5, 'No completed seed pairs', ha='center', transform=ax.transAxes)
        ax.axhline(0, color='gray', linewidth=1)
        ax.set(xlabel='Paired seed', ylabel='Layer4 − frozen', title=label)
        ax.grid(axis='y', alpha=0.2)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, 0.9, 'Same seed, two completed runs\nPositive = better for scores\nNegative = better for CE\nAccuracy delta 0.01 = 1 percentage point\nDo not trade segmentation away silently', va='top')
    fig.suptitle('Paired validation changes — additional variant, not a significance test')
    save(fig, out / 'paired_deltas.png')
    with (out / 'paired_deltas.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['seed', 'metric', 'delta_layer4_minus_frozen'])
        writer.writeheader()
        writer.writerows(deltas)
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = parser.parse_args()
    plot_results(args.root, args.seeds)
