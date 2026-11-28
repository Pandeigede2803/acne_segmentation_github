#!/usr/bin/env python3
"""Step 48: frozen-checkpoint test evaluation and final report.

Consumes the per-image predictions written while evaluating the locked
checkpoint (``testing_predictions.csv``). It never selects a checkpoint and
never changes training configuration. Optionally joins group ids to report
group-level bootstrap intervals.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from recovery_common import (LABELS, balanced_accuracy_from_report, classification_report,
                             cohen_kappa, confusion_matrix, macro_f1_from_report,
                             ordinal_bias, ordinal_mae, read_rows, weighted_f1_from_report,
                             wilson_interval, write_rows)


def read_predictions(path: Path) -> list[dict]:
    rows, _ = read_rows(path)
    if not rows:
        raise ValueError(f"Prediksi kosong: {path}")
    output = []
    for row in rows:
        try:
            true = int(row["true_label_index"])
            pred = int(row["pred_label_index"])
        except (KeyError, ValueError) as error:
            raise ValueError("CSV harus punya true_label_index dan pred_label_index.") from error
        if true not in range(4) or pred not in range(4):
            raise ValueError(f"Label di luar 0-3: {row}")
        output.append(dict(row, true_label_index=true, pred_label_index=pred,
                           correct=int(true == pred)))
    return output


def read_groups(manifest: Path | None) -> dict[str, str]:
    if manifest is None or not manifest.is_file():
        return {}
    rows, _ = read_rows(manifest)
    return {row["filename"]: row["group_id"] for row in rows if row.get("group_id")}


def percentile_ci(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"mean": sum(values) / len(values),
            "low": ordered[int(0.025 * len(ordered))],
            "high": ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))]}


def cluster_bootstrap(rows: list[dict], groups: dict[str, str], iterations: int,
                      seed: int) -> dict[str, dict[str, float]]:
    members: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        members[groups.get(row["filename"], f"row:{row['filename']}")].append(row)
    keys = sorted(members)
    rng = random.Random(seed)
    macro_values, recalls = [], {cls: [] for cls in range(4)}
    for _ in range(iterations):
        sampled: list[dict] = []
        for _ in range(len(keys)):
            sampled.extend(members[keys[rng.randrange(len(keys))]])
        report = classification_report(confusion_matrix(
            [row["true_label_index"] for row in sampled],
            [row["pred_label_index"] for row in sampled]))
        macro_values.append(macro_f1_from_report(report))
        for cls in range(4):
            recalls[cls].append(report[cls]["recall"])
    output = {"macro_f1": percentile_ci(macro_values)}
    for cls in range(4):
        output[f"recall_{cls}"] = percentile_ci(recalls[cls])
    return output


def plot_matrix(matrix: list[list[int]], path: Path, title: str, normalize: bool) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    array = np.asarray(matrix, dtype=float)
    if normalize:
        array = array / np.maximum(array.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(array, cmap="Blues")
    axis.set_xticks(range(4), LABELS)
    axis.set_yticks(range(4), LABELS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row in range(4):
        for col in range(4):
            value = array[row, col]
            axis.text(col, row, f"{value:.2f}" if normalize else f"{int(value)}",
                      ha="center", va="center",
                      color="white" if value > array.max() / 2 else "black")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def load_comparison(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("test_metrics") or payload.get("summary") or payload
    return {
        "source": str(path),
        "run": path.parent.name,
        "checkpoint_selection": payload.get("checkpoint_selection", ""),
        "best_epoch": payload.get("best_epoch", ""),
        "accuracy": metrics.get("accuracy", metrics.get("classification_accuracy", "")),
        "macro_f1": metrics.get("macro_f1", ""),
        "kappa": metrics.get("kappa", metrics.get("classification_kappa", "")),
        "dice": metrics.get("dice", metrics.get("mean_dice", "")),
        "iou": metrics.get("iou", metrics.get("mean_iou", "")),
    }



def count_metrics(rows: list[dict], true_labels: list[int]) -> dict | None:
    """Ringkas akurasi counting head bila kolom pred_count tersedia."""
    count_rows = [row for row in rows
                  if str(row.get("count_error", "")).strip() not in {"", "nan"}]
    if not count_rows:
        return None
    errors = [float(row["count_error"]) for row in count_rows]
    by_grade: dict[int, list[float]] = defaultdict(list)
    for row in count_rows:
        by_grade[int(row["true_label_index"])].append(float(row["count_error"]))
    payload = {
        "n": len(errors),
        "count_mae": sum(abs(value) for value in errors) / len(errors),
        "count_bias": sum(errors) / len(errors),
        "by_true_grade": {
            str(grade): {
                "n": len(values),
                "count_mae": sum(abs(value) for value in values) / len(values),
                "count_bias": sum(values) / len(values),
            }
            for grade, values in sorted(by_grade.items())
        },
    }
    grade_rows = [row for row in rows
                  if str(row.get("pred_count_grade", "")).strip() not in {"", "nan"}]
    if len(grade_rows) == len(rows):
        count_pred = [int(row["pred_count_grade"]) for row in grade_rows]
        if all(0 <= value <= 3 for value in count_pred):
            matrix = confusion_matrix(true_labels, count_pred)
            report = classification_report(matrix)
            payload["grade_from_count"] = {
                "accuracy": sum(1 for true, pred in zip(true_labels, count_pred)
                                if true == pred) / len(rows),
                "macro_f1": macro_f1_from_report(report),
                "confusion_matrix": matrix,
                "recall_ci95_wilson": {
                    str(cls): wilson_interval(matrix[cls][cls], report[cls]["support"])
                    for cls in range(4)
                },
            }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="testing_predictions.csv dari evaluasi checkpoint terkunci.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None,
                        help="metrics.json run untuk mencatat checkpoint_selection/best_epoch.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest test dengan kolom group_id untuk cluster bootstrap.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comparison", action="append", default=[],
                        help="NAMA=path/metrics.json, boleh berulang.")
    parser.add_argument("--title", default="Final test evaluation")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} sudah berisi file. Pakai folder baru.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_predictions(args.predictions)
    true_labels = [row["true_label_index"] for row in rows]
    pred_labels = [row["pred_label_index"] for row in rows]
    matrix = confusion_matrix(true_labels, pred_labels)
    report = classification_report(matrix)
    groups = read_groups(args.manifest)

    total = len(rows)
    assert sum(sum(row) for row in matrix) == total
    for cls in range(4):
        assert sum(matrix[cls]) == report[cls]["support"]
        assert report[cls]["predicted"] == sum(matrix[row][cls] for row in range(4))

    macro_f1 = macro_f1_from_report(report)
    weighted_f1 = weighted_f1_from_report(report)
    balanced = balanced_accuracy_from_report(report)
    kappa = cohen_kappa(matrix)
    accuracy = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred) / total
    mae = ordinal_mae(true_labels, pred_labels)
    bias = ordinal_bias(true_labels, pred_labels)
    intervals = {cls: wilson_interval(matrix[cls][cls], report[cls]["support"])
                 for cls in range(4)}
    group_ci = cluster_bootstrap(rows, groups, args.bootstrap, args.seed) if groups else {}
    dice = [float(row["dice"]) for row in rows if str(row.get("dice", "")) not in {"", "nan"}]
    iou = [float(row["iou"]) for row in rows if str(row.get("iou", "")) not in {"", "nan"}]
    counting = count_metrics(rows, true_labels)

    run_metrics = {}
    if args.metrics and args.metrics.is_file():
        run_metrics = json.loads(args.metrics.read_text(encoding="utf-8"))

    test_used_in_training = run_metrics.get("test_metrics") is not None
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": str(args.predictions.resolve()),
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "title": args.title,
        "test_images": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "ordinal_mae": mae,
        "ordinal_bias": bias,
        "mean_dice": sum(dice) / len(dice) if dice else None,
        "mean_iou": sum(iou) / len(iou) if iou else None,
        "counting": counting,
        "confusion_matrix": matrix,
        "classification_report": report,
        "recall_ci95_wilson": {str(cls): intervals[cls] for cls in range(4)},
        "bootstrap": {"iterations": args.bootstrap, "cluster_level": bool(groups),
                      "result": group_ci},
        "checkpoint_selection": run_metrics.get("checkpoint_selection", ""),
        "best_epoch": run_metrics.get("best_epoch", ""),
        "checkpoint_metric": run_metrics.get("checkpoint_metric", "val_loss"),
        "test_already_used_during_training": test_used_in_training,
        "limitations": [
            "Test internal, bukan test eksternal independen; test lama telah dipakai berulang kali.",
            "Level 3 support kecil; interval kepercayaan lebar.",
            "Perbandingan dengan run lama bersifat historical reference.",
        ],
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report_rows = []
    for row in report:
        cls = int(row["label"])
        report_rows.append({
            "label_index": cls, "label_name": LABELS[cls], "support": row["support"],
            "predicted": row["predicted"], "precision": round(row["precision"], 6),
            "recall": round(row["recall"], 6), "f1": round(row["f1"], 6),
            "recall_ci95_low": round(intervals[cls][0], 6),
            "recall_ci95_high": round(intervals[cls][1], 6),
        })
    write_rows(args.output_dir / "classification_report.csv", report_rows)
    write_rows(args.output_dir / "test_predictions.csv", [
        dict(row, true_label_name=LABELS[row["true_label_index"]],
             pred_label_name=LABELS[row["pred_label_index"]])
        for row in rows])

    plot_matrix(matrix, args.output_dir / "confusion_matrix.png", args.title, normalize=False)
    plot_matrix(matrix, args.output_dir / "confusion_matrix_normalized.png",
                f"{args.title} (row-normalized)", normalize=True)

    comparison_rows = []
    for item in args.comparison:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError("Format --comparison: NAMA=path/metrics.json")
        comparison_rows.append(dict(load_comparison(Path(path)), experiment=name))
    if comparison_rows:
        write_rows(args.output_dir / "experiment_comparison.csv", comparison_rows)

    lines = [
        f"# {args.title}", "",
        f"- Prediksi: `{args.predictions}`; gambar test: **{total}**.",
        f"- Checkpoint selection: **{payload['checkpoint_selection'] or 'tidak tercatat'}**; "
        f"best epoch: **{payload['best_epoch'] or 'tidak tercatat'}**.",
        f"- Test sudah pernah dipakai saat training: **{'ya' if test_used_in_training else 'tidak'}**.",
        "", "## Metrik utama", "",
        "| Metrik | Nilai |", "|---|---:|",
        f"| Macro F1 | {macro_f1:.4f} |",
        f"| Balanced accuracy | {balanced:.4f} |",
        f"| Accuracy | {accuracy:.4f} |",
        f"| Weighted F1 | {weighted_f1:.4f} |",
        f"| Cohen kappa (unweighted) | {kappa:.4f} |",
        f"| Ordinal MAE | {mae:.4f} |",
        f"| Mean Dice | {payload['mean_dice'] if payload['mean_dice'] is not None else 'n/a'} |",
        f"| Mean IoU | {payload['mean_iou'] if payload['mean_iou'] is not None else 'n/a'} |",
        "", "## Per kelas", "",
        "| Kelas | Support | Precision | Recall | F1 | Recall 95% CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report:
        cls = int(row["label"])
        lines.append(f"| {LABELS[cls]} | {row['support']} | {row['precision']:.4f} | "
                     f"{row['recall']:.4f} | {row['f1']:.4f} | "
                     f"{intervals[cls][0]:.3f}–{intervals[cls][1]:.3f} |")
    lines += [
        "", "## Confusion matrix (baris = true)", "",
        "| true \\ pred | " + " | ".join(LABELS) + " |",
        "|---|" + "---:|" * 4,
    ]
    for index, row in enumerate(matrix):
        lines.append(f"| {LABELS[index]} | " + " | ".join(str(value) for value in row) + " |")
    lines += [
        "", "## Ordinal bias", "",
        f"- underestimate: {bias['underestimated']}; exact: {bias['exact']}; "
        f"overestimate: {bias['overestimated']}.",
    ]
    if counting:
        lines += [
            "", "## Counting head", "",
            f"- Gambar dengan count: **{counting['n']}**; "
            f"MAE: **{counting['count_mae']:.3f}**; bias: **{counting['count_bias']:+.3f}**.",
            "", "| True grade | n | MAE | Bias |", "|---:|---:|---:|---:|",
        ]
        for grade, values in counting["by_true_grade"].items():
            lines.append(f"| {LABELS[int(grade)]} | {values['n']} | "
                         f"{values['count_mae']:.3f} | {values['count_bias']:+.3f} |")
        grade_metrics = counting.get("grade_from_count")
        if grade_metrics:
            lines += [
                "", f"- Grade dari count prediksi: accuracy "
                f"**{grade_metrics['accuracy']:.4f}**, Macro F1 "
                f"**{grade_metrics['macro_f1']:.4f}** (diagnostik, bukan model final)."
            ]
    lines += [
        "", "## Uncertainty", "",
    ]
    if group_ci:
        lines.append(f"- Cluster bootstrap ({args.bootstrap} iterasi) pada level group.")
        for key, value in group_ci.items():
            lines.append(f"  - {key}: {value['mean']:.4f} "
                         f"(95% {value['low']:.4f}–{value['high']:.4f})")
    else:
        lines.append("- Tidak ada kolom `group_id`; interval per kelas memakai Wilson. "
                     "Bootstrap group tidak dijalankan.")
    lines += [
        "", "## Batasan", "",
    ] + [f"- {item}" for item in payload["limitations"]] + [
        "", "## Kesimpulan", "",
        "- Konfigurasi dibekukan dari validation; test tidak dipakai untuk memilih checkpoint.",
        "- Angka ini adalah evaluasi internal, bukan validasi klinis.",
    ]
    (args.output_dir / "final_research_report.md").write_text("\n".join(lines) + "\n",
                                                              encoding="utf-8")

    print(json.dumps({"macro_f1": round(macro_f1, 4), "balanced_accuracy": round(balanced, 4),
                      "accuracy": round(accuracy, 4), "kappa": round(kappa, 4),
                      "test_images": total}, indent=2))
    print("Report:", args.output_dir / "final_research_report.md")


if __name__ == "__main__":
    main()
