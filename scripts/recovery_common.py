#!/usr/bin/env python3
"""Dependency-free helpers shared by the Step 42-48 recovery protocol.

Everything here is pure Python so the CPU-only steps can be unit tested without
numpy/PIL/torch. The CLI scripts add the image/IO layer on top.
"""
from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path

LABELS = ["levle0", "levle1", "levle2", "levle3"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0].keys()) if rows else [])
    if not fields:
        raise ValueError(f"Tidak ada kolom untuk ditulis: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Duplicate grouping
# ---------------------------------------------------------------------------

def stable_group_ids(keys) -> dict[str, str]:
    """Assign deterministic ``dup_0001`` ids ordered by the first sort key."""
    unique = sorted(set(keys))
    return {key: f"dup_{index:04d}" for index, key in enumerate(unique, start=1)}


def pairs_to_groups(nodes, pairs) -> dict[str, str]:
    """Union-find over ``(a, b)`` pairs; returns node -> ``grp_0001`` id."""
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for left, right in pairs:
        if left in parent and right in parent:
            union(left, right)

    roots = sorted({find(node) for node in parent})
    names = {root: f"grp_{index:04d}" for index, root in enumerate(roots, start=1)}
    return {node: names[find(node)] for node in parent}


# ---------------------------------------------------------------------------
# Controlled augmentation (Step 45)
# ---------------------------------------------------------------------------

def controlled_target(count: int, target: int = 300, cap: int = 7) -> int:
    """``max(n, min(target, cap * n))``: no undersampling, at most ``cap`` rows."""
    if count < 1:
        raise ValueError("Jumlah original per kelas harus >= 1.")
    if target < 1 or cap < 1:
        raise ValueError("target dan cap harus >= 1.")
    return max(count, min(target, cap * count))


def controlled_targets(counts: dict[int, int], target: int = 300, cap: int = 7) -> dict[int, int]:
    return {cls: controlled_target(count, target, cap) for cls, count in counts.items()}


def even_allocation(total: int, n_sources: int) -> list[int]:
    """Split ``total`` evenly over ``n_sources`` (difference at most one)."""
    if n_sources < 1:
        raise ValueError("Butuh minimal satu sumber.")
    if total < 0:
        raise ValueError("total tidak boleh negatif.")
    base, remainder = divmod(total, n_sources)
    return [base + (1 if index < remainder else 0) for index in range(n_sources)]


def max_synthetic_per_source(cap: int = 7) -> int:
    """Rows per source include the original, so synthetic copies are ``cap - 1``."""
    return max(0, cap - 1)


# ---------------------------------------------------------------------------
# Grouped split (Step 43)
# ---------------------------------------------------------------------------

def assign_grouped_splits(group_labels: dict[str, int], train_ratio: float = 0.6,
                          val_ratio: float = 0.2, test_ratio: float = 0.2,
                          seed: int = 42, min_val_per_class: int = 1,
                          min_test_per_class: int = 1) -> dict[str, str]:
    """Deterministic stratified split over groups, never over individual images."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train+val+test ratio harus 1.0.")
    if train_ratio <= 0 or test_ratio <= 0 or val_ratio < 0:
        raise ValueError("train dan test harus > 0; val boleh 0.")

    by_class: dict[int, list[str]] = {}
    for group, label in group_labels.items():
        by_class.setdefault(int(label), []).append(group)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for label in sorted(by_class):
        groups = sorted(by_class[label])
        rng.shuffle(groups)
        total = len(groups)
        if total < 3:
            raise ValueError(f"Kelas {label} hanya punya {total} grup; butuh >= 3 untuk train/val/test.")

        n_test = max(min_test_per_class, int(round(total * test_ratio)))
        n_val = 0 if val_ratio == 0 else max(min_val_per_class, int(round(total * val_ratio)))
        n_test = min(n_test, total - 1 - n_val)
        n_val = min(n_val, total - n_test - 1)
        n_train = total - n_test - n_val
        if n_train < 1:
            raise ValueError(f"Kelas {label}: grup tidak cukup untuk train/val/test.")

        for index, group in enumerate(groups):
            if index < n_train:
                assignment[group] = "train"
            elif index < n_train + n_val:
                assignment[group] = "val"
            else:
                assignment[group] = "test"
    return assignment


# ---------------------------------------------------------------------------
# Classification metrics (Step 46/48) - dependency-free
# ---------------------------------------------------------------------------

def confusion_matrix(true_labels: list[int], pred_labels: list[int], num_classes: int = 4) -> list[list[int]]:
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(true_labels, pred_labels):
        matrix[int(true)][int(pred)] += 1
    return matrix


def classification_report(matrix: list[list[int]]) -> list[dict[str, float | int | str]]:
    num_classes = len(matrix)
    supports = [sum(row) for row in matrix]
    predicted = [sum(matrix[row][col] for row in range(num_classes)) for col in range(num_classes)]
    report = []
    for cls in range(num_classes):
        tp = matrix[cls][cls]
        fp = predicted[cls] - tp
        fn = supports[cls] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / supports[cls] if supports[cls] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report.append({"label": cls, "label_name": LABELS[cls], "support": supports[cls],
                       "predicted": predicted[cls], "precision": precision,
                       "recall": recall, "f1": f1})
    return report


def macro_f1_from_report(report: list[dict]) -> float:
    values = [row["f1"] for row in report]
    return sum(values) / len(values) if values else 0.0


def weighted_f1_from_report(report: list[dict]) -> float:
    total = sum(row["support"] for row in report)
    if not total:
        return 0.0
    return sum(row["f1"] * row["support"] for row in report) / total


def balanced_accuracy_from_report(report: list[dict]) -> float:
    values = [row["recall"] for row in report if row["support"] > 0]
    return sum(values) / len(values) if values else 0.0


def cohen_kappa(matrix: list[list[int]]) -> float:
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return 0.0
    observed = sum(matrix[index][index] for index in range(len(matrix))) / total
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[row][col] for row in range(len(matrix))) for col in range(len(matrix))]
    expected = sum(row_sums[index] * col_sums[index] for index in range(len(matrix))) / (total * total)
    return 0.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)


def ordinal_mae(true_labels: list[int], pred_labels: list[int]) -> float:
    if not true_labels:
        return 0.0
    return sum(abs(int(true) - int(pred)) for true, pred in zip(true_labels, pred_labels)) / len(true_labels)


def ordinal_bias(true_labels: list[int], pred_labels: list[int]) -> dict[str, int]:
    under = sum(1 for true, pred in zip(true_labels, pred_labels) if pred < true)
    over = sum(1 for true, pred in zip(true_labels, pred_labels) if pred > true)
    return {"underestimated": under, "overestimated": over,
            "exact": len(true_labels) - under - over}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_mean(values: list[float], iterations: int = 2000, seed: int = 42) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean": sum(values) / n,
        "low": means[int(0.025 * len(means))],
        "high": means[min(len(means) - 1, int(0.975 * len(means)))],
    }
