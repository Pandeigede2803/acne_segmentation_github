#!/usr/bin/env python3
"""Audit split, offline augmentation, mask geometry, and MA-LDS targets before training.

The split and augmentation decisions are reproduced exactly (same seeds, same
order, same RNG call sequence) from:

  split_train_test_manifest.py   -> train/test 60/40, seed 42
  split_train_validation.py      -> 15% validation from train, seed 42
  augment_train_pairs_offline.py -> target 1000/class, seed 42

Mask geometry is measured on circle masks rasterized from the official
ACNE04-v2 annotations with the same letterbox geometry used by
refine_masks_color.py (target 512). The offline augmenter only rotates and
flips the mask, so area retention is geometry-only and does not depend on the
colour refinement step (V1/V2).

Outputs (in --output-dir):
  audit_metrics.json  all computed numbers
  audit_report.md     human-readable report

This script never writes to output/, never trains, and never touches test rows.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from audit_acne04v2 import audit_source  # noqa: E402
from augment_train_pairs_offline import class_targets  # noqa: E402
from split_train_test_manifest import assign_splits  # noqa: E402
from split_train_validation import split_train_validation  # noqa: E402

BASE_EPS = 0.05
MAX_EPS = 0.35


def load_annotations(path: Path) -> tuple[list[dict], dict[str, list[tuple[float, float, float]]],
                                          dict[str, tuple[int, int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = {row["id"]: row for row in data["images"]}
    circles: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    sizes: dict[str, tuple[int, int]] = {}
    for ann in data["annotations"]:
        meta = images[ann["image_id"]]
        cx, cy = ann["coordinates"]
        circles[meta["file_name"]].append((float(cx), float(cy), float(ann["radius"])))
        sizes[meta["file_name"]] = (int(meta["width"]), int(meta["height"]))
    return data["images"], circles, sizes


def circle_mask(circles: list[tuple[float, float, float]], width: int, height: int,
                size: int) -> np.ndarray:
    """Same letterbox geometry as refine_masks_color.build_circle_mask."""
    mask, _ = circle_mask_and_centers(circles, width, height, size)
    return mask


def circle_mask_and_centers(circles: list[tuple[float, float, float]], width: int, height: int,
                            size: int) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    scale = min(size / width, size / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    offset_x = (size - resized_w) // 2
    offset_y = (size - resized_h) // 2
    mask = np.zeros((size, size), dtype=np.uint8)
    centers: list[tuple[float, float, float]] = []
    for cx, cy, radius in circles:
        tx = int(cx * scale) + offset_x
        ty = int(cy * scale) + offset_y
        tr = max(1, int(radius * scale))
        cv2.circle(mask, (tx, ty), tr, 255, -1)
        centers.append((float(tx), float(ty), float(tr)))
    return mask, centers


def transform_centers(centers: list[tuple[float, float, float]], angle: float, size: int,
                      flip: bool) -> list[tuple[float, float, float]]:
    """Rotate/flip circle centers the same way PIL rotates the mask."""
    radians = np.deg2rad(angle)
    cos, sin = float(np.cos(radians)), float(np.sin(radians))
    center = (size - 1) / 2.0
    output = []
    for x, y, radius in centers:
        dx, dy = x - center, y - center
        rx = cos * dx + sin * dy + center
        ry = -sin * dx + cos * dy + center
        if flip:
            rx = (size - 1) - rx
        output.append((rx, ry, radius))
    return output


def fully_clipped(center: tuple[float, float, float], size: int) -> bool:
    x, y, radius = center
    return (x + radius <= 0.0 or x - radius >= size - 1
            or y + radius <= 0.0 or y - radius >= size - 1)


GRID_X: np.ndarray | None = None
GRID_Y: np.ndarray | None = None


def morphology(mask: np.ndarray) -> dict[str, float]:
    """Copy of train_proposal_hybrid_resnet50_malds.mask_morphology."""
    global GRID_X, GRID_Y
    probs = mask.astype(np.float32) / 255.0
    height, width = probs.shape
    if GRID_X is None or GRID_X.shape[1] != width:
        GRID_X = np.linspace(0.0, 1.0, width, dtype=np.float32).reshape(1, width)
        GRID_Y = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    mass = max(float(probs.sum()), 1e-6)
    area = float(probs.mean())
    density = float((probs > 0.5).mean())
    center_x = float((probs * GRID_X).sum() / mass)
    center_y = float((probs * GRID_Y).sum() / mass)
    spread_x = float((probs * np.abs(GRID_X - center_x)).sum() / mass)
    spread_y = float((probs * np.abs(GRID_Y - center_y)).sum() / mass)
    return {"area": area, "density": density, "centroid_x": center_x,
            "centroid_y": center_y, "spread": 0.5 * (spread_x + spread_y)}


def malds_epsilon(features: dict[str, float]) -> float:
    eps = BASE_EPS + 1.5 * features["area"] + 0.5 * features["density"] + 0.2 * features["spread"]
    return float(min(max(eps, BASE_EPS), MAX_EPS))


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "p1": float(np.percentile(array, 1)),
        "p5": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def rank_auc(positive: list[float], negative: list[float]) -> float:
    """AUC of positive scores above negative scores, ties handled by average rank."""
    if not positive or not negative:
        return float("nan")
    combined = [(value, 1) for value in positive] + [(value, 0) for value in negative]
    combined.sort(key=lambda item: item[0])
    ranks = np.empty(len(combined), dtype=np.float64)
    index = 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        ranks[index:end + 1] = average
        index = end + 1
    labels = np.asarray([label for _, label in combined], dtype=np.int64)
    positive_rank_sum = float(ranks[labels == 1].sum())
    n_pos, n_neg = len(positive), len(negative)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def simulate_augmentation(train_rows: list[dict], target_per_class: int, seed: int,
                          rotation_deg: float) -> tuple[list[dict], dict[int, dict]]:
    """Replay augment_train_pairs_offline.main() RNG calls in the same order."""
    counts = Counter(int(row["label_index"]) for row in train_rows)
    targets = class_targets(counts, 0.75, target_per_class)
    rng = random.Random(seed)
    synthetics: list[dict] = []
    multiplicity: dict[int, Counter] = defaultdict(Counter)
    for cls, count in counts.items():
        sources = [row for row in train_rows if int(row["label_index"]) == cls]
        for j in range(targets[cls] - count):
            source = rng.choice(sources)
            angle = rng.uniform(-rotation_deg, rotation_deg)
            flip = rng.random() < 0.5
            rng.uniform(0.85, 1.15)  # brightness
            rng.uniform(0.85, 1.15)  # contrast
            multiplicity[cls][source["filename"]] += 1
            synthetics.append({"source": source["filename"], "label_index": cls,
                               "synthetic_index": j, "angle": angle, "flip": flip})
    per_class = {}
    for cls in sorted(counts):
        copies = [multiplicity[cls][row["filename"]] for row in train_rows
                  if int(row["label_index"]) == cls]
        totals = [1 + value for value in copies]
        per_class[cls] = {
            "originals": counts[cls],
            "synthetic": targets[cls] - counts[cls],
            "total": targets[cls],
            "copies": summary([float(value) for value in copies]),
            "sources_unused": int(sum(1 for value in copies if value == 0)),
            "sources_above_10": int(sum(1 for value in copies if value > 10)),
            "sources_above_50": int(sum(1 for value in copies if value > 50)),
            "max_share_from_one_source": float(max(totals) / sum(totals)),
            "effective_sample_size": float(sum(totals) ** 2 / sum(value * value for value in totals)),
            "synthetic_share": float((sum(totals) - len(totals)) / sum(totals)),
        }
    return synthetics, per_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations-json", type=Path,
                        default=ROOT.parent / "acne04v2" / "Acne04-v2_annotations.json")
    parser.add_argument("--image-dir", type=Path,
                        default=ROOT / "datasetacne04" / "Classification" / "JPEGImages")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "audit_step34")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--target-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--rotation-deg", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.annotations_json.is_file():
        raise FileNotFoundError(args.annotations_json)
    if not args.image_dir.is_dir():
        raise FileNotFoundError(args.image_dir)

    source_audit = audit_source(args.image_dir, args.annotations_json)
    _, circles, sizes = load_annotations(args.annotations_json)
    names = sorted(circles)
    if len(names) != 1204:
        raise ValueError(f"Anotasi harus 1.204 citra; ditemukan {len(names)}.")

    rows = [{"filename": name, "label_name": name.split("_")[0],
             "label_index": name.split("_")[0][-1], "split": "train"} for name in names]

    # Step 31: train/test 60/40, seed 42 (same row order as the base manifest).
    split_rows = assign_splits(rows, args.train_ratio, args.seed)
    # Step 39a: carve validation out of train, seed 42.
    final_rows = split_train_validation(split_rows, args.val_fraction, args.seed)
    split_counts = Counter((row["split"], int(row["label_index"])) for row in final_rows)
    train_rows = [row for row in final_rows if row["split"] == "train"]

    synthetics, multiplicity = simulate_augmentation(
        train_rows, args.target_per_class, args.seed, args.rotation_deg)

    # One circle mask per train source, at the training resolution.
    source_masks: dict[str, np.ndarray] = {}
    source_features: dict[str, dict[str, float]] = {}
    source_centers: dict[str, list[tuple[float, float, float]]] = {}
    for row in train_rows:
        name = row["filename"]
        width, height = sizes[name]
        mask, centers = circle_mask_and_centers(circles[name], width, height, args.image_size)
        source_masks[name] = mask
        source_features[name] = morphology(mask)
        source_centers[name] = centers

    retention_by_class: dict[int, list[float]] = defaultdict(list)
    lost_lesions_by_class: dict[int, list[int]] = defaultdict(list)
    synth_features: dict[int, list[dict[str, float]]] = defaultdict(list)
    for item in synthetics:
        source = source_masks[item["source"]]
        image = Image.fromarray(source)
        rotated = image.rotate(item["angle"], resample=Image.Resampling.NEAREST, fillcolor=0)
        if item["flip"]:
            rotated = rotated.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = np.asarray(rotated, dtype=np.uint8)
        features = morphology(mask)
        synth_features[item["label_index"]].append(features)
        area_source = source_features[item["source"]]["area"]
        retention_by_class[item["label_index"]].append(
            features["area"] / area_source if area_source > 0 else 0.0)
        centers = transform_centers(source_centers[item["source"]], item["angle"],
                                    args.image_size, item["flip"])
        lost_lesions_by_class[item["label_index"]].append(
            sum(1 for center in centers if fully_clipped(center, args.image_size)))

    geometry: dict[int, dict] = {}
    for cls in sorted(multiplicity):
        retention = retention_by_class[cls]
        lost = np.asarray(lost_lesions_by_class[cls], dtype=np.int64)
        eps_original = [malds_epsilon(source_features[row["filename"]])
                        for row in train_rows if int(row["label_index"]) == cls]
        eps_synthetic = [malds_epsilon(features) for features in synth_features[cls]]
        geometry[cls] = {
            "retention": summary(retention),
            "retention_below_0.999": float(np.mean(np.asarray(retention) < 0.999)),
            "retention_below_0.99": float(np.mean(np.asarray(retention) < 0.99)),
            "retention_below_0.95": float(np.mean(np.asarray(retention) < 0.95)),
            "samples_with_fully_clipped_lesion": float(np.mean(lost > 0)),
            "fully_clipped_lesions_per_sample": summary([float(value) for value in lost]),
            "epsilon_original": summary(eps_original),
            "epsilon_synthetic": summary(eps_synthetic),
            "epsilon_synthetic_clipped_at_max": float(
                np.mean(np.asarray(eps_synthetic) >= MAX_EPS - 1e-9)),
            "area_original": summary([source_features[row["filename"]]["area"]
                                      for row in train_rows if int(row["label_index"]) == cls]),
            "area_synthetic": summary([features["area"] for features in synth_features[cls]]),
            "spread_original": summary([source_features[row["filename"]]["spread"]
                                        for row in train_rows if int(row["label_index"]) == cls]),
            "spread_synthetic": summary([features["spread"] for features in synth_features[cls]]),
        }

    all_eps = [malds_epsilon(source_features[row["filename"]]) for row in train_rows]
    eps_class = {cls: [malds_epsilon(source_features[row["filename"]])
                       for row in train_rows if int(row["label_index"]) == cls] for cls in range(4)}

    metrics = {
        "source": source_audit,
        "protocol": {"train_ratio": args.train_ratio, "val_fraction": args.val_fraction,
                     "target_per_class": args.target_per_class, "seed": args.seed,
                     "image_size": args.image_size, "rotation_deg": args.rotation_deg,
                     "base_eps": BASE_EPS, "max_eps": MAX_EPS},
        "split_counts": {f"{split}/{label}": count
                         for (split, label), count in sorted(split_counts.items())},
        "split_totals": {split: sum(count for (name, _), count in split_counts.items() if name == split)
                         for split in ("train", "val", "test")},
        "augmentation": {"synthetic_total": len(synthetics),
                         "synthetic_share_overall": len(synthetics) / (len(train_rows) + len(synthetics)),
                         "per_class": multiplicity},
        "geometry": geometry,
        "epsilon": {
            "per_class": {cls: summary(values) for cls, values in eps_class.items()},
            "auc_level2_vs_level3": rank_auc(eps_class[3], eps_class[2]),
            "level2_above_level3_threshold": {
                "level3_min": min(eps_class[3]),
                "level2_max": max(eps_class[2]),
                "level2_p95": pct(eps_class[2], 95),
                "level3_p05": pct(eps_class[3], 5),
                "overlap_ratio_level3_below_level2_p95": float(
                    np.mean(np.asarray(eps_class[3]) <= pct(eps_class[2], 95))),
            },
        },
        "val_files": sorted(row["filename"] for row in final_rows if row["split"] == "val"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "audit_report.md").write_text(render_report(metrics), encoding="utf-8")
    print(json.dumps({"split_totals": metrics["split_totals"],
                      "synthetic_total": len(synthetics)}, indent=2))
    print("Report:", args.output_dir / "audit_report.md")


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_report(metrics: dict) -> str:
    split = metrics["split_counts"]
    aug = metrics["augmentation"]
    geometry = metrics["geometry"]
    lines = [
        "# Audit Step 34+ — Split, Augmentasi, Geometri Mask, MA-LDS",
        "",
        "Dihasilkan otomatis oleh `scripts/audit_step34_dataset_augmentation.py`.",
        "Angka split/augmentasi adalah replay deterministik dari skrip pipeline",
        f"(seed {metrics['protocol']['seed']}, target {metrics['protocol']['target_per_class']}"
        "/kelas). Geometri diukur pada circle mask anotasi resmi yang dirasterisasi ke "
        f"{metrics['protocol']['image_size']}px letterbox.",
        "",
        "## 1. Jumlah per split/kelas",
        "",
        "| Split | Level 0 | Level 1 | Level 2 | Level 3 | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "val", "test"):
        cells = [split.get(f"{name}/{cls}", 0) for cls in range(4)]
        lines.append(f"| {name} | " + " | ".join(str(value) for value in cells)
                     + f" | {metrics['split_totals'][name]} |")
    lines += [
        "",
        "## 2. Penggandaan sintetis per citra asli",
        "",
        "| Level | Asli | Sintetis | Total | Median salinan | Max salinan | Sumber tanpa salinan "
        "| Sumber >50 salinan | Max share 1 sumber | ESS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cls in sorted(aug["per_class"], key=int):
        entry = aug["per_class"][cls]
        copies = entry["copies"]
        lines.append(
            f"| {cls} | {entry['originals']} | {entry['synthetic']} | {entry['total']} | "
            f"{fmt(copies['median'], 1)} | {fmt(copies['max'], 0)} | {entry['sources_unused']} | "
            f"{entry['sources_above_50']} | {fmt(entry['max_share_from_one_source'])} | "
            f"{fmt(entry['effective_sample_size'], 1)} |")
    lines += [
        "",
        f"Total baris train setelah augmentasi: **{sum(v['total'] for v in aug['per_class'].values())}**, "
        f"sintetis **{aug['synthetic_total']}** "
        f"({fmt(aug['synthetic_share_overall'] * 100, 1)}%).",
        "",
        "ESS (effective sample size) = (Σn)²/Σn², dengan n = jumlah baris per citra asli.",
        "ESS jauh di bawah jumlah baris berarti model melihat citra yang sama berulang kali.",
        "",
        "## 3. Retensi area mask dan lesi hilang setelah rotasi/flip offline",
        "",
        "| Level | Median | P1 | P5 | Min | <0,999 | <0,99 | <0,95 | Sampel dengan lesi terpotong total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cls in sorted(geometry, key=int):
        entry = geometry[cls]
        retention = entry["retention"]
        lines.append(
            f"| {cls} | {fmt(retention['median'])} | {fmt(retention['p1'])} | "
            f"{fmt(retention['p5'])} | {fmt(retention['min'])} | "
            f"{fmt(entry['retention_below_0.999'] * 100, 2)}% | "
            f"{fmt(entry['retention_below_0.99'] * 100, 2)}% | "
            f"{fmt(entry['retention_below_0.95'] * 100, 2)}% | "
            f"{fmt(entry['samples_with_fully_clipped_lesion'] * 100, 2)}% |")
    lines += [
        "",
        "Retensi = area(mask setelah rotasi+flip) / area(mask sumber). Nilai <1 berarti",
        "sebagian lesi terpotong oleh kanvas rotasi (fill hitam, tanpa expand).",
        "\"Lesi terpotong total\" = pusat+radius anotasi berada seluruhnya di luar kanvas",
        "setelah rotasi/flip. Retensi bisa sedikit >1 karena pembulatan rasterisasi.",
        "",
        "## 4. Fitur morfologi per kelas (median)",
        "",
        "| Level | area asli (P5–P95) | area sintetis | spread asli (P5–P95) | spread sintetis |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls in sorted(geometry, key=int):
        entry = geometry[cls]
        area_original = entry["area_original"]
        area_synthetic = entry["area_synthetic"]
        spread_original = entry["spread_original"]
        spread_synthetic = entry["spread_synthetic"]
        lines.append(
            f"| {cls} | {fmt(area_original['median'])} "
            f"({fmt(area_original['p5'])}–{fmt(area_original['p95'])}) | "
            f"{fmt(area_synthetic['median'])} | "
            f"{fmt(spread_original['median'])} "
            f"({fmt(spread_original['p5'])}–{fmt(spread_original['p95'])}) | "
            f"{fmt(spread_synthetic['median'])} |")
    lines += [
        "",
        "Untuk mask biner, `density` identik dengan `area`; karena itu epsilon efektif",
        "= `0.05 + 2.0·area + 0.2·spread`.",
        "",
        "## 5. Distribusi epsilon MA-LDS per kelas",
        "",
        "| Level | Epsilon asli (mean) | Epsilon asli (P5–P95) | Epsilon sintetis (mean) | "
        "Epsilon sintetis (P5–P95) | Sintetis kena clip 0,35 |",
        "|---|---:|---|---:|---|---:|",
    ]
    for cls in sorted(geometry, key=int):
        entry = geometry[cls]
        original = entry["epsilon_original"]
        synthetic = entry["epsilon_synthetic"]
        lines.append(
            f"| {cls} | {fmt(original['mean'])} | {fmt(original['p5'])}–{fmt(original['p95'])} | "
            f"{fmt(synthetic['mean'])} | {fmt(synthetic['p5'])}–{fmt(synthetic['p95'])} | "
            f"{fmt(entry['epsilon_synthetic_clipped_at_max'] * 100, 2)}% |")
    separation = metrics["epsilon"]["level2_above_level3_threshold"]
    lines += [
        "",
        f"AUC Level 3 vs Level 2 (epsilon): **{fmt(metrics['epsilon']['auc_level2_vs_level3'])}**.",
        f"Level 3 minimum {fmt(separation['level3_min'])}; Level 2 maksimum "
        f"{fmt(separation['level2_max'])}; P95 Level 2 {fmt(separation['level2_p95'])}. "
        f"{fmt(separation['overlap_ratio_level3_below_level2_p95'] * 100, 1)}% citra Level 3 "
        "berada di bawah P95 Level 2.",
        "",
        "## 6. Catatan metode",
        "",
        "- Split dan pemilihan sumber augmentasi direplay dengan urutan RNG yang sama seperti",
        "  `augment_train_pairs_offline.py` (choice, rotasi, flip, brightness, contrast).",
        "- Geometri memakai circle mask anotasi, bukan mask V1/V2 hasil refinement warna.",
        "  Rotasi/flip diterapkan pada mask yang sama; jadi retensi area di sini adalah",
        "  batas geometris murni, bukan hasil akhir V2.",
        "- Fitur morfologi dan epsilon dihitung dengan rumus trainer yang sama",
        "  (`area`, `density`, `centroid`, `spread`; eps = base + 1.5·area + 0.5·density + 0.2·spread).",
        "- Test tidak dipakai untuk menghitung apa pun di laporan ini.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
