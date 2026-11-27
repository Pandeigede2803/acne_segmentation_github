#!/usr/bin/env python3
"""Step 35/36: compare Circle, V1 and V2 masks without training.

Input is a manifest produced by ``refine_masks_color_v2.py``. The manifest must
contain ``circle_mask``, ``v1_mask`` and ``refined_mask`` (V2) paths plus
``preprocessed_path``. Statistics are descriptive: IoU against the annotation
circle measures agreement with the pseudo-annotation, not clinical accuracy.

Outputs:
  metrics_per_image.csv
  metrics_per_annotation.csv      (with --annotations)
  metrics_per_class.csv
  morphology_severity_correlation.csv
  audit_report.md
  comparison_grid.png             (with --grids N)
  config.json, provenance.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from . import refine_masks_color as v1
    from . import refine_masks_color_v2 as v2mod
except ImportError:
    import refine_masks_color as v1
    import refine_masks_color_v2 as v2mod

VARIANT_COLUMNS = {"circle": "circle_mask", "v1": "v1_mask", "v2": "refined_mask"}
LEVELS = ("levle0", "levle1", "levle2", "levle3")
FEATURES = ("area_frac", "area_ratio_circle", "density_frame", "components",
            "largest_component_frac", "centroid_x", "centroid_y", "spread",
            "compactness")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def eta_squared(values: list[float], labels: list[int]) -> float | None:
    if len(values) < 4 or len(set(labels)) < 2:
        return None
    grand = statistics.fmean(values)
    total = sum((value - grand) ** 2 for value in values)
    if total == 0:
        return None
    within = 0.0
    for label in set(labels):
        group = [value for value, item in zip(values, labels) if item == label]
        mean = statistics.fmean(group)
        within += sum((value - mean) ** 2 for value in group)
    return 1.0 - within / total


def format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def as_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def mask_features(mask: np.ndarray, circle_area: int) -> dict:
    binary = (mask > 127).astype(np.uint8)
    height, width = binary.shape
    frame_area = height * width
    area = int(binary.sum())
    ys, xs = np.nonzero(binary)
    if area:
        centroid_x = float(xs.mean() / max(width - 1, 1))
        centroid_y = float(ys.mean() / max(height - 1, 1))
        spread = float(np.mean(np.abs(xs / max(width - 1, 1) - centroid_x)) +
                       np.mean(np.abs(ys / max(height - 1, 1) - centroid_y))) / 2.0
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
        largest = float(component_areas.max() / area) if component_areas.size else 0.0
        components = max(0, count - 1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        compactness = (4.0 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    else:
        centroid_x = centroid_y = spread = largest = compactness = 0.0
        components = 0
    return {
        "area_px": area,
        "area_frac": area / frame_area,
        "area_ratio_circle": area / max(circle_area, 1),
        "density_frame": area / frame_area,
        "components": components,
        "largest_component_frac": largest,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "spread": spread,
        "compactness": compactness,
    }


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    a = mask_a > 127
    b = mask_b > 127
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def resolve_mask(row: dict, variant: str) -> Path:
    column = VARIANT_COLUMNS[variant]
    value = row.get(column) or ""
    if not value:
        raise KeyError(f"Kolom {column!r} tidak ada/kosong untuk {row.get('filename')}")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Mask {variant} tidak ditemukan: {path}")
    return path


def select_rows(rows: list[dict], split: str, max_images: int) -> list[dict]:
    if split and split != "all":
        rows = [row for row in rows if (row.get("split") or "").strip() == split]
    if max_images > 0:
        rows = rows[:max_images]
    if not rows:
        raise ValueError(f"Tidak ada baris untuk split={split!r}.")
    return rows


def per_annotation_rows(
    rows: list[dict],
    annotations: Path,
    variants: tuple[str, ...],
    masks_by_file: dict[str, dict[str, np.ndarray]],
    circle_masks_by_file: dict[str, np.ndarray],
) -> list[dict]:
    circles_by_file, sizes_by_file = v1.load_json_circle_annotations(annotations)
    output = []
    for row in rows:
        filename = row["filename"]
        if filename not in circles_by_file:
            continue
        source_w, source_h = sizes_by_file[filename]
        crop = v1.center_crop_box(source_w, source_h, 1.0)
        for index, circle in enumerate(circles_by_file[filename]):
            circle_mask = v1.build_circle_mask([circle], source_w, source_h, crop, v2mod.TARGET_SIZE, "letterbox")
            circle_area = int(np.count_nonzero(circle_mask))
            record = {
                "filename": filename,
                "label_name": row.get("label_name", ""),
                "split": row.get("split", ""),
                "annotation_index": index,
                "circle_area_px": circle_area,
                "out_of_bounds": int(circle_area == 0),
            }
            for variant in variants:
                mask = circle_masks_by_file[filename] if variant == "circle" else masks_by_file[filename][variant]
                overlap = int(np.count_nonzero(mask[circle_mask > 0]))
                record[f"{variant}_inside_px"] = overlap
                record[f"{variant}_retention"] = format_number(overlap / circle_area) if circle_area else ""
            output.append(record)
    return output


def summarize_classes(rows: list[dict], variants: tuple[str, ...]) -> list[dict]:
    classes = sorted({row["label_name"] for row in rows},
                     key=lambda name: (name != "levle3", name))
    output = []
    for label in classes:
        group = [row for row in rows if row["label_name"] == label]
        record = {
            "label_name": label,
            "label_index": group[0].get("label_index", ""),
            "images": len(group),
            "fallback_annotations": sum(as_int(row.get("fallback_annotations")) for row in group),
            "lost_annotations": sum(as_int(row.get("lost_annotations")) for row in group),
            "out_of_bounds_annotations": sum(as_int(row.get("out_of_bounds_annotations")) for row in group),
        }
        for variant in variants:
            for feature in ("area_frac", "area_ratio_circle", "components", "spread", "compactness"):
                values = [float(row[f"{variant}_{feature}"]) for row in group]
                record[f"{variant}_{feature}_median"] = format_number(statistics.median(values))
                record[f"{variant}_{feature}_mean"] = format_number(statistics.fmean(values))
            if variant != "circle":
                ious = [float(row[f"iou_circle_{variant}"]) for row in group
                        if row.get(f"iou_circle_{variant}") not in ("", None)]
                record[f"iou_circle_{variant}_median"] = format_number(statistics.median(ious)) if ious else ""
        output.append(record)
    return output


def correlation_rows(rows: list[dict], variants: tuple[str, ...]) -> list[dict]:
    labels = [int(row["label_index"]) for row in rows]
    output = []
    for variant in variants:
        for feature in FEATURES:
            values = [float(row[f"{variant}_{feature}"]) for row in rows]
            output.append({
                "variant": variant,
                "feature": feature,
                "n": len(values),
                "pearson_with_label": format_number(pearson(values, labels)),
                "spearman_with_label": format_number(spearman(values, labels)),
                "eta_squared_by_class": format_number(eta_squared(values, labels)),
            })
    return output


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    variants: tuple[str, ...],
    row_count: int,
    class_rows: list[dict],
    correlations: list[dict],
    annotation_count: int | None,
) -> None:
    lines = [
        "# Audit Statistik Mask - Circle vs V1 vs V2",
        "",
        "Eksperimen deskriptif tanpa training. IoU dihitung terhadap circle anotasi",
        "sebagai ukuran kesesuaian dengan pseudo-annotation, bukan akurasi klinis.",
        "",
        "## Konfigurasi",
        "",
        f"- Manifest: {args.manifest}",
        f"- Split: {args.split}",
        f"- Citra dianalisis: {row_count}",
        f"- Anotasi dianalisis: {annotation_count if annotation_count is not None else 'tidak dihitung'}",
        f"- Varian: {', '.join(variants)}",
        "",
        "## Ringkasan per kelas",
        "",
        "| Kelas | Citra | IoU V1 median | IoU V2 median | Area V1 / circle | Area V2 / circle |",
        "|---|---|---|---|---|---|",
    ]
    for row in class_rows:
        lines.append(
            f"| {row['label_name']} | {row['images']} | {row.get('iou_circle_v1_median', '')} | "
            f"{row.get('iou_circle_v2_median', '')} | {row.get('v1_area_ratio_circle_median', '')} | "
            f"{row.get('v2_area_ratio_circle_median', '')} |"
        )
    lines += [
        "",
        "## Korelasi fitur morfologi dengan label severity",
        "",
        "| Varian | Fitur | Pearson | Spearman | Eta squared |",
        "|---|---|---|---|---|",
    ]
    for row in correlations:
        lines.append(
            f"| {row['variant']} | {row['feature']} | {row['pearson_with_label']} | "
            f"{row['spearman_with_label']} | {row['eta_squared_by_class']} |"
        )
    lines += [
        "",
        "## Batasan",
        "",
        "- Korelasi/eta squared hanya asosiasi pada sampel ini, bukan kausalitas.",
        "- Circle hanyalah proxy anotasi; semua varian inherit batas anotasi.",
        "- Jangan memilih varian memakai test set. Gunakan train/validation.",
        "- Audit ini tidak mengubah mask, dataset, atau model apa pun.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def select_grid_rows(rows: list[dict], count: int) -> list[dict]:
    def score(row):
        try:
            delta = abs(float(row["iou_circle_v2"]) - float(row["iou_circle_v1"]))
        except (KeyError, ValueError):
            delta = 0.0
        try:
            iou_v2 = float(row["iou_circle_v2"])
        except (KeyError, ValueError):
            iou_v2 = 1.0
        return (row.get("label_name") != "levle3", -delta, iou_v2)

    return sorted(rows, key=score)[:count]


def write_grids(rows: list[dict], output_dir: Path, count: int, variants: tuple[str, ...]) -> None:
    selected = select_grid_rows(rows, count)
    grids = []
    for row in selected:
        image = cv2.imread(row["preprocessed_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        circle = cv2.imread(row["circle_mask"], cv2.IMREAD_GRAYSCALE)
        masks = {variant: cv2.imread(resolve_mask(row, variant), cv2.IMREAD_GRAYSCALE)
                 for variant in variants}
        stats = (f"V1 IoU {row.get('iou_circle_v1', '')} · "
                 f"V2 IoU {row.get('iou_circle_v2', '')} · "
                 f"fallback V2 {row.get('fallback_annotations', '?')}")
        grid = v2mod.comparison_row(
            image, circle, masks.get("v1", circle), masks.get("v2", circle),
            title=f"{row['filename']} | {row['label_name']} | {row.get('split', '')}",
            stats=stats, panel_size=320,
        )
        cv2.imwrite(str(output_dir / "comparison_grids" / f"{Path(row['filename']).stem}.png"), grid)
        grids.append((row.get("label_name", ""), row["filename"], grid))
    if grids:
        montage = v2mod.build_montage([grid for _, _, grid in grids])
        cv2.imwrite(str(output_dir / "comparison_grid.png"), montage)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--variants", default="circle,v1,v2")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--grids", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    features = mask_features(mask, 100)
    assert features["area_px"] == 25 and features["components"] == 1
    assert abs(features["area_ratio_circle"] - 0.25) < 1e-9
    assert pearson([1, 2, 3], [2, 4, 6]) is not None
    assert spearman([1, 2, 3], [2, 4, 6]) is not None
    assert eta_squared([1, 1, 3, 3], [0, 0, 1, 1]) is not None
    print("PASS self-test: mask features, correlation helpers")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.manifest is None or args.output_dir is None:
        raise SystemExit("Provide --manifest and --output-dir (or use --self-test).")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    if not variants or set(variants) - set(VARIANT_COLUMNS):
        raise ValueError(f"Varian tidak dikenal: {variants}; pilihan {list(VARIANT_COLUMNS)}")
    if args.annotations is not None and not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    if args.grids < 0:
        raise ValueError("--grids harus >= 0")

    fields, all_rows = read_manifest(args.manifest)
    required = {"filename", "label_name", "label_index", "preprocessed_path", "circle_mask"}
    missing = required - set(fields)
    if missing:
        raise KeyError(f"Manifest kurang kolom: {sorted(missing)}")
    rows = select_rows(all_rows, args.split.strip(), args.max_images)

    output_rows: list[dict] = []
    masks_by_file: dict[str, dict[str, np.ndarray]] = {}
    circles_by_file: dict[str, np.ndarray] = {}
    for row in rows:
        image = cv2.imread(row["preprocessed_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Foto tidak terbaca: {row['preprocessed_path']}")
        circle = cv2.imread(row["circle_mask"], cv2.IMREAD_GRAYSCALE)
        if circle is None:
            raise ValueError(f"Circle mask tidak terbaca: {row['circle_mask']}")
        circle_area = int(np.count_nonzero(circle))
        masks = {"circle": circle}
        for variant in variants:
            if variant == "circle":
                continue
            mask = cv2.imread(str(resolve_mask(row, variant)), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Mask {variant} tidak terbaca untuk {row['filename']}")
            if mask.shape != circle.shape:
                raise ValueError(f"Ukuran mask {variant} berbeda untuk {row['filename']}")
            masks[variant] = mask
        circles_by_file[row["filename"]] = circle
        masks_by_file[row["filename"]] = masks
        record = {
            "filename": row["filename"],
            "label_name": row.get("label_name", ""),
            "label_index": int(row["label_index"]),
            "split": row.get("split", ""),
            "n_circles": row.get("n_circles", ""),
            "fallback_annotations": row.get("fallback_annotations", ""),
            "lost_annotations": row.get("lost_annotations", ""),
            "out_of_bounds_annotations": row.get("out_of_bounds_annotations", ""),
            "preprocessed_path": row["preprocessed_path"],
            "circle_mask": row["circle_mask"],
            "v1_mask": row.get("v1_mask", ""),
            "refined_mask": row.get("refined_mask", ""),
        }
        for variant in variants:
            features = mask_features(masks[variant], circle_area)
            for key, value in features.items():
                record[f"{variant}_{key}"] = format_number(value) if isinstance(value, float) else value
            if variant != "circle":
                record[f"iou_circle_{variant}"] = format_number(iou(masks[variant], circle))
        output_rows.append(record)

    annotation_rows = []
    if args.annotations is not None:
        annotation_rows = per_annotation_rows(rows, args.annotations, variants, masks_by_file, circles_by_file)

    class_rows = summarize_classes(output_rows, variants)
    correlations = correlation_rows(output_rows, variants)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True)
    if args.grids > 0:
        (output_dir / "comparison_grids").mkdir()

    write_csv(output_dir / "metrics_per_image.csv", output_rows)
    if args.annotations is not None:
        write_csv(output_dir / "metrics_per_annotation.csv", annotation_rows)
    write_csv(output_dir / "metrics_per_class.csv", class_rows)
    write_csv(output_dir / "morphology_severity_correlation.csv", correlations)
    write_report(output_dir / "audit_report.md", args=args, variants=variants,
                 row_count=len(output_rows), class_rows=class_rows,
                 correlations=correlations,
                 annotation_count=len(annotation_rows) if args.annotations else None)
    if args.grids > 0:
        write_grids(output_rows, output_dir, args.grids, variants)

    config = {
        "manifest": str(args.manifest.resolve()),
        "annotations": str(args.annotations.resolve()) if args.annotations else "",
        "split": args.split, "variants": list(variants),
        "max_images": args.max_images, "grids": args.grids,
        "note": "Descriptive pseudo-mask audit; do not select parameters with the test set.",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "annotations_sha256": sha256(args.annotations) if args.annotations else "",
        "script_sha256": sha256(Path(__file__).resolve()),
        "images_analyzed": len(output_rows),
        "annotations_analyzed": len(annotation_rows) if args.annotations else 0,
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(output_rows)} images, {len(annotation_rows)} annotations")
    print("Report:", (output_dir / "audit_report.md").resolve())
    if args.grids > 0:
        print("Grid  :", (output_dir / "comparison_grid.png").resolve())


if __name__ == "__main__":
    main()
