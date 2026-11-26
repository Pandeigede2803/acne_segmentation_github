#!/usr/bin/env python3
"""Generate and audit the Step 34 V2 candidate masks without training.

V2 keeps the V1 letterbox geometry and annotation circles, but replaces the
single global skin baseline with a local annulus around each annotation.
Every annotation is refined independently and falls back to its own circle
when the candidate is too small, so an annotation can never disappear
silently.

Example (audit subset from a split manifest):
  python scripts/refine_masks_color_v2.py \
    --image-dir /path/to/original_images \
    --annotations /path/to/Acne04-v2_annotations.json \
    --manifest /path/to/split_60_40/manifest_split.csv \
    --split train --per-level 3 --seed 42 \
    --output-dir /path/to/v2_audit_run --write-grids

Outputs (all inside --output-dir, never overwriting the source):
  preprocessed_images/      letterbox 512 JPEG (original colours)
  circle_masks/             annotation circle union
  v1_masks/                 V1 control mask (global baseline)
  v2_masks/                 V2 candidate mask
  overlays/                 V2 overlay on the input image
  comparison_grids/         per-image 4-panel grid (with --write-grids)
  comparison_grid.png       montage of the first --grid-samples images
  refined_manifest_v2.csv   V2 manifest with valid absolute paths
  audit_per_image.csv       per-image retention, fallback, IoU proxy
  audit_per_annotation.csv  per-annotation baseline and fallback reasons
  audit_per_class.csv       per-class summary (Level 3 first in the report)
  audit_report.md           human-readable summary and limitations
  refinement_config.json    exact parameters used
  provenance.json           input/config/script hashes and selection
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

import cv2
import numpy as np

try:  # Supports both `python scripts/...py` and package imports.
    from . import face_skin_utils, refine_masks_color as v1
except ImportError:
    import face_skin_utils
    import refine_masks_color as v1

IMAGE_SUFFIXES = {".jpg", ".jpeg"}
TARGET_SIZE = 512
MIN_VALID_PIXELS = 50
SKIN_CR, SKIN_CB, SKIN_V_MIN, SKIN_MORPH = (133, 180), (77, 133), 30, 5
V1_SIGMA, V1_MORPH_KERNEL, V1_MIN_RATIO = 1.2, 5, 0.02
LEVELS = ("levle0", "levle1", "levle2", "levle3")

AUDIT_IMAGE_FIELDS = [
    "filename", "label_name", "label_index", "split", "annotations",
    "fallback_annotations", "lost_annotations", "out_of_bounds_annotations", "circle_area_px",
    "v1_area_px", "v2_area_px", "v1_circle_proxy_iou", "v2_circle_proxy_iou",
    "v1_fallback", "v2_unchanged_vs_circle_union", "image_path",
    "preprocessed_path", "circle_mask", "v1_mask", "v2_mask", "overlay_path",
]
ANNOTATION_FIELDS = [
    "filename", "label_name", "annotation_index", "circle_area_px",
    "baseline_source", "annulus_pixels", "sample_pixels", "threshold_lab_a",
    "candidate_area_px", "candidate_ratio", "final_area_px",
    "retained_area_ratio", "fallback", "fallback_reason",
]


def configure_step31b_hsv_defaults() -> None:
    """Mirror the V1 CLI HSV defaults used by Step 31b (no HSV overrides)."""
    v1.RED_LOWER_1 = np.array([0, 40, 50])
    v1.RED_UPPER_1 = np.array([12, 255, 255])
    v1.RED_LOWER_2 = np.array([155, 40, 50])
    v1.RED_UPPER_2 = np.array([180, 255, 255])
    v1.PUS_LOWER_WHITE = np.array([0, 0, 180])
    v1.PUS_UPPER_WHITE = np.array([180, 40, 255])
    v1.PUS_LOWER_YELLOW = np.array([15, 10, 160])
    v1.PUS_UPPER_YELLOW = np.array([40, 90, 255])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def letterbox_content_mask(crop_box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = crop_box
    crop_w, crop_h = max(1, right - left), max(1, bottom - top)
    scale = min(TARGET_SIZE / crop_w, TARGET_SIZE / crop_h)
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    x0, y0 = (TARGET_SIZE - resized_w) // 2, (TARGET_SIZE - resized_h) // 2
    mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    mask[y0:y0 + resized_h, x0:x0 + resized_w] = 255
    return mask


def robust_threshold(samples: np.ndarray, sigma: float) -> float:
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples.astype(np.float32) - median)))
    return median + sigma * max(1.0, 1.4826 * mad)


def hsv_candidates(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, v1.RED_LOWER_1, v1.RED_UPPER_1),
        cv2.inRange(hsv, v1.RED_LOWER_2, v1.RED_UPPER_2),
    )
    white = cv2.inRange(hsv, v1.PUS_LOWER_WHITE, v1.PUS_UPPER_WHITE)
    yellow = cv2.inRange(hsv, v1.PUS_LOWER_YELLOW, v1.PUS_UPPER_YELLOW)
    return cv2.bitwise_or(red, cv2.bitwise_or(white, yellow))


def refine_annotation(
    lab_a: np.ndarray,
    hsv_mask: np.ndarray,
    circle: np.ndarray,
    outer: np.ndarray,
    circle_union: np.ndarray,
    content: np.ndarray,
    skin: np.ndarray,
    *,
    sigma: float,
    min_candidate_ratio: float,
    morph_kernel: int,
    min_valid_pixels: int = MIN_VALID_PIXELS,
) -> tuple[np.ndarray, dict]:
    circle_area = int(np.count_nonzero(circle))
    if circle_area == 0:
        return circle.copy(), {
            "baseline_source": "outside_image",
            "annulus_pixels": 0,
            "sample_pixels": 0,
            "threshold_lab_a": "",
            "candidate_area_px": 0,
            "candidate_ratio": 0.0,
            "final_area_px": 0,
            "retained_area_ratio": 0.0,
            "fallback": 1,
            "fallback_reason": "annotation_outside_image",
        }

    ring = (outer > 0) & (circle == 0) & (circle_union == 0) & (content > 0)
    skin_samples = lab_a[ring & (skin > 0)]
    samples = skin_samples if skin_samples.size >= min_valid_pixels else lab_a[ring]
    baseline = "annulus_skin" if skin_samples.size >= min_valid_pixels else "annulus_without_skin"

    if samples.size < min_valid_pixels:
        return circle.copy(), {
            "baseline_source": baseline,
            "annulus_pixels": int(ring.sum()),
            "sample_pixels": int(samples.size),
            "threshold_lab_a": "",
            "candidate_area_px": 0,
            "candidate_ratio": 0.0,
            "final_area_px": circle_area,
            "retained_area_ratio": 1.0,
            "fallback": 1,
            "fallback_reason": "annulus_has_too_few_valid_pixels",
        }

    threshold = robust_threshold(samples, sigma)
    candidate = ((lab_a > threshold) | (hsv_mask > 0)).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(candidate, circle)
    candidate = cv2.bitwise_and(candidate, content)
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        candidate = cv2.bitwise_and(candidate, circle)
        candidate = cv2.bitwise_and(candidate, content)

    candidate_area = int(np.count_nonzero(candidate))
    ratio = candidate_area / max(circle_area, 1)
    fallback = ratio < min_candidate_ratio
    final = circle.copy() if fallback else candidate
    final_area = int(np.count_nonzero(final))
    return final, {
        "baseline_source": baseline,
        "annulus_pixels": int(ring.sum()),
        "sample_pixels": int(samples.size),
        "threshold_lab_a": f"{threshold:.6f}",
        "candidate_area_px": candidate_area,
        "candidate_ratio": f"{ratio:.8f}",
        "final_area_px": final_area,
        "retained_area_ratio": f"{final_area / max(circle_area, 1):.8f}",
        "fallback": int(fallback),
        "fallback_reason": "candidate_below_min_ratio" if fallback else "",
    }


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_images(
    image_names: list[str],
    manifest_rows: list[dict] | None,
    levels: tuple[str, ...],
    per_level: int,
    split: str,
    seed: int,
    max_images: int,
) -> list[dict]:
    """Deterministically pick a stratified subset; never invents filenames."""
    available = set(image_names)
    manifest = {}
    if manifest_rows:
        for row in manifest_rows:
            name = (row.get("filename") or "").strip()
            if name in available and name not in manifest:
                manifest[name] = row
        candidates = sorted(manifest)
        if split:
            candidates = [name for name in candidates
                          if (manifest[name].get("split") or "").strip() == split]
            if not candidates:
                raise ValueError(f"Tidak ada citra split={split!r} di manifest.")
    else:
        candidates = sorted(available)

    by_level: dict[str, list[str]] = {}
    for name in candidates:
        label = (manifest.get(name, {}).get("label_name")
                 or v1.level_from_filename(name))
        if levels and label not in levels:
            continue
        by_level.setdefault(label, []).append(name)

    if not by_level:
        raise ValueError("Tidak ada citra yang cocok dengan filter level/split.")

    rng = random.Random(seed)
    selected: list[dict] = []
    for label in sorted(by_level):
        pool = sorted(by_level[label])
        rng.shuffle(pool)
        take = len(pool) if per_level <= 0 else min(per_level, len(pool))
        for name in pool[:take]:
            selected.append({
                "filename": name,
                "label_name": label,
                "split": (manifest.get(name, {}).get("split") or "").strip(),
            })
    selected.sort(key=lambda row: (row["label_name"], row["filename"]))
    if max_images > 0:
        selected = selected[:max_images]
    return selected


def mask_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    color_layer = np.zeros_like(image)
    color_layer[mask > 0] = color
    return cv2.addWeighted(image, 0.65, color_layer, 0.35, 0)


def _label_panel(panel: np.ndarray, lines: list[str]) -> np.ndarray:
    bar = 36
    canvas = np.zeros((panel.shape[0] + bar, panel.shape[1], 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)
    canvas[bar:] = panel
    for index, line in enumerate(lines[:2]):
        cv2.putText(canvas, line, (6, 15 + index * 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def comparison_row(
    image: np.ndarray,
    circle: np.ndarray,
    v1_mask: np.ndarray,
    v2_mask: np.ndarray,
    *,
    title: str,
    stats: str,
    panel_size: int,
) -> np.ndarray:
    panels = [
        _label_panel(cv2.resize(image, (panel_size, panel_size)), [title, "input 512"]),
        _label_panel(cv2.resize(mask_overlay(image, circle, (255, 0, 255)),
                                (panel_size, panel_size)), [title, "circle (anotasi)"]),
        _label_panel(cv2.resize(mask_overlay(image, v1_mask, (0, 255, 255)),
                                (panel_size, panel_size)), [title, "V1 global baseline"]),
        _label_panel(cv2.resize(mask_overlay(image, v2_mask, (0, 220, 0)),
                                (panel_size, panel_size)), [title, stats]),
    ]
    return np.hstack(panels)


def build_montage(rows: list[np.ndarray]) -> np.ndarray:
    divider = np.full((6, rows[0].shape[1], 3), 90, dtype=np.uint8)
    pieces = []
    for index, row in enumerate(rows):
        if index:
            pieces.append(divider)
        pieces.append(row)
    return np.vstack(pieces)


def summarize_classes(image_rows: list[dict], annotation_rows: list[dict]) -> list[dict]:
    annotations_by_file: dict[str, list[dict]] = {}
    for row in annotation_rows:
        annotations_by_file.setdefault(row["filename"], []).append(row)

    grouped: dict[str, list[dict]] = {}
    for row in image_rows:
        grouped.setdefault(row["label_name"], []).append(row)

    summary = []
    for label in sorted(grouped, key=lambda name: (name != "levle3", name)):
        rows = grouped[label]
        anns = [ann for row in rows for ann in annotations_by_file.get(row["filename"], [])]
        retention = [float(ann["retained_area_ratio"]) for ann in anns
                     if not int(ann["fallback"])]
        ious = [float(row["v2_circle_proxy_iou"]) for row in rows]
        v1_ious = [float(row["v1_circle_proxy_iou"]) for row in rows]
        summary.append({
            "label_name": label,
            "label_index": rows[0]["label_index"],
            "images": len(rows),
            "annotations": len(anns),
            "fallback_annotations": sum(int(ann["fallback"]) for ann in anns),
            "lost_annotations": sum(int(ann["final_area_px"]) == 0 for ann in anns),
            "out_of_bounds_annotations": sum(int(ann["circle_area_px"]) == 0 for ann in anns),
            "fallback_rate": f"{(sum(int(ann['fallback']) for ann in anns) / max(len(anns), 1)):.4f}",
            "median_retention_v2": f"{statistics.median(retention):.4f}" if retention else "",
            "iqr_low_retention_v2": f"{float(np.percentile(retention, 25)):.4f}" if retention else "",
            "iqr_high_retention_v2": f"{float(np.percentile(retention, 75)):.4f}" if retention else "",
            "min_retention_v2": f"{min(retention):.4f}" if retention else "",
            "median_v1_circle_iou": f"{statistics.median(v1_ious):.4f}",
            "median_v2_circle_iou": f"{statistics.median(ious):.4f}",
            "mean_v2_area_px": f"{statistics.mean(float(row['v2_area_px']) for row in rows):.2f}",
            "mean_circle_area_px": f"{statistics.mean(float(row['circle_area_px']) for row in rows):.2f}",
        })
    return summary


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    summary: list[dict],
    image_rows: list[dict],
    annotation_rows: list[dict],
    selection: dict,
) -> None:
    fallback_rows = [row for row in annotation_rows if int(row["fallback"])]
    lost_rows = [row for row in annotation_rows if int(row["final_area_px"]) == 0]
    lines = [
        "# Step 34 - Audit Refine Mask V2",
        "",
        "Eksperimen pseudo-mask tanpa training. Tidak ada perubahan arsitektur,",
        "hyperparameter, atau artefak baseline.",
        "",
        "## Konfigurasi",
        "",
        f"- Target size: {TARGET_SIZE}x{TARGET_SIZE} (letterbox, warna asli)",
        f"- V2 sigma: {args.sigma}",
        f"- V2 morphology: {args.morph_kernel}x{args.morph_kernel}",
        f"- V2 min candidate ratio: {args.min_candidate_ratio}",
        f"- V2 annulus radius factor: {args.annulus_radius_factor}",
        f"- V2 minimum valid annulus pixels: {args.min_valid_pixels}",
        f"- V1 control: sigma {V1_SIGMA}, kernel {V1_MORPH_KERNEL}x{V1_MORPH_KERNEL}, ratio {V1_MIN_RATIO}",
        f"- Seed: {args.seed}",
        f"- Seleksi: split={selection['split'] or 'semua'}, level={selection['levels'] or 'semua'}, "
        f"per level={selection['per_level'] if selection['per_level'] else 'semua'}",
        f"- Citra diaudit: {len(image_rows)}",
        "",
        "## Ringkasan per kelas",
        "",
        "| Kelas | Citra | Anotasi | Fallback | Hilang | Di luar gambar | Median retention V2* | Median IoU V1** | Median IoU V2** |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['label_name']} | {row['images']} | {row['annotations']} | "
            f"{row['fallback_annotations']} | {row['lost_annotations']} | "
            f"{row['out_of_bounds_annotations']} | "
            f"{row['median_retention_v2']} | {row['median_v1_circle_iou']} | "
            f"{row['median_v2_circle_iou']} |"
        )
    lines += [
        "",
        "*Retention hanya dari anotasi non-fallback; fallback circle tidak dihitung sebagai refinement.",
        "**IoU dihitung terhadap circle union anotasi sebagai proxy, bukan akurasi klinis.",
        "",
        "## Temuan",
        "",
        f"- Anotasi fallback: {len(fallback_rows)} dari {len(annotation_rows)}.",
        f"- Anotasi hilang: {len(lost_rows)} (termasuk anotasi di luar batas gambar).",
        f"- Anotasi di luar batas gambar: {sum(int(row['circle_area_px']) == 0 for row in annotation_rows)}.",
        "- Fallback berarti V2 memakai circle anotasi; bukan bukti batas lesi yang benar.",
        "",
        "## Batasan",
        "",
        "- Refinement dibatasi pada circle anotasi, sehingga lesi di luar anotasi tidak",
        "  dipulihkan dan tidak diukur.",
        "- Kemerahan belum tentu lesi aktif; anotasi awal juga bisa tidak lengkap.",
        "- Parameter ini kandidat eksploratif, bukan nilai optimal. Jangan memilih",
        "  parameter memakai test set.",
        "- Tidak ada training atau evaluasi test pada step ini.",
        "",
        "## Berkas",
        "",
        "- `audit_per_image.csv`, `audit_per_annotation.csv`, `audit_per_class.csv`",
        "- `comparison_grid.png`, `comparison_grids/`",
        "- `refined_manifest_v2.csv`, `refinement_config.json`, `provenance.json`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_images(image_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name in found:
            raise ValueError(f"Nama file duplikat di folder foto: {path.name} ({found[path.name]} vs {path})")
        found[path.name] = path
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=Path, help="Folder of original annotated photos")
    parser.add_argument("--annotations", type=Path, help="ACNE04-v2_annotations.json")
    parser.add_argument("--output-dir", type=Path, help="New output folder; existing folders are refused")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional CSV used only to choose an audit subset")
    parser.add_argument("--split", default="",
                        help="Keep only rows with this split (e.g. train)")
    parser.add_argument("--levels", default="",
                        help="Comma-separated levels, e.g. levle3,levle2")
    parser.add_argument("--per-level", type=int, default=0,
                        help="Max images per level (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--annulus-radius-factor", type=float, default=2.0)
    parser.add_argument("--min-candidate-ratio", type=float, default=0.02)
    parser.add_argument("--min-valid-pixels", type=int, default=MIN_VALID_PIXELS,
                        help="Minimum annulus pixels for a local baseline")
    parser.add_argument("--morph-kernel", type=int, default=1)
    parser.add_argument("--grid-samples", type=int, default=9,
                        help="Rows in comparison_grid.png")
    parser.add_argument("--grid-focus", default="levle3",
                        help="Level shown first in comparison_grid.png")
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--write-grids", action="store_true",
                        help="Also write one 4-panel grid per image")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Skip per-image V2 overlay PNGs (useful for full datasets)")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.image_dir or not args.annotations or not args.output_dir:
        raise SystemExit("Provide --image-dir, --annotations, and --output-dir (or use --self-test).")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {args.image_dir}")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")
    if args.manifest is not None and not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.sigma <= 0 or args.annulus_radius_factor <= 1 or not 0 < args.min_candidate_ratio <= 1:
        raise ValueError("Require sigma > 0, annulus-radius-factor > 1, and min-candidate-ratio in (0, 1].")
    if args.morph_kernel < 1 or (args.morph_kernel > 1 and args.morph_kernel % 2 == 0):
        raise ValueError("Morphology kernel must be 1 or a positive odd integer.")
    if args.min_valid_pixels < 1:
        raise ValueError("min-valid-pixels minimal 1.")
    if args.per_level < 0 or args.max_images < 0 or args.grid_samples < 0:
        raise ValueError("Require per-level >= 0, max-images >= 0, grid-samples >= 0.")
    if args.panel_size < 128:
        raise ValueError("panel-size minimal 128.")
    levels = tuple(level.strip() for level in args.levels.split(",") if level.strip())
    unknown = set(levels) - set(LEVELS)
    if unknown:
        raise ValueError(f"Level tidak dikenal: {sorted(unknown)}. Pilihan: {list(LEVELS)}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output folder: {args.output_dir}")
    return levels


def self_test() -> None:
    a = np.full((64, 64), 140, dtype=np.uint8)
    cv2.circle(a, (32, 32), 3, 200, -1)
    hsv = np.zeros_like(a)
    circle = np.zeros_like(a)
    outer = np.zeros_like(a)
    cv2.circle(circle, (32, 32), 7, 255, -1)
    cv2.circle(outer, (32, 32), 15, 255, -1)
    content = np.full_like(a, 255)
    skin = content.copy()
    result, info = refine_annotation(
        a, hsv, circle, outer, circle, content, skin,
        sigma=1.2, min_candidate_ratio=0.02, morph_kernel=1,
    )
    again, _ = refine_annotation(
        a, hsv, circle, outer, circle, content, skin,
        sigma=1.2, min_candidate_ratio=0.02, morph_kernel=1,
    )
    assert info["fallback"] == 0 and np.array_equal(result, again)
    assert np.all(result[circle == 0] == 0) and not np.array_equal(result, circle)
    no_ring, fallback = refine_annotation(
        a, hsv, circle, outer, content, content, skin,
        sigma=1.2, min_candidate_ratio=0.02, morph_kernel=1,
    )
    assert fallback["fallback"] == 1 and np.array_equal(no_ring, circle)
    picked = select_images(
        ["levle3_a.jpg", "levle3_b.jpg", "levle2_c.jpg"],
        [{"filename": "levle3_a.jpg", "split": "train", "label_name": "levle3"},
         {"filename": "levle3_b.jpg", "split": "test", "label_name": "levle3"},
         {"filename": "levle2_c.jpg", "split": "train", "label_name": "levle2"}],
        ("levle3",), 1, "train", 42, 0,
    )
    assert [row["filename"] for row in picked] == ["levle3_a.jpg"]
    print("PASS self-test: deterministic, circle-confined, fallback, and selection behavior")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    levels = validate_args(args)

    configure_step31b_hsv_defaults()
    circles_by_file, sizes = v1.load_json_circle_annotations(args.annotations)
    images_by_name = collect_images(args.image_dir)
    annotated_names = sorted(name for name in images_by_name if name in circles_by_file)
    skipped = [name for name in images_by_name if name not in circles_by_file]
    if not annotated_names:
        raise ValueError("No annotated images found in --image-dir.")

    manifest_rows = read_manifest(args.manifest) if args.manifest else None
    selected = select_images(annotated_names, manifest_rows, levels, args.per_level,
                             args.split.strip(), args.seed, args.max_images)
    selected_names = [row["filename"] for row in selected]
    print(f"Selected {len(selected_names)} image(s) for audit.")

    args.output_dir.mkdir(parents=True)
    image_out = args.output_dir / "preprocessed_images"
    circle_out = args.output_dir / "circle_masks"
    v1_out = args.output_dir / "v1_masks"
    v2_out = args.output_dir / "v2_masks"
    overlay_out = args.output_dir / "overlays"
    grids_out = args.output_dir / "comparison_grids"
    for folder in (image_out, circle_out, v1_out, v2_out, overlay_out):
        folder.mkdir()
    if args.write_grids:
        grids_out.mkdir()

    image_rows: list[dict] = []
    annotation_rows: list[dict] = []
    manifest_out_rows: list[dict] = []
    grid_rows: list[tuple[str, str, np.ndarray]] = []

    for index, entry in enumerate(selected, 1):
        fname = entry["filename"]
        image_path = images_by_name[fname]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode image: {image_path}")
        width, height = sizes[fname]
        if (bgr.shape[1], bgr.shape[0]) != (width, height):
            raise ValueError(
                f"Dimensions disagree with annotations for {fname}: "
                f"{(bgr.shape[1], bgr.shape[0])} != {(width, height)}")

        crop = v1.center_crop_box(width, height, 1.0)
        left, top, right, bottom = crop
        resized = v1.resize_to_square(bgr[top:bottom, left:right], TARGET_SIZE, "letterbox")
        input_path = image_out / fname
        if not cv2.imwrite(str(input_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write preprocessed image: {input_path}")
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
            raise ValueError(f"Invalid preprocessed image: {input_path}")

        annotations = circles_by_file[fname]
        circle_union = v1.build_circle_mask(annotations, width, height, crop, TARGET_SIZE, "letterbox")
        content = letterbox_content_mask(crop)
        lab_a = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 1]
        hsv = hsv_candidates(image)
        skin = face_skin_utils.skin_map(image, None, SKIN_CR, SKIN_CB, SKIN_V_MIN, SKIN_MORPH)
        stem = image_path.stem
        label_name = entry["label_name"]

        # V1 control: identical geometry and V1 CLI parameters, global baseline.
        v1_mask, v1_fallback = v1.refine_mask_with_red(
            image, circle_union, V1_MORPH_KERNEL, V1_MIN_RATIO, V1_SIGMA)
        circle_path = circle_out / f"{stem}.png"
        v1_path = v1_out / f"{stem}.png"
        for path, mask in ((circle_path, circle_union), (v1_path, v1_mask)):
            if not cv2.imwrite(str(path), mask):
                raise OSError(f"Could not write mask: {path}")

        final_union = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        this_annotations = []
        for ann_index, source_circle in enumerate(annotations):
            circle = v1.build_circle_mask([source_circle], width, height, crop, TARGET_SIZE, "letterbox")
            outer = v1.build_circle_mask(
                [(source_circle[0], source_circle[1], source_circle[2] * args.annulus_radius_factor)],
                width, height, crop, TARGET_SIZE, "letterbox",
            )
            final, info = refine_annotation(
                lab_a, hsv, circle, outer, circle_union, content, skin,
                sigma=args.sigma, min_candidate_ratio=args.min_candidate_ratio,
                morph_kernel=args.morph_kernel, min_valid_pixels=args.min_valid_pixels,
            )
            final_union = cv2.bitwise_or(final_union, final)
            record = {"filename": fname, "label_name": label_name,
                      "annotation_index": ann_index, "circle_area_px": int(np.count_nonzero(circle)),
                      **info}
            annotation_rows.append(record)
            this_annotations.append(record)

        if np.any((final_union > 0) & (circle_union == 0)):
            raise AssertionError(f"V2 mask escaped circle union: {fname}")
        v2_path = v2_out / f"{stem}.png"
        overlay_path = overlay_out / f"{stem}.png"
        if not cv2.imwrite(str(v2_path), final_union):
            raise OSError(f"Could not write mask: {v2_path}")
        if args.no_overlays:
            overlay_path = ""
        elif not cv2.imwrite(str(overlay_path), mask_overlay(image, final_union, (0, 220, 0))):
            raise OSError(f"Could not write overlay: {overlay_path}")

        circle_area = int(np.count_nonzero(circle_union))
        v1_area = int(np.count_nonzero(v1_mask))
        v2_area = int(np.count_nonzero(final_union))
        fallback_count = sum(int(row["fallback"]) for row in this_annotations)
        lost_count = sum(int(row["final_area_px"]) == 0 for row in this_annotations)
        out_of_bounds = sum(int(row["circle_area_px"]) == 0 for row in this_annotations)
        stats = f"V2 \u00b7 fallback {fallback_count}/{len(this_annotations)} \u00b7 area {v2_area}/{circle_area}"
        title = f"{fname} | {label_name} | {entry['split'] or 'split?'}"

        image_rows.append({
            "filename": fname, "label_name": label_name,
            "label_index": v1.LABEL_TO_INDEX.get(label_name, -1),
            "split": entry["split"], "annotations": len(this_annotations),
            "fallback_annotations": fallback_count, "lost_annotations": lost_count,
            "out_of_bounds_annotations": out_of_bounds,
            "circle_area_px": circle_area, "v1_area_px": v1_area, "v2_area_px": v2_area,
            "v1_circle_proxy_iou": f"{v1_area / max(circle_area, 1):.8f}",
            "v2_circle_proxy_iou": f"{v2_area / max(circle_area, 1):.8f}",
            "v1_fallback": int(v1_fallback),
            "v2_unchanged_vs_circle_union": int(np.array_equal(final_union, circle_union)),
            "image_path": str(image_path.resolve()),
            "preprocessed_path": str(input_path.resolve()),
            "circle_mask": str(circle_path.resolve()),
            "v1_mask": str(v1_path.resolve()),
            "v2_mask": str(v2_path.resolve()),
            "overlay_path": str(overlay_path.resolve()) if overlay_path else "",
        })
        manifest_out_rows.append({
            "filename": fname, "label_name": label_name,
            "label_index": v1.LABEL_TO_INDEX.get(label_name, -1),
            "split": entry["split"], "image_path": str(image_path.resolve()),
            "preprocessed_path": str(input_path.resolve()),
            "refined_mask": str(v2_path.resolve()), "circle_mask": str(circle_path.resolve()),
            "v1_mask": str(v1_path.resolve()), "image_size": TARGET_SIZE,
            "n_circles": len(annotations), "circle_area": circle_area,
            "refined_area": v2_area, "v1_area": v1_area,
            "fallback_annotations": fallback_count, "lost_annotations": lost_count,
            "out_of_bounds_annotations": out_of_bounds,
            "resize_mode": "letterbox", "preprocessing_mode": "none",
        })

        grid = None
        if args.write_grids:
            grid = comparison_row(image, circle_union, v1_mask, final_union,
                                  title=title, stats=stats, panel_size=args.panel_size)
            if not cv2.imwrite(str(grids_out / f"{stem}.png"), grid):
                raise OSError(f"Could not write comparison grid for {fname}")
        elif len(grid_rows) < args.grid_samples:
            grid = comparison_row(image, circle_union, v1_mask, final_union,
                                  title=title, stats=stats, panel_size=args.panel_size)
        if grid is not None and args.grid_samples > 0:
            grid_rows.append((entry["label_name"], fname, grid))
        if index % 25 == 0:
            print(f"processed {index}/{len(selected)}", flush=True)

    if grid_rows and args.grid_samples > 0:
        ordered_grids = sorted(grid_rows, key=lambda item: (item[0] != args.grid_focus, item[0], item[1]))
        montage = build_montage([item[2] for item in ordered_grids[:args.grid_samples]])
        if not cv2.imwrite(str(args.output_dir / "comparison_grid.png"), montage):
            raise OSError("Could not write comparison_grid.png")

    summary = summarize_classes(image_rows, annotation_rows)
    write_csv(args.output_dir / "audit_per_image.csv", AUDIT_IMAGE_FIELDS, image_rows)
    write_csv(args.output_dir / "audit_per_annotation.csv", ANNOTATION_FIELDS, annotation_rows)
    write_csv(args.output_dir / "audit_per_class.csv", list(summary[0]) if summary else [], summary)
    manifest_fields = list(manifest_out_rows[0]) if manifest_out_rows else []
    write_csv(args.output_dir / "refined_manifest_v2.csv", manifest_fields, manifest_out_rows)

    selection = {
        "manifest": str(args.manifest.resolve()) if args.manifest else "",
        "manifest_sha256": sha256(args.manifest) if args.manifest else "",
        "split": args.split.strip(), "levels": list(levels),
        "per_level": args.per_level, "seed": args.seed, "max_images": args.max_images,
        "images": selected_names,
    }
    config = {
        "variant": "v2_local_annulus",
        "objective": "audit pseudo-mask candidate without training",
        "candidate": "local Lab median/MAD plus V1 CLI-default HSV candidates",
        "target_size": TARGET_SIZE, "resize_mode": "letterbox", "crop_ratio": 1.0,
        "jpeg_quality": 95, "sigma": args.sigma,
        "annulus_radius_factor": args.annulus_radius_factor,
        "min_valid_pixels": args.min_valid_pixels,
        "min_candidate_ratio": args.min_candidate_ratio,
        "morph_kernel": args.morph_kernel,
        "no_overlays": bool(args.no_overlays),
        "v1_control": {"sigma": V1_SIGMA, "morph_kernel": V1_MORPH_KERNEL,
                       "min_candidate_ratio": V1_MIN_RATIO},
        "skin_map": {"cr": list(SKIN_CR), "cb": list(SKIN_CB), "v_min": SKIN_V_MIN,
                     "morph": SKIN_MORPH, "use": "annulus sampling only"},
        "selection": selection,
        "fallback_note": "Annotations below the candidate-area threshold retain the original circle; "
                         "inspect fallback_annotations and unchanged_vs_circle_union.",
        "warning": "Exploratory pseudo-mask candidate, not clinical ground truth. Do not tune with the test set.",
    }
    config_text = json.dumps(config, indent=2) + "\n"
    (args.output_dir / "refinement_config.json").write_text(config_text, encoding="utf-8")
    (args.output_dir / "config.json").write_text(config_text, encoding="utf-8")
    write_report(args.output_dir / "audit_report.md", args=args, summary=summary,
                 image_rows=image_rows, annotation_rows=annotation_rows, selection=selection)

    provenance = {
        "variant": "v2_local_annulus",
        "input_dir": str(args.image_dir.resolve()),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations),
        "v1_script_sha256": sha256(Path(v1.__file__).resolve()),
        "v2_script_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": config_sha256(config),
        "selection": selection,
        "images_processed": len(image_rows),
        "annotations_processed": len(annotation_rows),
        "unannotated_images_skipped": len(skipped),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    unchanged = [row["filename"] for row in image_rows if row["v2_unchanged_vs_circle_union"]]
    print(f"PASS: {len(image_rows)} images processed; "
          f"{sum(row['fallback_annotations'] for row in image_rows)} annotation fallbacks.")
    print("Masks exactly unchanged from circle union:",
          ", ".join(unchanged) if unchanged else "none")
    print("Audit:", (args.output_dir / "audit_report.md").resolve())
    if (args.output_dir / "comparison_grid.png").is_file():
        print("Grid :", (args.output_dir / "comparison_grid.png").resolve())


if __name__ == "__main__":
    main()
