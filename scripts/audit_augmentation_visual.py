#!/usr/bin/env python3
"""Step 44: visual + statistical audit of offline augmentation (Level 2/3).

Uses the augmented dataset that already exists; it does not generate new pairs.
Produces per-original grids (original RGB, original mask overlay, augmented RGB,
augmented mask overlay), a scores CSV, and a quality report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from recovery_common import read_rows, sha256_file, write_rows

SUFFIX = re.compile(r"__offaug_c(\d)_(\d{5})$")
THUMB = 200


def row_image(row: dict) -> str:
    return row.get("preprocessed_path") or row.get("image_path") or ""


def row_mask(row: dict) -> str:
    return row.get("mask_path") or row.get("refined_mask") or row.get("circle_mask") or ""


def resolve(raw: str, manifest_dir: Path, fallback: Path | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_dir / path
    if path.is_file():
        return path
    if fallback is not None and (fallback / Path(raw).name).is_file():
        return fallback / Path(raw).name
    return None


def overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    tile = image.convert("RGB").resize((THUMB, THUMB), Image.Resampling.BILINEAR)
    alpha = mask.convert("L").resize((THUMB, THUMB), Image.Resampling.NEAREST)
    red = Image.new("RGB", (THUMB, THUMB), (225, 45, 45))
    return Image.composite(red, tile, alpha.point(lambda value: 110 if value > 127 else 0))


def panel(image: Image.Image | None, label: str, subtitle: str = "") -> Image.Image:
    tile = Image.new("RGB", (THUMB, THUMB + 30), (20, 20, 20))
    if image is not None:
        tile.paste(image, (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((3, THUMB + 2), label[:32], fill=(230, 230, 230))
    if subtitle:
        draw.text((3, THUMB + 15), subtitle[:40], fill=(180, 200, 255))
    return tile


def mask_area(path: Path | None) -> int:
    if path is None:
        return 0
    with Image.open(path) as mask:
        return int((np.asarray(mask.convert("L")) > 127).sum())


def image_hash(path: Path) -> str:
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").resize((64, 64)).tobytes()).hexdigest()


def mean_intensity(path: Path | None, mask_path: Path | None) -> float:
    if path is None or mask_path is None:
        return 0.0
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((THUMB, THUMB)), dtype=np.float32)
    with Image.open(mask_path) as mask:
        region = np.asarray(mask.convert("L").resize((THUMB, THUMB)), dtype=np.float32) > 127
    return float(pixels[region].mean()) if region.any() else 0.0


def grade(area_ratio: float, empty: bool, intensity_ratio: float | None) -> str:
    if empty or area_ratio < 0.85 or area_ratio > 1.15:
        return "fail"
    if area_ratio < 0.95 or area_ratio > 1.05:
        return "suspect"
    if intensity_ratio is not None and not 0.70 <= intensity_ratio <= 1.40:
        return "suspect"
    return "pass"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--augmented-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-original", type=int, default=10)
    parser.add_argument("--level2-originals", type=int, default=30, help="0 = semua original.")
    parser.add_argument("--level3-originals", type=int, default=0, help="0 = semua original.")
    parser.add_argument("--controls-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} sudah berisi file. Pakai folder baru.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.augmented_manifest.parent
    image_fallback = Path(args.image_dir) if args.image_dir else None
    mask_fallback = Path(args.mask_dir) if args.mask_dir else None

    rows, _ = read_rows(args.augmented_manifest)
    originals: dict[str, dict] = {}
    synthetics_by_source: dict[str, list[dict]] = defaultdict(list)
    synthetics: list[dict] = []
    for row in rows:
        stem = Path(row["filename"]).stem
        match = SUFFIX.search(stem)
        if match:
            entry = dict(row, source_stem=stem[:match.start()],
                         synthetic_index=int(match.group(2)))
            synthetics_by_source[entry["source_stem"]].append(entry)
            synthetics.append(entry)
        else:
            originals[Path(row["filename"]).stem] = row
    if not synthetics:
        raise ValueError("Tidak ada baris sintetis __offaug_c... di manifest.")
    unmatched = sorted(source for source in synthetics_by_source if source not in originals)
    if unmatched:
        raise ValueError(f"{len(unmatched)} sumber sintetis tidak ditemukan di manifest: {unmatched[:5]}")

    rng = random.Random(args.seed)
    area_by_synthetic: dict[str, tuple[int, int]] = {}
    representatives: dict[int, list[tuple[str, list[dict]]]] = defaultdict(list)
    for source, entries in sorted(synthetics_by_source.items()):
        label = int(originals[source]["label_index"])
        source_area = mask_area(resolve(row_mask(originals[source]), manifest_dir, mask_fallback))
        for entry in entries:
            area_by_synthetic[entry["filename"]] = (
                source_area, mask_area(resolve(row_mask(entry), manifest_dir, mask_fallback)))
        ordered = sorted(entries, key=lambda item: item["filename"])
        rng.shuffle(ordered)
        representatives[label].append((source, ordered[:args.per_original]))

    limits = {
        3: args.level3_originals or len(representatives[3]),
        2: args.level2_originals or len(representatives[2]),
        0: args.controls_per_class,
        1: args.controls_per_class,
    }
    chosen: dict[int, list[tuple[str, list[dict]]]] = {
        label: representatives[label][:limits[label]] for label in (0, 1, 2, 3)}

    score_rows: list[dict] = []
    seen_hashes: set[str] = set()
    for label, folder_name in ((3, "visual_audit_level3"), (2, "visual_audit_level2"),
                               (0, "visual_audit_controls"), (1, "visual_audit_controls")):
        folder = args.output_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        for source, sampled in chosen[label]:
            original = originals[source]
            image_path = resolve(row_image(original), manifest_dir, image_fallback)
            mask_path = resolve(row_mask(original), manifest_dir, mask_fallback)
            if image_path is None:
                continue
            with Image.open(image_path) as image:
                original_image = image.convert("RGB").resize((THUMB, THUMB), Image.Resampling.BILINEAR)
            panels = [panel(original_image, f"{source} original", row_image(original)[-24:])]
            if mask_path:
                with Image.open(mask_path) as mask_image:
                    panels.append(panel(overlay(original_image, mask_image), "original overlay"))
            else:
                panels.append(panel(None, "original overlay"))
            source_area = mask_area(mask_path)
            for entry in sampled:
                aug_image_path = resolve(row_image(entry), manifest_dir, image_fallback)
                aug_mask_path = resolve(row_mask(entry), manifest_dir, mask_fallback)
                if aug_image_path is None:
                    continue
                with Image.open(aug_image_path) as image:
                    aug_image = image.convert("RGB").resize((THUMB, THUMB), Image.Resampling.BILINEAR)
                area = mask_area(aug_mask_path)
                ratio = area / source_area if source_area else 0.0
                baseline = mean_intensity(image_path, mask_path)
                current = mean_intensity(aug_image_path, aug_mask_path)
                intensity_ratio = current / baseline if baseline else None
                panels.append(panel(aug_image, Path(entry["filename"]).stem[-28:],
                                    f"area {ratio:.3f}"))
                if aug_mask_path:
                    with Image.open(aug_mask_path) as aug_mask_image:
                        panels.append(panel(overlay(aug_image, aug_mask_image),
                                            "augmented overlay"))
                else:
                    panels.append(panel(None, "augmented overlay"))
                output_hash = image_hash(aug_image_path)
                score_rows.append({
                    "source": source, "synthetic": entry["filename"],
                    "label_index": entry["label_index"],
                    "area_source": source_area, "area_synthetic": area,
                    "area_ratio": round(ratio, 6),
                    "empty_mask": int(area == 0),
                    "intensity_ratio": "" if intensity_ratio is None else round(intensity_ratio, 4),
                    "duplicate_output": int(output_hash in seen_hashes),
                    "grade": grade(ratio, area == 0, intensity_ratio),
                })
                seen_hashes.add(output_hash)
            if len(panels) > 2:
                columns = 4
                rows_count = (len(panels) + columns - 1) // columns
                grid = Image.new("RGB", (THUMB * columns, (THUMB + 30) * rows_count), (12, 12, 12))
                for index, tile in enumerate(panels):
                    grid.paste(tile, ((index % columns) * THUMB,
                                      (index // columns) * (THUMB + 30)))
                grid.save(folder / f"grid_{source}.png")

    # Statistical pass over every synthetic row (area only, no extra image decode).
    all_scores = []
    for entry in synthetics:
        source_area, area = area_by_synthetic[entry["filename"]]
        ratio = area / source_area if source_area else 0.0
        all_scores.append({"source": entry["source_stem"], "synthetic": entry["filename"],
                           "label_index": entry["label_index"], "area_source": source_area,
                           "area_synthetic": area, "area_ratio": round(ratio, 6),
                           "empty_mask": int(area == 0),
                           "grade": grade(ratio, area == 0, None)})
    write_rows(args.output_dir / "visual_audit_scores.csv", all_scores)

    grades = Counter(row["grade"] for row in all_scores)
    by_class: dict[int, Counter] = defaultdict(Counter)
    for row in all_scores:
        by_class[int(row["label_index"])][row["grade"]] += 1
    ratios = [row["area_ratio"] for row in all_scores]
    reuse = Counter(row["source"] for row in all_scores)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "augmented_manifest": str(args.augmented_manifest.resolve()),
        "augmented_manifest_sha256": sha256_file(args.augmented_manifest),
        "synthetics": len(synthetics),
        "sources": len(synthetics_by_source),
        "sampled_sources": {str(label): len(chosen[label]) for label in (0, 1, 2, 3)},
        "sampled_synthetics": len(score_rows),
        "grades": dict(grades),
        "grades_by_class": {str(key): dict(value) for key, value in sorted(by_class.items())},
        "area_ratio": {"min": float(np.min(ratios)), "p1": float(np.percentile(ratios, 1)),
                       "median": float(np.median(ratios)),
                       "p99": float(np.percentile(ratios, 99)), "max": float(np.max(ratios))},
        "reuse": {"max": max(reuse.values()),
                  "median": float(np.median(list(reuse.values())))},
        "reuse_above_10": sum(1 for value in reuse.values() if value > 10),
        "duplicate_sampled_outputs": sum(row["duplicate_output"] for row in score_rows),
        "unmatched_sources": unmatched,
    }
    (args.output_dir / "step44_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Step 44 — Visual audit augmentasi Level 2/3", "",
        f"- Manifest: `{args.augmented_manifest}`",
        f"- Sintetis: **{len(synthetics)}**, sumber: **{len(synthetics_by_source)}**.",
        f"- Grid sampling: Level 3 {len(chosen[3])} sumber, Level 2 {len(chosen[2])} sumber, "
        f"kontrol {len(chosen[0]) + len(chosen[1])} sumber; {args.per_original} sintetis/sumber.",
        "", "## Statistik seluruh sintetis", "",
        "| Level | pass | suspect | fail |", "|---|---:|---:|---:|",
    ]
    for label in range(4):
        counts = by_class[label]
        report.append(f"| {label} | {counts.get('pass', 0)} | {counts.get('suspect', 0)} | "
                      f"{counts.get('fail', 0)} |")
    report += [
        f"| total | {grades.get('pass', 0)} | {grades.get('suspect', 0)} | "
        f"{grades.get('fail', 0)} |",
        "", f"- Area ratio: median **{metrics['area_ratio']['median']:.4f}**, "
        f"P1 {metrics['area_ratio']['p1']:.4f}, P99 {metrics['area_ratio']['p99']:.4f}, "
        f"min {metrics['area_ratio']['min']:.4f}, max {metrics['area_ratio']['max']:.4f}.",
        f"- Mask kosong: **{sum(row['empty_mask'] for row in all_scores)}** sintetis.",
        f"- Sumber dipakai >10×: **{metrics['reuse_above_10']}**; maksimum "
        f"**{metrics['reuse']['max']}** sintetis per sumber.",
        f"- Output duplikat pada sampel grid: **{metrics['duplicate_sampled_outputs']}**.",
        "", "## Aturan grading", "",
        "- `fail`: mask kosong atau area ratio <0,85 / >1,15.",
        "- `suspect`: area ratio <0,95 / >1,05 atau intensity ratio di luar 0,70–1,40.",
        "- `pass`: selain itu. Grading ini bukan penilaian klinis.",
        "", "## Batasan", "",
        "- Area ratio bukan pengganti jumlah lesi; transformasi per sampel hanya tersedia bila "
        "lineage dicatat (Step 45).",
        "- Penilaian visual manusia tetap diperlukan untuk kasus `suspect`/`fail`.",
        "- Tidak ada inference model pada test di step ini.",
        "", "## Output", "",
        "- `visual_audit_level3/`, `visual_audit_level2/`, `visual_audit_controls/`",
        "- `visual_audit_scores.csv`, `step44_metrics.json`",
    ]
    (args.output_dir / "augmentation_quality_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({key: metrics[key] for key in
                      ("synthetics", "sources", "grades", "duplicate_sampled_outputs")}, indent=2))
    print("Report:", args.output_dir / "augmentation_quality_report.md")


if __name__ == "__main__":
    main()
