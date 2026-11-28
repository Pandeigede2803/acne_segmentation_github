#!/usr/bin/env python3
"""Step 45: controlled offline augmentation (target 300/class, cap 7x).

Rows per class: ``max(n, min(target, cap * n))``. Augmented rows are distributed
as evenly as possible over the original train images of that class, with at most
``cap - 1`` synthetic copies per source. Test/val rows are copied unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from recovery_common import (controlled_targets, even_allocation, max_synthetic_per_source,
                             read_rows, sha256_file, write_rows)

SYNTHETIC_MARKERS = ("__offaug_", "__syn_", "__aug_")


def image_path(row: dict) -> str:
    return row.get("preprocessed_path") or row.get("image_path") or ""


def mask_path(row: dict) -> str:
    return row.get("mask_path") or row.get("refined_mask") or row.get("circle_mask") or ""


def resolve(raw: str, manifest_dir: Path, fallback: Path | None) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_dir / path
    if path.is_file():
        return path
    if fallback is not None and (fallback / Path(raw).name).is_file():
        return fallback / Path(raw).name
    raise FileNotFoundError(f"File tidak ditemukan: {raw}")


def mask_area(path: Path) -> int:
    with Image.open(path) as mask:
        return int((np.asarray(mask.convert("L")) > 127).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Split manifest original dari Step 43B.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-per-class", type=int, default=300)
    parser.add_argument("--cap", type=int, default=7, help="Maksimum baris per original, termasuk original.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotation-deg", type=float, default=15.0)
    parser.add_argument("--brightness", type=float, default=0.15)
    parser.add_argument("--contrast", type=float, default=0.15)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} sudah berisi file. Pakai folder baru.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.manifest.parent
    image_fallback = Path(args.image_dir) if args.image_dir else None
    mask_fallback = Path(args.mask_dir) if args.mask_dir else None

    rows, fields = read_rows(args.manifest)
    splits = {row.get("split") for row in rows}
    if splits != {"train", "val", "test"}:
        raise ValueError(f"Manifest harus train/val/test; ditemukan {sorted(splits)}.")
    if any(marker in row["filename"] for row in rows for marker in SYNTHETIC_MARKERS):
        raise ValueError("Manifest sudah memuat baris sintetis; pakai manifest original Step 43B.")

    train_rows = [row for row in rows if row["split"] == "train"]
    counts = Counter(int(row["label_index"]) for row in train_rows)
    if set(counts) != {0, 1, 2, 3}:
        raise ValueError("Train harus memuat keempat kelas.")
    targets = controlled_targets(dict(counts), args.target_per_class, args.cap)
    cap_synthetic = max_synthetic_per_source(args.cap)

    rng = random.Random(args.seed)
    aug_dir = args.output_dir / "aug_pairs"
    aug_dir.mkdir(parents=True, exist_ok=True)
    additions: list[dict] = []
    lineage: list[dict] = []

    for cls in sorted(targets):
        needed = targets[cls] - counts[cls]
        if needed < 0:
            raise ValueError(f"Target kelas {cls} lebih kecil dari jumlah asli.")
        sources = sorted([row for row in train_rows if int(row["label_index"]) == cls],
                         key=lambda row: row["filename"])
        if needed and not sources:
            raise ValueError(f"Kelas {cls} tidak punya sumber train.")
        allocation = even_allocation(needed, len(sources)) if sources else []
        if allocation and max(allocation) > cap_synthetic:
            raise ValueError(f"Alokasi kelas {cls} melebihi cap {args.cap} baris/original.")
        index = 0
        for source, quota in zip(sources, allocation):
            source_image = resolve(image_path(source), manifest_dir, image_fallback)
            source_mask = resolve(mask_path(source), manifest_dir, mask_fallback)
            area_before = mask_area(source_mask)
            for _ in range(quota):
                angle = rng.uniform(-args.rotation_deg, args.rotation_deg)
                flip = rng.random() < 0.5
                brightness = rng.uniform(1 - args.brightness, 1 + args.brightness)
                contrast = rng.uniform(1 - args.contrast, 1 + args.contrast)
                stem = Path(source["filename"]).stem
                suffix = f"__offaug_c{cls}_{index:05d}"
                image_out = aug_dir / f"{stem}{suffix}.jpg"
                mask_out = aug_dir / f"{stem}{suffix}_mask.png"
                with Image.open(source_image) as image:
                    image = image.convert("RGB").rotate(
                        angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                    if flip:
                        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    image = ImageEnhance.Brightness(image).enhance(brightness)
                    image = ImageEnhance.Contrast(image).enhance(contrast)
                    image.save(image_out, quality=95)
                with Image.open(source_mask) as mask:
                    mask = mask.convert("L").rotate(
                        angle, resample=Image.Resampling.NEAREST, fillcolor=0)
                    if flip:
                        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    mask.save(mask_out)
                area_after = mask_area(mask_out)
                row = dict(source)
                row["filename"] = image_out.name
                row["split"] = "train"
                row["augmentation_source"] = source["filename"]
                row["augmentation_index"] = index
                if "image_path" in row:
                    row["image_path"] = str(image_out.resolve())
                if "preprocessed_path" in row:
                    row["preprocessed_path"] = str(image_out.resolve())
                mask_field = next((key for key in ("mask_path", "refined_mask", "circle_mask")
                                   if key in row), None)
                if mask_field:
                    row[mask_field] = str(mask_out.resolve())
                additions.append(row)
                lineage.append({
                    "synthetic": image_out.name, "source": source["filename"],
                    "group_id": source.get("group_id", ""),
                    "label_index": cls, "label_name": f"levle{cls}",
                    "angle_deg": round(angle, 6), "flip": int(flip),
                    "brightness_factor": round(brightness, 6),
                    "contrast_factor": round(contrast, 6),
                    "area_source": area_before, "area_synthetic": area_after,
                    "area_ratio": round(area_after / area_before, 6) if area_before else 0.0,
                    "empty_mask": int(area_after == 0),
                })
                index += 1
            if quota > cap_synthetic:
                raise AssertionError("Batas synthetic per source dilanggar.")

    output_rows = rows + additions
    write_rows(args.output_dir / "controlled_aug_manifest.csv", output_rows,
               fields + [key for key in ("augmentation_source", "augmentation_index")
                         if key not in fields])
    write_rows(args.output_dir / "augmentation_lineage.csv", lineage)

    after = Counter(int(row["label_index"]) for row in output_rows if row["split"] == "train")
    per_source = Counter(entry["source"] for entry in lineage)
    val_test_names = {row["filename"] for row in output_rows if row["split"] != "train"}
    synthetic_names = {entry["synthetic"] for entry in lineage}
    ratios = [entry["area_ratio"] for entry in lineage]

    config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "formula": "target_c = max(n_c, min(target_per_class, cap * n_c))",
        "target_per_class": args.target_per_class, "cap": args.cap,
        "max_synthetic_per_source": cap_synthetic, "seed": args.seed,
        "rotation_deg": args.rotation_deg, "brightness": args.brightness,
        "contrast": args.contrast,
        "counts_before": {str(key): value for key, value in sorted(counts.items())},
        "targets": {str(key): value for key, value in sorted(targets.items())},
        "counts_after": {str(key): value for key, value in sorted(after.items())},
        "synthetic_total": len(additions),
    }
    (args.output_dir / "augmentation_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")

    integrity = {
        "status": "PASS",
        "counts_match_targets": dict(sorted(after.items())) == dict(sorted(targets.items())),
        "max_rows_per_source": max(per_source.values()) if per_source else 0,
        "cap_ok": (max(per_source.values()) if per_source else 0) <= cap_synthetic,
        "synthetics_only_from_train": all(entry["source"] in {row["filename"] for row in train_rows}
                                          for entry in lineage),
        "no_synthetic_in_val_test": not (synthetic_names & val_test_names),
        "unique_filenames": len({row["filename"] for row in output_rows}) == len(output_rows),
        "empty_masks": sum(entry["empty_mask"] for entry in lineage),
        "area_ratio": {"min": min(ratios) if ratios else 0.0,
                       "median": float(np.median(ratios)) if ratios else 0.0,
                       "max": max(ratios) if ratios else 0.0},
        "sources_used": len(per_source),
        "limitations": [
            "Augmentasi tidak menambah pasien baru; hanya variasi transformasi.",
            "Cap 7x dan target 300 adalah hipotesis awal, bukan angka optimal.",
        ],
    }
    integrity["status"] = "PASS" if all([
        integrity["counts_match_targets"], integrity["cap_ok"],
        integrity["synthetics_only_from_train"], integrity["no_synthetic_in_val_test"],
        integrity["unique_filenames"],
    ]) else "FAIL"
    (args.output_dir / "controlled_aug_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Step 45 — Controlled offline augmentation", "",
        f"- Manifest: `{args.manifest}`",
        f"- Formula: `target_c = max(n_c, min({args.target_per_class}, {args.cap} × n_c))`; "
        f"seed {args.seed}.",
        "", "| Level | Original train | Target | Synthetic | Rows/original (max) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls in sorted(targets):
        report.append(f"| {cls} | {counts[cls]} | {targets[cls]} | "
                      f"{targets[cls] - counts[cls]} | {args.cap} |")
    report += [
        f"| total | {sum(counts.values())} | {sum(targets.values())} | {len(additions)} | |",
        "", f"- Baris train baru: **{sum(after.values())}** "
        f"(turun {100 * (1 - sum(after.values()) / 4000):.2f}% vs 4000/class-1000).",
        f"- Maksimum sintetis per sumber: **{integrity['max_rows_per_source']}** "
        f"(cap {cap_synthetic}).",
        f"- Mask kosong: **{integrity['empty_masks']}**; area ratio median "
        f"**{integrity['area_ratio']['median']:.4f}**.",
        f"- Integritas: **{integrity['status']}**.",
        "", "## Catatan", "",
        "- Transformasi sama dengan augmenter lama (rotasi ±15°, flip 0,5, brightness/contrast "
        "0,85–1,15, mask nearest, image bilinear).",
        "- Lineage per sampel tersimpan di `augmentation_lineage.csv`.",
        "- Jika transform lama gagal audit Step 44, buat eksperimen transform terpisah; jangan "
        "mengubah transform dan oversampling sekaligus.",
        "", "## Output", "",
        "- `controlled_aug_manifest.csv`, `augmentation_lineage.csv`, "
        "`augmentation_config.json`, `controlled_aug_integrity.json`",
    ]
    (args.output_dir / "controlled_aug_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({"counts_after": dict(sorted(after.items())),
                      "synthetic": len(additions), "status": integrity["status"]}, indent=2))
    if integrity["status"] != "PASS":
        sys.exit("Integritas controlled augmentation gagal; lihat controlled_aug_integrity.json")


if __name__ == "__main__":
    main()
