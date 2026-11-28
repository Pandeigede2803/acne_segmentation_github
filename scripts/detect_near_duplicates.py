#!/usr/bin/env python3
"""Step 43A: near-duplicate candidates with dHash/pHash (flip-aware).

Similarity is only a candidate signal. Every pair must be reviewed visually in
``near_duplicate_review.csv`` before it can influence the grouped split.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from recovery_common import read_rows, sha256_file, write_rows

HASH_BITS = 64


def load_gray(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.uint8)


def dhash_image(path: Path) -> int:
    pixels = load_gray(path, 9)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def phash_image(path: Path) -> int:
    pixels = load_gray(path, 32).astype(np.float32)
    coefficients = cv2.dct(pixels)[:8, :8].flatten()
    median = float(np.median(coefficients[1:]))
    value = 0
    for coefficient in coefficients:
        value = (value << 1) | int(coefficient > median)
    return value


def hamming(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def flip_hash(value: int) -> int:
    """Mirror a 8x8 bit hash horizontally."""
    mirrored = 0
    for row in range(8):
        byte = (value >> (8 * (7 - row))) & 0xFF
        reversed_byte = int(f"{byte:08b}"[::-1], 2)
        mirrored = (mirrored << 8) | reversed_byte
    return mirrored


def resolve_path(row: dict, manifest_dir: Path, image_dir: Path | None) -> Path:
    raw = row.get("preprocessed_path") or row.get("image_path") or ""
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_dir / path
    if path.is_file():
        return path
    if image_dir is not None and (image_dir / row.get("filename", "")).is_file():
        return image_dir / row["filename"]
    raise FileNotFoundError(f"Foto tidak ditemukan: {row.get('filename')} ({raw})")


def pair_grid(pairs: list[dict], paths: dict[str, Path], output: Path, thumb: int = 150) -> None:
    if not pairs:
        return
    rows = min(len(pairs), 12)
    canvas = Image.new("RGB", (thumb * 2 + 40, (thumb + 30) * rows + 30), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Near-duplicate candidates (left/right)", fill=(235, 235, 235))
    for index, pair in enumerate(pairs[:rows]):
        y = 30 + index * (thumb + 30)
        for column, name in enumerate((pair["filename_a"], pair["filename_b"])):
            try:
                with Image.open(paths[name]) as image:
                    tile = image.convert("RGB")
                    tile.thumbnail((thumb, thumb))
            except OSError:
                tile = Image.new("RGB", (thumb, thumb), (70, 20, 20))
            canvas.paste(tile, (10 + column * (thumb + 20), y))
            draw.text((10 + column * (thumb + 20), y + thumb + 4), name[:28], fill=(225, 225, 225))
        draw.text((10, y - 12),
                  f"d={pair['dhash_distance']} p={pair['phash_distance']} "
                  f"flip={pair['flip_match']} | {pair['label_a']} vs {pair['label_b']}",
                  fill=(255, 220, 120))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest bersih dari Step 42 (satu baris per citra).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--dhash-threshold", type=int, default=6)
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=50000)
    parser.add_argument("--review-top", type=int, default=24)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} sudah berisi file. Pakai folder baru.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = args.manifest.parent
    rows, _ = read_rows(args.manifest)
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("Manifest harus satu baris unik per citra.")
    image_dir = Path(args.image_dir) if args.image_dir else None

    paths = {row["filename"]: resolve_path(row, manifest_dir, image_dir) for row in rows}
    hashes = {}
    for name, path in paths.items():
        dhash, phash = dhash_image(path), phash_image(path)
        hashes[name] = {"dhash": dhash, "phash": phash,
                        "dhash_flip": flip_hash(dhash), "phash_flip": flip_hash(phash)}

    names = sorted(rows, key=lambda row: row["filename"])
    by_name = {row["filename"]: row for row in names}
    candidates = []
    for left, right in combinations([row["filename"] for row in names], 2):
        left_hash, right_hash = hashes[left], hashes[right]
        d_distance = min(
            hamming(left_hash["dhash"], right_hash["dhash"]),
            hamming(left_hash["dhash_flip"], right_hash["dhash"]),
        )
        p_distance = min(
            hamming(left_hash["phash"], right_hash["phash"]),
            hamming(left_hash["phash_flip"], right_hash["phash"]),
        )
        if d_distance <= args.dhash_threshold or p_distance <= args.phash_threshold:
            candidates.append({
                "filename_a": left, "filename_b": right,
                "label_a": by_name[left].get("label_name", ""),
                "label_b": by_name[right].get("label_name", ""),
                "split_a": by_name[left].get("split", ""),
                "split_b": by_name[right].get("split", ""),
                "dhash_distance": d_distance, "phash_distance": p_distance,
                "flip_match": hamming(left_hash["dhash_flip"],
                                      right_hash["dhash"]) <= args.dhash_threshold,
                "candidate_reason": ("dhash" if d_distance <= args.dhash_threshold else "")
                                    + ("+phash" if p_distance <= args.phash_threshold
                                       and d_distance <= args.dhash_threshold
                                       else "phash" if p_distance <= args.phash_threshold else ""),
            })
        if len(candidates) >= args.max_candidates:
            break

    candidates.sort(key=lambda pair: (min(pair["dhash_distance"], pair["phash_distance"]),
                                      pair["filename_a"], pair["filename_b"]))
    review = [{
        "pair_id": f"pair_{index:05d}",
        "filename_a": pair["filename_a"], "filename_b": pair["filename_b"],
        "label_a": pair["label_a"], "label_b": pair["label_b"],
        "split_a": pair["split_a"], "split_b": pair["split_b"],
        "dhash_distance": pair["dhash_distance"], "phash_distance": pair["phash_distance"],
        "flip_match": pair["flip_match"],
        "same_label": pair["label_a"] == pair["label_b"],
        "status": "pending", "decision": "", "reviewer": "", "note": "",
    } for index, pair in enumerate(candidates)]

    write_rows(args.output_dir / "near_duplicate_candidates.csv", candidates)
    write_rows(args.output_dir / "near_duplicate_review.csv", review)
    pair_grid(candidates[:args.review_top], paths, args.output_dir / "near_duplicate_review_grid.png")

    cross_split = sum(1 for pair in candidates if pair["split_a"] != pair["split_b"])
    label_conflicts = sum(1 for pair in candidates if pair["label_a"] != pair["label_b"])
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "images": len(rows),
        "dhash_threshold": args.dhash_threshold,
        "phash_threshold": args.phash_threshold,
        "candidates": len(candidates),
        "cross_split_candidates": cross_split,
        "label_conflict_candidates": label_conflicts,
        "review_status": "pending",
        "note": "Kandidat belum boleh memengaruhi split sebelum direview.",
    }
    (args.output_dir / "step43_near_duplicate_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Step 43A — Near-duplicate candidates", "",
        f"- Citra: **{len(rows)}**; kandidat: **{len(candidates)}** "
        f"(dHash ≤ {args.dhash_threshold} atau pHash ≤ {args.phash_threshold}, termasuk flip).",
        f"- Kandidat lintas split: **{cross_split}**; beda label: **{label_conflicts}**.",
        "- Semua kandidat berstatus `pending` di `near_duplicate_review.csv`.",
        "- Isi `status` = `confirmed`/`rejected` setelah review visual; hanya `confirmed` yang "
        "dipakai Step 43B untuk menggabungkan grup.",
        "", "## Catatan", "",
        "- Kemiripan visual bukan identitas pasien. Tanpa patient ID, independensi pasien tetap "
        "tidak terbukti.",
        "- Threshold dicatat di `step43_near_duplicate_metrics.json`; jangan diubah tanpa alasan.",
        "", "## Output", "",
        "- `near_duplicate_candidates.csv`, `near_duplicate_review.csv`, "
        "`near_duplicate_review_grid.png`, `step43_near_duplicate_metrics.json`",
    ]
    (args.output_dir / "step43_near_duplicate_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({"images": len(rows), "candidates": len(candidates),
                      "cross_split": cross_split, "label_conflicts": label_conflicts}, indent=2))
    print("Review:", args.output_dir / "near_duplicate_review.csv")


if __name__ == "__main__":
    main()
