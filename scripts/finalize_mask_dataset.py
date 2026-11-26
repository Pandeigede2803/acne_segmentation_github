#!/usr/bin/env python3
"""Step 39: write a final, immutable mask manifest without training.

The selected masks already exist (circle, V1 or V2). This script does not copy
or modify image/mask pixels; it writes a new manifest whose ``refined_mask``
column points to the chosen variant, plus provenance and a decision report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2

VARIANTS = {"circle": "circle_mask", "v1": "v1_mask", "v2": "refined_mask"}
REQUIRED = {"filename", "label_name", "label_index", "split",
            "preprocessed_path", "circle_mask", "v1_mask", "refined_mask"}


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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        help="V2 manifest with circle_mask, v1_mask, refined_mask")
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--decision-note",
                        help="Why this variant was selected; recorded in provenance")
    parser.add_argument("--metrics", type=Path, default=None,
                        help="Optional metrics_per_image.csv from Step 35/36")
    parser.add_argument("--expected-images", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print("PASS self-test: finalizer has no computation to validate")
        return
    if args.manifest is None or args.output_dir is None or args.variant is None:
        raise SystemExit("Provide --manifest, --variant, and --output-dir (or use --self-test).")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    if not (args.decision_note or "").strip():
        raise ValueError("--decision-note wajib diisi; keputusan harus dapat diaudit.")
    if args.metrics is not None and not args.metrics.is_file():
        raise FileNotFoundError(args.metrics)

    fields, rows = read_manifest(args.manifest)
    missing = REQUIRED - set(fields)
    if missing:
        raise KeyError(f"Manifest kurang kolom: {sorted(missing)}")
    if not rows:
        raise ValueError("Manifest kosong.")
    if args.expected_images and len(rows) != args.expected_images:
        raise ValueError(f"Jumlah baris {len(rows)} != expected {args.expected_images}")

    column = VARIANTS[args.variant]
    output_rows = []
    split_counts: Counter = Counter()
    class_counts: Counter = Counter()
    selected_paths = set()
    for row in rows:
        selected = Path(row[column] or "")
        if not selected.is_file():
            raise FileNotFoundError(f"Mask {args.variant} tidak ditemukan: {selected}")
        image = cv2.imread(row["preprocessed_path"], cv2.IMREAD_COLOR)
        mask = cv2.imread(str(selected), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Foto/mask tidak terbaca untuk {row['filename']}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Ukuran foto/mask berbeda untuk {row['filename']}")
        updated = dict(row)
        updated["refined_mask"] = str(selected.resolve())
        updated["selected_mask"] = str(selected.resolve())
        updated["mask_variant"] = args.variant
        output_rows.append(updated)
        selected_paths.add(str(selected.resolve()))
        split_counts[(row.get("split") or "").strip()] += 1
        class_counts[(row.get("split") or "").strip(), row.get("label_name", "")] += 1

    if len(selected_paths) != len(output_rows):
        raise ValueError("Nama/path mask terpilih duplikat; cek manifest.")
    unknown_splits = set(split_counts) - {"train", "val", "test"}
    if unknown_splits:
        raise ValueError(f"Split tidak dikenal: {sorted(unknown_splits)}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True)
    final_manifest = output_dir / "manifest_final.csv"
    write_csv(final_manifest, output_rows, fields + ["selected_mask", "mask_variant"])

    count_rows = [
        {"split": split, "label_name": label, "images": count}
        for (split, label), count in sorted(class_counts.items())
    ]
    write_csv(output_dir / "final_counts.csv", count_rows,
              ["split", "label_name", "images"])

    if args.metrics is not None:
        (output_dir / "metrics_source.csv").write_bytes(args.metrics.read_bytes())

    provenance = {
        "variant": args.variant,
        "decision_note": args.decision_note.strip(),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "final_manifest": str(final_manifest.resolve()),
        "final_manifest_sha256": sha256(final_manifest),
        "images": len(output_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": {f"{split}/{label}": count
                         for (split, label), count in sorted(class_counts.items())},
        "metrics_source": str(args.metrics.resolve()) if args.metrics else "",
        "metrics_source_sha256": sha256(args.metrics) if args.metrics else "",
        "script_sha256": sha256(Path(__file__).resolve()),
        "note": "Masks are referenced in place; no pixel data was modified.",
    }
    (output_dir / "finalization.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Finalisasi Dataset Mask",
        "",
        f"- Varian terpilih: **{args.variant}**",
        f"- Manifest sumber: `{args.manifest}`",
        f"- Manifest final: `{final_manifest}`",
        f"- Jumlah citra: {len(output_rows)}",
        f"- Split: {dict(sorted(split_counts.items()))}",
        "",
        "## Alasan keputusan",
        "",
        args.decision_note.strip(),
        "",
        "## Catatan",
        "",
        "- Mask tidak disalin/diubah; manifest menunjuk path varian terpilih.",
        "- File lama tidak ditimpa.",
        "- Training tidak dijalankan oleh step ini.",
        "",
    ]
    (output_dir / "finalization_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"PASS: {len(output_rows)} images -> {final_manifest}")
    print("Variant:", args.variant, "| splits:", dict(split_counts))


if __name__ == "__main__":
    main()
