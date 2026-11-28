#!/usr/bin/env python3
"""Audit sumber label ACNE04: prefix nama file vs kolom label NNEW.

Tujuan: memisahkan fakta dari asumsi sebelum label dipakai untuk training.

Yang diperiksa:
1. Apakah label NNEW konsisten dengan jumlah lesi (skala Hayashi)?
2. File mana yang prefix nama filenya berbeda dari label NNEW?
3. Apakah jumlah lesi antar fold berubah untuk file yang sama?
4. Apakah `n_circles` Acne04-v2 bisa dipakai sebagai pengganti count NNEW?

Script ini read-only. Keluarannya berupa ringkasan teks dan (opsional) CSV
berisi file yang perlu ditinjau manual.

Contoh:
    python scripts/audit_label_sources.py \
        --annotations ../acne04v2/Acne04-v2_annotations.json \
        --out-csv docs/28-nov/label_mismatch_34_review.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lds_targets import count_to_severity_index  # noqa: E402

PREFIX_RE = re.compile(r"^levle(\d)_")
HAYASHI_TEXT = "1-5 -> 0, 6-20 -> 1, 21-50 -> 2, 51+ -> 3"


def read_nnew(path: Path) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Baris NNEW tidak valid di {path}: {line!r}")
        file_name, label, count = parts
        rows.append((file_name, int(label), int(count)))
    return rows


def prefix_label(file_name: str) -> int | None:
    match = PREFIX_RE.match(file_name)
    return int(match.group(1)) if match else None


def load_fold_rows(classification_dir: Path) -> dict[int, dict[str, tuple[int, int]]]:
    folds: dict[int, dict[str, tuple[int, int]]] = {}
    for fold in range(5):
        rows: dict[str, tuple[int, int]] = {}
        for split in ("trainval", "test"):
            path = classification_dir / f"NNEW_{split}_{fold}.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            for file_name, label, count in read_nnew(path):
                if file_name in rows:
                    raise ValueError(f"{file_name} muncul dua kali di fold {fold}")
                rows[file_name] = (label, count)
        folds[fold] = rows
    return folds


def load_circle_counts(annotations: Path) -> dict[str, int]:
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    counts: Counter[int] = Counter(annotation["image_id"] for annotation in payload["annotations"])
    return {image["file_name"]: counts.get(image["id"], 0) for image in payload["images"]}


def union_rows(folds: dict[int, dict[str, tuple[int, int]]]) -> dict[str, tuple[int, int]]:
    """Gabungkan fold; label wajib sama, count boleh berbeda (dicatat terpisah)."""
    merged: dict[str, tuple[int, int]] = {}
    for rows in folds.values():
        for file_name, (label, count) in rows.items():
            if file_name in merged and merged[file_name][0] != label:
                raise ValueError(f"{file_name} punya label berbeda antar fold")
            merged.setdefault(file_name, (label, count))
    return merged


def fmt_counter(counter: Counter) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items()))


def audit(annotations: Path, classification_dir: Path, out_csv: Path | None) -> int:
    folds = load_fold_rows(classification_dir)
    all_files = union_rows(folds)
    circles = load_circle_counts(annotations)
    v2_names = set(circles)

    inconsistent: list[tuple[str, dict[int, tuple[int, int]]]] = []
    for file_name in v2_names & set(all_files):
        values = {folds[fold][file_name] for fold in folds if file_name in folds[fold]}
        if len(values) > 1:
            inconsistent.append((file_name, {fold: folds[fold][file_name]
                                             for fold in folds if file_name in folds[fold]}))

    print("=== Cakupan ===")
    print(f"File di daftar NNEW (5 fold) : {len(all_files)}")
    print(f"File dengan anotasi v2       : {len(v2_names)}")
    print(f"File v2 tanpa baris NNEW     : {len(v2_names - set(all_files))}")
    print(f"File NNEW tanpa anotasi v2   : {len(set(all_files) - v2_names)}")

    consistent_v2 = {
        name: all_files[name] for name in v2_names
        if name in all_files
        and len({folds[fold][name][0] for fold in folds if name in folds[fold]}) == 1
    }
    print(f"File v2 dengan label stabil  : {len(consistent_v2)}")

    print("\n=== Label NNEW vs jumlah lesi (Hayashi) ===")
    print(f"Aturan: {HAYASHI_TEXT}")
    agree = sum(1 for label, count in consistent_v2.values()
                if count_to_severity_index(count) == label)
    print(f"Cocok: {agree}/{len(consistent_v2)}")
    if agree != len(consistent_v2):
        print("PERINGATAN: ada label NNEW yang tidak berasal dari count-nya.")

    print("\n=== Prefix nama file vs label NNEW (subset v2) ===")
    mismatches: list[dict[str, object]] = []
    for file_name, (label, count) in sorted(consistent_v2.items()):
        prefix = prefix_label(file_name)
        if prefix is None:
            print(f"PERINGATAN: prefix tidak dikenali: {file_name}")
            continue
        if prefix != label:
            mismatches.append({
                "file_name": file_name,
                "prefix_label": prefix,
                "nnew_label": label,
                "nnew_count": count,
                "hayashi_grade_from_count": count_to_severity_index(count),
                "v2_circles": circles[file_name],
            })
    pattern = Counter((row["prefix_label"], row["nnew_label"]) for row in mismatches)
    print(f"Jumlah file: {len(mismatches)}")
    print(f"Pola (prefix -> NNEW): {fmt_counter(pattern)}")
    if mismatches:
        print(f"Grade dari count == NNEW label : "
              f"{sum(1 for row in mismatches if row['hayashi_grade_from_count'] == row['nnew_label'])}"
              f"/{len(mismatches)}")
        print(f"Grade dari count == prefix     : "
              f"{sum(1 for row in mismatches if row['hayashi_grade_from_count'] == row['prefix_label'])}"
              f"/{len(mismatches)}")

    print("\n=== Prefix nama file vs label NNEW (seluruh ACNE04) ===")
    full_mismatch = [(name, values[0]) for name, values in all_files.items()
                     if prefix_label(name) is not None and prefix_label(name) != values[0]]
    print(f"Jumlah file: {len(full_mismatch)}")
    print(f"Pola: {fmt_counter(Counter((prefix_label(name), label) for name, label in full_mismatch))}")

    print("\n=== Count NNEW vs n_circles v2 ===")
    v2_grades = Counter(count_to_severity_index(circles[name]) for name in consistent_v2)
    agree_v2 = sum(1 for name, (label, _) in consistent_v2.items()
                   if count_to_severity_index(circles[name]) == label)
    print(f"Grade dari n_circles v2 == label NNEW: {agree_v2}/{len(consistent_v2)}")
    print(f"Distribusi grade dari n_circles v2   : {fmt_counter(v2_grades)}")
    nnew_counts = sorted(count for _, count in consistent_v2.values())
    v2_counts = sorted(circles[name] for name in consistent_v2)
    print(f"Median count NNEW: {nnew_counts[len(nnew_counts) // 2]} | "
          f"median n_circles v2: {v2_counts[len(v2_counts) // 2]}")

    print("\n=== Count berubah antar fold (subset v2) ===")
    print(f"File dengan (label, count) tidak stabil: {len(inconsistent)}")
    for file_name, values in sorted(inconsistent):
        labels = {label for label, _ in values.values()}
        counts = sorted(count for _, count in values.values())
        print(f"  {file_name}: label={sorted(labels)} count={counts}")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["file_name", "prefix_label", "nnew_label", "nnew_count",
                      "hayashi_grade_from_count", "v2_circles"]
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mismatches)
        print(f"\nCSV tinjauan manual ditulis ke: {out_csv}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path,
                        default=ROOT.parent / "acne04v2" / "Acne04-v2_annotations.json",
                        help="Path Acne04-v2_annotations.json")
    parser.add_argument("--classification-dir", type=Path,
                        default=ROOT / "datasetacne04" / "Classification",
                        help="Folder berisi NNEW_trainval_*.txt dan NNEW_test_*.txt")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Tulis CSV file yang perlu ditinjau manual.")
    args = parser.parse_args()
    return audit(args.annotations, args.classification_dir, args.out_csv)


if __name__ == "__main__":
    raise SystemExit(main())
