#!/usr/bin/env python3
"""Terapkan label ground truth (policy NNEW) ke manifest sumber Step 42.

Manifest asli (`refined_manifest_v2.csv`) memberi label dari prefix nama file.
Hasil audit menunjukkan 34 file v2 berbeda dari label NNEW, dan label NNEW
konsisten 100% dengan count-nya. Script ini menulis manifest **baru** dengan:

- ``label_name`` / ``label_index`` dari grade NNEW;
- kolom ``nnew_count`` (median count antar fold);
- kolom ``gt_source = nnew``.

Manifest asli tidak diubah. Semua kolom lain dipertahankan.

Contoh:
    python scripts/apply_ground_truth_labels.py \
        --manifest /path/refined_manifest_v2.csv \
        --output /path/refined_manifest_v2_nnew.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def load_ground_truth(path: Path) -> dict[str, dict[str, object]]:
    rows, fields = read_csv(path)
    required = {"file_name", "grade", "count_median"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Ground truth tidak punya kolom: {sorted(missing)}")
    truth: dict[str, dict[str, object]] = {}
    for row in rows:
        name = row["file_name"]
        grade = int(row["grade"])
        count = float(row["count_median"])
        if not 0 <= grade <= 3:
            raise ValueError(f"Grade tidak valid untuk {name}: {grade}")
        if name in truth and truth[name]["grade"] != grade:
            raise ValueError(f"Grade ground truth konflik untuk {name}")
        truth[name] = {"grade": grade, "count": count}
    return truth


def apply_labels(rows: list[dict[str, str]], fields: list[str],
                 truth: dict[str, dict[str, object]], name_column: str,
                 allow_missing: bool) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    if name_column not in fields:
        raise ValueError(f"Kolom nama '{name_column}' tidak ada di manifest.")

    output_fields = list(fields)
    for extra in ("nnew_count", "gt_source"):
        if extra not in output_fields:
            output_fields.append(extra)

    patched: list[dict[str, object]] = []
    report: list[dict[str, object]] = []
    missing: list[str] = []
    for row in rows:
        name = row.get(name_column, "")
        entry = truth.get(name)
        if entry is None:
            missing.append(name)
            if allow_missing:
                patched.append(dict(row))
                continue
            continue

        old_label = row.get("label_name", "")
        old_index = row.get("label_index", "")
        new_label = f"levle{entry['grade']}"
        new_index = str(entry["grade"])
        updated = dict(row)
        updated["label_name"] = new_label
        updated["label_index"] = new_index
        updated["nnew_count"] = f"{entry['count']:.1f}"
        updated["gt_source"] = "nnew"
        patched.append(updated)

        if old_label != new_label or str(old_index) != new_index:
            report.append({
                "filename": name,
                "old_label_name": old_label,
                "old_label_index": old_index,
                "new_label_name": new_label,
                "new_label_index": new_index,
                "old_n_circles": row.get("n_circles", ""),
                "nnew_count": f"{entry['count']:.1f}",
            })

    if missing and not allow_missing:
        example = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} file manifest tidak ada di ground truth "
                         f"(contoh: {example}). Pakai --allow-missing untuk melewati.")
    return patched, report, output_fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest sumber Step 42 (tidak diubah).")
    parser.add_argument("--ground-truth", type=Path,
                        default=ROOT / "output" / "ground_truth" / "ground_truth_manifest_nnew.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="Manifest baru dengan label NNEW.")
    parser.add_argument("--name-column", default="filename")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Lewati baris yang tidak ada di ground truth (mis. baris sintetis).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output sudah ada: {args.output}. Pakai --overwrite.")
    if args.output.resolve() == args.manifest.resolve():
        raise ValueError("Output tidak boleh sama dengan manifest sumber.")

    rows, fields = read_csv(args.manifest)
    truth = load_ground_truth(args.ground_truth)
    patched, report, output_fields = apply_labels(
        rows, fields, truth, args.name_column, args.allow_missing)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(patched)

    report_path = args.output.with_name(args.output.stem + "_patch_report.csv")
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "filename", "old_label_name", "old_label_index", "new_label_name",
            "new_label_index", "old_n_circles", "nnew_count"])
        writer.writeheader()
        writer.writerows(report)

    grades = Counter(int(row["label_index"]) for row in patched
                     if str(row.get("label_index", "")).strip() != "")
    print(f"Manifest       : {args.manifest} ({len(rows)} baris)")
    print(f"Output         : {args.output}")
    print(f"Label berubah  : {len(report)}")
    print(f"Distribusi     : {dict(sorted(grades.items()))}")
    print(f"Laporan patch  : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
