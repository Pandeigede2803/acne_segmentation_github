#!/usr/bin/env python3
"""Bangun manifest ground truth yang deterministik untuk Acne04-v2.

Dua kebijakan (policy):

- ``nnew`` (default): severity = label NNEW, count = count NNEW per fold.
  Ini yang kompatibel dengan Prokhorov & Kalinin dan dengan target LDS.
- ``v2``: severity dan count diturunkan dari anotasi circle v2.
  Konsisten secara internal, tetapi distribusinya berbeda dan tidak
  sebanding dengan Prokhorov.

Keduanya tidak boleh dicampur dalam satu eksperimen.

Selain manifest, script ini menandai file yang perlu ditinjau manual:

- ``flag_prefix_mismatch``: prefix nama file != grade policy.
- ``flag_count_unstable``: count NNEW berbeda antar fold.
- ``flag_v2_grade_disagrees``: grade dari circle v2 != grade policy.
- ``review_priority``: prefix mismatch ATAU count menyimpang dari hubungan
  normal count-NNEW vs circle-v2 (robust z > 3 di dalam grade yang sama).

Contoh:
    python scripts/build_ground_truth_manifest.py \
        --out-dir output/ground_truth \
        --audit-per-grade 15
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_label_sources import (load_circle_counts, load_fold_rows,
                                         prefix_label)
from scripts.lds_targets import count_to_severity_index  # noqa: E402

MANIFEST_FIELDS = [
    "file_name", "fold", "split", "policy", "grade", "count",
    "count_median", "count_min", "count_max", "count_n_distinct",
    "prefix_label", "v2_circles", "v2_grade", "count_v2_log_ratio_z",
    "flag_prefix_mismatch", "flag_count_unstable", "flag_v2_grade_disagrees",
    "review_priority", "image_path",
]
REVIEW_FIELDS = [
    "file_name", "grade", "count_median", "count_min", "count_max",
    "prefix_label", "v2_circles", "v2_grade", "count_v2_log_ratio_z", "reason",
]
AUDIT_FIELDS = [
    "file_name", "grade", "count_median", "nnew_count_fold0", "prefix_label",
    "v2_circles", "image_path", "manual_count", "manual_grade", "rater", "notes",
]


def _median_count(counts: list[int]) -> float:
    return float(statistics.median(counts))


def _log_ratio_z(values: dict[str, float], grades: dict[str, int]) -> dict[str, float]:
    """Robust z dari log-ratio count v2 vs count NNEW, dihitung per grade."""
    by_grade: dict[int, list[float]] = defaultdict(list)
    for name, value in values.items():
        by_grade[grades[name]].append(value)

    params: dict[int, tuple[float, float]] = {}
    for grade, entries in by_grade.items():
        center = statistics.median(entries)
        mad = statistics.median([abs(entry - center) for entry in entries])
        params[grade] = (center, max(mad, 1e-6))

    z: dict[str, float] = {}
    for name, value in values.items():
        center, mad = params[grades[name]]
        z[name] = (value - center) / (1.4826 * mad)
    return z


def build_manifest_rows(folds: dict[int, dict[str, tuple[int, int]]],
                        circles: dict[str, int], policy: str,
                        image_dir: Path | None = None) -> list[dict[str, object]]:
    if policy not in {"nnew", "v2"}:
        raise ValueError("policy harus 'nnew' atau 'v2'.")

    fold_ids = sorted(folds)
    names = set.intersection(*(set(folds[fold]) for fold in fold_ids)) & set(circles)
    if not names:
        raise ValueError("Tidak ada file yang muncul di semua fold dan punya anotasi v2.")

    meta: dict[str, dict[str, object]] = {}
    ratios: dict[str, float] = {}
    grades: dict[str, int] = {}
    for name in names:
        labels = {folds[fold][name][0] for fold in fold_ids}
        if len(labels) != 1:
            raise ValueError(f"Label {name} berbeda antar fold: {sorted(labels)}")
        nnew_label = next(iter(labels))
        counts = [folds[fold][name][1] for fold in fold_ids]
        median_count = _median_count(counts)
        v2_count = circles[name]
        if policy == "nnew":
            grade = nnew_label
        else:
            grade = count_to_severity_index(v2_count)
        meta[name] = {
            "grade": grade,
            "nnew_label": nnew_label,
            "counts": counts,
            "count_median": median_count,
            "count_min": min(counts),
            "count_max": max(counts),
            "count_n_distinct": len(set(counts)),
            "prefix": prefix_label(name),
            "v2_circles": v2_count,
            "v2_grade": count_to_severity_index(v2_count),
        }
        grades[name] = grade
        ratios[name] = math.log1p(v2_count) - math.log1p(median_count)

    z_scores = _log_ratio_z(ratios, grades)

    rows: list[dict[str, object]] = []
    for name in sorted(names):
        entry = meta[name]
        prefix_mismatch = entry["prefix"] != entry["grade"]
        unstable = entry["count_n_distinct"] > 1
        v2_disagrees = entry["v2_grade"] != entry["grade"]
        outlier = abs(z_scores[name]) > 3.0
        review = bool(outlier or (policy == "nnew" and prefix_mismatch))
        image_path = str((image_dir / name).resolve()) if image_dir else ""
        for fold in fold_ids:
            _, count = folds[fold][name]
            rows.append({
                "file_name": name,
                "fold": fold,
                "split": "",
                "policy": policy,
                "grade": entry["grade"],
                "count": circles[name] if policy == "v2" else count,
                "count_median": entry["count_median"],
                "count_min": entry["count_min"],
                "count_max": entry["count_max"],
                "count_n_distinct": entry["count_n_distinct"],
                "prefix_label": entry["prefix"],
                "v2_circles": entry["v2_circles"],
                "v2_grade": entry["v2_grade"],
                "count_v2_log_ratio_z": round(z_scores[name], 4),
                "flag_prefix_mismatch": int(prefix_mismatch),
                "flag_count_unstable": int(unstable),
                "flag_v2_grade_disagrees": int(v2_disagrees),
                "review_priority": int(review),
                "image_path": image_path,
            })
    return rows


def attach_splits(rows: list[dict[str, object]],
                  splits: dict[tuple[int, str], str]) -> None:
    for row in rows:
        row["split"] = splits[(int(row["fold"]), str(row["file_name"]))]


def unique_files(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: dict[str, dict[str, object]] = {}
    for row in rows:
        seen.setdefault(str(row["file_name"]), row)
    return [seen[name] for name in sorted(seen)]


def build_review_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    review: list[dict[str, object]] = []
    for row in unique_files(rows):
        if not row["review_priority"]:
            continue
        reasons = []
        if row["flag_prefix_mismatch"]:
            reasons.append("prefix_mismatch")
        if abs(float(row["count_v2_log_ratio_z"])) > 3.0:
            reasons.append("count_v2_outlier")
        review.append({
            "file_name": row["file_name"],
            "grade": row["grade"],
            "count_median": row["count_median"],
            "count_min": row["count_min"],
            "count_max": row["count_max"],
            "prefix_label": row["prefix_label"],
            "v2_circles": row["v2_circles"],
            "v2_grade": row["v2_grade"],
            "count_v2_log_ratio_z": row["count_v2_log_ratio_z"],
            "reason": ";".join(reasons),
        })
    review.sort(key=lambda item: (-abs(float(item["count_v2_log_ratio_z"])),
                                  str(item["file_name"])))
    return review


def sample_audit(rows: list[dict[str, object]], per_grade: int,
                 seed: int) -> list[dict[str, object]]:
    if per_grade <= 0:
        return []
    by_grade: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in unique_files(rows):
        by_grade[int(row["grade"])].append(row)

    rng = random.Random(seed)
    sample: list[dict[str, object]] = []
    for grade in sorted(by_grade):
        candidates = sorted(by_grade[grade], key=lambda item: str(item["file_name"]))
        chosen = candidates if len(candidates) <= per_grade else rng.sample(candidates, per_grade)
        for row in sorted(chosen, key=lambda item: str(item["file_name"])):
            sample.append({
                "file_name": row["file_name"],
                "grade": row["grade"],
                "count_median": row["count_median"],
                "nnew_count_fold0": row["count"],
                "prefix_label": row["prefix_label"],
                "v2_circles": row["v2_circles"],
                "image_path": row["image_path"],
                "manual_count": "",
                "manual_grade": "",
                "rater": "",
                "notes": "",
            })
    return sample


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], policy: str,
              review: list[dict[str, object]], audit: list[dict[str, object]]) -> dict:
    files = unique_files(rows)
    return {
        "policy": policy,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": len(files),
        "rows": len(rows),
        "grade_distribution": dict(sorted(Counter(int(r["grade"]) for r in files).items())),
        "prefix_mismatch": sum(int(r["flag_prefix_mismatch"]) for r in files),
        "count_unstable": sum(int(r["flag_count_unstable"]) for r in files),
        "v2_grade_disagrees": sum(int(r["flag_v2_grade_disagrees"]) for r in files),
        "review_priority": len(review),
        "audit_sample": len(audit),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classification-dir", type=Path,
                        default=ROOT / "datasetacne04" / "Classification")
    parser.add_argument("--annotations", type=Path,
                        default=ROOT.parent / "acne04v2" / "Acne04-v2_annotations.json")
    parser.add_argument("--image-dir", type=Path,
                        default=ROOT / "datasetacne04" / "Classification" / "JPEGImages")
    parser.add_argument("--policy", choices=("nnew", "v2"), default="nnew")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "ground_truth")
    parser.add_argument("--audit-per-grade", type=int, default=0,
                        help=">0: tulis sampel audit acak berstrata per grade.")
    parser.add_argument("--audit-seed", type=int, default=42)
    args = parser.parse_args()

    folds = load_fold_rows(args.classification_dir)
    circles = load_circle_counts(args.annotations)

    # Split per fold: trainval vs test.
    splits: dict[tuple[int, str], str] = {}
    for fold in folds:
        splits.update({(fold, name): "train" for name in folds[fold]})
    for fold in folds:
        test_path = (args.classification_dir / f"NNEW_test_{fold}.txt")
        for line in test_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                splits[(fold, line.split()[0])] = "test"

    image_dir = args.image_dir if args.image_dir.exists() else None
    rows = build_manifest_rows(folds, circles, args.policy, image_dir)
    attach_splits(rows, splits)

    missing = sum(1 for row in rows if image_dir and not Path(str(row["image_path"])).exists())
    if missing:
        raise FileNotFoundError(f"{missing} baris manifest menunjuk ke gambar yang tidak ada.")

    review = build_review_rows(rows)
    audit = sample_audit(rows, args.audit_per_grade, args.audit_seed)
    summary = summarize(rows, args.policy, review, audit)

    suffix = args.policy
    write_csv(args.out_dir / f"ground_truth_manifest_{suffix}.csv", MANIFEST_FIELDS, rows)
    write_csv(args.out_dir / f"ground_truth_review_{suffix}.csv", REVIEW_FIELDS, review)
    if audit:
        write_csv(args.out_dir / f"label_audit_sample_{suffix}.csv", AUDIT_FIELDS, audit)
    (args.out_dir / f"ground_truth_summary_{suffix}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutput: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
