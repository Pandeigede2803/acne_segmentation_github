#!/usr/bin/env python3
"""Step 43B: build a grouped, stratified split from reviewed duplicates.

Groups are formed from (a) exact duplicate groups from Step 42, (b) near-duplicate
pairs marked ``confirmed`` in the Step 43A review, and optionally (c) patient ids
from an official metadata CSV. Splits are assigned per group, never per image.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from recovery_common import (assign_grouped_splits, pairs_to_groups, read_rows,
                             sha256_file, write_rows)

SYNTHETIC_MARKERS = ("__offaug_", "__syn_", "__aug_")


def load_exact_groups(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    rows, _ = read_rows(path)
    return {row["filename"]: row["group_id"] for row in rows if row.get("group_id")}


def load_confirmed_pairs(path: Path | None) -> list[tuple[str, str]]:
    if path is None or not path.is_file():
        return []
    rows, _ = read_rows(path)
    pairs = []
    for row in rows:
        status = (row.get("status") or "").strip().lower()
        decision = (row.get("decision") or "").strip().lower()
        if status in {"confirmed", "accepted", "yes"} or decision in {"merge", "confirmed"}:
            pairs.append((row["filename_a"], row["filename_b"]))
    return pairs


def load_patients(path: Path | None, names: set[str]) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    rows, _ = read_rows(path)
    patients = {}
    for row in rows:
        filename = row.get("filename") or row.get("file_name") or ""
        patient = row.get("patient_id") or row.get("subject_id") or ""
        if filename in names and patient:
            patients[filename] = str(patient)
    return patients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest bersih Step 42.")
    parser.add_argument("--exact-groups", type=Path, default=None,
                        help="exact_duplicate_groups.csv dari Step 42.")
    parser.add_argument("--near-review", type=Path, default=None,
                        help="near_duplicate_review.csv Step 43A; hanya status confirmed dipakai.")
    parser.add_argument("--patients", type=Path, default=None,
                        help="CSV opsional filename,patient_id (metadata resmi).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="0 = tanpa validation (train/test saja).")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} sudah berisi file. Pakai folder baru.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, fields = read_rows(args.manifest)
    names = {row["filename"] for row in rows}
    if len(names) != len(rows):
        raise ValueError("Manifest harus satu baris unik per citra.")
    synthetic = sorted(name for name in names if any(marker in name for marker in SYNTHETIC_MARKERS))
    if synthetic:
        raise ValueError(f"Manifest memuat baris sintetis: {synthetic[:5]}. Split harus dari original.")

    labels = {}
    for row in rows:
        try:
            labels[row["filename"]] = int(row["label_index"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"label_index tidak valid: {row.get('filename')}") from error

    exact = load_exact_groups(args.exact_groups)
    pairs = load_confirmed_pairs(args.near_review)
    patients = load_patients(args.patients, names)

    # Nodes are exact-group ids when available, otherwise the filename itself.
    node_of = {name: exact.get(name, f"file:{name}") for name in names}
    group_pairs: list[tuple[str, str]] = []
    for left, right in pairs:
        if left in node_of and right in node_of:
            group_pairs.append((node_of[left], node_of[right]))
    for name, patient in patients.items():
        # Patient-level grouping: all images of one patient share a node.
        group_pairs.append((node_of[name], f"patient:{patient}"))

    nodes = set(node_of.values()) | {f"patient:{patient}" for patient in patients.values()}
    node_groups = pairs_to_groups(nodes, group_pairs)

    group_labels: dict[str, int] = {}
    mixed: dict[str, set[int]] = defaultdict(set)
    for name, node in node_of.items():
        group = node_groups[node]
        mixed[group].add(labels[name])
    for group, group_label_set in mixed.items():
        if len(group_label_set) > 1:
            raise ValueError(f"Grup {group} memuat lebih dari satu label: {sorted(group_label_set)}. "
                             "Selesaikan Step 42 sebelum split.")
        group_labels[group] = next(iter(group_label_set))

    assignment = assign_grouped_splits(
        group_labels, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    output_rows = []
    for row in rows:
        updated = dict(row)
        updated["group_id"] = node_groups[node_of[row["filename"]]]
        updated["split"] = assignment[updated["group_id"]]
        output_rows.append(updated)
    output_rows.sort(key=lambda row: (row["split"], row["label_index"], row["filename"]))
    write_rows(args.output_dir / "split_manifest.csv", output_rows, fields + ["group_id"])

    counts = Counter((row["split"], int(row["label_index"])) for row in output_rows)
    groups_by_split_label: dict[tuple[str, int], set[str]] = defaultdict(set)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        groups_by_split_label[(row["split"], int(row["label_index"]))].add(row["group_id"])
        groups_by_split[row["split"]].add(row["group_id"])
    split_totals = Counter(row["split"] for row in output_rows)
    count_rows = []
    for split in ("train", "val", "test"):
        for label in range(4):
            count_rows.append({"split": split, "label_index": label,
                               "label_name": f"levle{label}",
                               "images": counts.get((split, label), 0),
                               "groups": len(groups_by_split_label[(split, label)])})
        count_rows.append({"split": split, "label_index": "all", "label_name": "all",
                           "images": split_totals[split], "groups": len(groups_by_split[split])})
    write_rows(args.output_dir / "split_counts.csv", count_rows)

    # Integrity: a group id must never appear in two splits.
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        splits_by_group[row["group_id"]].add(row["split"])
    leaking = sorted(group for group, splits in splits_by_group.items() if len(splits) > 1)
    evaluated_splits = ["train", "test"] + (["val"] if args.val_ratio > 0 else [])
    missing_classes = [split for split in evaluated_splits
                       if any(counts.get((split, label), 0) == 0 for label in range(4))]

    integrity = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "split_manifest": str((args.output_dir / "split_manifest.csv").resolve()),
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "seed": args.seed,
        "images": len(output_rows),
        "groups": len(group_labels),
        "group_sizes": dict(sorted(Counter(
            Counter(node_groups[node_of[name]] for name in names).values()).items())),
        "counts": {f"{split}/{label}": counts.get((split, label), 0)
                   for split in ("train", "val", "test") for label in range(4)},
        "groups_per_split": {split: len(groups_by_split[split])
                             for split in ("train", "val", "test")},
        "duplicate_group_leak": leaking,
        "missing_classes": missing_classes,
        "patient_metadata": bool(patients),
        "near_duplicate_pairs_used": len(pairs),
        "near_review_sha256": (sha256_file(args.near_review)
                               if args.near_review and args.near_review.is_file() else None),
        "exact_groups_sha256": (sha256_file(args.exact_groups)
                                if args.exact_groups and args.exact_groups.is_file() else None),
        "patients_sha256": (sha256_file(args.patients)
                            if args.patients and args.patients.is_file() else None),
        "status": "PASS" if not leaking and not missing_classes else "FAIL",
        "limitations": [
            "Tidak ada patient ID resmi" if not patients
            else "Patient ID berasal dari metadata yang diberikan user; verifikasi sumbernya.",
            "Near-duplicate hanya dari pasangan yang direview; false negative mungkin ada.",
            "Test lama telah dipakai berulang kali; split baru bukan test eksternal.",
        ],
    }
    (args.output_dir / "split_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8")

    level3_val = counts.get(("val", 3), 0)
    report = [
        "# Step 43B — Split proposal (grouped)", "",
        f"- Input: `{args.manifest}`; citra **{len(output_rows)}**, grup **{len(group_labels)}**.",
        f"- Rasio: train {args.train_ratio:.0%} / val {args.val_ratio:.0%} / "
        f"test {args.test_ratio:.0%}; seed {args.seed}.",
        f"- Pasangan near-duplicate yang dipakai: **{len(pairs)}**; patient metadata: "
        f"**{'ya' if patients else 'tidak'}**.",
        "", "## Jumlah", "",
        "| Split | L0 | L1 | L2 | L3 | Total | Grup |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    report_splits = ["train", "test"] + (["val"] if args.val_ratio > 0 else [])
    for split in report_splits:
        report.append(f"| {split} | " + " | ".join(str(counts.get((split, label), 0))
                                                    for label in range(4))
                      + f" | {split_totals[split]} | {len(groups_by_split[split])} |")
    report += [
        "", f"- Level 3 validation: **{level3_val} citra**.",
        "", "## Status integritas", "",
        f"- Group overlap lintas split: **{len(leaking)}** (harus 0).",
        f"- Kelas hilang di split: **{missing_classes or 'tidak ada'}**.",
        f"- Status: **{integrity['status']}**.",
        "", "## Keterbatasan", "",
    ] + [f"- {item}" for item in integrity["limitations"]] + [
        "", "## Catatan", "",
        "- Split tidak memakai skor model apa pun.",
        "- Bila Level 3 validation terlalu kecil, pertimbangkan grouped cross-validation pada "
        "development set dan laporkan ketidakpastian; jangan menjanjikan recall stabil.",
        "- Bekukan `split_manifest.csv` dan hash-nya sebelum training.",
        "", "## Output", "",
        "- `split_manifest.csv`, `split_counts.csv`, `split_integrity.json`",
    ]
    (args.output_dir / "split_proposal.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({"counts": integrity["counts"], "status": integrity["status"]}, indent=2))
    print("Proposal:", args.output_dir / "split_proposal.md")
    if integrity["status"] != "PASS":
        raise SystemExit("Split gagal integritas; lihat split_integrity.json")


if __name__ == "__main__":
    main()
