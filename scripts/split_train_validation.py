#!/usr/bin/env python3
"""Step 39a: carve a stratified validation split from the final train rows.

Test rows are copied unchanged. No image/mask pixels are modified. The output
manifest keeps all original columns and only changes ``split`` for held-out
train rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_train_validation(rows: list[dict], val_fraction: float, seed: int) -> list[dict]:
    if not 0 < val_fraction < 0.5:
        raise ValueError("val-fraction harus di antara 0 dan 0.5.")
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        raise ValueError("Tidak ada baris train.")
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in train:
        by_label[int(row["label_index"])].append(row)
    if set(by_label) != {0, 1, 2, 3}:
        raise ValueError("Train harus berisi keempat kelas.")

    rng = random.Random(seed)
    val_names: set[str] = set()
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda row: row["filename"])
        rng.shuffle(group)
        count = max(1, int(round(len(group) * val_fraction)))
        count = min(count, len(group) - 1)
        val_names.update(row["filename"] for row in group[:count])

    output = []
    for row in rows:
        updated = dict(row)
        if row["split"] == "train" and row["filename"] in val_names:
            updated["split"] = "val"
        output.append(updated)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        rows = [{"filename": f"c{label}_{index}.jpg", "label_index": str(label), "split": "train"}
                for label in range(4) for index in range(10)]
        rows.append({"filename": "test.jpg", "label_index": "0", "split": "test"})
        output = split_train_validation(rows, 0.2, 42)
        counts = Counter((row["split"], row["label_index"]) for row in output)
        assert counts[("val", "3")] == 2 and counts[("test", "0")] == 1
        print("PASS self-test: stratified train/val split")
        return
    if args.manifest is None or args.output_manifest is None:
        raise SystemExit("Provide --manifest and --output-manifest (or use --self-test).")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.output_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_manifest}")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not {"filename", "label_index", "split"} <= set(fields):
        raise KeyError("Manifest butuh filename, label_index, split.")
    if "test" not in {row["split"] for row in rows}:
        raise ValueError("Manifest harus memiliki split test.")
    if {row["split"] for row in rows} - {"train", "test"}:
        raise ValueError("Manifest sumber harus train/test saja.")

    output_rows = split_train_validation(rows, args.val_fraction, args.seed)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    counts = Counter((row["split"], row["label_index"]) for row in output_rows)
    provenance = {
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "counts": {f"{split}/{label}": count for (split, label), count in sorted(counts.items())},
        "note": "Image-level stratified split; no patient IDs were available.",
    }
    (args.output_manifest.parent / "split_validation.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("PASS:", dict(sorted(Counter(row["split"] for row in output_rows).items())))


if __name__ == "__main__":
    main()
