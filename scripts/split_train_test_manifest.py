"""Create a stratified train/test-only manifest from an original manifest.

No validation split is produced. The original split column is ignored and
reassigned deterministically by label, so test rows stay original (they are
never passed to the offline augmenter).
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def assign_splits(rows, train_ratio, seed=42):
    """Return copies of rows with split in {"train", "test"} and no validation."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio harus di antara 0 dan 1.")
    filenames = [row.get("filename") for row in rows]
    if any(not name for name in filenames) or len(set(filenames)) != len(filenames):
        raise ValueError("Manifest harus memiliki filename unik dan tidak kosong.")
    grouped = defaultdict(list)
    for row in rows:
        try:
            label = int(row["label_index"])
        except (KeyError, ValueError) as error:
            raise ValueError("label_index harus 0 sampai 3.") from error
        if label not in range(4):
            raise ValueError("label_index harus 0 sampai 3.")
        grouped[label].append(row)
    if set(grouped) != set(range(4)):
        raise ValueError("Manifest harus berisi keempat kelas 0 sampai 3.")

    rng = random.Random(seed)
    assigned = {}
    for label in sorted(grouped):
        group = grouped[label][:]
        rng.shuffle(group)
        total = len(group)
        if total < 2:
            raise ValueError(f"Kelas {label} butuh minimal 2 citra untuk train dan test.")
        train_count = int(total * train_ratio)
        train_count = min(max(train_count, 1), total - 1)
        for index, row in enumerate(group):
            assigned[row["filename"]] = "train" if index < train_count else "test"
    return [dict(row, split=assigned[row["filename"]]) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest asli dengan image_path/preprocessed_path dan mask.")
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-images", type=int, default=0,
                        help="0 = tidak memeriksa jumlah; 1204 untuk audit ACNE04-v2.")
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if not rows:
        raise ValueError("Manifest kosong.")
    if args.expected_images and len(rows) != args.expected_images:
        raise ValueError(f"Jumlah citra {len(rows)} != {args.expected_images}.")
    output_rows = assign_splits(rows, args.train_ratio, args.seed)
    if {row["split"] for row in output_rows} != {"train", "test"}:
        raise ValueError("Split harus berisi train dan test saja.")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    train = Counter(int(row["label_index"]) for row in output_rows if row["split"] == "train")
    test = Counter(int(row["label_index"]) for row in output_rows if row["split"] == "test")
    print("Train counts:", dict(sorted(train.items())), "total", sum(train.values()))
    print("Test counts:", dict(sorted(test.items())), "total", sum(test.values()))
    print("Manifest:", args.output_manifest)


if __name__ == "__main__":
    main()
