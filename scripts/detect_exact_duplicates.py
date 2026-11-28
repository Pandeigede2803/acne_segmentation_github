#!/usr/bin/env python3
"""Step 42: exact duplicates, label conflicts, canonical manifest.

Reads the original (pre-augmentation) manifest, hashes every image twice
(byte SHA-256 and decoded-RGB SHA-256), groups pixel-identical images, reports
label/mask conflicts, writes a contact sheet, and produces a clean manifest
without deleting or modifying any source file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from recovery_common import read_rows, sha256_file, stable_group_ids, write_rows

ROOT = Path(__file__).resolve().parent.parent


def image_path(row: dict, manifest_dir: Path, image_dir: Path | None) -> Path:
    raw = row.get("preprocessed_path") or row.get("image_path") or ""
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_dir / path
    if path.is_file():
        return path
    if image_dir is not None and (image_dir / row.get("filename", "")).is_file():
        return image_dir / row["filename"]
    raise FileNotFoundError(f"Foto tidak ditemukan: {row.get('filename')} ({raw})")


def mask_path(row: dict, manifest_dir: Path) -> Path | None:
    raw = row.get("mask_path") or row.get("refined_mask") or row.get("circle_mask")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_dir / path
    return path if path.is_file() else None


def decoded_pixel_hash(path: Path) -> tuple[str, int, int]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256(f"{width}x{height}:".encode("ascii"))
        digest.update(rgb.tobytes())
    return digest.hexdigest(), width, height


def label_of(row: dict) -> str:
    name = row.get("label_name")
    if name in {"levle0", "levle1", "levle2", "levle3"}:
        return name
    index = row.get("label_index", "")
    return f"levle{index}" if index in {"0", "1", "2", "3"} else "unknown"


def canonical_choice(rows: list[dict], rule: str) -> dict:
    if rule == "first_filename":
        return sorted(rows, key=lambda row: row["filename"])[0]
    if rule == "most_circles":
        return sorted(rows, key=lambda row: (-int(float(row.get("n_circles") or 0)),
                                             row["filename"]))[0]
    raise ValueError(f"Aturan canonical tidak dikenal: {rule}")


def contact_sheet(groups: list[dict], manifest_dir: Path, image_dir: Path | None,
                  output: Path, thumb: int = 160, max_rows: int = 12) -> None:
    if not groups:
        return
    selected = sorted(groups, key=lambda group: (-len(group["rows"]), group["group_id"]))[:max_rows]
    columns = max(len(group["rows"]) for group in selected)
    width = thumb * columns + 20
    row_height = thumb + 34
    canvas = Image.new("RGB", (width + 20, row_height * len(selected) + 40), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Exact duplicate groups (overlay = mask, red)", fill=(235, 235, 235))
    for row_index, group in enumerate(selected):
        y = 40 + row_index * row_height
        for col_index, entry in enumerate(sorted(group["rows"], key=lambda r: r["filename"])):
            x = 10 + col_index * thumb
            try:
                with Image.open(image_path(entry, manifest_dir, image_dir)) as image:
                    tile = image.convert("RGB").resize((thumb, thumb), Image.Resampling.BILINEAR)
                mask_file = mask_path(entry, manifest_dir)
                if mask_file is not None:
                    with Image.open(mask_file) as mask:
                        overlay = mask.convert("L").resize((thumb, thumb), Image.Resampling.NEAREST)
                    red = Image.new("RGB", (thumb, thumb), (220, 40, 40))
                    tile = Image.composite(red, tile, overlay.point(lambda value: 90 if value > 127 else 0))
            except FileNotFoundError:
                tile = Image.new("RGB", (thumb, thumb), (70, 20, 20))
            canvas.paste(tile, (x, y))
            draw.text((x, y + thumb + 2), entry["filename"][:26], fill=(230, 230, 230))
            draw.text((x, y + thumb + 14), f"{label_of(entry)} | {entry.get('split', '?')}",
                      fill=(180, 200, 255))
        draw.text((10, y - 12), group["group_id"], fill=(255, 220, 120))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def resolve_image_dir(value: str | None, manifest_rows: list[dict], manifest_dir: Path) -> Path | None:
    if value:
        return Path(value)
    parents = {str(Path(row["image_path"]).parent) for row in manifest_rows if row.get("image_path")}
    if len(parents) == 1:
        candidate = Path(parents.pop())
        if candidate.is_dir():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest original SEBELUM augmentasi (satu baris per citra).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-dir", default=None,
                        help="Fallback folder foto bila path di manifest tidak tersedia.")
    parser.add_argument("--canonical-rule", choices=("first_filename", "most_circles"),
                        default="first_filename")
    parser.add_argument("--overrides", type=Path, default=None,
                        help="CSV opsional: group_id,canonical_filename,status,reason")
    parser.add_argument("--quarantine-mask-conflicts", action="store_true",
                        help="Karantina grup yang mask-nya berbeda (default: hanya label).")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} sudah berisi file. Pakai folder baru.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = args.manifest.parent
    rows, fields = read_rows(args.manifest)
    if not rows:
        raise ValueError("Manifest kosong.")
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("Manifest harus satu baris unik per citra; jalankan pada manifest original.")
    image_dir = resolve_image_dir(args.image_dir, rows, manifest_dir)

    hashes: dict[str, dict] = {}
    for row in rows:
        path = image_path(row, manifest_dir, image_dir)
        pixel_hash, width, height = decoded_pixel_hash(path)
        mask = mask_path(row, manifest_dir)
        hashes[row["filename"]] = {
            "bytes_sha256": sha256_file(path),
            "pixel_sha256": pixel_hash,
            "mask_sha256": sha256_file(mask) if mask else "",
            "width": width,
            "height": height,
        }

    group_names = stable_group_ids(entry["pixel_sha256"] for entry in hashes.values())
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[group_names[hashes[row["filename"]]["pixel_sha256"]]].append(row)

    overrides: dict[str, dict] = {}
    if args.overrides and args.overrides.is_file():
        override_rows, _ = read_rows(args.overrides)
        overrides = {row["group_id"]: row for row in override_rows}

    groups: list[dict] = []
    conflicts: list[dict] = []
    for group_id, group_rows in sorted(by_group.items()):
        labels = sorted({label_of(row) for row in group_rows})
        masks = sorted({hashes[row["filename"]]["mask_sha256"] for row in group_rows})
        splits = sorted({row.get("split", "?") for row in group_rows})
        files = {hashes[row["filename"]]["bytes_sha256"] for row in group_rows}
        status, reason = "unique", ""
        if len(labels) > 1:
            status, reason = "label_conflict", "|".join(labels)
        elif len(masks) > 1 and args.quarantine_mask_conflicts:
            status, reason = "mask_conflict", "different masks"
        elif len(masks) > 1:
            status, reason = "mask_variant", "kept for adjudication, not auto-quarantined"
        entry = {
            "group_id": group_id, "size": len(group_rows), "labels": labels, "splits": splits,
            "label_conflict": len(labels) > 1,
            "byte_identical": len(files) == 1,
            "pixel_identical": True,
            "mask_variants": len(masks), "status": status, "reason": reason,
            "rows": group_rows,
        }
        override = overrides.get(group_id)
        if override:
            if override.get("status"):
                entry["status"] = override["status"]
            if override.get("reason"):
                entry["reason"] = override["reason"]
        groups.append(entry)
        if len(group_rows) > 1 or entry["status"] in {"label_conflict", "mask_conflict"}:
            conflicts.append(entry)

    # One representative per group; quarantine conflicts that were not resolved.
    clean_rows: list[dict] = []
    quarantine_rows: list[dict] = []
    for entry in groups:
        if entry["status"] == "label_conflict" or entry["status"] == "mask_conflict" \
                or entry["status"] == "quarantined":
            for row in entry["rows"]:
                quarantine_rows.append(dict(row, group_id=entry["group_id"],
                                            quarantine_reason=entry["status"]))
            continue
        canonical_name = canonical_choice(entry["rows"], args.canonical_rule)["filename"]
        override = overrides.get(entry["group_id"])
        if override and override.get("canonical_filename"):
            canonical_name = override["canonical_filename"]
        for row in entry["rows"]:
            if row["filename"] == canonical_name:
                clean_rows.append(row)
                break

    clean_rows.sort(key=lambda row: row["filename"])
    duplicate_rows = []
    for entry in groups:
        if entry["size"] < 2:
            continue
        for row in sorted(entry["rows"], key=lambda item: item["filename"]):
            duplicate_rows.append({
                "group_id": entry["group_id"], "filename": row["filename"],
                "label_name": label_of(row), "split": row.get("split", ""),
                "bytes_sha256": hashes[row["filename"]]["bytes_sha256"],
                "pixel_sha256": hashes[row["filename"]]["pixel_sha256"],
                "mask_sha256": hashes[row["filename"]]["mask_sha256"],
                "width": hashes[row["filename"]]["width"],
                "height": hashes[row["filename"]]["height"],
                "n_circles": row.get("n_circles", ""),
                "status": entry["status"],
            })

    conflict_rows = [{
        "group_id": entry["group_id"], "size": entry["size"],
        "filenames": "|".join(sorted(row["filename"] for row in entry["rows"])),
        "labels": "|".join(entry["labels"]), "splits": "|".join(entry["splits"]),
        "mask_variants": entry["mask_variants"], "status": entry["status"],
        "reason": entry["reason"],
    } for entry in conflicts if entry["label_conflict"] or entry["mask_variants"] > 1]

    write_rows(args.output_dir / "exact_duplicate_groups.csv", duplicate_rows,
               ["group_id", "filename", "label_name", "split", "bytes_sha256",
                "pixel_sha256", "mask_sha256", "width", "height", "n_circles", "status"])
    write_rows(args.output_dir / "label_mask_conflicts.csv", conflict_rows,
               ["group_id", "size", "filenames", "labels", "splits", "mask_variants",
                "status", "reason"])
    write_rows(args.output_dir / "quarantine_manifest.csv", quarantine_rows, fields)
    write_rows(args.output_dir / "clean_original_manifest.csv", clean_rows, fields)

    contact_sheet([entry for entry in groups if entry["size"] > 1], manifest_dir, image_dir,
                  args.output_dir / "duplicate_review_grid.png")

    before = Counter(label_of(row) for row in rows)
    after = Counter(label_of(row) for row in clean_rows)
    cross_split = Counter()
    for entry in groups:
        if entry["size"] < 2:
            continue
        splits = set(entry["splits"])
        if {"train", "test"} <= splits:
            cross_split["train-test"] += 1
        if {"val", "test"} <= splits:
            cross_split["val-test"] += 1
        if {"train", "val"} <= splits:
            cross_split["train-val"] += 1

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "rows_before": len(rows), "rows_after": len(clean_rows),
        "quarantined_rows": len(quarantine_rows),
        "groups_total": len(groups),
        "groups_with_duplicates": sum(1 for entry in groups if entry["size"] > 1),
        "duplicate_rows": sum(entry["size"] - 1 for entry in groups if entry["size"] > 1),
        "label_conflicts": sum(1 for entry in groups if entry["label_conflict"]),
        "mask_variant_groups": sum(1 for entry in groups if entry["size"] > 1 and entry["mask_variants"] > 1),
        "cross_split_groups": dict(cross_split),
        "counts_before": dict(sorted(before.items())),
        "counts_after": dict(sorted(after.items())),
        "canonical_rule": args.canonical_rule,
        "quarantine_mask_conflicts": bool(args.quarantine_mask_conflicts),
    }
    (args.output_dir / "step42_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Step 42 — Exact duplicates dan konflik label", "",
        f"- Manifest: `{args.manifest}` (SHA-256 `{metrics['manifest_sha256'][:16]}…`)",
        f"- Baris sebelum: **{len(rows)}**; sesudah dedup: **{len(clean_rows)}**; "
        f"dikarantina: **{len(quarantine_rows)}**",
        f"- Grup duplikat: **{metrics['groups_with_duplicates']}** dari **{len(groups)}** grup; "
        f"{metrics['duplicate_rows']} baris berlebih.",
        f"- Konflik label: **{metrics['label_conflicts']}** grup; "
        f"grup dengan mask berbeda: **{metrics['mask_variant_groups']}**.",
        f"- Grup lintas split: {metrics['cross_split_groups'] or 'tidak ada'}.",
        "", "## Rekonsiliasi per kelas", "",
        "| Kelas | Sebelum | Sesudah | Selisih |", "|---|---:|---:|---:|",
    ]
    for label in sorted(before):
        report.append(f"| {label} | {before[label]} | {after.get(label, 0)} | "
                      f"{after.get(label, 0) - before[label]} |")
    report += [
        "", "## Aturan yang dipakai", "",
        f"- Duplikat = hash piksel RGB identik setelah decode (byte duplikat juga dicatat).",
        f"- Representasi canonical: `{args.canonical_rule}`; tidak menghapus file sumber.",
        "- Grup dengan konflik label dimasukkan ke `quarantine_manifest.csv` dan tidak masuk "
        "manifest bersih sampai ada adjudikasi.",
        "- Mask berbeda dicatat sebagai `mask_variant`; lihat `label_mask_conflicts.csv`.",
        "", "## Output", "",
        "- `exact_duplicate_groups.csv`, `label_mask_conflicts.csv`, "
        "`quarantine_manifest.csv`, `clean_original_manifest.csv`",
        "- `duplicate_review_grid.png`, `step42_metrics.json`",
    ]
    (args.output_dir / "step42_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({key: metrics[key] for key in
                      ("rows_before", "rows_after", "label_conflicts", "groups_with_duplicates")},
                     indent=2))
    print("Report:", args.output_dir / "step42_report.md")


if __name__ == "__main__":
    main()
