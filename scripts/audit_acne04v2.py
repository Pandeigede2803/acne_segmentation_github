"""Strict ACNE04-v2 source/manifest checks, without ML dependencies."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


OFFICIAL_COMMIT = "5930801fc75fb1a7084c26c1db3095d3e42d1f55"
OFFICIAL_SHA256 = "ae03fc23fed0925c254dc4c4c9e55b539bd199b236151cb649857378a43539e9"
OFFICIAL_URL = (
    f"https://raw.githubusercontent.com/AIpourlapeau/acne04v2/{OFFICIAL_COMMIT}/"
    "Acne04-v2_annotations.json"
)
EXPECTED_CLASSES = {"levle0": 470, "levle1": 579, "levle2": 133, "levle3": 22}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_official_annotations(path):
    path = Path(path)
    if not path.is_file() or sha256(path) != OFFICIAL_SHA256:
        raise ValueError("Anotasi bukan JSON v2 resmi yang dipin. Jangan gunakan XML original.")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_source(image_dir, annotations_path):
    data = load_official_annotations(annotations_path)
    images = data["images"]
    names = [row["file_name"] for row in images]
    if len(names) != 1204 or len(set(names)) != 1204 or len(data["annotations"]) != 32443:
        raise ValueError("Jumlah gambar/anotasi resmi tidak sesuai.")
    paths = {}
    for path in Path(image_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.setdefault(path.name, []).append(path)
    missing = sorted(set(names) - paths.keys())
    duplicates = sorted(name for name in names if len(paths.get(name, [])) > 1)
    if missing or duplicates:
        raise ValueError(f"Sumber v2 tidak lengkap/ambigu: missing={missing[:10]}, duplicate={duplicates[:10]}")
    return {
        "dataset": "ACNE04-v2", "official_commit": OFFICIAL_COMMIT,
        "annotations_sha256": OFFICIAL_SHA256,
        "source_image_files": sum(map(len, paths.values())),
        "eligible_images": len(names), "lesion_annotations": len(data["annotations"]),
        "excluded_source_files": sum(len(v) for k, v in paths.items() if k not in set(names)),
        "class_counts_from_filename": dict(Counter(n.split("_")[0] for n in names)),
        "severity_source": "original filename prefix; not recomputed from v2 lesion counts",
    }


def audit_manifest(manifest_path, annotations_path, image_dir):
    """Require complete original v2 manifest; never apply this to synthetic rows."""
    report = audit_source(image_dir, annotations_path)
    data = load_official_annotations(annotations_path)
    names = {row["file_name"] for row in data["images"]}
    source_paths = {
        p.name: p.resolve() for p in Path(image_dir).rglob("*")
        if p.is_file() and p.name in names
    }
    circles = Counter(row["image_id"] for row in data["annotations"])
    circles_by_name = {row["file_name"]: circles[row["id"]] for row in data["images"]}
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actual = [row.get("filename", "") for row in rows]
    if len(rows) != 1204 or len(set(actual)) != 1204 or set(actual) != names:
        raise ValueError("Manifest harus tepat 1.204 nama unik pada JSON resmi v2. Buat output baru; jangan pakai manifest 1.457 gambar.")
    for row in rows:
        name = row["filename"]
        prefix = name.split("_")[0]
        if row.get("label_name") != prefix or row.get("label_index") != prefix[-1]:
            raise ValueError(f"Label tidak sesuai sumber nama file: {name}")
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Split tidak valid: {name}")
        if row.get("n_circles") != str(circles_by_name[name]):
            raise ValueError(f"Jumlah lesi tidak sesuai JSON v2: {name}")
        if (row.get("resize_mode"), row.get("preprocessing_mode"), row.get("image_size")) != ("letterbox", "none", "320"):
            raise ValueError(f"Preprocessing manifest tidak sesuai protokol: {name}")
        for field in ("image_path", "preprocessed_path", "refined_mask", "circle_mask"):
            if not row.get(field) or not Path(row[field]).is_file():
                raise ValueError(f"File {field} hilang pada {name}")
        if Path(row["image_path"]).resolve() != source_paths[name]:
            raise ValueError(f"Sumber gambar manifest berbeda: {name}")
    split_classes = Counter((r["split"], r["label_name"]) for r in rows)
    if any(not split_classes[(split, label)] for split in ("train", "val", "test") for label in EXPECTED_CLASSES):
        raise ValueError("Setiap split wajib berisi keempat kelas.")
    report["manifest_sha256"] = sha256(manifest_path)
    report["split_counts"] = dict(Counter(r["split"] for r in rows))
    report["split_class_counts"] = {
        split: {label: split_classes[(split, label)] for label in EXPECTED_CLASSES}
        for split in ("train", "val", "test")
    }
    return report


def preprocessing_provenance(manifest_path, annotations_path, image_dir, refine_script_path):
    report = audit_manifest(manifest_path, annotations_path, image_dir)
    report["refine_script_sha256"] = sha256(refine_script_path)
    report["preprocessing"] = {"resize_mode": "letterbox", "mode": "none", "image_size": 320}
    return report


def audit_augmented_manifest(original_path, augmented_path):
    """Check train-only additions and exact preservation of original records."""
    def read_rows(path):
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames, list(reader)

    fields, original = read_rows(original_path)
    augmented_fields, augmented = read_rows(augmented_path)
    if set(fields or []) != set(augmented_fields or []) or not original:
        raise ValueError("Kolom manifest augmented berbeda atau sumber kosong.")
    def records(rows):
        return Counter(tuple(row.get(key, "") for key in fields) for row in rows)

    if records(original) - records(augmented):
        raise ValueError("Baris sumber asli hilang/berubah pada manifest augmented.")
    if records([r for r in original if r["split"] != "train"]) != records([r for r in augmented if r["split"] != "train"]):
        raise ValueError("Validation/test berubah atau split asing ditemukan.")
    before = Counter(r["label_index"] for r in original if r["split"] == "train")
    after = Counter(r["label_index"] for r in augmented if r["split"] == "train")
    if set(before) != {"0", "1", "2", "3"} or set(after) != set(before) or set(after.values()) != {max(before.values())}:
        raise ValueError("Augmentasi 100% wajib menyeimbangkan empat kelas ke kelas train asli terbesar.")
    names = [r["filename"] for r in augmented]
    if len(names) != len(set(names)):
        raise ValueError("Nama sampel augmented duplikat.")
    for row in augmented:
        for field in ("image_path", "preprocessed_path", "refined_mask"):
            if not row.get(field) or not Path(row[field]).is_file():
                raise ValueError(f"File augmented {field} hilang: {row['filename']}")
    return {"train_counts_before": dict(before), "train_counts_after": dict(after),
            "synthetic_train_pairs": len(augmented) - len(original),
            "augmented_manifest_sha256": sha256(augmented_path)}
