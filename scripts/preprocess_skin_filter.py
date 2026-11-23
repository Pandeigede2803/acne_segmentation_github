#!/usr/bin/env python3
"""Create a separate face/skin-filtered copy of a prepared 512 manifest.

Only pixels inside the face box or skin map are kept; the rest is filled.
Masks are copied unchanged by default so segmentation targets stay comparable.
This script never writes to the source manifest or source images.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from face_skin_utils import face_skin_region
except ImportError:  # Imported as scripts.preprocess_skin_filter
    from scripts.face_skin_utils import face_skin_region


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_image(row):
    return Path(row.get("preprocessed_path") or row.get("image_path") or "")


def row_mask(row):
    value = row.get("mask_path") or row.get("refined_mask") or row.get("circle_mask")
    return Path(value) if value else None


def apply_skin_filter(bgr, region, fill):
    output = bgr.copy()
    outside = region == 0
    if fill == "black":
        output[outside] = 0
    elif fill == "gray":
        output[outside] = 127
    elif fill == "mean":
        pixels = bgr[region > 0]
        color = pixels.mean(axis=0) if len(pixels) else np.array([127, 127, 127])
        output[outside] = color.astype(np.uint8)
    else:
        raise ValueError(f"Fill tidak dikenal: {fill}")
    return output


def select_examples(rows, limit):
    by_label = {}
    for row in rows:
        by_label.setdefault(int(row["label_index"]), row)
    chosen = [by_label[label] for label in sorted(by_label)]
    for row in rows:
        if len(chosen) >= limit:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen[:limit]


def draw_examples(rows, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in rows:
        image_path, mask_path = row_image(row), row_mask(row)
        bgr = cv2.imread(str(image_path))
        if bgr is None or mask_path is None:
            continue
        face_box, _, _, region = face_skin_region(bgr, args)
        filtered = apply_skin_filter(bgr, region, args.fill)
        coverage = 100.0 * float(np.count_nonzero(region)) / region.size
        before = bgr.copy()
        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(before, (x, y), (x + w, y + h), (0, 220, 255), 2)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(before, contours, -1, (255, 210, 0), 1)

        def header(canvas, text):
            banner = np.full((34, canvas.shape[1], 3), 24, dtype=np.uint8)
            cv2.putText(banner, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (245, 245, 245), 1, cv2.LINE_AA)
            return np.vstack([banner, canvas])

        before = header(before, f"SEBELUM | {row['filename']} | label {row['label_index']}")
        filtered = header(filtered, f"SESUDAH filter wajah+kulit | region {coverage:.1f}% | fill {args.fill}")
        separator = np.full((before.shape[0], 8, 3), 255, dtype=np.uint8)
        panel = np.hstack([before, separator, filtered])
        target_w = 1600
        if panel.shape[1] > target_w:
            scale = target_w / panel.shape[1]
            panel = cv2.resize(panel, (target_w, int(panel.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        path = output_dir / f"{int(row['label_index'])}_{Path(row['filename']).stem}_skin.png"
        cv2.imwrite(str(path), panel)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest siap training (mis. preprocess_512/refined_manifest.csv).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--fill", choices=("black", "gray", "mean"), default="black")
    parser.add_argument("--clip-masks", action="store_true",
                        help="Intersect mask with region. Default: masks copied unchanged.")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-face-size", type=int, default=30)
    parser.add_argument("--skin-cr-min", type=int, default=133)
    parser.add_argument("--skin-cr-max", type=int, default=180)
    parser.add_argument("--skin-cb-min", type=int, default=77)
    parser.add_argument("--skin-cb-max", type=int, default=133)
    parser.add_argument("--skin-v-min", type=int, default=30)
    parser.add_argument("--skin-morph", type=int, default=5)
    args = parser.parse_args()
    if cv2 is None or np is None:
        raise SystemExit("Butuh opencv-python dan numpy. Install: python3 -m pip install opencv-python-headless numpy")
    if args.output_dir.resolve() == args.manifest.resolve().parent:
        raise ValueError("Output tidak boleh menimpa folder sumber. Gunakan folder baru.")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if args.max_images > 0:
        rows = rows[: args.max_images]
    if not rows:
        raise ValueError("Manifest kosong.")
    mask_columns = [column for column in ("mask_path", "refined_mask") if column in fields]
    if not mask_columns and "circle_mask" not in fields:
        raise KeyError("Manifest harus punya mask_path, refined_mask, atau circle_mask.")

    images_dir = args.output_dir / "images"
    masks_dir = args.output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    output_rows, detected, region_pct = [], 0, []
    for index, row in enumerate(rows, start=1):
        image_path, mask_path = row_image(row), row_mask(row)
        if not image_path.is_file():
            raise FileNotFoundError(f"Foto tidak ditemukan: {image_path}")
        if mask_path is None or not mask_path.is_file():
            raise FileNotFoundError(f"Mask tidak ditemukan untuk {row['filename']}")
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Foto tidak dapat dibaca: {image_path}")
        face_box, _, _, region = face_skin_region(bgr, args)
        filtered = apply_skin_filter(bgr, region, args.fill)
        output_image = images_dir / Path(row["filename"]).name
        cv2.imwrite(str(output_image), filtered, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if args.clip_masks:
            mask = cv2.bitwise_and(mask, region)
        output_mask = masks_dir / f"{Path(row['filename']).stem}.png"
        cv2.imwrite(str(output_mask), mask)

        updated = dict(row)
        if "preprocessed_path" in fields:
            updated["preprocessed_path"] = str(output_image.resolve())
        else:
            updated["image_path"] = str(output_image.resolve())
        if "image_path" in fields and not updated.get("image_path"):
            updated["image_path"] = str(output_image.resolve())
        for column in mask_columns:
            updated[column] = str(output_mask.resolve())
        if not mask_columns:
            updated["circle_mask"] = str(output_mask.resolve())
        output_rows.append(updated)
        detected += int(face_box is not None)
        region_pct.append(100.0 * float(np.count_nonzero(region)) / region.size)
        if index % 100 == 0:
            print(f"  {index}/{len(rows)} citra selesai", flush=True)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    provenance = {
        "source_manifest": str(args.manifest), "source_manifest_sha256": file_hash(args.manifest),
        "output_manifest": str(args.output_manifest), "output_manifest_sha256": file_hash(args.output_manifest),
        "script_sha256": file_hash(__file__),
        "face_skin_utils_sha256": file_hash(Path(__file__).with_name("face_skin_utils.py")),
        "fill": args.fill, "clip_masks": bool(args.clip_masks),
        "face_backend": "opencv_haar", "n_images": len(output_rows),
        "face_detected": detected,
        "face_detection_rate_pct": round(100.0 * detected / len(output_rows), 3),
        "mean_region_pct": round(sum(region_pct) / len(region_pct), 3),
        "skin_thresholds": {"cr": [args.skin_cr_min, args.skin_cr_max],
                            "cb": [args.skin_cb_min, args.skin_cb_max],
                            "v_min": args.skin_v_min, "morph": args.skin_morph},
    }
    (args.output_dir / "skin_filter_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    if args.examples > 0:
        examples = draw_examples(select_examples(rows, args.examples),
                                 args.output_dir / "visual_examples", args)
        print("Contoh visual:", ", ".join(str(path) for path in examples))
    print(json.dumps({key: provenance[key] for key in (
        "n_images", "face_detected", "face_detection_rate_pct", "mean_region_pct")}, indent=2))
    print("Manifest:", args.output_manifest)
    print("Provenance:", args.output_dir / "skin_filter_provenance.json")


if __name__ == "__main__":
    main()
