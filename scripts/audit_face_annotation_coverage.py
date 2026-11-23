#!/usr/bin/env python3
"""Audit face/skin coverage of ACNE04-v2 lesion annotations.

Work order implemented here:
1. Detect one face per image and record its bbox + face-area percentage.
2. Measure how much annotation area falls inside the face box and skin map.
3. Render 4-6 before/after examples, always including Very Severe when present.

No training, no GPU, no writes to existing manifests.
"""

import argparse
import csv
import hashlib
import json
import os
import time
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from face_skin_utils import expand_box, detect_face, skin_map
except ImportError:  # Imported as scripts.audit_face_annotation_coverage
    from scripts.face_skin_utils import expand_box, detect_face, skin_map

try:
    import cv2
    import numpy as np
except ImportError:  # Geometry helpers stay importable without OpenCV.
    cv2 = None
    np = None


LABEL_NAMES = {"levle0": 0, "levle1": 1, "levle2": 2, "levle3": 3}
OUTPUT_FIELDS = (
    "filename", "label_name", "label_index", "split", "image_path",
    "image_width", "image_height", "face_detected",
    "face_x", "face_y", "face_w", "face_h", "face_area_pct",
    "n_circles", "annotation_area_px", "annotation_area_pct",
    "annotation_inside_face_px", "annotation_inside_face_pct",
    "annotation_inside_skin_px", "annotation_inside_skin_pct",
    "annotation_inside_union_pct", "circles_center_inside_union",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentage(part, whole):
    return round(100.0 * part / whole, 3) if whole else None


def load_circles(annotations_path):
    """Load ACNE04-v2 JSON circles keyed by file name, plus source sizes."""
    data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data["images"]}
    circles, sizes = defaultdict(list), {}
    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        name = image["file_name"]
        cx, cy = annotation["coordinates"]
        circles[name].append((float(cx), float(cy), float(annotation["radius"])))
        sizes[name] = (int(image["width"]), int(image["height"]))
    return dict(circles), sizes


def analyze_image(bgr, circles, args):
    """Return face/skin/annotation masks and per-image coverage values."""
    height, width = bgr.shape[:2]
    face_box = detect_face(bgr, args.min_neighbors, args.min_face_size,
                           getattr(args, "detect_max_size", 512))
    face_mask = np.zeros((height, width), dtype=np.uint8)
    if face_box is not None:
        x, y, w, h = face_box
        face_mask[y:y + h, x:x + w] = 255
    skin = skin_map(bgr, face_box, (args.skin_cr_min, args.skin_cr_max),
                    (args.skin_cb_min, args.skin_cb_max), args.skin_v_min, args.skin_morph)
    region = cv2.bitwise_or(face_mask, skin)

    annotation = np.zeros((height, width), dtype=np.uint8)
    for cx, cy, radius in circles:
        cv2.circle(annotation, (int(round(cx)), int(round(cy))),
                   max(1, int(round(radius))), 255, -1)
    annotation_px = int(np.count_nonzero(annotation))
    inside_face = int(np.count_nonzero(cv2.bitwise_and(annotation, face_mask)))
    inside_skin = int(np.count_nonzero(cv2.bitwise_and(annotation, skin)))
    inside_union = int(np.count_nonzero(cv2.bitwise_and(annotation, region)))
    centers_inside = 0
    for cx, cy, _ in circles:
        x, y = int(round(cx)), int(round(cy))
        if 0 <= x < width and 0 <= y < height and region[y, x]:
            centers_inside += 1
    metrics = {
        "face_detected": int(face_box is not None),
        "face_x": face_box[0] if face_box else "",
        "face_y": face_box[1] if face_box else "",
        "face_w": face_box[2] if face_box else "",
        "face_h": face_box[3] if face_box else "",
        "face_area_pct": percentage(face_box[2] * face_box[3], width * height) if face_box else 0.0,
        "n_circles": len(circles),
        "annotation_area_px": annotation_px,
        "annotation_area_pct": percentage(annotation_px, width * height),
        "annotation_inside_face_px": inside_face,
        "annotation_inside_face_pct": percentage(inside_face, annotation_px),
        "annotation_inside_skin_px": inside_skin,
        "annotation_inside_skin_pct": percentage(inside_skin, annotation_px),
        "annotation_inside_union_pct": percentage(inside_union, annotation_px),
        "circles_center_inside_union": centers_inside,
        "region": region,
    }
    return metrics


def select_examples(records, limit=6):
    """Pick one worst-coverage image per label, then fill with the worst cases."""
    annotated = [row for row in records
                 if int(row["n_circles"]) > 0
                 and row["annotation_inside_union_pct"] is not None]
    pool = [row for row in annotated if int(row["face_detected"]) == 1]
    chosen, seen = [], set()
    for label in range(4):
        candidates = [row for row in annotated if int(row["label_index"]) == label]
        if candidates:
            # Prefer a detected face, but still show Very Severe when detection fails.
            detected = [row for row in candidates if int(row["face_detected"]) == 1]
            row = min(detected or candidates,
                      key=lambda item: item["annotation_inside_union_pct"])
            chosen.append(row)
            seen.add(row["filename"])
    for row in sorted(pool, key=lambda item: item["annotation_inside_union_pct"]):
        if len(chosen) >= limit:
            break
        if row["filename"] not in seen:
            chosen.append(row)
            seen.add(row["filename"])
    return chosen[:limit]


def mean(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(records):
    detected = [row for row in records if int(row["face_detected"]) == 1]
    annotated = [row for row in records if int(row["n_circles"]) > 0]
    per_label = {}
    for label in range(4):
        subset = [row for row in annotated if int(row["label_index"]) == label]
        per_label[str(label)] = {
            "images": sum(1 for row in records if int(row["label_index"]) == label),
            "annotated_images": len(subset),
            "mean_face_area_pct": mean(row["face_area_pct"] for row in detected
                                       if int(row["label_index"]) == label),
            "mean_inside_face_pct": mean(row["annotation_inside_face_pct"] for row in subset),
            "mean_inside_skin_pct": mean(row["annotation_inside_skin_pct"] for row in subset),
            "mean_inside_union_pct": mean(row["annotation_inside_union_pct"] for row in subset),
        }
    worst = sorted(annotated, key=lambda row: row["annotation_inside_union_pct"])[:10]
    return {
        "n_images": len(records),
        "face_detected": len(detected),
        "face_detection_rate_pct": percentage(len(detected), len(records)),
        "mean_face_area_pct": mean(row["face_area_pct"] for row in detected),
        "annotated_images": len(annotated),
        "mean_annotation_area_pct": mean(row["annotation_area_pct"] for row in annotated),
        "mean_inside_face_pct": mean(row["annotation_inside_face_pct"] for row in annotated),
        "mean_inside_skin_pct": mean(row["annotation_inside_skin_pct"] for row in annotated),
        "mean_inside_union_pct": mean(row["annotation_inside_union_pct"] for row in annotated),
        "images_fully_inside_union": sum(1 for row in annotated
                                         if (row["annotation_inside_union_pct"] or 0) >= 99.999),
        "images_below_90pct": sum(1 for row in annotated
                                  if (row["annotation_inside_union_pct"] or 0) < 90),
        "images_below_50pct": sum(1 for row in annotated
                                  if (row["annotation_inside_union_pct"] or 0) < 50),
        "images_zero_coverage": sum(1 for row in annotated
                                    if (row["annotation_inside_union_pct"] or 0) == 0),
        "per_label": per_label,
        "worst_union_coverage": [
            {"filename": row["filename"], "label_index": row["label_index"],
             "inside_union_pct": row["annotation_inside_union_pct"],
             "inside_face_pct": row["annotation_inside_face_pct"],
             "inside_skin_pct": row["annotation_inside_skin_pct"]}
            for row in worst
        ],
    }


def draw_examples(examples, output_dir, circles_by_file, args):
    """Render before/after panels: annotations + face/skin, then filtered view."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in examples:
        image_path = Path(row["image_path"])
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            continue
        circles = circles_by_file.get(row["filename"], [])
        result = analyze_image(bgr, circles, args)
        region = result["region"]
        height, width = bgr.shape[:2]
        before, after = bgr.copy(), bgr.copy()
        after[region == 0] = (after[region == 0] * 0.22).astype(np.uint8)

        for cx, cy, radius in circles:
            x, y = int(round(cx)), int(round(cy))
            inside = 0 <= x < width and 0 <= y < height and region[y, x]
            color = (60, 200, 60) if inside else (40, 40, 230)
            if inside:
                cv2.circle(after, (x, y), max(1, int(round(radius))), color, 2)
            cv2.circle(before, (x, y), max(1, int(round(radius))), color, 2)
        if result["face_detected"]:
            fx, fy, fw, fh = result["face_x"], result["face_y"], result["face_w"], result["face_h"]
            cv2.rectangle(before, (fx, fy), (fx + fw, fy + fh), (0, 220, 255), 2)
            cv2.rectangle(after, (fx, fy), (fx + fw, fy + fh), (0, 220, 255), 2)
        contours, _ = cv2.findContours(region if result["face_detected"] else np.zeros_like(region),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(before, contours, -1, (255, 210, 0), 1)
        cv2.drawContours(after, contours, -1, (255, 210, 0), 1)

        def header(canvas, text):
            banner = np.full((34, canvas.shape[1], 3), 24, dtype=np.uint8)
            cv2.putText(banner, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (245, 245, 245), 1, cv2.LINE_AA)
            return np.vstack([banner, canvas])

        coverage = row["annotation_inside_union_pct"]
        before = header(before, f"SEBELUM | {row['filename']} | label {row['label_index']} | "
                                f"face {row['face_area_pct']}% | circles {row['n_circles']}")
        after = header(after, f"SESUDAH filter wajah+kulit | anotasi tersisa {coverage}% | "
                              f"hijau=masuk, merah=keluar")
        separator = np.full((before.shape[0], 8, 3), 255, dtype=np.uint8)
        panel = np.hstack([before, separator, after])
        target_w = 1600
        if panel.shape[1] > target_w:
            scale = target_w / panel.shape[1]
            panel = cv2.resize(panel, (target_w, int(panel.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        name = f"{int(row['label_index'])}_{Path(row['filename']).stem}_coverage.png"
        path = output_dir / name
        cv2.imwrite(str(path), panel)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="refined_manifest.csv asli (1.204 baris).")
    parser.add_argument("--annotations-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Fallback folder foto asli bila image_path manifest tidak ada.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=6, help="0 = tanpa visual.")
    parser.add_argument("--max-images", type=int, default=0, help="0 = semua.")
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-face-size", type=int, default=30)
    parser.add_argument("--detect-max-size", type=int, default=512,
                        help="Sisi terpanjang citra untuk deteksi wajah; lebih kecil = lebih cepat.")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Simpan checkpoint parsial setiap N citra.")
    parser.add_argument("--skin-cr-min", type=int, default=133)
    parser.add_argument("--skin-cr-max", type=int, default=180)
    parser.add_argument("--skin-cb-min", type=int, default=77)
    parser.add_argument("--skin-cb-max", type=int, default=133)
    parser.add_argument("--skin-v-min", type=int, default=30)
    parser.add_argument("--skin-morph", type=int, default=5)
    args = parser.parse_args()
    if cv2 is None or np is None:
        raise SystemExit("Butuh opencv-python dan numpy. Install: python3 -m pip install opencv-python-headless numpy")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_images > 0:
        rows = rows[: args.max_images]
    if not rows:
        raise ValueError("Manifest kosong.")
    circles_by_file, sizes = load_circles(args.annotations_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "audit_config.json"
    summary_path = args.output_dir / "face_annotation_summary.json"
    final_csv = args.output_dir / "face_annotation_audit.csv"
    checkpoint_path = args.output_dir / "face_annotation_audit.partial.jsonl"
    config = {
        "manifest": str(args.manifest), "manifest_sha256": file_hash(args.manifest),
        "annotations": str(args.annotations_json), "annotations_sha256": file_hash(args.annotations_json),
        "script_sha256": file_hash(__file__), "face_backend": "opencv_haar",
        "detect_max_size": args.detect_max_size, "min_neighbors": args.min_neighbors,
        "min_face_size": args.min_face_size,
        "skin_thresholds": {"cr": [args.skin_cr_min, args.skin_cr_max],
                            "cb": [args.skin_cb_min, args.skin_cb_max],
                            "v_min": args.skin_v_min, "morph": args.skin_morph},
    }
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Manifest/anotasi/parameter audit berbeda. Gunakan output dir baru.")
    else:
        if final_csv.is_file() or summary_path.is_file():
            raise ValueError("Output berisi hasil audit lama tanpa config; gunakan output dir baru.")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("complete") and previous.get("provenance") == config:
            print("Audit lengkap sudah tersedia; tidak dijalankan ulang.")
            print("CSV:", final_csv)
            return

    records = []
    if checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    wanted = {row["filename"] for row in rows}
    records = [record for record in records if record.get("filename") in wanted]
    unique = {record["filename"]: record for record in records}
    records = list(unique.values())
    done = {record["filename"] for record in records}
    remaining = [row for row in rows if row["filename"] not in done]
    if records:
        print(f"♻️ Resume audit: {len(records)}/{len(rows)} citra sudah ada.", flush=True)
    print(f"🧭 Audit {len(remaining)} citra tersisa (deteksi maks {args.detect_max_size}px, "
          f"{args.checkpoint_every} citra/checkpoint).", flush=True)

    missing = []
    started = time.monotonic()
    processed = 0
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for row in remaining:
            image_path = Path(row.get("image_path", ""))
            if not image_path.is_file() and args.image_dir is not None:
                image_path = args.image_dir / row["filename"]
            if not image_path.is_file():
                missing.append(row["filename"])
                continue
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                missing.append(row["filename"])
                continue
            height, width = bgr.shape[:2]
            if row["filename"] in sizes and sizes[row["filename"]] != (width, height):
                raise ValueError(f"Ukuran {row['filename']} {width}x{height} != metadata {sizes[row['filename']]}.")
            metrics = analyze_image(bgr, circles_by_file.get(row["filename"], []), args)
            metrics.pop("region")
            record = {
                "filename": row["filename"], "label_name": row.get("label_name", ""),
                "label_index": int(row["label_index"]), "split": row.get("split", ""),
                "image_path": str(image_path), "image_width": width, "image_height": height,
                **metrics,
            }
            records.append(record)
            checkpoint.write(json.dumps(record) + "\n")
            processed += 1
            if processed % max(1, args.checkpoint_every) == 0:
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
                elapsed = max(1e-6, time.monotonic() - started)
                rate = processed / elapsed
                eta = (len(remaining) - processed) / rate if rate else 0
                print(f"  {len(records)}/{len(rows)} citra | {rate:.1f} img/s | sisa ±{eta:.0f}s", flush=True)

    order = {row["filename"]: index for index, row in enumerate(rows)}
    records.sort(key=lambda record: order.get(record["filename"], 10**9))
    with final_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records)
    complete = len(records) == len(rows) and not missing
    summary["complete"] = complete
    summary["n_requested"] = len(rows)
    summary["n_processed"] = len(records)
    summary["missing_images"] = missing
    summary["provenance"] = config
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if missing:
        print(f"⚠️ {len(missing)} foto tidak ditemukan (contoh: {missing[:3]}). "
              "Perbaiki --image-dir, lalu jalankan ulang untuk melanjutkan.", flush=True)
    if complete and args.examples > 0:
        examples = select_examples(records, args.examples)
        written = draw_examples(examples, args.output_dir / "visual_examples", circles_by_file, args)
        print("Contoh visual:", ", ".join(str(path) for path in written))
    elif not complete:
        print("⏳ Belum lengkap. Jalankan ulang Step 32; checkpoint parsial akan dilanjutkan.", flush=True)
    print(json.dumps({key: summary[key] for key in (
        "n_images", "face_detection_rate_pct", "mean_face_area_pct",
        "mean_inside_face_pct", "mean_inside_skin_pct", "mean_inside_union_pct",
        "images_fully_inside_union", "images_below_90pct", "images_below_50pct",
        "images_zero_coverage", "complete", "n_processed", "n_requested")}, indent=2))
    print("CSV:", final_csv)
    print("Ringkasan:", summary_path)


if __name__ == "__main__":
    main()
