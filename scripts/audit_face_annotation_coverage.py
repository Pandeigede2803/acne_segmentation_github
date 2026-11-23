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
from collections import Counter, defaultdict
from pathlib import Path

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


def expand_box(box, scale, width, height):
    """Expand a face box by scale, clamped to the image."""
    x, y, w, h = box
    extra_x, extra_y = int(round(w * (scale - 1) / 2)), int(round(h * (scale - 1) / 2))
    left, top = max(0, x - extra_x), max(0, y - extra_y)
    right, bottom = min(width, x + w + extra_x), min(height, y + h + extra_y)
    return left, top, max(1, right - left), max(1, bottom - top)


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


def detect_face(bgr, min_neighbors=5, min_size=30):
    """Detect the largest face using the Haar cascades shipped with OpenCV."""
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    boxes = []
    for name in ("haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml",
                 "haarcascade_profileface.xml"):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            continue
        for x, y, w, h in cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(min_size, min_size)):
            boxes.append((int(x), int(y), int(w), int(h)))
        if name == "haarcascade_profileface.xml":
            flipped = cv2.flip(gray, 1)
            for x, y, w, h in cascade.detectMultiScale(
                    flipped, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(min_size, min_size)):
                boxes.append((int(gray.shape[1] - x - w), int(y), int(w), int(h)))
    return max(boxes, key=lambda box: box[2] * box[3]) if boxes else None


def skin_map(bgr, face_box, cr_range, cb_range, v_min, morph_size):
    """YCrCb skin threshold, restricted to the expanded face when available."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, cr_range[0], cb_range[0]),
                       (255, cr_range[1], cb_range[1]))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_and(mask, cv2.inRange(hsv, (0, 0, v_min), (180, 255, 255)))
    limit = np.zeros(mask.shape, dtype=np.uint8)
    if face_box is not None:
        x, y, w, h = expand_box(face_box, 1.1, bgr.shape[1], bgr.shape[0])
        limit[y:y + h, x:x + w] = 255
    else:
        limit[:] = 255
    mask = cv2.bitwise_and(mask, limit)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def analyze_image(bgr, circles, args):
    """Return face/skin/annotation masks and per-image coverage values."""
    height, width = bgr.shape[:2]
    face_box = detect_face(bgr, args.min_neighbors, args.min_face_size)
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

    records = []
    missing = []
    for index, row in enumerate(rows, start=1):
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
        records.append({
            "filename": row["filename"], "label_name": row.get("label_name", ""),
            "label_index": int(row["label_index"]), "split": row.get("split", ""),
            "image_path": str(image_path), "image_width": width, "image_height": height,
            **metrics,
        })
        if index % 100 == 0:
            print(f"  {index}/{len(rows)} citra selesai", flush=True)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} foto asli tidak ditemukan (contoh: {missing[:3]}). "
            "Jalankan Step 5/6 atau berikan --image-dir.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "face_annotation_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records)
    summary["provenance"] = {
        "manifest": str(args.manifest), "manifest_sha256": file_hash(args.manifest),
        "annotations": str(args.annotations_json), "annotations_sha256": file_hash(args.annotations_json),
        "script_sha256": file_hash(__file__), "face_backend": "opencv_haar",
        "skin_thresholds": {"cr": [args.skin_cr_min, args.skin_cr_max],
                            "cb": [args.skin_cb_min, args.skin_cb_max],
                            "v_min": args.skin_v_min, "morph": args.skin_morph},
    }
    (args.output_dir / "face_annotation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    examples = select_examples(records, args.examples) if args.examples > 0 else []
    written = draw_examples(examples, args.output_dir / "visual_examples", circles_by_file, args)
    print(json.dumps({key: summary[key] for key in (
        "n_images", "face_detection_rate_pct", "mean_face_area_pct",
        "mean_inside_face_pct", "mean_inside_skin_pct", "mean_inside_union_pct",
        "images_fully_inside_union", "images_below_90pct", "images_below_50pct",
        "images_zero_coverage")}, indent=2))
    print("Contoh visual:", ", ".join(str(path) for path in written))
    print("CSV:", args.output_dir / "face_annotation_audit.csv")
    print("Ringkasan:", args.output_dir / "face_annotation_summary.json")


if __name__ == "__main__":
    main()
