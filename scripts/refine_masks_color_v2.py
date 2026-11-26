#!/usr/bin/env python3
"""Generate the Step 34 V2 candidate masks without training.

Example:
  python scripts/refine_masks_color_v2.py \
    --image-dir /path/to/original_images \
    --annotations /path/to/Acne04-v2_annotations.json \
    --output-dir /path/to/v2_masks_run

The input folder should contain original images whose filenames occur in the
annotation JSON. Images are letterboxed to 512x512 and saved as JPEG quality
95 before mask refinement, matching the Step 31b model-input image format.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

try:  # Supports both `python scripts/...py` and package imports.
    from . import face_skin_utils, refine_masks_color as v1
except ImportError:
    import face_skin_utils
    import refine_masks_color as v1

IMAGE_SUFFIXES = {".jpg", ".jpeg"}
TARGET_SIZE = 512
MIN_VALID_PIXELS = 50
SKIN_CR, SKIN_CB, SKIN_V_MIN, SKIN_MORPH = (133, 180), (77, 133), 30, 5


def configure_step31b_hsv_defaults() -> None:
    """Mirror the V1 CLI HSV defaults used by Step 31b (no HSV overrides)."""
    v1.RED_LOWER_1 = np.array([0, 40, 50])
    v1.RED_UPPER_1 = np.array([12, 255, 255])
    v1.RED_LOWER_2 = np.array([155, 40, 50])
    v1.RED_UPPER_2 = np.array([180, 255, 255])
    v1.PUS_LOWER_WHITE = np.array([0, 0, 180])
    v1.PUS_UPPER_WHITE = np.array([180, 40, 255])
    v1.PUS_LOWER_YELLOW = np.array([15, 10, 160])
    v1.PUS_UPPER_YELLOW = np.array([40, 90, 255])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def letterbox_content_mask(crop_box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = crop_box
    crop_w, crop_h = max(1, right - left), max(1, bottom - top)
    scale = min(TARGET_SIZE / crop_w, TARGET_SIZE / crop_h)
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    x0, y0 = (TARGET_SIZE - resized_w) // 2, (TARGET_SIZE - resized_h) // 2
    mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    mask[y0:y0 + resized_h, x0:x0 + resized_w] = 255
    return mask


def robust_threshold(samples: np.ndarray, sigma: float) -> float:
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples.astype(np.float32) - median)))
    return median + sigma * max(1.0, 1.4826 * mad)


def hsv_candidates(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, v1.RED_LOWER_1, v1.RED_UPPER_1),
        cv2.inRange(hsv, v1.RED_LOWER_2, v1.RED_UPPER_2),
    )
    white = cv2.inRange(hsv, v1.PUS_LOWER_WHITE, v1.PUS_UPPER_WHITE)
    yellow = cv2.inRange(hsv, v1.PUS_LOWER_YELLOW, v1.PUS_UPPER_YELLOW)
    return cv2.bitwise_or(red, cv2.bitwise_or(white, yellow))


def refine_annotation(
    lab_a: np.ndarray,
    hsv_mask: np.ndarray,
    circle: np.ndarray,
    outer: np.ndarray,
    circle_union: np.ndarray,
    content: np.ndarray,
    skin: np.ndarray,
    *,
    sigma: float,
    min_candidate_ratio: float,
    morph_kernel: int,
) -> tuple[np.ndarray, dict]:
    ring = (outer > 0) & (circle == 0) & (circle_union == 0) & (content > 0)
    skin_samples = lab_a[ring & (skin > 0)]
    samples = skin_samples if skin_samples.size >= MIN_VALID_PIXELS else lab_a[ring]
    baseline = "annulus_skin" if skin_samples.size >= MIN_VALID_PIXELS else "annulus_without_skin"
    circle_area = int(np.count_nonzero(circle))

    if samples.size < MIN_VALID_PIXELS:
        return circle.copy(), {
            "baseline_source": baseline,
            "annulus_pixels": int(ring.sum()),
            "sample_pixels": int(samples.size),
            "threshold_lab_a": "",
            "candidate_area_px": 0,
            "candidate_ratio": 0.0,
            "final_area_px": circle_area,
            "retained_area_ratio": 1.0,
            "fallback": 1,
            "fallback_reason": "annulus_has_too_few_valid_pixels",
        }

    threshold = robust_threshold(samples, sigma)
    candidate = ((lab_a > threshold) | (hsv_mask > 0)).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(candidate, circle)
    candidate = cv2.bitwise_and(candidate, content)
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        candidate = cv2.bitwise_and(candidate, circle)
        candidate = cv2.bitwise_and(candidate, content)

    candidate_area = int(np.count_nonzero(candidate))
    ratio = candidate_area / max(circle_area, 1)
    fallback = ratio < min_candidate_ratio
    final = circle.copy() if fallback else candidate
    final_area = int(np.count_nonzero(final))
    return final, {
        "baseline_source": baseline,
        "annulus_pixels": int(ring.sum()),
        "sample_pixels": int(samples.size),
        "threshold_lab_a": f"{threshold:.6f}",
        "candidate_area_px": candidate_area,
        "candidate_ratio": f"{ratio:.8f}",
        "final_area_px": final_area,
        "retained_area_ratio": f"{final_area / max(circle_area, 1):.8f}",
        "fallback": int(fallback),
        "fallback_reason": "candidate_below_min_ratio" if fallback else "",
    }


def self_test() -> None:
    a = np.full((64, 64), 140, dtype=np.uint8)
    cv2.circle(a, (32, 32), 3, 200, -1)
    hsv = np.zeros_like(a)
    circle = np.zeros_like(a)
    outer = np.zeros_like(a)
    cv2.circle(circle, (32, 32), 7, 255, -1)
    cv2.circle(outer, (32, 32), 15, 255, -1)
    content = np.full_like(a, 255)
    skin = content.copy()
    result, info = refine_annotation(
        a, hsv, circle, outer, circle, content, skin,
        sigma=1.2, min_candidate_ratio=0.02, morph_kernel=1,
    )
    again, _ = refine_annotation(
        a, hsv, circle, outer, circle, content, skin,
        sigma=1.2, min_candidate_ratio=0.02, morph_kernel=1,
    )
    assert info["fallback"] == 0 and np.array_equal(result, again)
    assert np.all(result[circle == 0] == 0) and not np.array_equal(result, circle)
    no_ring, fallback = refine_annotation(
        a, hsv, circle, outer, content, content, skin,
        sigma=1.2, min_candidate_ratio=0.02, morph_kernel=1,
    )
    assert fallback["fallback"] == 1 and np.array_equal(no_ring, circle)
    print("PASS self-test: deterministic, circle-confined, fallback behavior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, help="Folder of original annotated photos")
    parser.add_argument("--annotations", type=Path, help="ACNE04-v2_annotations.json")
    parser.add_argument("--output-dir", type=Path, help="New output folder; existing folders are refused")
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--annulus-radius-factor", type=float, default=2.0)
    parser.add_argument("--min-candidate-ratio", type=float, default=0.02)
    parser.add_argument("--morph-kernel", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.image_dir or not args.annotations or not args.output_dir:
        raise SystemExit("Provide --image-dir, --annotations, and --output-dir (or use --self-test).")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {args.image_dir}")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")
    if args.sigma <= 0 or args.annulus_radius_factor <= 1 or not 0 < args.min_candidate_ratio <= 1:
        raise ValueError("Require sigma > 0, annulus-radius-factor > 1, and min-candidate-ratio in (0, 1].")
    if args.morph_kernel < 1 or (args.morph_kernel > 1 and args.morph_kernel % 2 == 0):
        raise ValueError("Morphology kernel must be 1 or a positive odd integer.")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output folder: {args.output_dir}")

    configure_step31b_hsv_defaults()
    circles_by_file, sizes = v1.load_json_circle_annotations(args.annotations)
    all_image_paths = sorted(
        p for p in args.image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    image_paths = [p for p in all_image_paths if p.name in circles_by_file]
    skipped = [p for p in all_image_paths if p.name not in circles_by_file]
    if not image_paths:
        raise ValueError("No annotated images found in --image-dir.")

    args.output_dir.mkdir(parents=True)
    image_out = args.output_dir / "preprocessed_images"
    mask_out = args.output_dir / "v2_masks"
    overlay_out = args.output_dir / "overlays"
    image_out.mkdir(); mask_out.mkdir(); overlay_out.mkdir()
    image_rows, annotation_rows = [], []
    for index, image_path in enumerate(image_paths, 1):
        fname = image_path.name
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode image: {image_path}")
        width, height = sizes[fname]
        if (bgr.shape[1], bgr.shape[0]) != (width, height):
            raise ValueError(f"Dimensions disagree with annotations for {fname}: {(bgr.shape[1], bgr.shape[0])} != {(width, height)}")

        crop = v1.center_crop_box(width, height, 1.0)
        left, top, right, bottom = crop
        resized = v1.resize_to_square(bgr[top:bottom, left:right], TARGET_SIZE, "letterbox")
        input_path = image_out / fname
        if not cv2.imwrite(str(input_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write preprocessed image: {input_path}")
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
            raise ValueError(f"Invalid preprocessed image: {input_path}")

        annotations = circles_by_file[fname]
        circle_union = v1.build_circle_mask(annotations, width, height, crop, TARGET_SIZE, "letterbox")
        content = letterbox_content_mask(crop)
        lab_a = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 1]
        hsv = hsv_candidates(image)
        skin = face_skin_utils.skin_map(image, None, SKIN_CR, SKIN_CB, SKIN_V_MIN, SKIN_MORPH)
        final_union = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        this_annotations = []

        for ann_index, source_circle in enumerate(annotations):
            circle = v1.build_circle_mask([source_circle], width, height, crop, TARGET_SIZE, "letterbox")
            outer = v1.build_circle_mask(
                [(source_circle[0], source_circle[1], source_circle[2] * args.annulus_radius_factor)],
                width, height, crop, TARGET_SIZE, "letterbox",
            )
            final, info = refine_annotation(
                lab_a, hsv, circle, outer, circle_union, content, skin,
                sigma=args.sigma, min_candidate_ratio=args.min_candidate_ratio,
                morph_kernel=args.morph_kernel,
            )
            final_union = cv2.bitwise_or(final_union, final)
            record = {"filename": fname, "label_name": v1.level_from_filename(fname),
                      "annotation_index": ann_index, "circle_area_px": int(np.count_nonzero(circle)), **info}
            annotation_rows.append(record); this_annotations.append(record)

        if np.any((final_union > 0) & (circle_union == 0)):
            raise AssertionError(f"V2 mask escaped circle union: {fname}")
        stem = image_path.stem
        mask_path = mask_out / f"{stem}.png"
        overlay_path = overlay_out / f"{stem}_v2_overlay.png"
        if not cv2.imwrite(str(mask_path), final_union):
            raise OSError(f"Could not write mask: {mask_path}")
        color = np.zeros_like(image); color[final_union > 0] = (0, 220, 0)
        overlay = cv2.addWeighted(image, 0.65, color, 0.35, 0)
        if not cv2.imwrite(str(overlay_path), overlay):
            raise OSError(f"Could not write overlay: {overlay_path}")
        circle_area = int(np.count_nonzero(circle_union))
        final_area = int(np.count_nonzero(final_union))
        image_rows.append({
            "filename": fname, "label_name": v1.level_from_filename(fname),
            "annotations": len(this_annotations),
            "fallback_annotations": sum(int(r["fallback"]) for r in this_annotations),
            "lost_annotations": sum(int(r["final_area_px"]) == 0 for r in this_annotations),
            "circle_union_area_px": circle_area, "v2_area_px": final_area,
            "unchanged_vs_circle_union": int(np.array_equal(final_union, circle_union)),
            "v2_circle_proxy_iou": f"{final_area / max(circle_area, 1):.8f}",
            "mask_path": str(mask_path.resolve()), "overlay_path": str(overlay_path.resolve()),
        })
        if index % 100 == 0:
            print(f"processed {index}/{len(image_paths)}", flush=True)

    for filename, rows in (("per_image.csv", image_rows), ("per_annotation.csv", annotation_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    config = {
        "candidate": "local Lab median/MAD plus V1 CLI-default HSV candidates",
        "target_size": TARGET_SIZE, "resize_mode": "letterbox", "crop_ratio": 1.0,
        "jpeg_quality": 95, "sigma": args.sigma,
        "annulus_radius_factor": args.annulus_radius_factor,
        "min_valid_pixels": MIN_VALID_PIXELS,
        "min_candidate_ratio": args.min_candidate_ratio,
        "morph_kernel": args.morph_kernel,
        "skin_map": {"cr": list(SKIN_CR), "cb": list(SKIN_CB), "v_min": SKIN_V_MIN,
                      "morph": SKIN_MORPH, "use": "annulus sampling only"},
        "fallback_note": "Annotations below the candidate-area threshold retain the original circle; inspect fallback_annotations and unchanged_vs_circle_union.",
        "warning": "Exploratory pseudo-mask candidate, not clinical ground truth. Do not tune with the test set.",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "input_dir": str(args.image_dir.resolve()),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations),
        "v1_script_sha256": sha256(Path(v1.__file__).resolve()),
        "v2_script_sha256": sha256(Path(__file__).resolve()),
        "images_processed": len(image_rows),
        "unannotated_images_skipped": len(skipped),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    unchanged = [r["filename"] for r in image_rows if r["unchanged_vs_circle_union"]]
    print(f"PASS: {len(image_rows)} images processed; {sum(r['fallback_annotations'] for r in image_rows)} annotation fallbacks.")
    print("Masks exactly unchanged from circle union:", ", ".join(unchanged) if unchanged else "none")
    print("Output:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
