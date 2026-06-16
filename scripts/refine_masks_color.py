#!/usr/bin/env python3
"""
Refine segmentation masks menggunakan deteksi warna merah (HSV).

Masalah mask lingkaran lama:
- Dibuat murni dari (cx, cy, radius) geometris
- Tidak mengikuti batas visual lesi
- Memasukkan piksel kulit normal di sekitar lesi

Pendekatan baru:
- Gunakan circle sebagai ROI
- Di dalam ROI, deteksi piksel merah di HSV
- Intersection = refined mask yang lebih natural

Output:
- circle_masks/<filename>.png        : mask lingkaran asli
- refined_masks/<filename>.png       : mask yang sudah diperhalus
- preprocessed_images/<filename>.jpg : gambar preprocessed (opsional, --apply-preprocessing)
- visualizations/<filename>.png      : overlay mask langsung di atas gambar asli
- refined_manifest.csv               : manifest baru siap untuk training

Catatan preprocessing:
  Mask SELALU digenerate dari gambar original (warna asli).
  Preprocessing (equalize, denoise, sharpen) hanya diterapkan untuk
  menyimpan gambar input training, bukan untuk deteksi mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

LABEL_TO_INDEX = {"levle0": 0, "levle1": 1, "levle2": 2, "levle3": 3}


# ---------------------------------------------------------------------------
# Range warna merah di HSV — ada DUA range karena hue wrap di 0/180
# ---------------------------------------------------------------------------
RED_LOWER_1 = np.array([0,   20,  40])
RED_UPPER_1 = np.array([20, 255, 255])

RED_LOWER_2 = np.array([145,  20,  40])
RED_UPPER_2 = np.array([180, 255, 255])

# ---------------------------------------------------------------------------
# Range warna pus (nanah) di HSV
# Pustule: kepala putih-kuning, S rendah, V tinggi
# ---------------------------------------------------------------------------
PUS_LOWER_WHITE  = np.array([0,   0,  180])
PUS_UPPER_WHITE  = np.array([180, 40, 255])

PUS_LOWER_YELLOW = np.array([15,  10, 160])
PUS_UPPER_YELLOW = np.array([40,  90, 255])


def detect_lesion_mask(
    bgr_image: np.ndarray,
    circle_mask: np.ndarray,
    lab_sigma: float = 1.2,
) -> np.ndarray:
    """
    Deteksi lesi dengan dua metode yang digabungkan (OR):

    1. Lab relative redness (PRIMER)
       Channel 'a' pada Lab color space merepresentasikan sumbu merah-hijau.
       Baseline warna kulit diambil dari piksel DI LUAR circle mask pada
       gambar yang sama, sehingga threshold menyesuaikan otomatis dengan
       tone kulit — merah gelap pada kulit cokelat tetap tertangkap.

    2. HSV absolute (SEKUNDER)
       Untuk lesi merah cerah dan pus (kepala pustule putih/kuning)
       yang tidak tertangkap Lab.
    """
    # ── Lab: relative redness ────────────────────────────────────────────
    lab       = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
    a_channel = lab[:, :, 1].astype(np.float32)  # >128 = merah, <128 = hijau

    # Kulit di luar semua circle = baseline tone kulit gambar ini
    skin_pixels = a_channel[circle_mask == 0]
    if skin_pixels.size > 50:
        skin_mean = float(np.mean(skin_pixels))
        skin_std  = max(1.0, float(np.std(skin_pixels)))
    else:
        skin_mean, skin_std = 128.0, 5.0

    # Piksel yang jauh lebih merah dari kulit sekitar
    threshold  = skin_mean + lab_sigma * skin_std
    lab_red    = ((a_channel > threshold) * 255).astype(np.uint8)

    # ── HSV: absolute (merah cerah + pus) ────────────────────────────────
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    red2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    hsv_red = cv2.bitwise_or(red1, red2)

    pus_white  = cv2.inRange(hsv, PUS_LOWER_WHITE,  PUS_UPPER_WHITE)
    pus_yellow = cv2.inRange(hsv, PUS_LOWER_YELLOW, PUS_UPPER_YELLOW)
    pus = cv2.bitwise_or(pus_white, pus_yellow)

    # Gabungkan semua sinyal
    return cv2.bitwise_or(lab_red, cv2.bitwise_or(hsv_red, pus))


def build_circle_mask(
    circles: list[tuple[float, float, float]],
    src_w: int,
    src_h: int,
    crop_box: tuple[int, int, int, int],
    target_size: int,
) -> np.ndarray:
    left, top, right, bottom = crop_box
    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    scale_x = target_size / crop_w
    scale_y = target_size / crop_h
    r_scale = 0.5 * (scale_x + scale_y)

    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    for cx, cy, radius in circles:
        tx = int((cx - left) * scale_x)
        ty = int((cy - top) * scale_y)
        tr = max(1, int(radius * r_scale))
        cv2.circle(mask, (tx, ty), tr, 255, -1)
    return mask


def refine_mask_with_red(
    bgr_image: np.ndarray,
    circle_mask: np.ndarray,
    morph_kernel_size: int = 5,
    min_pixel_ratio: float = 0.02,
    lab_sigma: float = 1.2,
) -> np.ndarray:
    """
    Interseksi circle mask dengan deteksi piksel lesi (merah + pus).
    Fallback ke circle mask jika area yang terdeteksi terlalu kecil.
    """
    red_mask = detect_lesion_mask(bgr_image, circle_mask, lab_sigma)
    refined  = cv2.bitwise_and(red_mask, circle_mask)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN,  kernel)

    circle_area  = int(np.sum(circle_mask > 0))
    refined_area = int(np.sum(refined > 0))

    if circle_area > 0 and (refined_area / circle_area) < min_pixel_ratio:
        return circle_mask.copy(), True   # fallback

    return refined, False


def apply_preprocessing(
    bgr: np.ndarray,
    denoise_size: int = 3,
    sharpen_radius: float = 1.6,
    sharpen_percent: int = 140,
) -> np.ndarray:
    """
    Preprocessing sama persis dengan preprocess_hybrid_dataset.py:
    1. Histogram equalization per channel BGR
    2. Median filter (denoise)
    3. Unsharp mask (sharpen)

    TIDAK dipakai untuk deteksi mask — hanya untuk menyimpan
    gambar input training yang konsisten dengan pipeline lama.
    """
    # Equalize per channel (setara ImageOps.equalize PIL)
    b, g, r = cv2.split(bgr)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    equalized = cv2.merge([b, g, r])

    # Median filter
    denoised = cv2.medianBlur(equalized, denoise_size)

    # Unsharp mask: sharpened = original + amount * (original - blurred)
    blurred  = cv2.GaussianBlur(denoised, (0, 0), sharpen_radius)
    amount   = sharpen_percent / 100.0
    sharpened = cv2.addWeighted(denoised, 1.0 + amount, blurred, -amount, 0)

    return sharpened


def center_crop_box(w: int, h: int, crop_ratio: float) -> tuple[int, int, int, int]:
    cw = max(1, int(w * crop_ratio))
    ch = max(1, int(h * crop_ratio))
    left = (w - cw) // 2
    top  = (h - ch) // 2
    return left, top, left + cw, top + ch


def overlay_mask_on_image(
    bgr: np.ndarray,
    mask: np.ndarray,
    fill_color_bgr: tuple[int, int, int],
    contour_color_bgr: tuple[int, int, int],
    alpha: float = 0.40,
    contour_thickness: int = 2,
) -> np.ndarray:
    """
    Overlay mask di atas gambar:
    - Area mask    : fill semi-transparan
    - Batas kontur : garis solid tebal
    """
    out = bgr.copy()

    # Fill semi-transparan
    colored = np.zeros_like(bgr)
    colored[mask > 0] = fill_color_bgr
    out = cv2.addWeighted(out, 1.0, colored, alpha, 0)

    # Kontur tegas di tepi mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, contour_color_bgr, contour_thickness)

    return out


def make_visualization(
    bgr_resized: np.ndarray,
    circle_mask: np.ndarray,
    refined_mask: np.ndarray,
    fname: str,
    level: str,
    n_circles: int,
    circle_area: int,
    refined_area: int,
    is_fallback: bool,
) -> np.ndarray:
    """
    3-panel horizontal overlay langsung di atas gambar asli:
      [1] Original
      [2] Circle mask (overlay merah)
      [3] Refined mask (overlay hijau)

    Setiap panel dilengkapi label dan statistik.
    """
    H, W = bgr_resized.shape[:2]

    # Panel 1 – gambar original bersih
    p1 = bgr_resized.copy()

    # Panel 2 – circle mask overlay (merah transparan + kontur merah terang)
    p2 = overlay_mask_on_image(
        bgr_resized, circle_mask,
        fill_color_bgr=(0, 0, 200),
        contour_color_bgr=(0, 0, 255),
        alpha=0.35,
    )

    # Panel 3 – refined mask overlay (hijau transparan + kontur hijau terang)
    p3 = overlay_mask_on_image(
        bgr_resized, refined_mask,
        fill_color_bgr=(0, 180, 0),
        contour_color_bgr=(0, 255, 0),
        alpha=0.40,
    )

    def put_header(img: np.ndarray, lines: list[str], bg_color=(30, 30, 30)) -> np.ndarray:
        pad_h   = 26 * len(lines) + 6
        header  = np.full((pad_h, W, 3), bg_color, dtype=np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(header, line, (6, 20 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
        return np.vstack([header, img])

    p1 = put_header(p1, [f"[1] Original   |  {fname}", f"Level: {level}  |  Lesi: {n_circles}"])
    p2 = put_header(p2, ["[2] Circle mask (lama)", f"Area: {circle_area} px"],
                    bg_color=(50, 10, 10))
    fb_tag = "  [FALLBACK]" if is_fallback else ""
    p3 = put_header(p3, [f"[3] Refined mask{fb_tag}", f"Area: {refined_area} px"],
                    bg_color=(10, 40, 10))

    return np.hstack([p1, p2, p3])


def level_from_filename(fname: str) -> str:
    """Ekstrak level dari nama file, misal 'levle2_034.jpg' → 'levle2'."""
    stem = Path(fname).stem
    parts = stem.split("_")
    return parts[0] if parts else "unknown"


def assign_splits(
    fnames: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Stratified split per level → return {filename: 'train'|'val'|'test'}."""
    by_level: dict[str, list[str]] = defaultdict(list)
    for fname in fnames:
        by_level[level_from_filename(fname)].append(fname)

    rng = random.Random(seed)
    splits: dict[str, str] = {}
    for level_fnames in by_level.values():
        rng.shuffle(level_fnames)
        n = len(level_fnames)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        for i, fname in enumerate(level_fnames):
            if i < n_train:
                splits[fname] = "train"
            elif i < n_train + n_val:
                splits[fname] = "val"
            else:
                splits[fname] = "test"
    return splits


def pick_vis_samples(
    image_paths: list[Path],
    circles_by_file: dict,
    vis_per_level: int,
    seed: int = 42,
) -> set[str]:
    """
    Pilih sampel visualisasi secara merata dari setiap level.
    Return set of filename strings.
    """
    by_level: dict[str, list[Path]] = defaultdict(list)
    for p in image_paths:
        if p.name in circles_by_file:
            lv = level_from_filename(p.name)
            by_level[lv].append(p)

    rng = random.Random(seed)
    selected: set[str] = set()
    for lv, paths in sorted(by_level.items()):
        rng.shuffle(paths)
        for p in paths[:vis_per_level]:
            selected.add(p.name)
        print(f"  Vis sampling level {lv}: {min(len(paths), vis_per_level)} gambar dipilih dari {len(paths)}")

    return selected


def process_dataset(
    image_dir: Path,
    annotations_json: Path,
    output_dir: Path,
    target_size: int,
    crop_ratio: float,
    morph_kernel: int,
    min_pixel_ratio: float,
    max_images: int,
    vis_per_level: int,
    seed: int,
    args_lab_sigma: float = 1.2,
    do_preprocessing: bool = False,
    denoise_size: int = 3,
    sharpen_radius: float = 1.6,
    sharpen_percent: int = 140,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    circle_dir  = output_dir / "circle_masks"
    refined_dir = output_dir / "refined_masks"
    vis_dir     = output_dir / "visualizations"
    circle_dir.mkdir(exist_ok=True)
    refined_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    preproc_dir = None
    if do_preprocessing:
        preproc_dir = output_dir / "preprocessed_images"
        preproc_dir.mkdir(exist_ok=True)

    print(f"Membaca anotasi dari {annotations_json}")
    with open(annotations_json) as f:
        ann_data = json.load(f)

    circles_by_file: dict[str, list[tuple[float, float, float]]] = {}
    size_by_file:    dict[str, tuple[int, int]] = {}

    images_meta = {img["id"]: img for img in ann_data["images"]}
    for ann in ann_data["annotations"]:
        img_meta = images_meta[ann["image_id"]]
        fname    = img_meta["file_name"]
        cx, cy   = ann["coordinates"]
        r        = ann["radius"]
        circles_by_file.setdefault(fname, []).append((cx, cy, r))
        size_by_file[fname] = (img_meta["width"], img_meta["height"])

    # Kumpulkan path gambar
    all_paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"):
        all_paths.extend(image_dir.rglob(ext))
    all_paths.sort()

    # Stratified sampling per level — HANYA dari gambar yang punya anotasi
    if max_images > 0:
        n_levels  = len(LABEL_TO_INDEX)
        per_level = max_images // n_levels
        by_level: dict[str, list[Path]] = defaultdict(list)
        for p in all_paths:
            lv = level_from_filename(p.name)
            if lv in LABEL_TO_INDEX and p.name in circles_by_file:
                by_level[lv].append(p)

        rng = random.Random(seed)
        image_paths = []
        for lv in sorted(by_level):
            pool = by_level[lv][:]
            rng.shuffle(pool)
            selected = pool[:per_level]
            image_paths.extend(selected)
            print(f"  Sampling level {lv}: {len(selected)} dari {len(pool)} gambar beranotasi")
        image_paths.sort()
    else:
        image_paths = all_paths

    print(f"\nTotal gambar ditemukan : {len(all_paths)}")
    print(f"Target size            : {target_size}x{target_size}")
    print(f"Crop ratio             : {crop_ratio}")
    print(f"Vis per level          : {vis_per_level}\n")

    # Assign train/val/test split secara stratified per level
    annotated_fnames = [p.name for p in image_paths if p.name in circles_by_file]
    split_map = assign_splits(annotated_fnames, train_ratio, val_ratio, seed)

    # Tentukan gambar mana yang perlu divisualisasi (merata per level)
    vis_set = pick_vis_samples(image_paths, circles_by_file, vis_per_level, seed)
    print(f"\nTotal gambar untuk visualisasi: {len(vis_set)}\n")

    manifest_rows: list[dict] = []
    stats = {"fallback": 0, "refined": 0, "no_annotation": 0}

    for idx, img_path in enumerate(image_paths):
        fname = img_path.name

        if fname not in circles_by_file:
            stats["no_annotation"] += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [SKIP] Tidak bisa baca: {img_path}")
            continue

        src_w, src_h = size_by_file[fname]
        circles      = circles_by_file[fname]
        crop_box     = center_crop_box(src_w, src_h, crop_ratio)

        left, top, right, bottom = crop_box
        bgr_cropped  = bgr[top:bottom, left:right]
        bgr_resized  = cv2.resize(bgr_cropped, (target_size, target_size),
                                   interpolation=cv2.INTER_LANCZOS4)

        # Mask digenerate dari gambar ORIGINAL (warna asli, belum di-preprocess)
        circle_mask           = build_circle_mask(circles, src_w, src_h, crop_box, target_size)
        refined_mask, fallback = refine_mask_with_red(bgr_resized, circle_mask,
                                                       morph_kernel, min_pixel_ratio,
                                                       args_lab_sigma)

        # Simpan gambar preprocessed untuk input training (opsional)
        preproc_path = None
        if do_preprocessing and preproc_dir is not None:
            bgr_preprocessed = apply_preprocessing(bgr_resized, denoise_size,
                                                    sharpen_radius, sharpen_percent)
            preproc_path = preproc_dir / fname
            cv2.imwrite(str(preproc_path), bgr_preprocessed,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        if fallback:
            stats["fallback"] += 1
        else:
            stats["refined"] += 1

        c_area = int(np.sum(circle_mask  > 0))
        r_area = int(np.sum(refined_mask > 0))
        stem   = Path(fname).stem
        level  = level_from_filename(fname)

        # Simpan mask
        circle_path  = circle_dir  / f"{stem}.png"
        refined_path = refined_dir / f"{stem}.png"
        cv2.imwrite(str(circle_path),  circle_mask)
        cv2.imwrite(str(refined_path), refined_mask)

        # Simpan visualisasi overlay untuk gambar yang terpilih
        if fname in vis_set:
            vis_img = make_visualization(
                bgr_resized, circle_mask, refined_mask,
                fname=fname, level=level, n_circles=len(circles),
                circle_area=c_area, refined_area=r_area, is_fallback=fallback,
            )
            cv2.imwrite(str(vis_dir / f"{level}_{stem}.png"), vis_img)

        manifest_rows.append({
            "filename":          fname,
            "label_name":        level,
            "label_index":       LABEL_TO_INDEX.get(level, -1),
            "split":             split_map.get(fname, "train"),
            "image_path":        str(img_path),
            "preprocessed_path": str(preproc_path) if preproc_path else "",
            "refined_mask":      str(refined_path),
            "circle_mask":       str(circle_path),
            "image_size":        target_size,
            "n_circles":         len(circles),
            "circle_area":       c_area,
            "refined_area":      r_area,
            "fallback":          int(fallback),
        })

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(image_paths)} selesai...")

    # Tulis manifest
    manifest_path = output_dir / "refined_manifest.csv"
    if manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    print("\n=== Selesai ===")
    print(f"  Gambar diproses      : {len(manifest_rows)}")
    print(f"  Refined berhasil     : {stats['refined']}")
    print(f"  Fallback ke circle   : {stats['fallback']}  (merah terdeteksi < {min_pixel_ratio*100:.0f}% area circle)")
    print(f"  Tanpa anotasi        : {stats['no_annotation']}")
    print(f"  Manifest             : {manifest_path}")
    print(f"  Visualisasi          : {vis_dir}/")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Refine segmentation masks menggunakan deteksi warna merah (HSV)."
    )
    parser.add_argument(
        "--image-dir", type=Path,
        default=root.parent / "datasetacne04" / "acne_1024" / "all_1024",
        help="Folder gambar original (bukan preprocessed)",
    )
    parser.add_argument(
        "--annotations-json", type=Path,
        default=root.parent / "acne04v2" / "Acne04-v2_annotations.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "refined_masks_output",
    )
    parser.add_argument("--target-size",     type=int,   default=320)
    parser.add_argument("--crop-ratio",      type=float, default=1.0)
    parser.add_argument("--morph-kernel",    type=int,   default=5,
                        help="Ukuran kernel morphological closing/opening")
    parser.add_argument("--min-pixel-ratio", type=float, default=0.02,
                        help="Ratio minimum piksel merah dalam circle sebelum fallback ke circle mask")
    parser.add_argument("--lab-sigma",       type=float, default=1.2,
                        help="Sensitivitas Lab relative redness: lebih kecil = lebih banyak piksel tertangkap")
    parser.add_argument("--max-images",      type=int,   default=0,
                        help="Total gambar yang diproses dengan stratified sampling per kelas (0 = semua). "
                             "Misal --max-images 400 → 100 gambar per level")
    parser.add_argument("--vis-per-level",   type=int,   default=5,
                        help="Jumlah gambar visualisasi per level (levle0/1/2/3)")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--train-ratio",     type=float, default=0.7)
    parser.add_argument("--val-ratio",       type=float, default=0.15)

    # Preprocessing gambar untuk training
    parser.add_argument("--apply-preprocessing", action="store_true",
                        help="Simpan juga gambar preprocessed (equalize+denoise+sharpen) untuk input training")
    parser.add_argument("--denoise-size",    type=int,   default=3,
                        help="Ukuran kernel median filter")
    parser.add_argument("--sharpen-radius",  type=float, default=1.6,
                        help="Radius unsharp mask")
    parser.add_argument("--sharpen-percent", type=int,   default=140,
                        help="Kekuatan sharpen dalam persen")

    # Tuning HSV merah
    parser.add_argument("--red-h-low1",  type=int, default=0)
    parser.add_argument("--red-h-high1", type=int, default=12)
    parser.add_argument("--red-h-low2",  type=int, default=155)
    parser.add_argument("--red-h-high2", type=int, default=180)
    parser.add_argument("--red-s-min",   type=int, default=40)
    parser.add_argument("--red-v-min",   type=int, default=50)

    # Tuning HSV pus (nanah) — putih/kuning di kepala pustule
    parser.add_argument("--pus-s-max-white",   type=int, default=40,
                        help="Saturasi maksimum untuk pus putih")
    parser.add_argument("--pus-v-min-white",   type=int, default=180,
                        help="Brightness minimum untuk pus putih")
    parser.add_argument("--pus-h-low-yellow",  type=int, default=15,
                        help="Hue bawah untuk pus kuning-cream")
    parser.add_argument("--pus-h-high-yellow", type=int, default=40,
                        help="Hue atas untuk pus kuning-cream")
    parser.add_argument("--pus-s-max-yellow",  type=int, default=90,
                        help="Saturasi maksimum untuk pus kuning")
    parser.add_argument("--pus-v-min-yellow",  type=int, default=160,
                        help="Brightness minimum untuk pus kuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global RED_LOWER_1, RED_UPPER_1, RED_LOWER_2, RED_UPPER_2
    global PUS_LOWER_WHITE, PUS_UPPER_WHITE, PUS_LOWER_YELLOW, PUS_UPPER_YELLOW

    RED_LOWER_1 = np.array([args.red_h_low1,  args.red_s_min, args.red_v_min])
    RED_UPPER_1 = np.array([args.red_h_high1, 255,            255           ])
    RED_LOWER_2 = np.array([args.red_h_low2,  args.red_s_min, args.red_v_min])
    RED_UPPER_2 = np.array([args.red_h_high2, 255,            255           ])

    PUS_LOWER_WHITE  = np.array([0,                      0,                    args.pus_v_min_white ])
    PUS_UPPER_WHITE  = np.array([180,                    args.pus_s_max_white,  255                 ])
    PUS_LOWER_YELLOW = np.array([args.pus_h_low_yellow,  10,                   args.pus_v_min_yellow])
    PUS_UPPER_YELLOW = np.array([args.pus_h_high_yellow, args.pus_s_max_yellow, 255                 ])

    print("=== Refine Mask: Lab Relative Redness + HSV + Pus ===")
    print(f"Lab sigma      : {args.lab_sigma}  (threshold = mean_kulit + sigma * std_kulit)")
    print(f"Merah HSV 1    : H=[{args.red_h_low1}, {args.red_h_high1}]  S>={args.red_s_min}  V>={args.red_v_min}")
    print(f"Merah HSV 2    : H=[{args.red_h_low2}, {args.red_h_high2}]  S>={args.red_s_min}  V>={args.red_v_min}")
    print(f"Pus putih      : S<={args.pus_s_max_white}  V>={args.pus_v_min_white}")
    print(f"Pus kuning     : H=[{args.pus_h_low_yellow}, {args.pus_h_high_yellow}]  S<={args.pus_s_max_yellow}  V>={args.pus_v_min_yellow}")
    print()

    process_dataset(
        image_dir         = args.image_dir,
        annotations_json  = args.annotations_json,
        output_dir        = args.output_dir,
        target_size       = args.target_size,
        crop_ratio        = args.crop_ratio,
        morph_kernel      = args.morph_kernel,
        min_pixel_ratio   = args.min_pixel_ratio,
        max_images        = args.max_images,
        vis_per_level     = args.vis_per_level,
        seed              = args.seed,
        args_lab_sigma    = args.lab_sigma,
        do_preprocessing  = args.apply_preprocessing,
        denoise_size      = args.denoise_size,
        sharpen_radius    = args.sharpen_radius,
        sharpen_percent   = args.sharpen_percent,
        train_ratio       = args.train_ratio,
        val_ratio         = args.val_ratio,
    )


if __name__ == "__main__":
    main()
