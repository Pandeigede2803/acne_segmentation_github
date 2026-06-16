#!/usr/bin/env python3
"""
Unified ACNE04-v2 pipeline:

Preprocessing -> Dilated U-Net Encoder -> Bottleneck -> Spatial Attention
-> Segmentation Decoder -> Mask Lesi -> Fitur Morfologi + Fitur Global
-> AMFM -> Classification Head -> Severity Hayashi
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageFilter, ImageOps
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Butuh torch dan pillow di venv. Install dengan: python3 -m pip install torch pillow"
    ) from exc


LABEL_TO_INDEX = {
    "levle0": 0,
    "levle1": 1,
    "levle2": 2,
    "levle3": 3,
}


# ============================================================
# SECTION 1 - STRUKTUR DATA
# ------------------------------------------------------------
# Bagian ini menyiapkan bentuk data yang akan dibawa sepanjang
# pipeline: path gambar, label severity, ukuran gambar asli,
# anotasi lingkaran lesi, dan split train/val/test.
# ============================================================
@dataclass
class Sample:
    image_path: Path
    label_name: str
    label_index: int
    width: int
    height: int
    circles: list[tuple[float, float, float]]
    split: str


# ============================================================
# SECTION 2 - ARGUMEN EKSEKUSI
# ------------------------------------------------------------
# Semua parameter eksperimen diatur di sini:
# - lokasi dataset
# - jumlah epoch
# - batch size
# - device
# - ukuran input
# - rasio split
# - parameter preprocessing
# ============================================================
def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train Dilated U-Net + AMFM using ACNE04-v2 circular annotations."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=base_dir.parent / "datasetacne04" / "acne_1024" / "small_1024",
        help="Folder image ACNE04-v2.",
    )
    parser.add_argument(
        "--annotations-json",
        type=Path,
        default=base_dir.parent / "acne04v2" / "Acne04-v2_annotations.json",
        help="JSON anotasi sirkular ACNE04-v2.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--crop-ratio", type=float, default=1.0)
    parser.add_argument("--denoise-size", type=int, default=3)
    parser.add_argument("--sharpen-radius", type=float, default=1.6)
    parser.add_argument("--sharpen-percent", type=int, default=140)
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="Aktifkan augmentasi hanya pada data train.",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=15.0,
        help="Rotasi acak maksimum untuk augmentasi train. Contoh 15 = -15 s/d +15 derajat.",
    )
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.5,
        help="Probabilitas horizontal flip pada augmentasi train.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "baseline_runs" / "dilated_unet_acne04v2",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Batasi jumlah image untuk eksperimen cepat. 0 = pakai semua.",
    )
    return parser.parse_args()


# ============================================================
# SECTION 3 - UTILITAS UMUM
# ------------------------------------------------------------
# Fungsi-fungsi dasar untuk:
# - memastikan random seed konsisten
# - memilih device training (CPU / CUDA / MPS)
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# SECTION 4 - MEMBACA DATASET ACNE04-v2
# ------------------------------------------------------------
# Tujuan bagian ini:
# 1. membaca file JSON anotasi ACNE04-v2
# 2. menghubungkan image dengan anotasi lingkaran lesinya
# 3. mengambil label severity dari nama file
# 4. membagi data menjadi train / val / test secara stratified
# ============================================================
def load_acne04v2_samples(
    image_dir: Path,
    annotations_json: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    max_images: int,
) -> list[Sample]:
    # Tahap 1: baca semua metadata gambar dan anotasi lingkaran ACNE04-v2.
    if not image_dir.exists():
        raise FileNotFoundError(f"Folder image tidak ditemukan: {image_dir}")
    if not annotations_json.exists():
        raise FileNotFoundError(f"JSON anotasi tidak ditemukan: {annotations_json}")

    obj = json.loads(annotations_json.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in obj["images"]}
    grouped_circles: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for ann in obj["annotations"]:
        cx, cy = ann["coordinates"]
        radius = ann["radius"]
        grouped_circles[ann["image_id"]].append((float(cx), float(cy), float(radius)))

    raw_samples: list[Sample] = []
    for image_id, meta in images.items():
        file_name = meta["file_name"]
        label_name = Path(file_name).stem.split("_")[0]
        if label_name not in LABEL_TO_INDEX:
            continue
        image_path = image_dir / file_name
        if not image_path.exists():
            continue
        raw_samples.append(
            Sample(
                image_path=image_path,
                label_name=label_name,
                label_index=LABEL_TO_INDEX[label_name],
                width=int(meta["width"]),
                height=int(meta["height"]),
                circles=grouped_circles.get(image_id, []),
                split="",
            )
        )

    raw_samples.sort(key=lambda sample: sample.image_path.name)
    if max_images > 0:
        raw_samples = stratified_subset(raw_samples, max_images=max_images, seed=seed)
    return assign_stratified_splits(raw_samples, seed, train_ratio, val_ratio, test_ratio)


def stratified_subset(samples: list[Sample], max_images: int, seed: int) -> list[Sample]:
    if max_images <= 0 or max_images >= len(samples):
        return samples

    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label_name].append(sample)

    total = len(samples)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for label_name, group in grouped.items():
        exact = len(group) * max_images / total
        base = int(exact)
        quotas[label_name] = base
        remainders.append((exact - base, label_name))

    assigned = sum(quotas.values())
    remaining = max_images - assigned
    for _, label_name in sorted(remainders, reverse=True):
        if remaining == 0:
            break
        quotas[label_name] += 1
        remaining -= 1

    rng = random.Random(seed)
    subset: list[Sample] = []
    for label_name, group in sorted(grouped.items()):
        shuffled = group[:]
        rng.shuffle(shuffled)
        subset.extend(shuffled[: quotas[label_name]])

    subset.sort(key=lambda sample: sample.image_path.name)
    return subset


def assign_stratified_splits(
    rows: list[Sample],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> list[Sample]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio harus = 1.0")

    rng = random.Random(seed)
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for row in rows:
        grouped[row.label_name].append(row)

    assigned: list[Sample] = []
    for label_name, group in sorted(grouped.items()):
        shuffled = group[:]
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        if total >= 3:
            train_count = max(train_count, 1)
            val_count = max(val_count, 1)
        test_count = total - train_count - val_count
        if total >= 3 and test_count <= 0:
            test_count = 1
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1

        for idx, sample in enumerate(shuffled):
            split = "test"
            if idx < train_count:
                split = "train"
            elif idx < train_count + val_count:
                split = "val"
            assigned.append(
                Sample(
                    image_path=sample.image_path,
                    label_name=sample.label_name,
                    label_index=sample.label_index,
                    width=sample.width,
                    height=sample.height,
                    circles=sample.circles,
                    split=split,
                )
            )
    return assigned


# ============================================================
# SECTION 5 - PREPROCESSING DAN PEMBENTUKAN MASK
# ------------------------------------------------------------
# Bagian ini adalah implementasi tahap awal flow penelitian:
# Input Citra -> Preprocessing
#
# Yang dilakukan:
# - center crop opsional
# - RGB conversion
# - histogram equalization
# - denoising
# - sharpening
# - resize ke ukuran input model
#
# Selain itu, anotasi lingkaran diubah menjadi:
# - binary mask segmentasi
# - vektor statistik morfologi target
# ============================================================
def center_crop_box(width: int, height: int, crop_ratio: float) -> tuple[int, int, int, int]:
    if not 0 < crop_ratio <= 1:
        raise ValueError("crop_ratio harus di rentang 0 < x <= 1")
    crop_w = max(1, int(width * crop_ratio))
    crop_h = max(1, int(height * crop_ratio))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return left, top, left + crop_w, top + crop_h


def preprocess_image(
    image: Image.Image,
    crop_ratio: float,
    denoise_size: int,
    sharpen_radius: float,
    sharpen_percent: int,
    target_size: int,
) -> Image.Image:
    # Tahap preprocessing citra:
    # 1. ubah ke RGB
    # 2. crop area tengah bila diperlukan
    # 3. equalization untuk meratakan kontras
    # 4. denoise untuk mengurangi noise lokal
    # 5. sharpen untuk menegaskan detail lesi
    # 6. resize ke ukuran input model
    rgb = image.convert("RGB")
    left, top, right, bottom = center_crop_box(rgb.size[0], rgb.size[1], crop_ratio)
    cropped = rgb.crop((left, top, right, bottom))
    equalized = Image.merge("RGB", [ImageOps.equalize(channel) for channel in cropped.split()])
    denoised = equalized.filter(ImageFilter.MedianFilter(size=denoise_size))
    sharpened = denoised.filter(
        ImageFilter.UnsharpMask(radius=sharpen_radius, percent=sharpen_percent, threshold=3)
    )
    return sharpened.resize((target_size, target_size), Image.Resampling.LANCZOS)


def circles_to_mask(
    circles: list[tuple[float, float, float]],
    source_width: int,
    source_height: int,
    crop_ratio: float,
    target_size: int,
) -> torch.Tensor:
    # Mengubah anotasi lingkaran (cx, cy, radius) menjadi binary mask segmentasi.
    left, top, right, bottom = center_crop_box(source_width, source_height, crop_ratio)
    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    scale_x = target_size / crop_w
    scale_y = target_size / crop_h
    radius_scale = 0.5 * (scale_x + scale_y)

    yy = torch.arange(target_size, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(target_size, dtype=torch.float32).view(1, -1)
    mask = torch.zeros((target_size, target_size), dtype=torch.float32)

    for cx, cy, radius in circles:
        tx = (cx - left) * scale_x
        ty = (cy - top) * scale_y
        tr = radius * radius_scale
        if tx < -tr or ty < -tr or tx > target_size + tr or ty > target_size + tr:
            continue
        circle = ((xx - tx) ** 2 + (yy - ty) ** 2) <= (tr ** 2)
        mask = torch.maximum(mask, circle.float())
    return mask.unsqueeze(0)


def circle_stats(
    circles: list[tuple[float, float, float]],
    source_width: int,
    source_height: int,
    crop_ratio: float,
) -> torch.Tensor:
    # Mengubah anotasi lingkaran menjadi fitur morfologi target:
    # jumlah lesi, ukuran rata-rata, posisi rata-rata, dan kepadatan area lesi.
    left, top, right, bottom = center_crop_box(source_width, source_height, crop_ratio)
    crop_w = max(1.0, float(right - left))
    crop_h = max(1.0, float(bottom - top))

    kept: list[tuple[float, float, float]] = []
    for cx, cy, radius in circles:
        if left <= cx <= right and top <= cy <= bottom:
            kept.append((cx, cy, radius))

    if not kept:
        return torch.zeros(5, dtype=torch.float32)

    count = float(len(kept))
    norm_count = count / 100.0
    avg_radius = sum(r for _, _, r in kept) / count
    norm_radius = avg_radius / max(crop_w, crop_h)
    mean_x = sum((cx - left) / crop_w for cx, _, _ in kept) / count
    mean_y = sum((cy - top) / crop_h for _, cy, _ in kept) / count
    area_density = sum(math.pi * (r ** 2) for _, _, r in kept) / (crop_w * crop_h)

    return torch.tensor(
        [norm_count, norm_radius, mean_x, mean_y, area_density],
        dtype=torch.float32,
    )


def apply_train_augmentation(
    image: Image.Image,
    mask: torch.Tensor,
    rotation_deg: float,
    hflip_prob: float,
) -> tuple[Image.Image, torch.Tensor]:
    # Augmentasi ringan hanya untuk data train:
    # - rotasi acak agar model tidak terpaku pada orientasi wajah yang sama
    # - horizontal flip agar variasi sisi wajah bertambah
    angle = random.uniform(-rotation_deg, rotation_deg)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

    mask_uint8 = (mask.squeeze(0).clamp(0, 1) * 255).to(torch.uint8).contiguous()
    mask_image = Image.frombytes(
        "L",
        (mask_uint8.shape[1], mask_uint8.shape[0]),
        bytes(mask_uint8.view(-1).tolist()),
    )
    mask_image = mask_image.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

    if random.random() < hflip_prob:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask_image = mask_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    mask_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(mask_image.tobytes()))
    mask_tensor = mask_tensor.view(mask_image.size[1], mask_image.size[0]).float() / 255.0
    return image, mask_tensor.unsqueeze(0)


# ============================================================
# SECTION 6 - DATASET PYTORCH
# ------------------------------------------------------------
# Kelas ini bertugas mengambil satu sample lalu menyiapkan:
# - image hasil preprocessing
# - mask segmentasi dari anotasi lingkaran
# - label severity
# - target morfologi
#
# Jadi pada titik ini, data sudah siap masuk ke model.
# ============================================================
class Acne04v2Dataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int,
        crop_ratio: float,
        denoise_size: int,
        sharpen_radius: float,
        sharpen_percent: int,
        augment: bool,
        rotation_deg: float,
        hflip_prob: float,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.crop_ratio = crop_ratio
        self.denoise_size = denoise_size
        self.sharpen_radius = sharpen_radius
        self.sharpen_percent = sharpen_percent
        self.augment = augment
        self.rotation_deg = rotation_deg
        self.hflip_prob = hflip_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            # Preprocessing dilakukan langsung saat data dibaca oleh DataLoader.
            processed = preprocess_image(
                image,
                crop_ratio=self.crop_ratio,
                denoise_size=self.denoise_size,
                sharpen_radius=self.sharpen_radius,
                sharpen_percent=self.sharpen_percent,
                target_size=self.image_size,
            )

        mask = circles_to_mask(
            sample.circles,
            source_width=sample.width,
            source_height=sample.height,
            crop_ratio=self.crop_ratio,
            target_size=self.image_size,
        )
        morph_target = circle_stats(
            sample.circles,
            source_width=sample.width,
            source_height=sample.height,
            crop_ratio=self.crop_ratio,
        )

        if self.augment:
            processed, mask = apply_train_augmentation(
                processed,
                mask,
                rotation_deg=self.rotation_deg,
                hflip_prob=self.hflip_prob,
            )

        image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(processed.tobytes()))
        image_tensor = image_tensor.view(self.image_size, self.image_size, 3)
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
        return {
            "image": image_tensor,
            "mask": mask,
            "label": torch.tensor(sample.label_index, dtype=torch.long),
            "morph_target": morph_target,
        }


# ============================================================
# SECTION 7 - BLOK DASAR ARSITEKTUR
# ------------------------------------------------------------
# Blok-blok kecil pembangun model:
# - ConvBlock untuk ekstraksi fitur
# - SpatialAttention untuk fokus ke area penting
# - UpBlock untuk decoder segmentasi
# - AMFM untuk fusi fitur morfologi dan fitur global
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = torch.sigmoid(self.conv(x))
        return x * attention, attention


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class AMFM(nn.Module):
    def __init__(self, morph_dim: int, global_dim: int) -> None:
        super().__init__()
        self.alpha = nn.Sequential(
            nn.Linear(morph_dim, global_dim),
            nn.ReLU(inplace=True),
            nn.Linear(global_dim, global_dim),
            nn.Sigmoid(),
        )

    def forward(self, global_feature: torch.Tensor, morph_feature: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha(morph_feature)
        weighted_global = alpha * global_feature
        return torch.cat([weighted_global, morph_feature], dim=1)


# ============================================================
# SECTION 8 - MODEL UTAMA
# ------------------------------------------------------------
# Ini implementasi inti flow:
#
# Preprocessing
# -> Dilated U-Net Encoder
# -> Bottleneck
# -> Spatial Attention
# -> Segmentation Decoder
# -> Mask Lesi
# -> Fitur Morfologi + Fitur Global
# -> AMFM
# -> Classification Head
# -> Severity Hayashi
#
# Alur di forward():
# 1. Encoder menangkap fitur lokal jerawat
# 2. Attention menyorot area lesi penting
# 3. Decoder menghasilkan mask lesi
# 4. Mask dipakai menghitung fitur morfologi
# 5. Bottleneck menghasilkan fitur global
# 6. AMFM menggabungkan morfologi + global
# 7. Classifier memprediksi level severity
# ============================================================
class DilatedUNetAMFM(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, 32, dilation=1)
        self.enc2 = ConvBlock(32, 64, dilation=1)
        self.enc3 = ConvBlock(64, 128, dilation=2)
        self.enc4 = ConvBlock(128, 256, dilation=2)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512, dilation=4)
        self.attention = SpatialAttention(512)

        self.dec4 = UpBlock(512, 256, 256)
        self.dec3 = UpBlock(256, 128, 128)
        self.dec2 = UpBlock(128, 64, 64)
        self.dec1 = UpBlock(64, 32, 32)
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Linear(512, 128)
        self.morph_proj = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )
        self.amfm = AMFM(morph_dim=32, global_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def morphology_from_mask(self, seg_logits: torch.Tensor) -> torch.Tensor:
        # Mengambil fitur morfologi langsung dari mask prediksi model.
        probs = torch.sigmoid(seg_logits)
        bsz, _, h, w = probs.shape
        x_grid = torch.linspace(0.0, 1.0, w, device=probs.device).view(1, 1, 1, w)
        y_grid = torch.linspace(0.0, 1.0, h, device=probs.device).view(1, 1, h, 1)
        mass = probs.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)

        area = probs.mean(dim=(2, 3))
        density = (probs > 0.5).float().mean(dim=(2, 3))
        center_x = (probs * x_grid).sum(dim=(2, 3), keepdim=True) / mass
        center_y = (probs * y_grid).sum(dim=(2, 3), keepdim=True) / mass
        spread_x = (probs * (x_grid - center_x).abs()).sum(dim=(2, 3), keepdim=True) / mass
        spread_y = (probs * (y_grid - center_y).abs()).sum(dim=(2, 3), keepdim=True) / mass

        return torch.cat(
            [
                area,
                density,
                center_x.view(bsz, 1),
                center_y.view(bsz, 1),
                (0.5 * (spread_x + spread_y)).view(bsz, 1),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encoder Dilated U-Net: menangkap fitur lokal jerawat dari resolusi tinggi ke rendah.
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck + spatial attention: memfokuskan model ke area lesi yang dianggap penting.
        bottleneck = self.bottleneck(self.pool(s4))
        attended, attention_map = self.attention(bottleneck)

        # Decoder segmentasi: membangun kembali mask lesi dari fitur bottleneck.
        d4 = self.dec4(attended, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        seg_logits = self.seg_head(d1)

        # Fitur global diambil dari bottleneck yang sudah diberi attention.
        global_feature = self.global_proj(self.global_pool(attended).flatten(1))

        # Fitur morfologi dihitung dari mask prediksi, lalu diproyeksikan ke ruang fitur kecil.
        morph_feature = self.morph_proj(self.morphology_from_mask(seg_logits))

        # AMFM menggabungkan fitur global dan fitur morfologi agar klasifikasi lebih sadar lesi.
        fused = self.amfm(global_feature, morph_feature)
        class_logits = self.classifier(fused)

        return {
            "seg_logits": seg_logits,
            "class_logits": class_logits,
            "attention_map": attention_map,
            "fused_feature": fused,
        }


# ============================================================
# SECTION 9 - METRIK EVALUASI
# ------------------------------------------------------------
# Metrik yang dipakai untuk membaca performa model:
# - Dice: kualitas segmentasi
# - IoU : kualitas overlap segmentasi
# - Accuracy: kualitas klasifikasi severity
# - Cohen's Kappa: tingkat kesepakatan klasifikasi setelah koreksi peluang acak
# ============================================================
def dice_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / (union + 1e-6)).mean()


def iou_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == targets).float().mean()


def cohen_kappa(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=logits.device)

    for true_label, pred_label in zip(targets.view(-1), preds.view(-1)):
        confusion[true_label.long(), pred_label.long()] += 1.0

    total = confusion.sum()
    if total <= 0:
        return torch.tensor(0.0, device=logits.device)

    observed = confusion.diag().sum() / total
    row_marginals = confusion.sum(dim=1)
    col_marginals = confusion.sum(dim=0)
    expected = (row_marginals * col_marginals).sum() / (total * total)

    if torch.isclose(1.0 - expected, torch.tensor(0.0, device=logits.device)):
        return torch.tensor(0.0, device=logits.device)
    return (observed - expected) / (1.0 - expected)


# ============================================================
# SECTION 10 - TRAIN / VALIDATION / TEST LOOP
# ------------------------------------------------------------
# Bagian ini menjalankan satu epoch penuh.
#
# Pada setiap batch:
# - gambar masuk ke model
# - model menghasilkan mask dan prediksi severity
# - dihitung 3 loss:
#   1. segmentation loss
#   2. classification loss
#   3. morphology loss
# - jika mode train, bobot model diperbarui
# ============================================================
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    # Satu epoch = model melihat semua batch pada split train/val/test.
    is_train = optimizer is not None
    model.train(is_train)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    totals = {
        "loss": 0.0,
        "seg_loss": 0.0,
        "cls_loss": 0.0,
        "morph_loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "acc": 0.0,
        "kappa": 0.0,
        "count": 0,
    }

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)
        morph_targets = batch["morph_target"].to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        # Loss segmentasi: cocokkan mask prediksi dengan mask lingkaran dari ACNE04-v2.
        pred_morph = model.morphology_from_mask(outputs["seg_logits"])
        seg_loss = bce(outputs["seg_logits"], masks)

        # Loss klasifikasi: cocokkan prediksi severity dengan label level 0-3.
        cls_loss = ce(outputs["class_logits"], labels)

        # Loss morfologi: cocokkan statistik mask prediksi dengan statistik anotasi lingkaran.
        morph_loss = mse(pred_morph, morph_targets)
        loss = seg_loss + cls_loss + 0.2 * morph_loss

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["seg_loss"] += seg_loss.item() * batch_size
        totals["cls_loss"] += cls_loss.item() * batch_size
        totals["morph_loss"] += morph_loss.item() * batch_size
        totals["dice"] += dice_score(outputs["seg_logits"], masks).item() * batch_size
        totals["iou"] += iou_score(outputs["seg_logits"], masks).item() * batch_size
        totals["acc"] += accuracy(outputs["class_logits"], labels).item() * batch_size
        totals["kappa"] += cohen_kappa(outputs["class_logits"], labels).item() * batch_size
        totals["count"] += batch_size

    count = max(1, totals["count"])
    return {key: value / count for key, value in totals.items() if key != "count"}


# ============================================================
# SECTION 11 - MEMBANGUN DATALOADER
# ------------------------------------------------------------
# Data yang sudah dibagi train/val/test dibungkus menjadi
# DataLoader agar training bisa berjalan per batch.
# ============================================================
def build_loaders(args: argparse.Namespace, samples: list[Sample]) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    test_samples = [sample for sample in samples if sample.split == "test"]

    common = dict(
        image_size=args.image_size,
        crop_ratio=args.crop_ratio,
        denoise_size=args.denoise_size,
        sharpen_radius=args.sharpen_radius,
        sharpen_percent=args.sharpen_percent,
        rotation_deg=args.rotation_deg,
        hflip_prob=args.hflip_prob,
    )
    train_loader = DataLoader(
        Acne04v2Dataset(train_samples, augment=args.augment_train, **common),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Acne04v2Dataset(val_samples, augment=False, **common),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        Acne04v2Dataset(test_samples, augment=False, **common),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_test_result_csv(
    csv_path: Path,
    config: dict,
    best_val_loss: float,
    test_metrics: dict[str, float],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "encoder",
        "image_size",
        "batch_size",
        "epochs",
        "lr",
        "augment_train",
        "rotation_deg",
        "hflip_prob",
        "max_images",
        "device",
        "best_val_loss",
        "test_loss",
        "test_seg_loss",
        "test_cls_loss",
        "test_morph_loss",
        "test_acc",
        "test_kappa",
        "test_dice",
        "test_iou",
    ]
    row = {
        "run_name": config.get("run_name", ""),
        "encoder": config.get("encoder", "dilated_unet"),
        "image_size": config.get("image_size", ""),
        "batch_size": config.get("batch_size", ""),
        "epochs": config.get("epochs", ""),
        "lr": config.get("lr", ""),
        "augment_train": config.get("augment_train", ""),
        "rotation_deg": config.get("rotation_deg", ""),
        "hflip_prob": config.get("hflip_prob", ""),
        "max_images": config.get("max_images", ""),
        "device": config.get("device", ""),
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics.get("loss", ""),
        "test_seg_loss": test_metrics.get("seg_loss", ""),
        "test_cls_loss": test_metrics.get("cls_loss", ""),
        "test_morph_loss": test_metrics.get("morph_loss", ""),
        "test_acc": test_metrics.get("acc", ""),
        "test_kappa": test_metrics.get("kappa", ""),
        "test_dice": test_metrics.get("dice", ""),
        "test_iou": test_metrics.get("iou", ""),
    }
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# SECTION 12 - MAIN PIPELINE
# ------------------------------------------------------------
# Ini urutan besar saat script dijalankan:
#
# Step 1. Baca data ACNE04-v2
# Step 2. Bagi data train / val / test
# Step 3. Bangun model Dilated U-Net + AMFM
# Step 4. Jalankan preprocessing otomatis via DataLoader
# Step 5. Training per epoch
# Step 6. Evaluasi final pada test set
# Step 7. Simpan best model dan metrics
# ============================================================
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    print("🚀 Memulai pipeline ACNE04-v2")
    print(f"📁 Folder image: {args.image_dir}")
    print(f"📝 File anotasi: {args.annotations_json}")
    print(f"🖥️ Device: {device}")
    print("🔎 Step 1/6: membaca metadata gambar dan anotasi lingkaran")
    samples = load_acne04v2_samples(
        image_dir=args.image_dir,
        annotations_json=args.annotations_json,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_images=args.max_images,
    )
    print(f"✅ Total sample yang siap dipakai: {len(samples)}")

    print("🧪 Step 2/6: membagi data menjadi train / val / test")
    train_loader, val_loader, test_loader = build_loaders(args, samples)
    print(
        f"✅ Split data: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    print("🏗️ Step 3/6: membangun model Dilated U-Net + Spatial Attention + AMFM")
    model = DilatedUNetAMFM(num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_model.pt"

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")

    print("🧼 Step 4/6: preprocessing berjalan otomatis di DataLoader")
    print("   RGB -> equalization -> denoise -> sharpen -> resize")
    if args.augment_train:
        print(
            f"   🔁 Augmentasi train aktif -> rotasi acak ±{args.rotation_deg} derajat, hflip_prob={args.hflip_prob}"
        )
    else:
        print("   🔁 Augmentasi train tidak aktif")
    print("🧠 Step 5/6: training model dimulai")
    print("   Flow: preprocessing -> dilated unet -> attention -> segmentation -> morphology/global -> AMFM")

    for epoch in range(1, args.epochs + 1):
        print(f"📘 Epoch {epoch:02d}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_seg_loss": train_metrics["seg_loss"],
            "train_cls_loss": train_metrics["cls_loss"],
            "train_morph_loss": train_metrics["morph_loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_acc": train_metrics["acc"],
            "train_kappa": train_metrics["kappa"],
            "val_loss": val_metrics["loss"],
            "val_seg_loss": val_metrics["seg_loss"],
            "val_cls_loss": val_metrics["cls_loss"],
            "val_morph_loss": val_metrics["morph_loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_acc": val_metrics["acc"],
            "val_kappa": val_metrics["kappa"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_kappa={train_metrics['kappa']:.4f} train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_kappa={val_metrics['kappa']:.4f} val_dice={val_metrics['dice']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)
            print("💾 Model terbaik diperbarui berdasarkan val_loss")

    print("🧪 Step 6/6: evaluasi akhir pada data test")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, device, optimizer=None)

    summary = {
        "config": {
            "image_dir": str(args.image_dir),
            "annotations_json": str(args.annotations_json),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "crop_ratio": args.crop_ratio,
            "augment_train": args.augment_train,
            "rotation_deg": args.rotation_deg,
            "hflip_prob": args.hflip_prob,
            "seed": args.seed,
            "device": str(device),
            "max_images": args.max_images,
            "run_name": args.output_dir.name,
            "encoder": "dilated_unet",
        },
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "history": history,
    }
    save_json(summary, args.output_dir / "metrics.json")
    append_test_result_csv(
        args.output_dir.parent / "experiment_results.csv",
        summary["config"],
        best_val_loss,
        test_metrics,
    )

    print("✅ Training selesai.")
    print(f"🏁 Best model: {best_model_path}")
    print(f"📊 Metrics: {args.output_dir / 'metrics.json'}")
    print(f"🗂️ CSV log: {args.output_dir.parent / 'experiment_results.csv'}")
    print(
        "🧾 Test | "
        f"loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['acc']:.4f} "
        f"kappa={test_metrics['kappa']:.4f} "
        f"dice={test_metrics['dice']:.4f} "
        f"iou={test_metrics['iou']:.4f}"
    )


if __name__ == "__main__":
    main()
