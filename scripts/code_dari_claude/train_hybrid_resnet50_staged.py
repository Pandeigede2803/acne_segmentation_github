#!/usr/bin/env python3
"""
Train the staged hybrid model described in code.md using preprocessed inputs.

Saved outputs per stage:
- metrics.json
- experiment_results.csv
- predicted masks for selected samples
- backbone / attention / morph / fusion / logits tensors for selected samples
- stage_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_baseline_dilated_unet as base

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from PIL import Image, ImageFilter, ImageOps
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Butuh torch, torchvision, dan pillow di venv. Install dengan: python3 -m pip install torch torchvision pillow"
    ) from exc


@dataclass
class ManifestRow:
    file_name: str
    label_name: str
    label_index: int
    split: str
    image_path: Path
    preprocessed_path: Path | None   # None jika tidak ada
    mask_path: Path
    image_size: int


def preprocess_pil(image: Image.Image, denoise_size: int = 3,
                   sharpen_radius: float = 1.6, sharpen_percent: int = 140) -> Image.Image:
    """Preprocessing on-the-fly: equalize + denoise + sharpen (sama dengan pipeline lama)."""
    rgb       = image.convert("RGB")
    equalized = Image.merge("RGB", [ImageOps.equalize(c) for c in rgb.split()])
    denoised  = equalized.filter(ImageFilter.MedianFilter(size=denoise_size))
    sharpened = denoised.filter(
        ImageFilter.UnsharpMask(radius=sharpen_radius, percent=sharpen_percent, threshold=3)
    )
    return sharpened


class PreprocessedHybridDataset(Dataset):
    def __init__(self, rows: list[ManifestRow], augment: bool,
                 rotation_deg: float, hflip_prob: float,
                 preprocess_on_the_fly: bool = False) -> None:
        self.rows = rows
        self.augment = augment
        self.rotation_deg = rotation_deg
        self.hflip_prob = hflip_prob
        self.preprocess_on_the_fly = preprocess_on_the_fly

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]

        # Gunakan preprocessed_path jika tersedia, fallback ke image_path
        if row.preprocessed_path is not None and row.preprocessed_path.exists():
            with Image.open(row.preprocessed_path) as image:
                image = image.convert("RGB")
        else:
            with Image.open(row.image_path) as image:
                image = image.convert("RGB")
            if self.preprocess_on_the_fly:
                image = preprocess_pil(image)

        with Image.open(row.mask_path) as mask:
            mask = mask.convert("L")

        mask_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
        mask_tensor = mask_tensor.view(mask.size[1], mask.size[0]).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)

        if self.augment:
            image, mask_tensor = base.apply_train_augmentation(
                image,
                mask_tensor,
                rotation_deg=self.rotation_deg,
                hflip_prob=self.hflip_prob,
            )

        image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        image_tensor = image_tensor.view(image.size[1], image.size[0], 3)
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(row.label_index, dtype=torch.long),
            "image_id": row.file_name,
        }


class SharedBackbone(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e0, e1, e2, e3, e4


class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DilatedUNetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec4 = DilatedConvBlock(1024 + 1024, 512, dilation=2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DilatedConvBlock(256 + 512, 256, dilation=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DilatedConvBlock(128 + 256, 128, dilation=4)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DilatedConvBlock(64 + 64, 64, dilation=4)
        self.final = nn.Conv2d(64, 1, 1)

    @staticmethod
    def _resize_to_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            return nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, e0: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor, e4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u4 = self._resize_to_skip(self.up4(e4), e3)
        d4 = self.dec4(torch.cat([u4, e3], dim=1))
        u3 = self._resize_to_skip(self.up3(d4), e2)
        d3 = self.dec3(torch.cat([u3, e2], dim=1))
        u2 = self._resize_to_skip(self.up2(d3), e1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        u1 = self._resize_to_skip(self.up1(d2), e0)
        d1 = self.dec1(torch.cat([u1, e0], dim=1))
        seg_mask = torch.sigmoid(self.final(d1))
        v_morph = d4.mean(dim=[2, 3])
        return seg_mask, v_morph


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_map = self.conv(x)
        return x * attn_map, attn_map


class AMFM(nn.Module):
    def __init__(self, feat_dim: int = 512) -> None:
        super().__init__()
        self.W_m = nn.Linear(feat_dim, feat_dim)
        self.W_g = nn.Linear(feat_dim, feat_dim)
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, v_global: torch.Tensor, v_morph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.W_m(v_morph) + self.W_g(v_global) + self.bias)
        v_fused = alpha * v_global + (1 - alpha) * v_morph
        return v_fused, alpha


class MALDSLoss(nn.Module):
    def __init__(self, num_classes: int = 4, base_eps: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.base_eps = base_eps

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, v_morph: torch.Tensor) -> torch.Tensor:
        morph_score = torch.sigmoid(v_morph.mean(dim=1))
        eps = self.base_eps * (1 + 0.5 * morph_score)

        dist = torch.zeros(labels.shape[0], self.num_classes, device=labels.device)
        for b in range(labels.shape[0]):
            for c in range(self.num_classes):
                dist[b, c] = eps[b] * torch.exp(-0.5 * (labels[b].float() - c) ** 2)
        dist = dist / dist.sum(dim=1, keepdim=True).clamp_min(1e-8)

        log_probs = torch.log_softmax(logits, dim=1)
        kl_loss = dist * (torch.log(dist + 1e-8) - log_probs)
        return kl_loss.sum(dim=1).mean()


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal Loss untuk segmentasi dengan extreme pixel imbalance.
    - alpha=0.75 : bobot lebih besar ke piksel positif (lesi)
    - gamma=2.0  : down-weight easy negatives (background yang mudah diprediksi)
    Efek: model dipaksa fokus ke lesi kecil yang sulit, bukan background yang mudah.
    """
    eps  = 1e-6
    pred = pred.clamp(eps, 1 - eps)

    # Cross entropy per piksel
    ce_pos = -torch.log(pred)
    ce_neg = -torch.log(1 - pred)

    # Focal weighting
    fl_pos = alpha       * (1 - pred) ** gamma * ce_pos
    fl_neg = (1 - alpha) * pred       ** gamma * ce_neg

    loss = target * fl_pos + (1 - target) * fl_neg
    return loss.mean()


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_seg: float = 0.6,
        lambda_cls: float = 0.4,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_seg  = lambda_seg
        self.lambda_cls  = lambda_cls
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.malds = MALDSLoss()

    def forward(
        self,
        seg_pred: torch.Tensor,
        seg_gt: torch.Tensor,
        cls_logits: torch.Tensor,
        cls_labels: torch.Tensor,
        v_morph: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Dice loss
        intersection = 2 * (seg_pred * seg_gt).sum(dim=(1, 2, 3))
        denom        = seg_pred.sum(dim=(1, 2, 3)) + seg_gt.sum(dim=(1, 2, 3)) + 1e-6
        dice_loss    = 1 - (intersection / denom).mean()

        # Focal loss — lebih robust untuk extreme imbalance dibanding BCE
        fl = focal_loss(seg_pred, seg_gt, self.focal_alpha, self.focal_gamma)

        seg_loss = 0.5 * dice_loss + 0.5 * fl
        cls_loss = self.malds(cls_logits, cls_labels, v_morph)
        total    = self.lambda_seg * seg_loss + self.lambda_cls * cls_loss
        return total, {"seg_loss": seg_loss, "cls_loss": cls_loss, "dice_loss": dice_loss, "focal_loss": fl}


class HybridAcneNet(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = SharedBackbone(pretrained=pretrained)
        self.seg_decoder = DilatedUNetDecoder()
        self.attention = SpatialAttention(in_channels=2048)
        self.amfm = AMFM(feat_dim=512)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_proj = nn.Linear(2048, 512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e0, e1, e2, e3, e4 = self.backbone(x)
        e4_attn, attn_map = self.attention(e4)
        seg_mask, v_morph = self.seg_decoder(e0, e1, e2, e3, e4_attn)
        if seg_mask.shape[-2:] != x.shape[-2:]:
            seg_mask = nn.functional.interpolate(seg_mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        v_global = self.global_pool(e4_attn).flatten(1)
        v_global = self.cls_proj(v_global)
        v_fused, alpha = self.amfm(v_global, v_morph)
        cls_logits = self.classifier(v_fused)
        return {
            "seg_mask": seg_mask,
            "cls_logits": cls_logits,
            "e4": e4,
            "e4_attn": e4_attn,
            "attn_map": attn_map,
            "v_morph": v_morph,
            "v_global": v_global,
            "v_fused": v_fused,
            "alpha": alpha,
        }


def dice_score_from_probs(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / (union + 1e-6)).mean()


def iou_score_from_probs(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean()


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train staged hybrid ResNet50+DilatedUNet from preprocessed manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "refined_masks_output" / "refined_manifest.csv",
        help="Path ke refined_manifest.csv dari refine_masks_color.py",
    )
    parser.add_argument(
        "--preprocess-on-the-fly",
        action="store_true",
        help="Terapkan equalize+denoise+sharpen saat loading jika preprocessed_path tidak tersedia",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-phase1", type=int, default=10)
    parser.add_argument("--epochs-phase2", type=int, default=10)
    parser.add_argument("--lr-phase1", type=float, default=1e-3)
    parser.add_argument("--lr-phase2", type=float, default=1e-4)
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--rotation-deg", type=float, default=15.0)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--save-stage-samples", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "hybrid_resnet50_staged",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=ROOT / "baseline_runs" / "experiment_results.csv",
        help="CSV utama untuk rekap semua eksperimen.",
    )
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--lambda-seg",   type=float, default=0.6)
    parser.add_argument("--lambda-cls",   type=float, default=0.4)
    parser.add_argument("--focal-alpha",  type=float, default=0.75,
                        help="Focal loss alpha: bobot piksel positif (lesi). Default 0.75")
    parser.add_argument("--focal-gamma",  type=float, default=2.0,
                        help="Focal loss gamma: down-weight easy negatives. Default 2.0")
    return parser.parse_args()


def load_manifest_rows(manifest_path: Path) -> list[ManifestRow]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Support format lama (mask_path) dan format baru (refined_mask)
            mask_path = Path(row.get("refined_mask") or row.get("mask_path", ""))

            # preprocessed_path opsional — None kalau kolom tidak ada atau kosong
            raw_preproc = row.get("preprocessed_path", "").strip()
            preprocessed_path = Path(raw_preproc) if raw_preproc else None

            # file_name: format baru pakai "filename", lama pakai "file_name"
            file_name = row.get("filename") or row.get("file_name", "")

            # label_name: format baru pakai "label_name", lama sama
            label_name = row.get("label_name", "")

            rows.append(
                ManifestRow(
                    file_name=file_name,
                    label_name=label_name,
                    label_index=int(row["label_index"]),
                    split=row["split"],
                    image_path=Path(row["image_path"]),
                    preprocessed_path=preprocessed_path,
                    mask_path=mask_path,
                    image_size=int(row["image_size"]),
                )
            )
    return rows


def build_loaders(args: argparse.Namespace, rows: list[ManifestRow]) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_rows = [row for row in rows if row.split == "train"]
    val_rows   = [row for row in rows if row.split == "val"]
    test_rows  = [row for row in rows if row.split == "test"]
    otf = getattr(args, "preprocess_on_the_fly", False)
    train_loader = DataLoader(
        PreprocessedHybridDataset(train_rows, args.augment_train, args.rotation_deg,
                                   args.hflip_prob, preprocess_on_the_fly=otf),
        batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        PreprocessedHybridDataset(val_rows, False, args.rotation_deg,
                                   args.hflip_prob, preprocess_on_the_fly=otf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        PreprocessedHybridDataset(test_rows, False, args.rotation_deg,
                                   args.hflip_prob, preprocess_on_the_fly=otf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader


def save_mask_preview(mask_tensor: torch.Tensor, path: Path) -> None:
    mask_uint8 = (mask_tensor.squeeze(0).clamp(0, 1) * 255).to(torch.uint8).cpu()
    img = Image.frombytes("L", (mask_uint8.shape[1], mask_uint8.shape[0]), bytes(mask_uint8.contiguous().view(-1).tolist()))
    img.save(path, format="PNG")


def tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def save_stage_outputs(output_dir: Path, batch: dict, outputs: dict[str, torch.Tensor], limit: int, collected: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_ids = batch["image_id"]
    masks = batch["mask"]
    for idx in range(min(limit, len(image_ids))):
        image_id = Path(str(image_ids[idx])).stem
        sample_dir = output_dir / image_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        stage_pack = {
            "backbone_e4": outputs["e4"][idx].detach().cpu(),
            "attention_map": outputs["attn_map"][idx].detach().cpu(),
            "attended_e4": outputs["e4_attn"][idx].detach().cpu(),
            "pred_mask": outputs["seg_mask"][idx].detach().cpu(),
            "gt_mask": masks[idx].detach().cpu(),
            "v_morph": outputs["v_morph"][idx].detach().cpu(),
            "v_global": outputs["v_global"][idx].detach().cpu(),
            "alpha": outputs["alpha"][idx].detach().cpu(),
            "v_fused": outputs["v_fused"][idx].detach().cpu(),
            "cls_logits": outputs["cls_logits"][idx].detach().cpu(),
        }
        torch.save(stage_pack, sample_dir / "stage_outputs.pt")
        save_mask_preview(stage_pack["pred_mask"], sample_dir / "pred_mask.png")
        save_mask_preview(stage_pack["gt_mask"], sample_dir / "gt_mask.png")

        for stage_name, tensor in stage_pack.items():
            collected.append(
                {
                    "image_id": image_id,
                    "stage": stage_name,
                    **tensor_stats(tensor.float()),
                }
            )


def write_stage_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "stage", "shape", "mean", "std", "min", "max"])
        writer.writeheader()
        writer.writerows(rows)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: MultiTaskLoss,
    stage_output_dir: Path | None = None,
    save_stage_samples: int = 0,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "seg_loss": 0.0,
        "cls_loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "acc": 0.0,
        "kappa": 0.0,
        "count": 0,
    }
    collected_stage_rows: list[dict[str, object]] = []
    saved = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss, parts = criterion(outputs["seg_mask"], masks, outputs["cls_logits"], labels, outputs["v_morph"])

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["seg_loss"] += parts["seg_loss"].item() * batch_size
        totals["cls_loss"] += parts["cls_loss"].item() * batch_size
        totals["dice"] += dice_score_from_probs(outputs["seg_mask"], masks).item() * batch_size
        totals["iou"] += iou_score_from_probs(outputs["seg_mask"], masks).item() * batch_size
        totals["acc"] += classification_accuracy(outputs["cls_logits"], labels).item() * batch_size
        totals["kappa"] += cohen_kappa(outputs["cls_logits"], labels).item() * batch_size
        totals["count"] += batch_size

        if stage_output_dir is not None and saved < save_stage_samples:
            remaining = save_stage_samples - saved
            save_stage_outputs(stage_output_dir, batch, outputs, remaining, collected_stage_rows)
            saved += min(remaining, batch_size)

    if stage_output_dir is not None and collected_stage_rows:
        write_stage_summary(stage_output_dir / "stage_summary.csv", collected_stage_rows)

    count = max(1, totals["count"])
    return {key: value / count for key, value in totals.items() if key != "count"}


def freeze_backbone(model: HybridAcneNet, freeze: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = not freeze


def main() -> None:
    args = parse_args()
    base.set_seed(args.seed)
    device = base.resolve_device(args.device)
    rows = load_manifest_rows(args.manifest)
    train_loader, val_loader, test_loader = build_loaders(args, rows)

    print("🚀 Memulai hybrid ResNet50 staged training")
    print(f"🧾 Manifest: {args.manifest}")
    print(f"🖥️ Device: {device}")
    print(f"✅ Split data: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    model = HybridAcneNet(num_classes=4, pretrained=args.pretrained_backbone).to(device)
    criterion = MultiTaskLoss(
        lambda_seg  = args.lambda_seg,
        lambda_cls  = args.lambda_cls,
        focal_alpha = args.focal_alpha,
        focal_gamma = args.focal_gamma,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_model.pt"
    history: list[dict[str, float | int | str]] = []
    best_val_loss = float("inf")

    phase_settings = [
        {"name": "phase1_freeze_backbone", "epochs": args.epochs_phase1, "lr": args.lr_phase1, "freeze": True},
        {"name": "phase2_full_finetune", "epochs": args.epochs_phase2, "lr": args.lr_phase2, "freeze": False},
    ]

    epoch_counter = 0
    for phase in phase_settings:
        freeze_backbone(model, phase["freeze"])
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=phase["lr"], momentum=0.9)
        print(f"🧠 {phase['name']} | freeze_backbone={phase['freeze']} | epochs={phase['epochs']} | lr={phase['lr']}")

        for _ in range(phase["epochs"]):
            epoch_counter += 1
            print(f"📘 Epoch {epoch_counter:02d}/{args.epochs_phase1 + args.epochs_phase2}")
            train_metrics = run_epoch(model, train_loader, device, optimizer, criterion)
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, device, optimizer=None, criterion=criterion)

            row = {
                "phase": phase["name"],
                "epoch": epoch_counter,
                "train_loss": train_metrics["loss"],
                "train_seg_loss": train_metrics["seg_loss"],
                "train_cls_loss": train_metrics["cls_loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "train_acc": train_metrics["acc"],
                "train_kappa": train_metrics["kappa"],
                "val_loss": val_metrics["loss"],
                "val_seg_loss": val_metrics["seg_loss"],
                "val_cls_loss": val_metrics["cls_loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_acc": val_metrics["acc"],
                "val_kappa": val_metrics["kappa"],
            }
            history.append(row)

            print(
                f"Epoch {epoch_counter:02d} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_kappa={train_metrics['kappa']:.4f} train_dice={train_metrics['dice']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_kappa={val_metrics['kappa']:.4f} val_dice={val_metrics['dice']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), best_model_path)
                print("💾 Model terbaik diperbarui berdasarkan val_loss")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            test_loader,
            device,
            optimizer=None,
            criterion=criterion,
            stage_output_dir=args.output_dir / "stage_outputs_test",
            save_stage_samples=args.save_stage_samples,
        )

    summary = {
        "config": {
            "manifest": str(args.manifest),
            "batch_size": args.batch_size,
            "epochs": args.epochs_phase1 + args.epochs_phase2,
            "epochs_phase1": args.epochs_phase1,
            "epochs_phase2": args.epochs_phase2,
            "lr": args.lr_phase2,
            "lr_phase1": args.lr_phase1,
            "lr_phase2": args.lr_phase2,
            "augment_train": args.augment_train,
            "rotation_deg": args.rotation_deg,
            "hflip_prob": args.hflip_prob,
            "seed": args.seed,
            "device": str(device),
            "max_images": "",
            "pretrained_backbone": args.pretrained_backbone,
            "save_stage_samples": args.save_stage_samples,
            "run_name": args.output_dir.name,
            "encoder": "resnet50_hybrid_staged",
        },
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "history": history,
    }
    base.save_json(summary, args.output_dir / "metrics.json")
    base.append_test_result_csv(args.results_csv, summary["config"], best_val_loss, test_metrics)

    print("✅ Training selesai.")
    print(f"🏁 Best model: {best_model_path}")
    print(f"📊 Metrics: {args.output_dir / 'metrics.json'}")
    print(f"🗂️ CSV log: {args.results_csv}")
    print(f"🧪 Stage outputs: {args.output_dir / 'stage_outputs_test'}")
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
