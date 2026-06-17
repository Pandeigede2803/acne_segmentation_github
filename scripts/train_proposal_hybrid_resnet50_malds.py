#!/usr/bin/env python3
"""
Proposal-style hybrid training:

- Preprocessed ACNE04-v2 images + masks from manifest.csv
- ResNet-50 encoder
- Spatial attention
- Segmentation decoder
- Adaptive Morphology-Aware Fusion Module (AMFM)
- BCE + Dice segmentation loss
- Morphology-aware label distribution smoothing (MA-LDS style) + KL loss
- Two-phase training:
  phase 1: freeze ResNet encoder
  phase 2: unfreeze full network
- Class imbalance weighting
- Checkpoint per epoch with resume support
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ResNet50_Weights, resnet50


LABEL_NAMES = ["levle0", "levle1", "levle2", "levle3"]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train proposal-style ResNet50 + Dilated decoder + AMFM + MA-LDS.")
    parser.add_argument("--manifest", type=Path, default=base_dir / "acne04v2_preprocessed_all" / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "baseline_runs" / "proposal_hybrid_resnet50_malds")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--phase1-epochs", type=int, default=30)
    parser.add_argument("--phase2-epochs", type=int, default=90)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=1e-4)
    parser.add_argument("--seg-weight", type=float, default=1.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--morph-weight", type=float, default=0.2)
    parser.add_argument("--base-smoothing", type=float, default=0.05)
    parser.add_argument("--max-smoothing", type=float, default=0.35)
    parser.add_argument("--rotation-deg", type=float, default=15.0)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--brightness", type=float, default=0.15)
    parser.add_argument("--contrast", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet ResNet-50 weights if available.")
    parser.add_argument("--weighted-sampler", action="store_true", help="Oversample minority classes in train split.")
    parser.add_argument("--resume", action="store_true", help="Resume from last_checkpoint.pt in output-dir.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        image_path = row.get("preprocessed_path") or row.get("image_path")
        mask_path = row.get("mask_path") or row.get("refined_mask") or row.get("circle_mask")
        if not image_path:
            raise KeyError("Manifest harus punya kolom preprocessed_path atau image_path")
        if not mask_path:
            raise KeyError("Manifest harus punya kolom mask_path, refined_mask, atau circle_mask")
        normalized = dict(row)
        normalized["image_path"] = image_path
        normalized["mask_path"] = mask_path
        normalized_rows.append(normalized)
    return normalized_rows


def tensor_from_pil_rgb(image: Image.Image, image_size: int) -> torch.Tensor:
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(image_size, image_size, 3)
    return tensor.permute(2, 0, 1).float() / 255.0


def tensor_from_pil_mask(mask: Image.Image, image_size: int) -> torch.Tensor:
    if mask.size != (image_size, image_size):
        mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    tensor = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(image_size, image_size).float() / 255.0
    return (tensor > 0.5).float().unsqueeze(0)


def mask_morphology(mask: torch.Tensor) -> torch.Tensor:
    probs = mask.float()
    _, h, w = probs.shape
    x_grid = torch.linspace(0.0, 1.0, w).view(1, 1, w)
    y_grid = torch.linspace(0.0, 1.0, h).view(1, h, 1)
    mass = probs.sum().clamp_min(1e-6)

    area = probs.mean()
    density = (probs > 0.5).float().mean()
    center_x = (probs * x_grid).sum() / mass
    center_y = (probs * y_grid).sum() / mass
    spread_x = (probs * (x_grid - center_x).abs()).sum() / mass
    spread_y = (probs * (y_grid - center_y).abs()).sum() / mass
    return torch.stack([area, density, center_x, center_y, 0.5 * (spread_x + spread_y)])


def augment_pair(
    image: Image.Image,
    mask: Image.Image,
    rotation_deg: float,
    hflip_prob: float,
    brightness: float,
    contrast: float,
) -> tuple[Image.Image, Image.Image]:
    angle = random.uniform(-rotation_deg, rotation_deg)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

    if random.random() < hflip_prob:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if brightness > 0:
        factor = random.uniform(1.0 - brightness, 1.0 + brightness)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if contrast > 0:
        factor = random.uniform(1.0 - contrast, 1.0 + contrast)
        image = ImageEnhance.Contrast(image).enhance(factor)
    return image, mask


class PreprocessedAcneDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: int, augment: bool, args: argparse.Namespace) -> None:
        self.rows = rows
        self.image_size = image_size
        self.augment = augment
        self.args = args

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
        with Image.open(row["mask_path"]) as mask:
            mask = mask.convert("L")

        if self.augment:
            image, mask = augment_pair(
                image,
                mask,
                rotation_deg=self.args.rotation_deg,
                hflip_prob=self.args.hflip_prob,
                brightness=self.args.brightness,
                contrast=self.args.contrast,
            )

        mask_tensor = tensor_from_pil_mask(mask, self.image_size)
        return {
            "image": tensor_from_pil_rgb(image, self.image_size),
            "mask": mask_tensor,
            "label": torch.tensor(int(row["label_index"]), dtype=torch.long),
            "morph_target": mask_morphology(mask_tensor),
        }


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.stem(x)
        x = self.maxpool(s1)
        s2 = self.layer1(x)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        bottleneck = self.layer4(s4)
        return s1, s2, s3, s4, bottleneck


class SpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = torch.sigmoid(self.conv(x))
        return x * attention, attention


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ProposalHybridResNet50(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        self.attention = SpatialAttention(2048)
        self.dec4 = DecoderBlock(2048, 1024, 512, dilation=2)
        self.dec3 = DecoderBlock(512, 512, 256, dilation=2)
        self.dec2 = DecoderBlock(256, 256, 128, dilation=1)
        self.dec1 = DecoderBlock(128, 64, 64, dilation=1)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Linear(2048, 128)
        self.morph_proj = nn.Sequential(nn.Linear(5, 64), nn.ReLU(inplace=True), nn.Linear(64, 128))
        self.amfm_alpha = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def set_encoder_trainable(self, trainable: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = trainable

    def morphology_from_mask(self, seg_logits: torch.Tensor) -> torch.Tensor:
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
        return torch.cat([area, density, center_x.view(bsz, 1), center_y.view(bsz, 1), (0.5 * (spread_x + spread_y)).view(bsz, 1)], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1, s2, s3, s4, bottleneck = self.encoder(x)
        attended, attention_map = self.attention(bottleneck)
        d4 = self.dec4(attended, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.final_up(d1)
        seg_logits = self.seg_head(d0)
        if seg_logits.shape[-2:] != x.shape[-2:]:
            seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        global_feature = self.global_proj(self.global_pool(attended).flatten(1))
        morph_vector = self.morphology_from_mask(seg_logits)
        morph_feature = self.morph_proj(morph_vector)
        alpha = self.amfm_alpha(torch.cat([global_feature, morph_feature], dim=1))
        fused = alpha * global_feature + (1.0 - alpha) * morph_feature
        class_logits = self.classifier(fused)
        return {"seg_logits": seg_logits, "class_logits": class_logits, "attention_map": attention_map, "morph_vector": morph_vector}


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice.mean()


def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    preds = (torch.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    dice_union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    iou_union = dice_union - intersection
    dice = ((2.0 * intersection + 1e-6) / (dice_union + 1e-6)).mean()
    iou = ((intersection + 1e-6) / (iou_union + 1e-6)).mean()
    return dice, iou


def make_malds_targets(labels: torch.Tensor, morph_targets: torch.Tensor, base_eps: float, max_eps: float) -> torch.Tensor:
    batch_size = labels.numel()
    targets = torch.zeros((batch_size, 4), device=labels.device)
    area = morph_targets[:, 0].clamp(0, 1)
    density = morph_targets[:, 1].clamp(0, 1)
    spread = morph_targets[:, 4].clamp(0, 1)
    eps_values = (base_eps + 1.5 * area + 0.5 * density + 0.2 * spread).clamp(base_eps, max_eps)

    for i, label in enumerate(labels.tolist()):
        eps = eps_values[i]
        targets[i, label] = 1.0 - eps
        neighbors: list[int] = []
        if label - 1 >= 0:
            neighbors.append(label - 1)
        if label + 1 <= 3:
            neighbors.append(label + 1)
        if neighbors:
            for neighbor in neighbors:
                targets[i, neighbor] = eps / len(neighbors)
        else:
            targets[i, label] = 1.0
    return targets


def cohen_kappa_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=logits.device)
    for true_label, pred_label in zip(labels.view(-1), preds.view(-1)):
        confusion[true_label.long(), pred_label.long()] += 1.0
    total = confusion.sum()
    if total <= 0:
        return torch.tensor(0.0, device=logits.device)
    observed = confusion.diag().sum() / total
    expected = (confusion.sum(dim=1) * confusion.sum(dim=0)).sum() / (total * total)
    if torch.isclose(1.0 - expected, torch.tensor(0.0, device=logits.device)):
        return torch.tensor(0.0, device=logits.device)
    return (observed - expected) / (1.0 - expected)


def run_epoch(
    model: ProposalHybridResNet50,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    class_weights: torch.Tensor,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    totals = {key: 0.0 for key in ["loss", "seg_loss", "cls_loss", "morph_loss", "dice", "iou", "acc", "kappa"]}
    count = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)
        morph_targets = batch["morph_target"].to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        outputs = model(images)
        seg_loss = bce(outputs["seg_logits"], masks) + dice_loss_from_logits(outputs["seg_logits"], masks)
        target_dist = make_malds_targets(labels, morph_targets, args.base_smoothing, args.max_smoothing)
        per_sample_kl = F.kl_div(F.log_softmax(outputs["class_logits"], dim=1), target_dist, reduction="none").sum(dim=1)
        cls_loss = (per_sample_kl * class_weights[labels]).mean()
        morph_loss = mse(outputs["morph_vector"], morph_targets)
        loss = args.seg_weight * seg_loss + args.cls_weight * cls_loss + args.morph_weight * morph_loss

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        dice, iou = dice_iou_from_logits(outputs["seg_logits"], masks)
        acc = (outputs["class_logits"].argmax(dim=1) == labels).float().mean()
        kappa = cohen_kappa_from_logits(outputs["class_logits"], labels)
        batch_size = images.size(0)
        count += batch_size
        totals["loss"] += loss.item() * batch_size
        totals["seg_loss"] += seg_loss.item() * batch_size
        totals["cls_loss"] += cls_loss.item() * batch_size
        totals["morph_loss"] += morph_loss.item() * batch_size
        totals["dice"] += dice.item() * batch_size
        totals["iou"] += iou.item() * batch_size
        totals["acc"] += acc.item() * batch_size
        totals["kappa"] += kappa.item() * batch_size

    return {key: value / max(1, count) for key, value in totals.items()}


def build_loaders(args: argparse.Namespace, rows: list[dict[str, str]]) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    train_counts = Counter(int(row["label_index"]) for row in train_rows)
    total_train = sum(train_counts.values())
    class_weights = torch.tensor(
        [total_train / max(1, 4 * train_counts.get(idx, 0)) for idx in range(4)],
        dtype=torch.float32,
    )

    sampler = None
    shuffle = True
    if args.weighted_sampler:
        row_weights = [class_weights[int(row["label_index"])].item() for row in train_rows]
        sampler = WeightedRandomSampler(row_weights, num_samples=len(row_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        PreprocessedAcneDataset(train_rows, args.image_size, augment=True, args=args),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        PreprocessedAcneDataset(val_rows, args.image_size, augment=False, args=args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        PreprocessedAcneDataset(test_rows, args.image_size, augment=False, args=args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader, class_weights


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_model.pt"
    last_checkpoint_path = args.output_dir / "last_checkpoint.pt"

    rows = load_rows(args.manifest)
    train_loader, val_loader, test_loader, class_weights = build_loaders(args, rows)
    class_weights = class_weights.to(device)

    print("🚀 Proposal hybrid training: ResNet50 + Spatial Attention + Decoder + AMFM + MA-LDS")
    print(f"🧾 Manifest: {args.manifest}")
    print(f"🖥️ Device: {device}")
    print(f"✅ Split: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"⚖️ Class weights: {[round(x, 4) for x in class_weights.detach().cpu().tolist()]}")
    print(f"🧠 Phase 1: freeze encoder, epochs={args.phase1_epochs}, lr={args.phase1_lr}")
    print(f"🧠 Phase 2: full fine-tuning, epochs={args.phase2_epochs}, lr={args.phase2_lr}")

    model = ProposalHybridResNet50(num_classes=4, pretrained=args.pretrained).to(device)
    start_epoch = 1
    best_val_loss = float("inf")
    history: list[dict[str, float | int | str]] = []
    total_epochs = args.phase1_epochs + args.phase2_epochs

    optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=args.phase1_lr, momentum=0.9)

    if args.resume and last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        history = checkpoint["history"]
        print(f"🔁 Resume dari epoch {start_epoch}")

    for epoch in range(start_epoch, total_epochs + 1):
        phase = "phase1_freeze" if epoch <= args.phase1_epochs else "phase2_unfreeze"
        if phase == "phase1_freeze":
            model.set_encoder_trainable(False)
            lr = args.phase1_lr
        else:
            model.set_encoder_trainable(True)
            lr = args.phase2_lr
        optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=lr, momentum=0.9)

        print(f"📘 Epoch {epoch:03d}/{total_epochs} | {phase} | lr={lr}")
        train_metrics = run_epoch(model, train_loader, device, optimizer, args, class_weights)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, optimizer=None, args=args, class_weights=class_weights)

        row = {
            "epoch": epoch,
            "phase": phase,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_kappa={train_metrics['kappa']:.4f} train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_kappa={val_metrics['kappa']:.4f} val_dice={val_metrics['dice']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)
            print("💾 Best model diperbarui")

        torch.save(
            {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "model_state_dict": model.state_dict(),
                "history": history,
                "args": vars(args),
            },
            last_checkpoint_path,
        )
        save_json({"history": history, "best_val_loss": best_val_loss}, args.output_dir / "history_partial.json")

    print("🧪 Evaluasi akhir test set")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, device, optimizer=None, args=args, class_weights=class_weights)

    summary = {
        "config": {
            **vars(args),
            "manifest": str(args.manifest),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "encoder": "resnet50",
            "fusion": "amfm",
            "classification_loss": "morphology_aware_label_distribution_kl",
            "segmentation_loss": "bce_plus_dice",
        },
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "history": history,
    }
    save_json(summary, args.output_dir / "metrics.json")
    print("✅ Training selesai.")
    print(f"🏁 Best model: {best_model_path}")
    print(f"🔁 Last checkpoint: {last_checkpoint_path}")
    print(f"📊 Metrics: {args.output_dir / 'metrics.json'}")
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
