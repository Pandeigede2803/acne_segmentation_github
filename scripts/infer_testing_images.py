#!/usr/bin/env python3
"""
Run inference for testing images listed in a manifest CSV.

Default input:
- projectsegmentasi/refined_masks_output/refined_manifest.csv
- projectsegmentasi/baseline_runs/dilated_unet_acne04v2_bce_dice/best_model.pt

Default output:
- projectsegmentasi/testing_predictions/dilated_unet_acne04v2_bce_dice/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

import train_baseline_dilated_unet as base
from train_proposal_hybrid_resnet50_malds import ProposalHybridResNet50


INDEX_TO_LABEL = {
    0: "levle0",
    1: "levle1",
    2: "levle2",
    3: "levle3",
}


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Infer all testing images from manifest and save masks, overlays, and metrics."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=base_dir / "refined_masks_output" / "refined_manifest.csv",
        help="Manifest CSV with split, image/preprocessed path, mask path, and label columns.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base_dir / "baseline_runs" / "dilated_unet_acne04v2_bce_dice" / "best_model.pt",
        help="Model checkpoint (.pt) to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "testing_predictions" / "dilated_unet_acne04v2_bce_dice",
        help="Folder output for predicted masks, overlays, CSV, and summary JSON.",
    )
    parser.add_argument("--split", default="test", help="Split to infer. Use 'all' for every row.")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--model-type",
        choices=("baseline", "proposal"),
        default="baseline",
        help="Architecture used by the checkpoint.",
    )
    return parser.parse_args()


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    base_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / path,
        base_dir.parent / path,
        base_dir / path,
        manifest_path.parent / path,
    ]
    if path.parts and path.parts[0] == base_dir.name:
        candidates.append(base_dir / Path(*path.parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_rows(manifest_path: Path, split: str) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if split != "all":
        rows = [row for row in rows if row.get("split") == split]
    if not rows:
        raise ValueError(f"Tidak ada data untuk split='{split}' di {manifest_path}")
    return rows


def row_image_path(row: dict[str, str], manifest_path: Path) -> Path:
    value = row.get("preprocessed_path") or row.get("image_path")
    if not value:
        raise KeyError("Manifest harus punya kolom preprocessed_path atau image_path")
    return resolve_manifest_path(value, manifest_path)


def row_mask_path(row: dict[str, str], manifest_path: Path) -> Path | None:
    value = row.get("refined_mask") or row.get("mask_path") or row.get("circle_mask")
    if not value:
        return None
    return resolve_manifest_path(value, manifest_path)


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    image_tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    image_tensor = image_tensor.view(image_size, image_size, 3)
    return image_tensor.permute(2, 0, 1).float() / 255.0


def mask_to_tensor(mask_path: Path, image_size: int) -> torch.Tensor:
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        if mask.size != (image_size, image_size):
            mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        mask_tensor = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
    mask_tensor = mask_tensor.view(image_size, image_size).float() / 255.0
    return (mask_tensor > 0.5).float()


def tensor_to_mask_image(mask: torch.Tensor) -> Image.Image:
    mask = (mask.detach().cpu().clamp(0, 1) * 255).to(torch.uint8).contiguous()
    return Image.frombytes("L", (mask.shape[1], mask.shape[0]), bytes(mask.view(-1).tolist()))


def make_overlay(image: Image.Image, pred_mask: torch.Tensor, gt_mask: torch.Tensor | None) -> Image.Image:
    base_image = image.convert("RGB")
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    pred = pred_mask.detach().cpu()

    pred_pixels = pred > 0.5
    overlay_pixels = overlay.load()
    for y in range(base_image.size[1]):
        for x in range(base_image.size[0]):
            if pred_pixels[y, x]:
                overlay_pixels[x, y] = (0, 255, 80, 120)

    if gt_mask is not None:
        gt = gt_mask.detach().cpu() > 0.5
        for y in range(base_image.size[1]):
            for x in range(base_image.size[0]):
                if gt[y, x] and pred_pixels[y, x]:
                    overlay_pixels[x, y] = (255, 255, 0, 140)
                elif gt[y, x]:
                    overlay_pixels[x, y] = (255, 0, 0, 120)

    return Image.alpha_composite(base_image.convert("RGBA"), overlay).convert("RGB")


def dice_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> tuple[float, float]:
    pred = pred_mask.float()
    gt = gt_mask.float()
    intersection = (pred * gt).sum()
    dice_union = pred.sum() + gt.sum()
    iou_union = pred.sum() + gt.sum() - intersection
    dice = (2.0 * intersection + 1e-6) / (dice_union + 1e-6)
    iou = (intersection + 1e-6) / (iou_union + 1e-6)
    return float(dice.item()), float(iou.item())


def cohen_kappa_from_pairs(true_labels: list[int], pred_labels: list[int], num_classes: int = 4) -> float:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for true_label, pred_label in zip(true_labels, pred_labels):
        confusion[true_label, pred_label] += 1.0

    total = confusion.sum()
    if total <= 0:
        return 0.0
    observed = confusion.diag().sum() / total
    expected = (confusion.sum(dim=1) * confusion.sum(dim=0)).sum() / (total * total)
    if torch.isclose(1.0 - expected, torch.tensor(0.0)):
        return 0.0
    return float(((observed - expected) / (1.0 - expected)).item())


def main() -> None:
    args = parse_args()
    device = base.resolve_device(args.device)
    rows = load_rows(args.manifest, args.split)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.checkpoint}")

    pred_dir = args.output_dir / "pred_masks"
    overlay_dir = args.output_dir / "overlays"
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type == "proposal":
        model = ProposalHybridResNet50(num_classes=4, pretrained=False).to(device)
    else:
        model = base.DilatedUNetAMFM(num_classes=4).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    results: list[dict[str, object]] = []
    true_labels: list[int] = []
    pred_labels: list[int] = []

    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            filename = row.get("filename") or Path(row_image_path(row, args.manifest)).name
            image_path = row_image_path(row, args.manifest)
            mask_path = row_mask_path(row, args.manifest)

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if image.size != (args.image_size, args.image_size):
                    image = image.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
                image_tensor = image_to_tensor(image, args.image_size).unsqueeze(0).to(device)

                outputs = model(image_tensor)
                seg_prob = torch.sigmoid(outputs["seg_logits"])[0, 0].cpu()
                pred_mask = (seg_prob > args.threshold).float()
                class_prob = torch.softmax(outputs["class_logits"], dim=1)[0].cpu()
                pred_label = int(class_prob.argmax().item())
                confidence = float(class_prob[pred_label].item())

                gt_mask = mask_to_tensor(mask_path, args.image_size) if mask_path and mask_path.exists() else None
                dice = ""
                iou = ""
                if gt_mask is not None:
                    dice, iou = dice_iou(pred_mask, gt_mask)

                true_label = int(row.get("label_index", -1))
                if true_label >= 0:
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)

                stem = Path(filename).stem
                pred_mask_path = pred_dir / f"{stem}_pred.png"
                overlay_path = overlay_dir / f"{stem}_overlay.png"
                tensor_to_mask_image(pred_mask).save(pred_mask_path)
                make_overlay(image, pred_mask, gt_mask).save(overlay_path)

            results.append(
                {
                    "filename": filename,
                    "split": row.get("split", ""),
                    "true_label_index": true_label,
                    "true_label_name": row.get("label_name", ""),
                    "pred_label_index": pred_label,
                    "pred_label_name": INDEX_TO_LABEL[pred_label],
                    "confidence": confidence,
                    "dice": dice,
                    "iou": iou,
                    "gt_area": int(gt_mask.sum().item()) if gt_mask is not None else "",
                    "pred_area": int(pred_mask.sum().item()),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path else "",
                    "pred_mask_path": str(pred_mask_path),
                    "overlay_path": str(overlay_path),
                }
            )
            print(f"[{idx}/{len(rows)}] {filename} -> {INDEX_TO_LABEL[pred_label]} ({confidence:.4f})")

    csv_path = args.output_dir / "testing_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    rows_with_metrics = [row for row in results if row["dice"] != ""]
    correct = sum(1 for true_label, pred_label in zip(true_labels, pred_labels) if true_label == pred_label)
    summary = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "image_size": args.image_size,
        "threshold": args.threshold,
        "device": str(device),
        "total_images": len(results),
        "classification_accuracy": correct / max(1, len(true_labels)),
        "classification_kappa": cohen_kappa_from_pairs(true_labels, pred_labels) if true_labels else "",
        "mean_dice": sum(float(row["dice"]) for row in rows_with_metrics) / max(1, len(rows_with_metrics)),
        "mean_iou": sum(float(row["iou"]) for row in rows_with_metrics) / max(1, len(rows_with_metrics)),
        "csv_path": str(csv_path),
        "pred_mask_dir": str(pred_dir),
        "overlay_dir": str(overlay_dir),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Selesai.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Predicted masks: {pred_dir}")
    print(f"Overlays: {overlay_dir}")
    print(
        "Metrics: "
        f"acc={summary['classification_accuracy']:.4f}, "
        f"kappa={summary['classification_kappa']:.4f}, "
        f"dice={summary['mean_dice']:.4f}, "
        f"iou={summary['mean_iou']:.4f}"
    )


if __name__ == "__main__":
    main()
