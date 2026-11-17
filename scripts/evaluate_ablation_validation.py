"""Summarize completed Step 20 ablation runs using validation metrics only."""

import csv
import hashlib
import json
import math
from pathlib import Path


VARIANTS = ("baseline", "no_attention", "no_dilation", "no_amfm", "lds", "no_fusion", "proposed")
METRICS = ("val_acc", "val_kappa", "val_dice", "val_iou")
SHARED_CONFIG = ("image_size", "batch_size", "phase1_epochs", "phase2_epochs",
                 "phase1_lr", "phase2_lr", "seg_weight", "cls_weight", "morph_weight",
                 "base_smoothing", "max_smoothing", "rotation_deg", "hflip_prob",
                 "brightness", "contrast", "seed", "pretrained", "input_normalization")


def evaluate(ablation_dir):
    """Return rows and deltas vs the separately trained proposed control.

    Fail closed on missing runs or mismatched manifest/config instead of silently
    comparing incomplete experiments or the unrelated Step 11 proposed run.
    """
    root = Path(ablation_dir)
    provenance = json.loads((root / "ablation_provenance.json").read_text(encoding="utf-8"))
    if provenance.get("variants") != list(VARIANTS) or provenance.get("skip_test_eval") is not True:
        raise ValueError("Provenance bukan ablation Step 20 lengkap tanpa test.")
    manifest = root / "train_val_manifest.csv"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != provenance.get("manifest_sha256"):
        raise ValueError("Hash manifest ablation tidak cocok dengan provenance.")
    with manifest.open(newline="", encoding="utf-8") as file:
        splits = {row["split"] for row in csv.DictReader(file)}
    if not splits or not splits <= {"train", "val"} or "val" not in splits:
        raise ValueError("Manifest harus berisi train/validation saja, tanpa test.")

    rows = []
    shared_config = None
    for variant in VARIANTS:
        path = root / variant / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Run belum selesai: {path}. Selesaikan Step 20 dahulu.")
        summary = json.loads(path.read_text(encoding="utf-8"))
        config = summary["config"]
        if (config.get("variant") != variant or Path(config.get("manifest", "")) != manifest
                or config.get("skip_test_eval") is not True
                or config.get("weighted_sampler") is not False
                or config.get("class_weighting") != "balanced"):
            raise ValueError(f"Konfigurasi run {variant} tidak sesuai protokol Step 20.")
        current_config = {key: config.get(key) for key in SHARED_CONFIG}
        if shared_config is not None and current_config != shared_config:
            raise ValueError(f"Konfigurasi bersama berbeda pada varian {variant}.")
        shared_config = current_config
        if summary.get("test_metrics") is not None:
            raise ValueError(f"Run {variant} memuat hasil test; bukan run validation-only.")
        history = summary.get("history")
        if not history:
            raise ValueError(f"History kosong: {variant}.")
        best = min(history, key=lambda item: item["val_loss"])
        if not math.isclose(float(best["val_loss"]), float(summary["best_val_loss"]), rel_tol=1e-6):
            raise ValueError(f"Best validation loss tidak cocok: {variant}.")
        row = {"variant": variant, "best_epoch": best["epoch"],
               "val_loss": best["val_loss"], "train_accuracy": best["train_acc"]}
        row.update({metric.replace("val_acc", "val_accuracy"): best[metric] for metric in METRICS})
        rows.append(row)

    proposed = rows[-1]
    for row in rows:
        for metric in ("val_accuracy", "val_kappa", "val_dice", "val_iou"):
            row[f"delta_{metric}_vs_proposed"] = row[metric] - proposed[metric]
    return rows
