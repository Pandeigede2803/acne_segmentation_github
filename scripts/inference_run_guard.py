"""Resolve verified proposed test inputs from a saved training run (no ML imports)."""

import csv
import hashlib
import json
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_run(checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.name != "best_model.pt" or not checkpoint.is_file():
        raise FileNotFoundError("Pilih best_model.pt dari run training yang sudah selesai.")
    metrics_path = checkpoint.parent / "metrics.json"
    provenance_path = checkpoint.parent / "augmentation_training_provenance.json"
    if not metrics_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("metrics.json dan provenance Step 11 harus berada di samping checkpoint.")
    metrics = json.loads(metrics_path.read_text())
    config = metrics["config"]
    provenance = json.loads(provenance_path.read_text())
    manifest = Path(config["manifest"])
    source_command = provenance["command"]
    if "--manifest" not in source_command or manifest != Path(source_command[source_command.index("--manifest") + 1]):
        raise ValueError("Manifest metrics.json berbeda dari provenance training.")
    if not manifest.is_file() or provenance["sha256"].get(str(manifest)) != file_hash(manifest):
        raise ValueError("Manifest training hilang/berubah. Jangan menggantinya dengan manifest runtime/eksperimen lain.")
    if config.get("variant") != "proposed" or config.get("encoder") != "resnet50":
        raise ValueError("Guard ini khusus proposed ResNet-50 Step 11.")
    audit = provenance.get("dataset_audit", {})
    if audit.get("eligible_images") != 1204 or audit.get("annotations_sha256") != "ae03fc23fed0925c254dc4c4c9e55b539bd199b236151cb649857378a43539e9":
        raise ValueError("Run tidak memiliki provenance audit ACNE04-v2 resmi.")
    normalization = config.get("resolved_input_normalization", config.get("input_normalization"))
    if normalization not in {"imagenet", "none"}:
        raise ValueError("Normalisasi input training tidak diketahui; tidak boleh fallback diam-diam.")
    if config.get("skip_test_eval") or not metrics.get("test_metrics"):
        raise ValueError("Run belum memiliki hasil test final; jangan memakai checkpoint ablation sebagai final run.")
    with manifest.open(newline="", encoding="utf-8") as handle:
        test_rows = [r for r in csv.DictReader(handle) if r.get("split") == "test"]
    if not test_rows or len(test_rows) != audit.get("split_counts", {}).get("test"):
        raise ValueError("Jumlah test berbeda dari audit training.")
    if len({r["filename"] for r in test_rows}) != len(test_rows):
        raise ValueError("Nama sampel test duplikat.")
    for row in test_rows:
        image = row.get("preprocessed_path") or row.get("image_path")
        # Same priority as the training loader.
        mask = row.get("mask_path") or row.get("refined_mask") or row.get("circle_mask")
        if not image or not mask or not Path(image).is_file() or not Path(mask).is_file():
            raise FileNotFoundError(f"Foto/mask test tidak tersedia: {row.get('filename')}")
        if row.get("label_index") not in set("0123"):
            raise ValueError("Label test tidak valid.")
    best = min(metrics["history"], key=lambda r: r["val_loss"])
    if abs(best["val_loss"] - metrics["best_val_loss"]) > 1e-8:
        raise ValueError("best_val_loss tidak konsisten dengan history.")
    return {"manifest": str(manifest), "checkpoint": str(checkpoint), "split": "test",
            "image_size": int(config["image_size"]), "threshold": 0.5,
            "input_normalization": normalization, "variant": "proposed", "model_type": "proposal",
            "total_images": len(test_rows), "checkpoint_sha256": file_hash(checkpoint),
            "manifest_sha256": file_hash(manifest), "metrics_sha256": file_hash(metrics_path),
            "training_provenance_sha256": file_hash(provenance_path),
            "best_epoch_from_history": best["epoch"], "reference_test_metrics": metrics["test_metrics"]}


def validate_arguments(args):
    plan = resolve_run(args.checkpoint)
    if Path(args.manifest).resolve() != Path(plan["manifest"]).resolve():
        raise ValueError("--manifest bukan manifest yang dipakai checkpoint saat training.")
    for field in ("split", "image_size", "threshold", "model_type"):
        if getattr(args, field) != plan[field]:
            raise ValueError(f"Konfigurasi inferensi {field} berbeda dari evaluasi training.")
    for field in ("variant", "input_normalization"):
        if getattr(args, field) not in {"auto", plan[field]}:
            raise ValueError(f"Konfigurasi inferensi {field} berbeda dari training.")
    return plan
