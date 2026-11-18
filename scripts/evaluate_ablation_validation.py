"""Summarize completed Step 20 ablation runs using validation metrics only.

Also evaluates the seven locked ablation checkpoints once on the shared test
split. Test inference reuses the trained architecture and the same 0.5 mask
threshold as training; no threshold tuning is performed.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


VARIANTS = ("baseline", "no_attention", "no_dilation", "no_amfm", "lds", "no_fusion", "proposed")
METRICS = ("val_acc", "val_kappa", "val_dice", "val_iou")
SHARED_CONFIG = ("image_size", "batch_size", "phase1_epochs", "phase2_epochs",
                 "phase1_lr", "phase2_lr", "seg_weight", "cls_weight", "morph_weight",
                 "base_smoothing", "max_smoothing", "rotation_deg", "hflip_prob",
                 "brightness", "contrast", "seed", "pretrained", "input_normalization")
TEST_MANIFEST_NAME = "refined_manifest_offline_augmented_100pct.csv"
EXPECTED_TEST_IMAGES = 184
TRAINER_NAME = "train_proposal_hybrid_resnet50_malds.py"
TEST_RESULT_FIELDS = ("variant", "acc", "kappa", "macro_f1", "dice", "iou")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scores(actual, predicted):
    """Classification metrics copied from evaluate_single_task_test.py."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Label/prediksi harus tidak kosong dan berpasangan.")
    matrix = [[0] * 4 for _ in range(4)]
    for a, p in zip(actual, predicted):
        if a not in range(4) or p not in range(4):
            raise ValueError("Label/prediksi di luar 0–3.")
        matrix[a][p] += 1
    total = len(actual)
    accuracy = sum(matrix[i][i] for i in range(4)) / total
    expected = sum(sum(matrix[i]) * sum(r[i] for r in matrix) for i in range(4)) / total**2
    classes = []
    for i in range(4):
        support, predicted_count = sum(matrix[i]), sum(r[i] for r in matrix)
        precision = matrix[i][i] / predicted_count if predicted_count else 0.0
        recall = matrix[i][i] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes.append({"class": i, "precision": precision, "recall": recall, "f1": f1, "support": support})
    return {"acc": accuracy, "kappa": (accuracy - expected) / (1 - expected) if expected < 1 else 0.0,
            "macro_f1": sum(r["f1"] for r in classes) / 4,
            "balanced_accuracy": sum(r["recall"] for r in classes) / 4,
            "correct": sum(matrix[i][i] for i in range(4)), "total_images": total,
            "confusion_matrix": matrix, "per_class": classes}


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


def _manifest_records(rows):
    fields = ("filename", "split", "label_index", "image_path", "preprocessed_path")
    return Counter(tuple(row.get(key, "") for key in fields) for row in rows)


def _source_identity(provenance):
    """Recover the augmented manifest and trainer identity pinned by Step 11."""
    step11 = provenance.get("step11")
    if not isinstance(step11, dict):
        raise ValueError("Provenance ablation tidak menyimpan identitas sumber Step 11.")
    command = step11.get("command") or []
    hashes = step11.get("sha256") or {}
    if "--manifest" not in command:
        raise ValueError("Provenance sumber tidak mencatat --manifest; manifest test tidak bisa dipastikan.")
    source_manifest = Path(command[command.index("--manifest") + 1])
    manifest_hash = hashes.get(str(source_manifest))
    if not manifest_hash:
        raise ValueError("Provenance sumber tidak memuat SHA-256 manifest augmentasi.")
    trainers = [(Path(path), digest) for path, digest in hashes.items() if Path(path).name == TRAINER_NAME]
    if len(trainers) != 1:
        raise ValueError(f"Identitas {TRAINER_NAME} tidak tunggal pada provenance sumber.")
    return source_manifest, manifest_hash, trainers[0]


def prepare_test(ablation_dir, manifest_path=None):
    """Validate the shared test split and the seven locked checkpoints only."""
    root = Path(ablation_dir)
    provenance_path = root / "ablation_provenance.json"
    train_val_path = root / "train_val_manifest.csv"
    for path in (provenance_path, train_val_path):
        if not path.is_file():
            raise FileNotFoundError(f"Artefak ablation belum tersedia: {path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("variants") != list(VARIANTS) or provenance.get("skip_test_eval") is not True:
        raise ValueError("Provenance bukan ablation Step 20 lengkap tanpa test.")
    train_val_hash = file_hash(train_val_path)
    if train_val_hash != provenance.get("manifest_sha256"):
        raise ValueError("Hash manifest train/validation tidak cocok dengan provenance.")

    source_manifest, expected_manifest_hash, (trainer_source, trainer_hash) = _source_identity(provenance)
    manifest = Path(manifest_path) if manifest_path else source_manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest test tidak ditemukan: {manifest}")
    if manifest.name != TEST_MANIFEST_NAME:
        raise ValueError(f"Manifest test wajib bernama {TEST_MANIFEST_NAME}.")
    manifest_hash = file_hash(manifest)
    if manifest_hash != expected_manifest_hash:
        raise ValueError("Manifest test bukan manifest aug100 yang dipakai sumber Step 11.")

    all_rows = read_rows(manifest)
    train_val_rows = read_rows(train_val_path)
    if {row["split"] for row in train_val_rows} - {"train", "val"}:
        raise ValueError("Manifest ablation harus berisi train/validation saja.")
    if _manifest_records(train_val_rows) != _manifest_records(
            [row for row in all_rows if row.get("split") in {"train", "val"}]):
        raise ValueError("Train/validation ablation bukan bagian dari manifest test yang sama.")
    test_rows = [dict(row) for row in all_rows if row.get("split") == "test"]
    if len(test_rows) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Test wajib {EXPECTED_TEST_IMAGES} citra; ditemukan {len(test_rows)}.")
    if len({row.get("filename") for row in test_rows}) != len(test_rows):
        raise ValueError("Nama file test duplikat.")
    for row in test_rows:
        row["image_path"] = row.get("preprocessed_path") or row.get("image_path")
        row["mask_path"] = row.get("mask_path") or row.get("refined_mask") or row.get("circle_mask")
        if row.get("label_index") not in {"0", "1", "2", "3"}:
            raise ValueError(f"Label test tidak valid: {row.get('filename')}")
        if not row["image_path"] or not Path(row["image_path"]).is_file():
            raise ValueError(f"Foto test tidak tersedia: {row.get('filename')}")
        if not row["mask_path"] or not Path(row["mask_path"]).is_file():
            raise ValueError(f"Mask test tidak tersedia: {row.get('filename')}")

    variant_rows = []
    preprocessing = None
    for variant in VARIANTS:
        run_dir = root / variant
        metrics_path, checkpoint_path = run_dir / "metrics.json", run_dir / "best_model.pt"
        for path in (metrics_path, checkpoint_path):
            if not path.is_file():
                raise FileNotFoundError(f"Run test belum lengkap: {path}")
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = summary["config"]
        if (config.get("variant") != variant or Path(config.get("manifest", "")) != train_val_path
                or config.get("skip_test_eval") is not True
                or config.get("weighted_sampler") is not False
                or config.get("class_weighting") != "balanced"):
            raise ValueError(f"Konfigurasi run {variant} tidak sesuai protokol Step 20.")
        if summary.get("test_metrics") is not None:
            raise ValueError(f"Run {variant} sudah memiliki hasil test; bukan checkpoint netral.")
        recorded_manifest = config.get("manifest_used")
        if recorded_manifest is not None:
            recorded_path = Path(recorded_manifest)
            if not recorded_path.is_file() or file_hash(recorded_path) != train_val_hash:
                raise ValueError(f"manifest_used run {variant} bukan manifest train/validation run ini.")
        history = summary.get("history")
        if not history:
            raise ValueError(f"History kosong: {variant}.")
        best = min(history, key=lambda item: item["val_loss"])
        if not math.isclose(float(best["val_loss"]), float(summary["best_val_loss"]), rel_tol=1e-6):
            raise ValueError(f"Best validation loss tidak cocok: {variant}.")
        normalization = config.get("resolved_input_normalization", config.get("input_normalization"))
        if normalization not in {"imagenet", "none"}:
            raise ValueError(f"Normalisasi run {variant} tidak diketahui: {normalization}")
        current = {"image_size": int(config["image_size"]), "batch_size": int(config["batch_size"]),
                   "input_normalization": normalization}
        if preprocessing is not None and current != preprocessing:
            raise ValueError(f"Preprocessing run {variant} berbeda dari varian lain.")
        preprocessing = current
        variant_rows.append({"variant": variant, "best_epoch": best["epoch"],
                             "val_loss": best["val_loss"], "checkpoint": str(checkpoint_path),
                             "checkpoint_sha256": file_hash(checkpoint_path)})

    plan = {
        "run": root.name, "ablation_dir": str(root), "split": "test",
        "manifest": str(manifest), "manifest_sha256": manifest_hash,
        "train_val_manifest": str(train_val_path), "train_val_manifest_sha256": train_val_hash,
        "total_images": len(test_rows),
        "image_size": preprocessing["image_size"], "batch_size": preprocessing["batch_size"],
        "input_normalization": preprocessing["input_normalization"],
        "trainer_source": str(trainer_source), "trainer_sha256": trainer_hash,
        "variants": variant_rows,
        "test_records_sha256": hashlib.sha256(json.dumps(
            [[row["filename"], row["label_index"], row["image_path"]] for row in test_rows],
            sort_keys=True).encode()).hexdigest(),
    }
    return plan, test_rows


def _resolve_device(name):
    import torch
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA diminta tetapi tidak tersedia.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_trainer(plan, trainer_path=None):
    import importlib.util
    candidates = []
    if trainer_path:
        candidates.append(Path(trainer_path))
    else:
        root = Path(plan["ablation_dir"])
        candidates.extend([Path(plan["trainer_source"]),
                           root.parent.parent / "ablation_scripts" / TRAINER_NAME,
                           Path(__file__).resolve().parent / TRAINER_NAME])
    existing = next((path for path in candidates if path.is_file()), None)
    if existing is None:
        raise FileNotFoundError(
            f"Trainer {TRAINER_NAME} tidak ditemukan. Sediakan salinan SHA-256 {plan['trainer_sha256']}.")
    if file_hash(existing) != plan["trainer_sha256"]:
        raise ValueError(
            f"Trainer {existing} berbeda dari trainer Step 11 (SHA-256 {plan['trainer_sha256']}). "
            "Pulihkan salinan asli; jangan evaluasi dengan arsitektur lain.")
    spec = importlib.util.spec_from_file_location("ablation_proposal_trainer", existing)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_test_results(path, rows):
    """Write exactly the requested columns: variant, acc, kappa, macro_f1, dice, iou."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEST_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_test(ablation_dir, manifest_path=None, trainer_path=None, device="auto"):
    """Evaluate the seven locked ablation checkpoints once on the shared test split."""
    plan, rows = prepare_test(ablation_dir, manifest_path)
    root = Path(ablation_dir)
    result_path = root / "ablation_test_results.csv"
    provenance_path = root / "ablation_test_provenance.json"
    if provenance_path.is_file():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing.get("plan") != plan:
            raise ValueError("Sumber/checkpoint/manifest evaluasi berbeda. Gunakan folder output baru.")
        if result_path.is_file():
            print("Evaluasi test yang sama sudah tersedia; menjalankan ulang dilewati.")
            return read_rows(result_path)

    import torch
    from torch.utils.data import DataLoader

    trainer = _load_trainer(plan, trainer_path)
    resolved = _resolve_device(device)
    print(f"TEST ONLY | run={plan['run']} | {plan['total_images']} citra | threshold 0.5 | device={resolved}",
          flush=True)
    dataset = trainer.PreprocessedAcneDataset(rows, plan["image_size"], False, argparse.Namespace(
        resolved_input_normalization=plan["input_normalization"]))
    loader = DataLoader(dataset, batch_size=plan["batch_size"], shuffle=False, num_workers=0)

    results, matrices = [], {}
    for item in plan["variants"]:
        variant = item["variant"]
        state = torch.load(item["checkpoint"], map_location="cpu", weights_only=True)
        weights = state.get("model_state", state.get("model_state_dict", state))
        model = trainer.ProposalHybridResNet50(num_classes=4, pretrained=False, variant=variant)
        model.load_state_dict(weights, strict=True)
        del state, weights
        model.to(resolved).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        actual, predicted = [], []
        dice_sum = iou_sum = 0.0
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(resolved)
                masks = batch["mask"].to(resolved)
                output = model(images)
                labels = batch["label"]
                actual.extend(labels.tolist())
                predicted.extend(output["class_logits"].argmax(dim=1).cpu().tolist())
                dice, iou = trainer.dice_iou_from_logits(output["seg_logits"], masks)
                batch_size = labels.numel()
                dice_sum += dice.item() * batch_size
                iou_sum += iou.item() * batch_size
        classification = scores(actual, predicted)
        row = {"variant": variant, "acc": classification["acc"], "kappa": classification["kappa"],
               "macro_f1": classification["macro_f1"],
               "dice": dice_sum / len(rows), "iou": iou_sum / len(rows)}
        results.append(row)
        matrices[variant] = classification["confusion_matrix"]
        print(f"✅ {variant}: acc={row['acc']:.4f} kappa={row['kappa']:.4f} "
              f"macro_f1={row['macro_f1']:.4f} dice={row['dice']:.4f} iou={row['iou']:.4f}", flush=True)
        del model
    if getattr(torch, "cuda", None) and torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload = {"plan": plan, "results": results, "confusion_matrices": matrices}
    temp = provenance_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(provenance_path)
    write_test_results(result_path, results)
    print(f"Hasil test tersimpan: {result_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest aug100; wajib sama SHA-256 dengan sumber Step 11.")
    parser.add_argument("--trainer", type=Path, default=None,
                        help="Salinan trainer dengan SHA-256 identik trainer Step 11.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    results = evaluate_test(args.ablation_dir, args.manifest, args.trainer, args.device)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
