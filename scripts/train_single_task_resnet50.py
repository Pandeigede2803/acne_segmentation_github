"""Image-only ResNet-50 severity control. Never loads masks or evaluates test."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def recover_source_run(output_dir):
    """Recover image-only inputs from trusted Step 11 records after runtime reset.

    The original raw images/annotation/trainer may have lived on /content and
    disappeared. They are not needed by this classifier. Their historical
    identity stays in provenance; persisted manifests must still match exactly.
    """
    output_dir = Path(output_dir)
    record_path = output_dir / "augmentation_training_provenance.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"Provenance Step 11 tidak ditemukan: {record_path}. Periksa ST_SOURCE_RUN, bukan mengulang training.")
    protocol = json.loads(record_path.read_text())
    command = protocol["command"]
    def option(flag):
        if flag not in command:
            raise ValueError(f"Provenance Step 11 tidak memiliki {flag}.")
        return command[command.index(flag) + 1]

    if (option("--variant") != "proposed" or option("--label-distribution") != "malds"
            or option("--class-weighting") != "balanced" or protocol["weighted_sampler"]
            or protocol["target_ratio"] != 1.0 or "--weighted-sampler" in command
            or "--pretrained" not in command):
        raise ValueError("Sumber bukan proposed augmented 100% balanced tanpa sampler.")
    hashes = protocol["sha256"]
    augmented = Path(option("--manifest"))
    originals = [Path(p) for p in hashes if Path(p).name == "refined_manifest.csv"]
    if len(originals) != 1:
        raise ValueError("Manifest asli tidak dapat diidentifikasi dari provenance.")
    original = originals[0]
    for path in (original, augmented):
        if not path.is_file() or hashes.get(str(path)) != file_hash(path):
            raise ValueError(f"Manifest hilang/berubah: {path}. Pulihkan file run asli; jangan mengganti dengan manifest lain.")
    audit = protocol.get("dataset_audit", {})
    official_hash = "ae03fc23fed0925c254dc4c4c9e55b539bd199b236151cb649857378a43539e9"
    if audit.get("eligible_images") != 1204 or audit.get("annotations_sha256") != official_hash:
        raise ValueError("Provenance tidak memiliki bukti audit v2 resmi 1.204 gambar.")
    def read_csv(path):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames, list(reader)

    fields, original_rows = read_csv(original)
    augmented_fields, augmented_rows = read_csv(augmented)
    expected = {"levle0": 470, "levle1": 579, "levle2": 133, "levle3": 22}
    if (len(original_rows) != 1204 or len({r["filename"] for r in original_rows}) != 1204
            or Counter(r["label_name"] for r in original_rows) != expected
            or set(fields) != set(augmented_fields)):
        raise ValueError("Struktur/jumlah sampel manifest v2 tidak sesuai.")
    def records(rows):
        return Counter(tuple(r.get(k, "") for k in fields) for r in rows)

    if records(original_rows) - records(augmented_rows):
        raise ValueError("Baris asli hilang/berubah pada manifest augmented.")
    if records([r for r in original_rows if r["split"] != "train"]) != records([r for r in augmented_rows if r["split"] != "train"]):
        raise ValueError("Validation/test berubah pada manifest augmented.")
    before = Counter(r["label_index"] for r in original_rows if r["split"] == "train")
    after = Counter(r["label_index"] for r in augmented_rows if r["split"] == "train")
    if set(before) != set("0123") or set(after) != set(before) or set(after.values()) != {max(before.values())}:
        raise ValueError("Train augmented belum seimbang 100%.")
    if len({r["filename"] for r in augmented_rows}) != len(augmented_rows):
        raise ValueError("Nama sampel augmented duplikat.")
    # Only preprocessed train/val photos must remain readable; no raw/test/mask access.
    read_train_val_rows(augmented)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        config = json.loads(metrics_path.read_text())["config"]
        if Path(config["manifest"]) != augmented or config["variant"] != "proposed":
            raise ValueError("metrics.json bukan run sumber yang sama.")
    return {"protocol": protocol, "source_record": record_path,
            "original_manifest": original, "augmented_manifest": augmented}


def read_train_val_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        # Test is removed before checking paths or constructing a dataset.
        rows = [r for r in csv.DictReader(handle) if r.get("split") in {"train", "val"}]
    for row in rows:
        row["input_path"] = row.get("preprocessed_path") or row.get("image_path")
        if not row["input_path"] or not Path(row["input_path"]).is_file():
            raise ValueError("Foto train/validation tidak tersedia.")
        if row.get("label_index") not in {"0", "1", "2", "3"}:
            raise ValueError("Label train/validation harus 0–3.")
    for split in ("train", "val"):
        if {r["label_index"] for r in rows if r["split"] == split} != set("0123"):
            raise ValueError(f"Split {split} wajib berisi keempat kelas.")
    return rows


def classification_summary(y_true, y_pred):
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Prediksi harus tidak kosong dan berpasangan.")
    cm = [[0] * 4 for _ in range(4)]
    for actual, predicted in zip(y_true, y_pred):
        cm[actual][predicted] += 1
    n = len(y_true)
    accuracy = sum(cm[i][i] for i in range(4)) / n
    expected = sum(sum(cm[i]) * sum(r[i] for r in cm) for i in range(4)) / n**2
    per_class = []
    for i in range(4):
        support, predicted = sum(cm[i]), sum(r[i] for r in cm)
        precision = cm[i][i] / predicted if predicted else 0.0
        recall = cm[i][i] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class": i, "precision": precision, "recall": recall,
                          "f1": f1, "support": support})
    return {"acc": accuracy, "kappa": (accuracy - expected) / (1 - expected) if expected < 1 else 0.0,
            "macro_f1": sum(r["f1"] for r in per_class) / 4,
            "confusion_matrix": cm, "per_class": per_class}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase1-epochs", type=int, default=30)
    parser.add_argument("--phase2-epochs", type=int, default=90)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=1e-4)
    parser.add_argument("--rotation-deg", type=float, default=15.0)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--brightness", type=float, default=0.15)
    parser.add_argument("--contrast", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_train_val_rows(args.manifest)
    if min(args.phase1_epochs, args.phase2_epochs) < 0 or args.phase1_epochs + args.phase2_epochs <= 0:
        raise ValueError("Jumlah epoch tidak valid.")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    config.update({"variant": "resnet50_single_task", "classification_loss": "class_weighted_ce",
                   "input_normalization": "imagenet", "weighted_sampler": False,
                   "class_weighting": "balanced", "test_evaluated": False,
                   "manifest_sha256": file_hash(args.manifest), "script_sha256": file_hash(__file__)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / "protocol.json"
    checkpoint_path = args.output_dir / "last_checkpoint.pt"
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text()) != config:
            raise ValueError("Folder/protokol berbeda. Gunakan folder baru atau resume dengan konfigurasi sama.")
    elif any(args.output_dir.iterdir()):
        raise ValueError("Folder berisi file tanpa protokol; gunakan folder baru.")
    elif args.resume:
        raise ValueError("Resume memerlukan protokol dan checkpoint run ini.")
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint resume tidak tersedia.")

    # Lazy imports keep manifest/metric helpers testable without ML dependencies.
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet50_Weights, resnet50
    from PIL import Image, ImageEnhance

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    class ImageDataset(Dataset):
        def __init__(self, selected, augment):
            self.rows, self.augment = selected, augment

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(row["input_path"]) as photo:
                photo = photo.convert("RGB")
            if self.augment:
                photo = photo.rotate(random.uniform(-args.rotation_deg, args.rotation_deg),
                                     resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                if random.random() < args.hflip_prob:
                    photo = photo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if args.brightness > 0:
                    photo = ImageEnhance.Brightness(photo).enhance(random.uniform(1 - args.brightness, 1 + args.brightness))
                if args.contrast > 0:
                    photo = ImageEnhance.Contrast(photo).enhance(random.uniform(1 - args.contrast, 1 + args.contrast))
            if photo.size != (args.image_size, args.image_size):
                photo = photo.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
            tensor = torch.frombuffer(bytearray(photo.tobytes()), dtype=torch.uint8)
            tensor = tensor.view(args.image_size, args.image_size, 3).permute(2, 0, 1).float() / 255
            tensor = (tensor - tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]) / tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
            return tensor, int(row["label_index"])

    class SingleTaskResNet50(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT if args.pretrained else None)
            self.encoder.fc = nn.Identity()
            # Match proposed global projection and classifier widths, without attention/fusion.
            self.global_proj = nn.Linear(2048, 128)
            self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 4))

        def forward(self, images):
            return self.classifier(self.global_proj(self.encoder(images)))

    selected = {split: [r for r in rows if r["split"] == split] for split in ("train", "val")}
    loaders = {split: DataLoader(ImageDataset(selected[split], split == "train"),
                               batch_size=args.batch_size, shuffle=split == "train",
                               num_workers=args.num_workers) for split in selected}
    counts = Counter(int(r["label_index"]) for r in selected["train"])
    weights = torch.tensor([len(selected["train"]) / (4 * counts[i]) for i in range(4)], device=device)
    model = SingleTaskResNet50().to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    phase1 = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=args.phase1_lr, momentum=0.9)
    phase2 = torch.optim.SGD(model.parameters(), lr=args.phase2_lr, momentum=0.9)
    history, best_loss, best_epoch, best_state, best_predictions = [], float("inf"), None, None, None
    start_epoch = 1
    if args.resume:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("Konfigurasi checkpoint berbeda.")
        model.load_state_dict(state["model"])
        phase1.load_state_dict(state["phase1_optimizer"])
        phase2.load_state_dict(state["phase2_optimizer"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        history, best_loss, best_epoch = state["history"], state["best_val_loss"], state["best_epoch"]
        best_state, best_predictions = state["best_state"], state["best_predictions"]
        start_epoch = state["epoch"] + 1
    else:
        write_json(protocol_path, config)

    def atomic_save(value, path):
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temporary)
        temporary.replace(path)

    def run_epoch(loader, optimizer=None):
        model.train(optimizer is not None)
        loss_sum, y_true, y_pred = 0.0, [], []
        with torch.set_grad_enabled(optimizer is not None):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                if optimizer is not None:
                    optimizer.zero_grad()
                logits = model(images)
                # Per-sample weighting matches the hybrid's loss reduction for hard labels.
                loss = (F.cross_entropy(logits, labels, reduction="none") * weights[labels]).mean()
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * len(labels)
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(logits.detach().argmax(1).cpu().tolist())
        result = classification_summary(y_true, y_pred)
        result["loss"] = loss_sum / len(y_true)
        return result, {"true": y_true, "predicted": y_pred}

    total_epochs = args.phase1_epochs + args.phase2_epochs
    print(f"Single-task ResNet50 | train={len(selected['train'])} val={len(selected['val'])} | no masks, no test", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} | epochs={total_epochs}", flush=True)
    for epoch in range(start_epoch, total_epochs + 1):
        frozen = epoch <= args.phase1_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad = not frozen
        phase = "phase1_freeze" if frozen else "phase2_unfreeze"
        print(f"Epoch {epoch}/{total_epochs} | {phase}", flush=True)
        train, _ = run_epoch(loaders["train"], phase1 if frozen else phase2)
        val, predictions = run_epoch(loaders["val"])
        row = {"epoch": epoch, "phase": phase,
               **{f"train_{k}": train[k] for k in ("loss", "acc", "kappa", "macro_f1")},
               **{f"val_{k}": val[k] for k in ("loss", "acc", "kappa", "macro_f1")}}
        history.append(row)
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_predictions = predictions
        state = {"config": config, "epoch": epoch, "model": model.state_dict(),
                 "phase1_optimizer": phase1.state_dict(), "phase2_optimizer": phase2.state_dict(),
                 "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "history": history, "best_val_loss": best_loss, "best_epoch": best_epoch,
                 "best_state": best_state, "best_predictions": best_predictions}
        atomic_save(state, checkpoint_path)
        # Best state is also embedded in the atomic last checkpoint for recovery.
        if epoch == best_epoch:
            atomic_save(best_state, args.output_dir / "best_model.pt")
        write_json(args.output_dir / "history_partial.json", {"history": history, "best_val_loss": best_loss})
        print(f"train_acc={train['acc']:.4f} val_acc={val['acc']:.4f} val_kappa={val['kappa']:.4f} val_loss={val['loss']:.4f} best_epoch={best_epoch}", flush=True)

    atomic_save(best_state, args.output_dir / "best_model.pt")
    best_report = classification_summary(best_predictions["true"], best_predictions["predicted"])
    best_report["loss"] = best_loss
    with (args.output_dir / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "true_label", "predicted_label"])
        writer.writerows((r.get("filename", Path(r["input_path"]).name), actual, predicted)
                         for r, actual, predicted in zip(selected["val"], best_predictions["true"], best_predictions["predicted"]))
    write_json(args.output_dir / "metrics.json", {"config": config, "status": "completed", "history": history,
               "best_epoch": best_epoch, "best_val_loss": best_loss,
               "validation_metrics": best_report, "test_metrics": None})
    print("Selesai. Hasil hanya validation; Dice/IoU tidak berlaku untuk single-task klasifikasi.", flush=True)


if __name__ == "__main__":
    main()
