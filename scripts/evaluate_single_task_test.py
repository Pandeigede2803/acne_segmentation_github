"""Evaluate a locked single-task checkpoint on the proposed run's exact test split.

No training, optimizer, augmentation, or writes to the original training runs.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


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


def prepare_test(single_task_run, proposed_run):
    single_task_run, proposed_run = Path(single_task_run), Path(proposed_run)
    st_metrics_path = single_task_run / "metrics.json"
    proposed_metrics_path = proposed_run / "metrics.json"
    source_record = proposed_run / "augmentation_training_provenance.json"
    bundle_record = single_task_run.parent / "source_provenance.json"
    required = [st_metrics_path, proposed_metrics_path, source_record, bundle_record,
                single_task_run / "protocol.json", single_task_run / "best_model.pt",
                single_task_run / "last_checkpoint.pt"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Artefak belum tersedia: {path}. Periksa path; tidak perlu training ulang jika run sudah selesai.")
    single = json.loads(st_metrics_path.read_text())
    proposed = json.loads(proposed_metrics_path.read_text())
    source = json.loads(source_record.read_text())
    bundle = json.loads(bundle_record.read_text())
    config, proposed_config = single["config"], proposed["config"]
    if single.get("status") != "completed" or config.get("variant") != "resnet50_single_task":
        raise ValueError("Single-task belum selesai atau varian salah.")
    if json.loads((single_task_run / "protocol.json").read_text()) != config:
        raise ValueError("Config single-task berbeda dari protocol.json.")
    if bundle["source_record_sha256"] != file_hash(source_record):
        raise ValueError("Single-task tidak berasal dari provenance proposed yang dipilih.")
    if config.get("script_sha256") != bundle.get("script_sha256"):
        raise ValueError("Identitas trainer single-task tidak konsisten.")
    if proposed_config.get("variant") != "proposed" or not proposed.get("test_metrics"):
        raise ValueError("Proposed belum memiliki evaluasi test final.")
    audit = source.get("dataset_audit", {})
    if audit.get("eligible_images") != 1204 or audit.get("annotations_sha256") != "ae03fc23fed0925c254dc4c4c9e55b539bd199b236151cb649857378a43539e9":
        raise ValueError("Provenance audit v2 resmi tidak tersedia.")
    manifest = Path(proposed_config["manifest"])
    source_cmd = source["command"]
    if "--manifest" not in source_cmd or manifest != Path(source_cmd[source_cmd.index("--manifest") + 1]):
        raise ValueError("Manifest proposed berbeda dari sumber training.")
    if source["sha256"].get(str(manifest)) != file_hash(manifest):
        raise ValueError("Manifest proposed berubah.")
    st_manifest = Path(config["manifest"])
    if config.get("manifest_sha256") != file_hash(st_manifest) or bundle.get("manifest_sha256") != file_hash(st_manifest):
        raise ValueError("Manifest train/validation single-task berubah.")
    all_rows, training_rows = read_rows(manifest), read_rows(st_manifest)
    fields = ("filename", "split", "label_index", "image_path", "preprocessed_path")
    def records(rows):
        return Counter(tuple(r.get(k, "") for k in fields) for r in rows)

    if records(training_rows) != records([r for r in all_rows if r["split"] in {"train", "val"}]):
        raise ValueError("Single-task dan proposed tidak memakai train/validation yang sama.")
    normalization = proposed_config.get("resolved_input_normalization", proposed_config.get("input_normalization"))
    if config.get("input_normalization") != normalization or normalization != "imagenet" or config["image_size"] != proposed_config["image_size"]:
        raise ValueError("Preprocessing single-task dan proposed tidak sesuai.")
    best = min(single["history"], key=lambda row: row["val_loss"])
    if best["epoch"] != single["best_epoch"] or abs(best["val_loss"] - single["best_val_loss"]) > 1e-8:
        raise ValueError("Checkpoint terpilih tidak konsisten dengan validation loss minimum.")
    test_rows = [dict(r) for r in all_rows if r["split"] == "test"]
    if not test_rows or len(test_rows) != audit.get("split_counts", {}).get("test"):
        raise ValueError("Jumlah test tidak sesuai audit proposed.")
    if len({r["filename"] for r in test_rows}) != len(test_rows):
        raise ValueError("Nama test duplikat.")
    for row in test_rows:
        row["input_path"] = row.get("preprocessed_path") or row.get("image_path")
        if row.get("label_index") not in set("0123") or not row["input_path"] or not Path(row["input_path"]).is_file():
            raise ValueError(f"Foto/label test tidak tersedia: {row.get('filename')}")
    counts = Counter(int(r["label_index"]) for r in training_rows if r["split"] == "train")
    if set(counts) != set(range(4)):
        raise ValueError("Train single-task tidak memiliki empat kelas.")
    plan = {"single_task_run": str(single_task_run), "proposed_run": str(proposed_run),
            "manifest": str(manifest), "split": "test", "total_images": len(test_rows),
            "single_task_best_epoch": best["epoch"], "single_task_best_val_loss": best["val_loss"],
            "image_size": int(config["image_size"]), "input_normalization": normalization,
            "batch_size": int(config["batch_size"]), "train_class_counts": dict(counts),
            "sha256": {str(p): file_hash(p) for p in required + [manifest, st_manifest]},
            "test_records_sha256": hashlib.sha256(json.dumps([
                [r["filename"], r["label_index"], r["input_path"]] for r in test_rows
            ], sort_keys=True).encode()).hexdigest(),
            "proposed_test_metrics": proposed["test_metrics"]}
    return plan, test_rows, single


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-task-run", type=Path, required=True)
    parser.add_argument("--proposed-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    plan, rows, single = prepare_test(args.single_task_run, args.proposed_run)
    plan["evaluation_script_sha256"] = file_hash(__file__)
    # JSON round-trip canonicalizes int keys in train_class_counts before comparisons.
    plan = json.loads(json.dumps(plan))
    provenance_path = args.output_dir / "test_evaluation_provenance.json"
    if provenance_path.is_file():
        if json.loads(provenance_path.read_text()) != plan:
            raise ValueError("Sumber/checkpoint/config evaluasi berbeda. Gunakan folder output baru.")
        if (args.output_dir / "metrics.json").is_file():
            print("Evaluasi yang sama sudah tersedia; tidak menjalankan test lagi.")
            print((args.output_dir / "metrics.json").read_text())
            return
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Folder output berisi hasil tanpa provenance; gunakan folder baru.")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import resnet50
    from PIL import Image

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = resnet50(weights=None)
            self.encoder.fc = nn.Identity()
            self.global_proj = nn.Linear(2048, 128)
            self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 4))

        def forward(self, images):
            return self.classifier(self.global_proj(self.encoder(images)))

    best_weights = torch.load(args.single_task_run / "best_model.pt", map_location="cpu", weights_only=True)
    # Only load trusted local checkpoints; full checkpoint contains Python RNG metadata.
    saved = torch.load(args.single_task_run / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    if saved["config"] != single["config"] or saved["best_epoch"] != plan["single_task_best_epoch"] or abs(saved["best_val_loss"] - plan["single_task_best_val_loss"]) > 1e-8:
        raise ValueError("Metadata checkpoint berbeda dari model terbaik yang dikunci.")
    if best_weights.keys() != saved["best_state"].keys() or any(not torch.equal(v, saved["best_state"][k]) for k, v in best_weights.items()):
        raise ValueError("best_model.pt berbeda dari best_state checkpoint. Jangan mengevaluasi epoch terakhir sebagai model terbaik.")
    del saved
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model = Model()
    model.load_state_dict(best_weights, strict=True)
    del best_weights
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    class TestImages(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            with Image.open(rows[index]["input_path"]) as photo:
                photo = photo.convert("RGB")
            size = plan["image_size"]
            if photo.size != (size, size):
                photo = photo.resize((size, size), Image.Resampling.LANCZOS)
            tensor = torch.frombuffer(bytearray(photo.tobytes()), dtype=torch.uint8)
            tensor = tensor.view(size, size, 3).permute(2, 0, 1).float() / 255
            tensor = (tensor - tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]) / tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
            return tensor

    print(f"TEST ONLY: {len(rows)} gambar yang sama dengan proposed; checkpoint epoch {plan['single_task_best_epoch']}; tanpa training.", flush=True)
    predictions, confidence = [], []
    with torch.inference_mode():
        for images in DataLoader(TestImages(), batch_size=plan["batch_size"], shuffle=False, num_workers=0):
            probabilities = model(images.to(device)).softmax(1)
            conf, predicted = probabilities.max(1)
            predictions.extend(predicted.cpu().tolist())
            confidence.extend(conf.cpu().tolist())
    result = scores([int(r["label_index"]) for r in rows], predictions)
    comparison = [
        {"model": "ResNet50 single-task", "split": "test", "n": len(rows), "acc": result["acc"], "kappa": result["kappa"]},
        {"model": "Proposed augmented", "split": "test", "n": len(rows), "acc": plan["proposed_test_metrics"]["acc"], "kappa": plan["proposed_test_metrics"]["kappa"]},
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(provenance_path, plan)
    with (args.output_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "split", "true_label_index", "pred_label_index", "confidence", "image_path"])
        writer.writerows((r["filename"], "test", r["label_index"], p, c, r["input_path"]) for r, p, c in zip(rows, predictions, confidence))
    with (args.output_dir / "test_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    write_json(args.output_dir / "metrics.json", {"status": "completed", "split": "test", "single_task_best_epoch": plan["single_task_best_epoch"],
               "test_metrics": result, "comparison": comparison, "dice": None, "iou": None})
    print(json.dumps({"test_metrics": result, "comparison": comparison}, indent=2), flush=True)
    print("Dice/IoU single-task tidak berlaku. Hasil training/validation asli tidak diubah.", flush=True)


if __name__ == "__main__":
    main()
