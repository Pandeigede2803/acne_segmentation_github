"""Restore ablation inputs from a completed Step 11 run, without training it."""

import ast
import importlib.util
import json
from pathlib import Path
from urllib.request import urlopen


def prepare_source(source_run, scripts_dir, allow_download=True):
    source_run, scripts_dir = Path(source_run), Path(scripts_dir)
    if not (source_run / "metrics.json").is_file():
        raise FileNotFoundError(f"Hasil Step 11 tidak ditemukan di {source_run}. Periksa ABLATION_SOURCE_RUN; jangan training ulang.")
    helper_path = scripts_dir / "train_single_task_resnet50.py"
    if not helper_path.is_file():
        raise FileNotFoundError(f"Helper pemulihan belum tersedia: {helper_path}. Tidak perlu menjalankan training single-task.")
    spec = importlib.util.spec_from_file_location("ablation_source_recovery", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if not hasattr(helper, "recover_source_run"):
        raise ValueError(f"Helper lama; unggah train_single_task_resnet50.py terbaru ke {helper_path}.")
    recovered = helper.recover_source_run(source_run)
    config = json.loads((source_run / "metrics.json").read_text())["config"]
    if (config.get("variant") != "proposed" or not config.get("pretrained")
            or config.get("class_weighting") != "balanced" or config.get("weighted_sampler")):
        raise ValueError("Konfigurasi sumber bukan proposed pretrained/balanced tanpa sampler.")
    normalization = config.get("resolved_input_normalization", config.get("input_normalization"))
    if normalization not in {"imagenet", "none"}:
        raise ValueError("Normalisasi sumber tidak diketahui.")
    expected = [(p, value) for p, value in recovered["protocol"]["sha256"].items()
                if Path(p).name == "train_proposal_hybrid_resnet50_malds.py"]
    if len(expected) != 1:
        raise ValueError("Hash trainer hybrid sumber tidak dapat ditentukan.")
    historical_path, expected_hash = expected[0]
    trainer_path = scripts_dir / "train_proposal_hybrid_resnet50_malds.py"
    if not trainer_path.is_file():
        historical = Path(historical_path)
        if historical.is_file() and helper.file_hash(historical) == expected_hash:
            trainer_bytes = historical.read_bytes()
        elif allow_download:
            with urlopen("https://raw.githubusercontent.com/Pandeigede2803/acne_segmentation_github/main/scripts/train_proposal_hybrid_resnet50_malds.py", timeout=60) as response:
                trainer_bytes = response.read()
        else:
            raise FileNotFoundError("Trainer sumber belum tersedia pada cache ablation.")
        import hashlib
        if hashlib.sha256(trainer_bytes).hexdigest() != expected_hash:
            raise ValueError(f"Trainer unduhan berbeda dari Step 11. Pulihkan salinan trainer dengan SHA-256 {expected_hash} ke {trainer_path}; jangan mengulang Step 11 atau melewati audit.")
        ast.parse(trainer_bytes.decode("utf-8"))
        scripts_dir.mkdir(parents=True, exist_ok=True)
        trainer_path.write_bytes(trainer_bytes)
    if helper.file_hash(trainer_path) != expected_hash:
        raise ValueError(f"Trainer cache berbeda dari Step 11. Pulihkan versi yang sama di {trainer_path}.")
    # Training needs stored masks, but not raw /content images or annotation JSON.
    rows = helper.read_train_val_rows(recovered["augmented_manifest"])
    for row in rows:
        mask = row.get("mask_path") or row.get("refined_mask") or row.get("circle_mask")
        if not mask or not Path(mask).is_file():
            raise FileNotFoundError(f"Mask train/validation tidak tersedia: {row.get('filename')}. Pulihkan file Drive asli, bukan training Step 11 ulang.")
    recovered.update({"config": config, "normalization": normalization,
                      "trainer_path": trainer_path, "trainer_sha256": expected_hash})
    return recovered
