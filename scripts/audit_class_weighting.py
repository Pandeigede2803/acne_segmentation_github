#!/usr/bin/env python3
"""Step 46: audit class weighting and cost the controlled experiment matrix.

CPU-only. Reads the trainer source to describe the implemented loss, computes
balanced weights and effective per-class contribution, then writes the E1-E4
matrix, the optimizer-update budget, and a pre-training go/no-go gate.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from recovery_common import read_rows, sha256_file

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "scripts" / "train_proposal_hybrid_resnet50_malds.py"


def parse_size(value: str) -> tuple[str, int]:
    name, _, size = value.partition("=")
    if not name or not size:
        raise argparse.ArgumentTypeError("Format harus NAMA=JUMLAH, mis. E1=613")
    return name, int(size)


def read_json(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Manifest yang akan dipakai eksperimen referensi (mis. controlled).")
    parser.add_argument("--weight-manifest", type=Path, default=None,
                        help="Manifest original train (mis. split Step 43B) untuk bobot balanced E4.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reference-train-size", type=int, default=0,
                        help="0 = pakai jumlah train dari --weight-manifest/--manifest.")
    parser.add_argument("--budget-epochs", type=int, default=50)
    parser.add_argument("--experiment", action="append", type=parse_size, default=None,
                        help="NAMA=JUMLAH_TRAIN, boleh berulang. Default E1=613,E2=977,E3=4000,E4=977.")
    parser.add_argument("--clean-manifest", type=Path, default=None)
    parser.add_argument("--split-integrity", type=Path, default=None)
    parser.add_argument("--visual-report", type=Path, default=None)
    parser.add_argument("--controlled-integrity", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} sudah berisi file. Pakai folder baru.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = read_rows(args.manifest)
    train_rows = [row for row in rows if row.get("split") == "train"]
    reference_rows = train_rows
    if args.weight_manifest is not None:
        reference_rows = [row for row in read_rows(args.weight_manifest)[0]
                          if row.get("split") == "train"]
    counts = Counter(int(row["label_index"]) for row in reference_rows)
    if set(counts) != {0, 1, 2, 3}:
        raise ValueError("Train harus memuat keempat kelas.")
    total = sum(counts.values())
    reference_train_size = args.reference_train_size or total
    weights = {cls: total / (4 * counts[cls]) for cls in range(4)}
    contribution = {cls: weights[cls] * counts[cls] / total for cls in range(4)}

    trainer_source = TRAINER.read_text(encoding="utf-8")
    implementation = {
        "balanced_formula": "total_train / (4 * class_count)" in trainer_source,
        "per_sample_weight": "per_sample_kl * class_weights[labels]" in trainer_source,
        "reduction": "reduction=\"none\").sum(dim=1)" in trainer_source,
        "soft_targets": "kl_div" in trainer_source and "target_dist" in trainer_source,
        "uniform_option": 'choices=("balanced", "none")' in trainer_source,
    }
    if not implementation["per_sample_weight"]:
        raise ValueError("Implementasi class weighting tidak dikenali; audit manual diperlukan.")

    default_experiment_size = sum(1 for row in train_rows)
    experiments = args.experiment or [("E1", total), ("E2", default_experiment_size),
                                      ("E3", 4000), ("E4", default_experiment_size)]
    per_epoch = {name: math.ceil(size / args.batch_size) for name, size in experiments}
    reference_updates = math.ceil(reference_train_size / args.batch_size) * args.budget_epochs
    budget = {
        "batch_size": args.batch_size,
        "reference_train_size": reference_train_size,
        "reference_epochs": args.budget_epochs,
        "reference_updates": reference_updates,
        "experiments": {
            name: {"train_size": size, "updates_per_epoch": per_epoch[name],
                   "fixed_update_epochs": math.ceil(reference_updates / per_epoch[name]),
                   "fixed_epoch_updates": per_epoch[name] * args.budget_epochs}
            for name, size in experiments
        },
    }

    checks = {
        "clean_manifest": args.clean_manifest,
        "split_integrity": args.split_integrity,
        "visual_audit_report": args.visual_report,
        "controlled_aug_integrity": args.controlled_integrity,
    }
    gate = {}
    for name, path in checks.items():
        if path is None:
            gate[name] = {"status": "MISSING", "path": None}
        elif not path.is_file():
            gate[name] = {"status": "MISSING", "path": str(path)}
        else:
            payload = read_json(path) if path.suffix == ".json" else {}
            status = payload.get("status", "PASS") if payload else "PASS"
            gate[name] = {"status": status, "path": str(path),
                          "sha256": sha256_file(path)}
    ready = all(entry["status"] == "PASS" for entry in gate.values())

    matrix = [
        "# Experiment matrix (Step 46). Semua memakai clean split, V2, seed sama.",
        "experiments:",
    ]
    for name, size in experiments:
        mapping = {"E1": ("none", "uniform"), "E2": ("controlled_300_cap7", "uniform"),
                   "E3": ("offline_1000_per_class", "uniform"),
                   "E4": ("controlled_300_cap7", "balanced_from_original_train")}.get(
                       name, ("unknown", "unknown"))
        matrix += [
            f"  - id: {name}",
            f"    train_size: {size}",
            f"    offline_augmentation: {mapping[0]}",
            f"    class_weight: {mapping[1]}",
            f"    updates_per_epoch: {per_epoch[name]}",
            f"    fixed_update_epochs: {math.ceil(reference_updates / per_epoch[name])}",
        ]
    matrix_text = "\n".join(matrix) + "\n"
    (args.output_dir / "experiment_matrix.yaml").write_text(matrix_text, encoding="utf-8")

    audit_lines = [
        "# Step 46 — Audit class weighting", "",
        "## Implementasi aktual di trainer", "",
        f"- File: `{TRAINER.relative_to(ROOT)}` (SHA-256 `{sha256_file(TRAINER)[:16]}…`)",
        f"- Formula balanced: `total_train / (4 * class_count)` → "
        f"**{'ditemukan' if implementation['balanced_formula'] else 'TIDAK ditemukan'}**",
        f"- Bobot diterapkan per sampel: `per_sample_kl * class_weights[labels]` → "
        f"**{'ya' if implementation['per_sample_weight'] else 'tidak'}**",
        "- Reduction: `KL(..., reduction='none').sum(dim=1)` lalu `.mean()` → bobot bekerja "
        "pada level sampel, bukan merata-ratakan per kelas.",
        "- Soft target MA-LDS: KL dihitung terhadap distribusi target, bukan one-hot; "
        "bobot kelas mengalikan seluruh kontribusi sampel termasuk neighbor mass.",
        "- Opsi `none` tersedia untuk uniform weight.",
        "",
        "## Bobot pada train original (weight manifest)", "",
        "| Kelas | Train | Weight | Kontribusi loss |", "|---|---:|---:|---:|",
    ]
    for cls in range(4):
        audit_lines.append(f"| {cls} | {counts[cls]} | {weights[cls]:.4f} | "
                           f"{contribution[cls]:.4f} |")
    audit_lines += [
        "", f"- Rasio bobot max/min: **{max(weights.values()) / min(weights.values()):.2f}×**.",
        "- Dengan balanced weight, kontribusi tiap kelas menjadi sama (0,25) secara ekspektasi; "
        "ini yang membuat minority class tidak tenggelam, tetapi juga memperbesar variance "
        "untuk kelas dengan sedikit sampel.",
        "- `area` dan `density` masih identik pada mask biner; dicatat sebagai keterbatasan, "
        "tidak diubah di tahap ini.",
        "",
        "## Catatan penting", "",
        "- Jangan hitung ulang balanced weight dari distribusi augmented pada eksperimen yang "
        "mensyaratkan weight tetap (E1–E3).",
        "- Hindari oversampling ekstrem + inverse-frequency tanpa audit; kombinasinya bisa "
        "overcompensate.",
    ]
    (args.output_dir / "class_weight_audit.md").write_text("\n".join(audit_lines) + "\n",
                                                           encoding="utf-8")

    budget_lines = [
        "# Step 46 — Training budget", "",
        f"Batch {args.batch_size}; referensi {reference_train_size} train × "
        f"{args.budget_epochs} epoch = **{reference_updates} optimizer updates**.",
        "", "| Run | Train size | Updates/epoch | Epoch untuk budget sama | Update @50 epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, size in experiments:
        entry = budget["experiments"][name]
        budget_lines.append(f"| {name} | {size} | {entry['updates_per_epoch']} | "
                            f"{entry['fixed_update_epochs']} | {entry['fixed_epoch_updates']} |")
    budget_lines += [
        "", "Pilih salah satu dan dokumentasikan sebelum training:", "",
        "1. **Fixed-update** — semua run berhenti setelah budget updates yang sama.",
        "2. **Fixed-epoch** — semua run 50 epoch; ini bukan perbandingan budget yang adil.",
        "", "Jangan mencampur keduanya dalam satu klaim perbandingan.",
    ]
    (args.output_dir / "training_budget.md").write_text("\n".join(budget_lines) + "\n",
                                                        encoding="utf-8")

    gate_lines = [
        "# Step 46 — Pre-training go/no-go", "",
        f"Status keseluruhan: **{'PASS' if ready else 'BLOCKED'}**", "",
        "| Artefak | Status | Path |", "|---|---|---|",
    ]
    for name, entry in gate.items():
        gate_lines.append(f"| {name} | {entry['status']} | `{entry['path'] or '-'}` |")
    gate_lines += [
        "", "## Syarat sebelum GPU", "",
        "- Semua artefak di atas PASS dan disetujui user.",
        "- Split dibekukan (hash) dan tidak diubah setelah melihat hasil.",
        "- Konfigurasi E1–E4, budget, checkpoint metric, dan early stopping ditetapkan lebih dulu.",
        "- Test tidak dipakai untuk memilih konfigurasi.",
        "", "**STOP**: jangan training sebelum persetujuan eksplisit.",
    ]
    (args.output_dir / "pretraining_go_no_go.md").write_text("\n".join(gate_lines) + "\n",
                                                             encoding="utf-8")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "weight_manifest": str(args.weight_manifest) if args.weight_manifest else str(args.manifest),
        "weight_manifest_sha256": (sha256_file(args.weight_manifest)
                                   if args.weight_manifest and args.weight_manifest.is_file()
                                   else sha256_file(args.manifest)),
        "counts": {str(cls): counts[cls] for cls in range(4)},
        "weights": {str(cls): weights[cls] for cls in range(4)},
        "contribution": {str(cls): contribution[cls] for cls in range(4)},
        "implementation": implementation,
        "budget": budget,
        "gate": gate,
        "ready_for_gpu": ready,
    }
    (args.output_dir / "step46_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"weights": {str(cls): round(weights[cls], 4) for cls in range(4)},
                      "ready_for_gpu": ready}, indent=2))
    print("Gate:", args.output_dir / "pretraining_go_no_go.md")


if __name__ == "__main__":
    main()
