# Implementasi Step 31 — Split 60/40 & 70/30 + Augmentasi 1000/Kelas

Ringkasan implementasi dari `improvement.md`. Semua training memakai protokol
**train/test tanpa validasi** dan checkpoint tetap disimpan.

## Protokol

| Setup | Train | Test | Train setelah augmentasi |
|---|---:|---:|---:|
| 60/40 | 721 | 483 | 4.000 (1.000/kelas) |
| 70/30 | 842 | 362 | 4.000 (1.000/kelas) |

- Split stratified dari **1.204 citra asli**, seed 42, tanpa baris `val`.
- Hanya baris `train` yang diaugmentasi; baris `test` disalin tanpa perubahan.
- Target per kelas diatur `AUG_PER_CLASS_TARGET` (1000–2000).
- Checkpoint: `last_checkpoint.pt` tiap epoch; `best_model.pt` = epoch akhir
  karena tidak ada validation untuk memilih checkpoint.
- Test dievaluasi sekali setelah checkpoint akhir tersimpan.

## File

- `scripts/split_train_test_manifest.py` — split ulang manifest asli.
- `scripts/augment_train_pairs_offline.py` — opsi baru `--target-per-class`.
- `scripts/train_proposal_hybrid_resnet50_malds.py` — opsi baru
  `--no-validation`, Macro F1, pemilihan checkpoint epoch akhir.
- Notebook **Step 31** — orkestrasi dua setup + ringkasan.

## Output per setup

```text
<source_run>_split_aug1000_v1/
├── split_60_40/
│   ├── manifest_split.csv
│   ├── manifest_aug.csv
│   ├── aug_pairs/
│   ├── split_aug_provenance.json
│   └── run/{metrics.json,best_model.pt,last_checkpoint.pt}
├── split_70_30/...
└── split_aug_results.csv
```

## Menjalankan

1. Pastikan artefak Step 11 tersedia di Drive.
2. Buka notebook, jalankan Step 31 dengan `RUN_SPLIT_AUG = True`.
3. Run yang sudah punya `metrics.json` dilewati; training terputus di-resume.
4. Untuk mengulang, ganti suffix `_v1` pada `SPLIT_AUG_BASE`; jangan menimpa
   folder lama.

## Catatan

- Angka Step 31 adalah run baru; jangan dicampur dengan T1/T2/T3.
- Satu seed per setup belum membuktikan signifikansi statistik.
- Augmentasi offline menghasilkan citra sintetis dari train, bukan pasien baru.
- Trainer harus mendukung `--no-validation`; cell akan menolak versi lama.
