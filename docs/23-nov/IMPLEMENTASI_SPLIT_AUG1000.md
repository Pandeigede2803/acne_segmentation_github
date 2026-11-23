# Implementasi Step 31 — Split 60/40 & 70/30 + Augmentasi 1000/Kelas

Ringkasan implementasi dari `improvement.md`. Semua training memakai protokol
**train/test tanpa validasi** dan checkpoint tetap disimpan.

## Protokol

| Setup | Train | Test | Train setelah augmentasi | Input |
|---|---:|---:|---:|---:|
| 60/40 | 721 | 483 | 4.000 (1.000/kelas) | 512×512 |
| 70/30 | 842 | 362 | 4.000 (1.000/kelas) | 512×512 |

- Split stratified dari **1.204 citra asli**, seed 42, tanpa baris `val`.
- Preprocessing dibangun ulang pada **512×512** (letterbox, warna asli,
  `--preprocessing-mode none`) sebelum split. Tanpa ini, training 512 hanya
  meng-upscale citra 320 dan bukan eksperimen 512 yang sah.
- Hanya baris `train` yang diaugmentasi; baris `test` disalin tanpa perubahan.
- Target per kelas diatur `AUG_PER_CLASS_TARGET` (1000–2000).
- Batch size 8 untuk 512.
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
output/experiments_512/split_aug1000_img512_v1/
├── preprocess_512/
│   ├── refined_manifest.csv
│   ├── preprocessed_images/
│   ├── refined_masks/
│   └── preprocess_provenance.json
├── split_60_40/
│   ├── manifest_split.csv
│   ├── manifest_aug.csv
│   ├── aug_pairs/
│   ├── split_aug_provenance.json
│   └── run/{metrics.json,best_model.pt,last_checkpoint.pt}
├── split_70_30/...
└── split_aug_results.csv
```

Run lama hanya dibaca sebagai sumber. Semua output 512 ditulis ke root baru
`output/experiments_512/`; cell menolak output yang berada di dalam folder run
lama.

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
