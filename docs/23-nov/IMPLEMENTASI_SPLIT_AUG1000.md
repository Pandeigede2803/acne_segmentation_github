# Implementasi Step 31 (persiapan data) & Step 31b (training)

Ringkasan implementasi dari `improvement.md`. Data disiapkan dulu di **Step 31**
tanpa training, lalu model dilatih di **Step 31b**. Protokol training adalah
**train/test tanpa validasi** dengan checkpoint yang tetap disimpan.

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

## Pembagian step

| Step | Isi | Training? |
|---|---|---|
| **31** | Preprocessing 512, split 60/40 & 70/30, augmentasi 1000/kelas, validasi manifest | ❌ tidak |
| **31b** | Training proposed pada data Step 31, checkpoint, evaluasi test sekali | ✅ ya |
| **33** | Salinan data 512 dengan filter wajah+kulit (step terpisah) | ❌ tidak |

Step 31 menulis `split_aug_preparation.csv`; Step 31b menulis
`split_aug_results.csv`. Untuk training data filter Step 33, jalankan Step 31b
dengan `TRAINING_VARIANT = 'skinfilter'`.

## File

- `scripts/split_train_test_manifest.py` — split ulang manifest asli.
- `scripts/augment_train_pairs_offline.py` — opsi `--target-per-class`.
- `scripts/train_proposal_hybrid_resnet50_malds.py` — `--no-validation`,
  Macro F1, checkpoint epoch akhir.
- `scripts/preprocess_skin_filter.py` + `scripts/face_skin_utils.py` —
  filter wajah+kulit untuk Step 33.
- Notebook **Step 31** — persiapan data saja.
- Notebook **Step 31b** — training dan ringkasan test.
- Notebook **Step 33** — persiapan data filter wajah+kulit.

## Output

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
├── split_aug_preparation.csv
└── split_aug_results.csv   # setelah Step 31b
```

Run lama hanya dibaca sebagai sumber. Semua output 512 ditulis ke root baru
`output/experiments_512/`; cell menolak output yang berada di dalam folder run
lama.

## Menjalankan

1. Pastikan artefak Step 11, foto asli, dan JSON anotasi tersedia.
2. **Step 31:** set `RUN_SPLIT_AUG = True` → hanya menyiapkan data.
3. Periksa `split_aug_preparation.csv` dan contoh manifest.
4. **Step 31b:** set `RUN_SPLIT_AUG_TRAINING = True` → training.
5. Run dengan `metrics.json` sudah ada dilewati; training terputus di-resume.

## Catatan

- Angka Step 31b adalah run baru; jangan dicampur dengan T1/T2/T3.
- Satu seed per setup belum membuktikan signifikansi statistik.
- Augmentasi offline menghasilkan citra sintetis dari train, bukan pasien baru.
- Checkpoint tanpa validation: `best_model.pt` = epoch akhir.
