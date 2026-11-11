# Step 12–17 — Hasil Akhir Eksperimen Proposed (ACNE04-v2)

**Run ID:** `2026-11-11_acne04v2_official1204_proposed_offline_aug100_balanced_v4`
**Tanggal eksekusi:** 11 November 2026
**Runtime:** Google Colab, Tesla T4 15,6 GB, CUDA aktif

---

## 12. Monitoring Training

Training berjalan 120 epoch (30 phase-1 + 90 phase-2) dan **selesai normal** (exit 0). Beberapa kali runtime terputus dan dilanjutkan dengan `TRAINING_MODE='resume'` dari checkpoint, tanpa mengulang dari epoch 1.

| Item | Nilai |
|---|---|
| Total epoch | 120 |
| Best epoch (val_loss terendah) | 119 |
| Best val_loss | 1,0741 |
| Best val_acc | 76,40% (epoch 119) |
| Best val_kappa | 0,5926 (epoch 119) |
| Best val_dice | 0,3648 (epoch 119) |

Catatan: epoch 120 val_loss naik ke 1,1030, jadi `best_model.pt` = epoch 119, bukan epoch terakhir.

### Kurva akhir (epoch 114–120)

| Epoch | train_loss | train_acc | train_kappa | val_loss | val_acc | val_kappa | val_dice |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 0,6781 | 90,99% | 0,8798 | 1,0979 | 74,72% | 0,5619 | 0,3593 |
| 115 | 0,6826 | 89,07% | 0,8543 | 1,0970 | 74,16% | 0,5515 | 0,3410 |
| 116 | 0,6739 | 90,68% | 0,8757 | 1,0892 | 72,47% | 0,5216 | 0,3544 |
| 117 | 0,6984 | 87,16% | 0,8288 | 1,0905 | 74,16% | 0,5515 | 0,3565 |
| 118 | 0,6938 | 87,59% | 0,8346 | 1,0886 | 75,28% | 0,5725 | 0,3573 |
| **119** | 0,6831 | 88,89% | 0,8519 | **1,0741** | **76,40%** | **0,5926** | **0,3648** |
| 120 | 0,6729 | 89,57% | 0,8609 | 1,1030 | 72,47% | 0,5216 | 0,3463 |

---

## 13. Test Metrics

Sumber: `metrics.json` pada folder run (evaluasi internal script training, split test 184 gambar).

| Metrik | Nilai |
|---|---:|
| Loss | 1,1853 |
| Seg loss | 0,6881 |
| Cls loss | 0,4956 |
| Morph loss | 0,0082 |
| **Accuracy (klasifikasi)** | **71,20%** |
| **Cohen's Kappa** | **0,5138** |
| **Mean Dice (segmentasi)** | **0,3402** |
| **Mean IoU** | **0,2358** |

Konfigurasi training yang terekam di `metrics.json`:

| Parameter | Nilai |
|---|---|
| image_size | 320 |
| batch_size | 16 |
| phase1_epochs / phase2_epochs | 30 / 90 |
| phase1_lr / phase2_lr | 1e-3 / 1e-4 |
| seg_weight / cls_weight / morph_weight | 1,0 / 1,0 / 0,2 |
| label_smoothing | 0,05 → 0,35 (dinamis) |
| augmentasi | rotasi 15°, hflip 0,5, brightness 0,15, contrast 0,15 |
| input_normalization | imagenet |
| variant | proposed |
| pretrained | true |
| seed | 42 |

---

## 14. Inference — Prediksi Test Set

Hasil inferensi ulang dengan checkpoint yang sama pada **manifest yang benar**:

| Item | Nilai |
|---|---|
| Manifest | `refined_manifest_offline_augmented_100pct.csv` |
| Split | test (184 gambar) |
| Checkpoint | `best_model.pt` (epoch 119) |
| threshold | 0,5 |
| Output | `output/predictions/proposal_hybrid_resnet50_malds_v4_fixed/` |

**Hasil inferensi = hasil evaluasi training (konsisten):** acc 0,7120 · kappa 0,5138 · dice 0,3402 · iou 0,2358.

### ⚠️ Temuan teknis penting (wajib dicatat di tesis bagian metode/keterbatasan)

Inferensi pertama memakai manifest yang **salah** (`refined_masks_output/refined_manifest.csv`, split test 224 gambar) sehingga menghasilkan acc 0,25 dan dice 0,24 — jauh di bawah evaluasi training. Setelah memakai manifest yang sama dengan training (`..._offline_augmented_100pct.csv`, 184 test), angkanya identik dengan `metrics.json`.

Pelajaran: **evaluasi hanya sah jika manifest dan checkpoint berasal dari eksperimen yang sama.**

---

## 15. Visualisasi Hasil Segmentasi

Artefak tersedia di `output/predictions/proposal_hybrid_resnet50_malds_v4_fixed/`:

- `overlays/` — 184 overlay prediksi vs ground truth
- `pred_masks/` — 184 mask prediksi
- `testing_predictions.csv` — per gambar: label benar, label prediksi, confidence, dice, iou, luas area
- `summary.json` — metrik agregat

Contoh kasus terbaik per kelas (dikirim terpisah ke Telegram):

| Gambar | Label benar | Prediksi | Dice |
|---|---|---|---:|
| levle0_168 | levle0 | levle0 | 0,878 |
| levle1_299 | levle1 | levle1 | 0,786 |
| levle2_147 | levle2 | levle2 | 0,711 |

### Confusion matrix (test 184, baris = aktual, kolom = prediksi)

| Aktual \ Prediksi | levle0 | levle1 | levle2 | levle3 |
|---|---:|---:|---:|---:|
| levle0 (71) | **50** | 21 | 0 | 0 |
| levle1 (88) | 16 | **69** | 3 | 0 |
| levle2 (21) | 0 | 9 | **12** | 0 |
| levle3 (4) | 0 | 0 | 4 | **0** |

### Mean Dice per kelas

| Kelas | Mean Dice | n |
|---|---:|---:|
| levle0 | 0,3203 | 71 |
| levle1 | 0,3316 | 88 |
| levle2 | 0,4351 | 21 |
| levle3 | 0,3835 | 4 |

---

## 16. Rekap Semua Eksperimen

- Rekap otomatis (`experiment_results.csv`) dijalankan oleh cell 39 dan menyalin artefak checkpoint yang tersebar di `/content/` ke Drive (34 file).
- Folder rekap berisi riwayat run lama (baseline `dilated_unet`, `resnet50_amfm`, dan run proposed sebelumnya), bukan bagian dari evaluasi ini.

Pembanding yang relevan untuk tesis (perlu diambil dari `experiment_results.csv` bila ingin ditampilkan sebagai tabel perbandingan):

| Model | Status |
|---|---|
| dilated_unet_acne04v2 | baseline tersedia |
| dilated_unet_acne04v2_bce_dice | baseline tersedia |
| resnet50_amfm_acne04v2 | baseline tersedia |
| dilated_unet_preprocessed_all_bce_dice | baseline tersedia |
| proposal_hybrid_resnet50_malds (v4, aug 100%) | **run ini** |

---

## 17. Backup Checkpoint

| File | Lokasi |
|---|---|
| `best_model.pt` | `output/training_runs/..._balanced_v4/` (165,7 MB) |
| `last_checkpoint.pt` | idem |
| `metrics.json` | idem |
| `history_partial.json` | idem (120 epoch) |

Cell 17 (backup) menyalin 34 file checkpoint dari `/content/` ke Drive. Tidak ada checkpoint yang tertinggal di runtime.

---

## Ringkasan untuk narasi tesis

**Terbukti (angka nyata):**
- Model proposed mencapai akurasi klasifikasi 71,20% (kappa 0,5138) pada test 184 gambar.
- Segmentasi lemah: mean Dice 0,3402 dan mean IoU 0,2358.
- Terjadi gap train–val besar (train acc ~89% vs val acc ~74%), indikasi overfitting.
- Kelas levle3 tidak terprediksi sama sekali (4 gambar test semuanya diprediksi levle2); levle0 paling banyak tertukar ke levle1 (21 dari 71).

**Dugaan (belum diuji, jangan ditulis sebagai temuan):**
- Dice rendah kemungkinan dipengaruhi kualitas mask refined dan ambang 0,5 yang belum dituning.
- levle3 gagal karena sampel sangat sedikit (4 test, 22 di seluruh subset).
- Overfitting kemungkinan karena train hanya 842 gambar unik walau sudah diaugmentasi menjadi 1.620 baris.

**Belum dikerjakan:**
- Perbandingan kuantitatif vs baseline pada split yang sama.
- Tuning threshold dan uji signifikansi statistik.
