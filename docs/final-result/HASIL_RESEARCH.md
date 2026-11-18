# Hasil Penelitian

Dokumen ini memuat angka keluaran eksperimen apa adanya, tanpa interpretasi.
Seluruh nilai dibaca langsung dari `metrics.json` masing-masing run dan
`ablation_results.csv`. Setiap tabel mencantumkan sumber dan cakupannya.

## 1. Daftar run dan sumber data

| Kode | Run | Path artefak | Himpunan dievaluasi |
|---|---|---|---|
| T1 | Proposed utama (Step 11 v4) | `2026-11-11_acne04v2_official1204_proposed_offline_aug100_balanced_v4/metrics.json` | validasi + test |
| T2 | Ablation study, 7 varian | `..._balanced_v4_ablation_aug100_v1/<varian>/metrics.json` | validasi saja |
| T2a | Ringkasan ablation | `..._ablation_aug100_v1/ablation_results.csv` | validasi saja |
| T2b | Evaluasi test ablation (Step 20b) | `..._ablation_aug100_v1/ablation_test_results.csv` | test |
| T3 | Single-task ResNet-50 | `..._balanced_v4_single_task_v1/run/metrics.json` | validasi + test |
| T3a | Evaluasi test single-task | `..._single_task_v1/test_evaluation_v1/metrics.json` | test |

Protokol yang berlaku pada ketiga rangkaian training:

| Parameter | Nilai |
|---|---|
| Ukuran input | 320 x 320 piksel, letterbox |
| Normalisasi | ImageNet (mean, std) |
| Batch size | 16 |
| Total epoch | 120 (30 freeze + 90 unfreeze) |
| Optimizer | SGD, momentum 0,9 |
| LR fase 1 / fase 2 | 1e-3 / 1e-4 |
| Seed | 42 |
| Augmentasi online | rotasi 15 derajat, hflip 0,5, brightness 0,15, contrast 0,15 |
| Weighted sampler | tidak dipakai |
| Class weighting | aktif, bobot efektif [1,1,1,1] (train seimbang 405 per kelas) |
| Kriteria pemilihan checkpoint | validation loss minimum |
| Split | stratified 70/15/15, seed 42 |

Jumlah data per himpunan pada rangkaian ablation: train 1.620 baris,
validasi 178 citra, test 0 (tidak dievaluasi saat training). Evaluasi test
T2b memakai split test 184 citra dari manifest aug100 yang sama. Test set juga
dievaluasi pada T1 dan T3, masing-masing satu kali setelah checkpoint dikunci.

## 2. Training 1 - Proposed utama (T1)

Checkpoint terbaik dipilih berdasarkan validation loss minimum, yaitu pada epoch 112.

| Metrik | Nilai (epoch 112) |
|---|---:|
| train_loss | 0.6926 |
| train_acc | 89.14% |
| train_kappa | 0.8551 |
| train_dice | 0.5606 |
| train_iou | 0.4196 |
| val_loss | 1.0694 |
| val_acc | 75.84% |
| val_kappa | 0.5838 |
| val_dice | 0.3715 |
| val_iou | 0.2597 |
| gap train-val accuracy | 13.29 pp |

### 2.1 Hasil pada test set

Test set berisi 184 citra. Evaluasi dilakukan satu kali setelah checkpoint dikunci.

| Metrik | Nilai |
|---|---:|
| total loss | 1.1853 |
| segmentation loss | 0.6881 |
| classification loss | 0.4956 |
| morphological loss | 0.0082 |
| Dice | 0.3402 |
| IoU | 0.2358 |
| Accuracy | 71.20% |
| Cohen's kappa | 0.5138 |

Catatan: `test_metrics` pada T1 tidak mencatat macro F1, balanced accuracy,
maupun confusion matrix. Nilai tersebut tidak dihitung ulang di sini.

## 3. Training 2 - Ablation study (T2)

Tujuh varian dilatih dengan manifest, protokol, seed, dan kriteria checkpoint
yang sama. Saat training tidak ada varian yang mengevaluasi test set.
Evaluasi test terpisah dijalankan satu kali lewat Step 20b (T2b), memakai
checkpoint terbaik yang sama dan threshold segmentasi 0,5 tanpa tuning.

### 3.1 Komponen yang diubah per varian

| Varian | Perubahan terhadap proposed |
|---|---|
| `baseline` | Spatial attention nonaktif, dilation 1, AMFM nonaktif, fusion nonaktif, label distribution smoothing nonaktif, classifier hanya memakai fitur global; decoder segmentasi tetap ada |
| `no_attention` | Hanya blok spatial attention dinonaktifkan |
| `no_dilation` | Dilation pada `dec4` dan `dec3` diubah dari 2 menjadi 1 |
| `no_amfm` | AMFM diganti konkatenasi fitur global dan morfologi; kanal masukan classifier 256 (proposed: 128) |
| `lds` | MA-LDS diturunkan menjadi LDS dengan epsilon tetap 0,05 |
| `no_fusion` | Fitur morfologi tidak dimasukkan ke classifier; target MA-LDS tetap |
| `proposed` | Seluruh komponen aktif, sebagai kontrol |

### 3.2 Metrik lengkap pada checkpoint terbaik

Checkpoint terbaik tiap varian = epoch dengan `val_loss` minimum.

| Varian | best_epoch | train_loss | train_acc | train_kappa | train_dice | train_iou | val_loss | val_acc | val_kappa | val_dice | val_iou |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `proposed` | 112 | 0.6912 | 89.07% | 0.8543 | 0.5610 | 0.4201 | 1.0649 | 78.65% | 0.6332 | 0.3665 | 0.2552 |
| `no_dilation` | 119 | 0.6767 | 88.40% | 0.8453 | 0.5703 | 0.4299 | 1.0673 | 75.84% | 0.5818 | 0.3711 | 0.2594 |
| `lds` | 112 | 0.7025 | 89.88% | 0.8650 | 0.5614 | 0.4208 | 1.1065 | 74.72% | 0.5648 | 0.3665 | 0.2566 |
| `no_attention` | 119 | 0.8004 | 90.49% | 0.8733 | 0.5201 | 0.3816 | 1.1365 | 74.72% | 0.5524 | 0.3614 | 0.2527 |
| `no_amfm` | 115 | 0.6227 | 94.26% | 0.9235 | 0.5642 | 0.4241 | 1.0694 | 74.72% | 0.5524 | 0.3698 | 0.2582 |
| `no_fusion` | 110 | 0.6163 | 94.88% | 0.9317 | 0.5595 | 0.4198 | 1.1791 | 73.03% | 0.5250 | 0.3874 | 0.2720 |
| `baseline` | 110 | 0.7354 | 96.98% | 0.9597 | 0.5148 | 0.3765 | 1.3804 | 73.03% | 0.5200 | 0.3803 | 0.2663 |

### 3.3 Selisih terhadap varian proposed

| Varian | delta val_kappa | delta val_acc (pp) | delta val_dice | delta val_iou | delta train_acc (pp) |
|---|---:|---:|---:|---:|---:|
| `no_dilation` | -0.0514 | -2.81 | +0.0046 | +0.0043 | -0.68 |
| `lds` | -0.0684 | -3.93 | -0.0000 | +0.0014 | +0.80 |
| `no_attention` | -0.0808 | -3.93 | -0.0051 | -0.0025 | +1.42 |
| `no_amfm` | -0.0808 | -3.93 | +0.0033 | +0.0031 | +5.19 |
| `no_fusion` | -0.1082 | -5.62 | +0.0208 | +0.0168 | +5.80 |
| `baseline` | -0.1132 | -5.62 | +0.0138 | +0.0111 | +7.90 |

### 3.4 Gap train dan validasi

| Varian | train_acc | val_acc | gap accuracy (pp) | train_kappa | val_kappa | gap kappa |
|---|---:|---:|---:|---:|---:|---:|
| `proposed` | 89.07% | 78.65% | 10.42 | 0.8543 | 0.6332 | 0.2211 |
| `no_dilation` | 88.40% | 75.84% | 12.55 | 0.8453 | 0.5818 | 0.2635 |
| `lds` | 89.88% | 74.72% | 15.16 | 0.8650 | 0.5648 | 0.3002 |
| `no_attention` | 90.49% | 74.72% | 15.77 | 0.8733 | 0.5524 | 0.3209 |
| `no_amfm` | 94.26% | 74.72% | 19.54 | 0.9235 | 0.5524 | 0.3711 |
| `no_fusion` | 94.88% | 73.03% | 21.84 | 0.9317 | 0.5250 | 0.4067 |
| `baseline` | 96.98% | 73.03% | 23.94 | 0.9597 | 0.5200 | 0.4396 |

### 3.5 Epoch terakhir versus checkpoint terbaik

| Varian | best_epoch | val_loss best | val_loss epoch 120 | val_acc best | val_acc epoch 120 | selisih val_acc (pp) |
|---|---:|---:|---:|---:|---:|---:|
| `no_fusion` | 110 | 1.1791 | 1.2265 | 73.03% | 71.35% | -1.69 |
| `baseline` | 110 | 1.3804 | 1.4147 | 73.03% | 72.47% | -0.56 |
| `proposed` | 112 | 1.0649 | 1.0888 | 78.65% | 75.84% | -2.81 |
| `lds` | 112 | 1.1065 | 1.1427 | 74.72% | 74.16% | -0.56 |
| `no_amfm` | 115 | 1.0694 | 1.1059 | 74.72% | 74.16% | -0.56 |
| `no_dilation` | 119 | 1.0673 | 1.0948 | 75.84% | 74.16% | -1.69 |
| `no_attention` | 119 | 1.1365 | 1.1393 | 74.72% | 75.28% | +0.56 |

### 3.6 Segmentasi pada checkpoint terbaik

| Varian | train_dice | val_dice | train_iou | val_iou | val_seg_loss |
|---|---:|---:|---:|---:|---:|
| `no_fusion` | 0.5595 | 0.3874 | 0.4198 | 0.2720 | 0.6393 |
| `baseline` | 0.5148 | 0.3803 | 0.3765 | 0.2663 | 0.7135 |
| `no_dilation` | 0.5703 | 0.3711 | 0.4299 | 0.2594 | 0.6525 |
| `no_amfm` | 0.5642 | 0.3698 | 0.4241 | 0.2582 | 0.6532 |
| `proposed` | 0.5610 | 0.3665 | 0.4201 | 0.2552 | 0.6564 |
| `lds` | 0.5614 | 0.3665 | 0.4208 | 0.2566 | 0.6563 |
| `no_attention` | 0.5201 | 0.3614 | 0.3816 | 0.2527 | 0.7241 |

### 3.7 Status artefak

| Varian | metrics.json | best_model.pt | last_checkpoint.pt | test_metrics |
|---|---|---|---|---|
| `proposed` | ada | ada (165-166 MB) | ada | `null` |
| `no_dilation` | ada | ada (165-166 MB) | ada | `null` |
| `lds` | ada | ada (165-166 MB) | ada | `null` |
| `no_attention` | ada | ada (165-166 MB) | ada | `null` |
| `no_amfm` | ada | ada (165-166 MB) | ada | `null` |
| `no_fusion` | ada | ada (165-166 MB) | ada | `null` |
| `baseline` | ada | ada (165-166 MB) | ada | `null` |

Jumlah epoch tercatat pada `history` setiap varian: 120 baris.
Konsistensi dicek: `best_epoch` pada `ablation_results.csv` sama dengan
epoch `argmin(val_loss)` pada `history`, cocok untuk ketujuh varian.

## 4. Training 3 - Single-task ResNet-50 (T3)

Model: ResNet-50 dengan kepala klasifikasi saja. Tanpa decoder segmentasi,
tanpa AMFM, tanpa fitur morfologi, tanpa loss segmentasi. Loss yang dipakai:
`class_weighted_ce`.

Checkpoint terbaik dipilih berdasarkan validation loss minimum, yaitu pada epoch 77.

| Metrik | Nilai (epoch 77) |
|---|---:|
| train_loss | 0.1847 |
| train_acc | 92.84% |
| train_kappa | 0.9045 |
| train_macro_f1 | 0.9284 |
| val_loss | 0.6906 |
| val_acc | 71.91% |
| val_kappa | 0.4952 |
| val_macro_f1 | 0.4541 |
| gap train-val accuracy | 20.93 pp |

### 4.1 Hasil pada test set

| Metrik | Nilai |
|---|---:|
| Accuracy | 69.02% |
| Cohen's kappa | 0.4562 |
| Macro F1 | 0.5306 |
| Balanced accuracy | 49.57% |
| Benar / total | 127 / 184 |

Metrik per kelas:

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Mild (kelas 0) | 0.8148 | 0.6197 | 0.7040 | 71 |
| Moderate (kelas 1) | 0.6417 | 0.8750 | 0.7404 | 88 |
| Severe (kelas 2) | 0.6250 | 0.2381 | 0.3448 | 21 |
| Very Severe (kelas 3) | 0.5000 | 0.2500 | 0.3333 | 4 |

Confusion matrix test (baris = aktual, kolom = prediksi):

| Aktual \ Prediksi | Mild | Moderate | Severe | Very Severe |
|---|---:|---:|---:|---:|
| **Mild** | 44 | 27 | 0 | 0 |
| **Moderate** | 10 | 77 | 1 | 0 |
| **Severe** | 0 | 15 | 5 | 1 |
| **Very Severe** | 0 | 1 | 2 | 1 |

Dice dan IoU tidak berlaku untuk model single-task karena tidak memiliki
keluaran segmentasi. Nilainya `null` pada `metrics.json`.

## 5. Perbandingan T1 dan T3 pada test set

Kedua model dievaluasi pada himpunan test yang sama, 184 citra.

| Model | Run | Accuracy | Cohen's kappa | Dice | IoU | Macro F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Proposed multi-task (T1) | `..._proposed_offline_aug100_balanced_v4` | 71.20% | 0.5138 | 0.3402 | 0.2358 | tidak dicatat | tidak dicatat |
| ResNet-50 single-task (T3) | `..._single_task_v1/run` | 69.02% | 0.4562 | tidak berlaku | tidak berlaku | 0.5306 | 49.57% |
| **Selisih (T1 - T3)** | dua run terpisah | **+2.17 pp** | **+0.0576** | | | | |

Nilai `correct` pada T1 tidak tersedia di `test_metrics`; persentase dihitung
dari field `acc` yang dilaporkan run, bukan dari confusion matrix.

Perbandingan validation antara proposed pada T1 dan varian `proposed` pada T2:

| Model | Run | val_loss | val_acc | val_kappa | val_dice | val_iou |
|---|---|---:|---:|---:|---:|---:|
| T1 proposed (epoch 112) | `..._proposed_offline_aug100_balanced_v4` | 1.0694 | 75.84% | 0.5838 | 0.3715 | 0.2597 |
| T2 `proposed` (epoch 112) | `..._balanced_v4_ablation_aug100_v1/proposed` | 1.0649 | 78.65% | 0.6332 | 0.3665 | 0.2552 |

Kedua run memakai seed, hyperparameter, augmentasi, dan kriteria checkpoint
yang sama, tetapi merupakan dua proses pelatihan terpisah sehingga angka
validasi keduanya tidak identik. Label run wajib disertakan; angka T1 dan T2
tidak boleh dicampur atau dirata-ratakan dalam satu sel.

## 6. Chart dan alur pembuatannya

### 6.1 Daftar chart

| No | Berkas | Sumber data | Field | Jenis |
|---|---|---|---|---|
| 01 | `charts/01_val_loss_7varian.png` | 7 x metrics.json T2 | `history[].val_loss` | line, 7 seri |
| 02 | `charts/02_loss_train_vs_val.png` | metrics.json T2 proposed dan baseline | `history[].train_loss, val_loss` | line 2 panel |
| 03 | `charts/03_val_acc_7varian.png` | 7 x metrics.json T2 | `history[].val_acc` | line, 7 seri |
| 04 | `charts/04_val_kappa_7varian.png` | 7 x metrics.json T2 | `history[].val_kappa` | line, 7 seri |
| 05 | `charts/05_val_dice_iou_7varian.png` | 7 x metrics.json T2 | `history[].val_dice, val_iou` | line 2 panel |
| 06 | `charts/06_bar_kappa_acc.png` | ablation_results.csv | `val_kappa, val_accuracy` | bar, sumbu ganda |
| 07 | `charts/07_gap_overfitting.png` | ablation_results.csv | `train_accuracy, val_accuracy` | grouped bar |
| 08 | `charts/08_scatter_dice_kappa.png` | ablation_results.csv | `val_dice, val_kappa` | scatter |
| 09 | `charts/09_single_task_curves.png` | run/metrics.json T3 | `history[].loss, acc, kappa` | line 3 panel |
| 10 | `charts/10_single_task_macroF1.png` | run/metrics.json T3 | `history[].macro_f1` | line |
| 11 | `charts/11_test_comparison.png` | test_metrics T1 dan T3 | `acc, kappa` | bar |
| 12 | `charts/12_cm_single_task.png` | test_evaluation_v1/metrics.json | `confusion_matrix` | heatmap |

### 6.2 Alur

```text
Google Drive
│
├── T2 ablation/<varian>/metrics.json ──┐
├── T1 proposed/metrics.json ───────────┤
├── T3 single_task/run/metrics.json ────┤
└── T2 ablation_results.csv ────────────┤
                                        │
                                        ▼
                             dump_hist.py (Colab kernel)
                             baca history[] + test_metrics
                                        │
                                        ▼
                             hist_dump.json (621 KB)
                             semua run, 120 epoch/run
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
            plot_charts.py                          gen_tables.py
            kurva per epoch                         tabel angka
                    │                                       │
                    ▼                                       ▼
            charts/01-12.png                        HASIL_ANGKA.md
```

Aturan pemakaian angka:

1. Kurva memakai seluruh 120 epoch dari `history`.
2. Tabel memakai satu epoch per varian, yaitu `argmin(val_loss)`.
3. Angka metrik tidak dicampur antar sumber. Ablation selalu dilabeli
   "validasi"; hanya T1 dan T3 yang punya angka test.
4. Nilai tidak dihaluskan, tidak diinterpolasi, tidak dirata-rata antar epoch.

## 7. Catatan cakupan dan keterbatasan angka

| No | Catatan |
|---|---|
| 1 | Tabel T2/T2a berasal dari set validasi. Angka test tujuh varian (T2b) dicatat terpisah dan tidak dicampur dengan test T1/T3 dalam tabel yang sama. |
| 2 | Satu seed per varian. Selisih kappa di bawah 0,06 tidak diuji signifikansinya. |
| 3 | `no_amfm` mengubah ukuran kanal masukan classifier dari 128 menjadi 256, sehingga penurunannya tidak murni berasal dari hilangnya AMFM. |
| 4 | `proposed` pada T1 dan T2 adalah dua proses pelatihan terpisah dengan angka validasi berbeda. |
| 5 | T1 tidak mencatat macro F1, balanced accuracy, dan confusion matrix pada test. |
| 6 | T3 tidak memiliki Dice dan IoU karena tidak ada keluaran segmentasi. |
| 7 | Checkpoint dipilih dari validasi; angka epoch 120 bukan angka yang dilaporkan pada tabel utama. |
| 8 | Gap train-validasi tidak sama dengan overfitting terukur; nilainya dilaporkan apa adanya dari dua himpunan berbeda. |

---

Sumber angka: `metrics.json` dan `ablation_results.csv` pada Google Drive,
dibaca langsung dari kernel Colab yang menjalankan training. Tidak ada angka
yang diketik manual. Dokumen dihasilkan oleh `gen_hasil_md.py`.

Dibuat: 2026-11-13 13:17 WIB

