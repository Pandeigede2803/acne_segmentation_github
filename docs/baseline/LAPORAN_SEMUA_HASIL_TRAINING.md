# Laporan Ringkas Seluruh Hasil Training

Dokumen ini merangkum hasil yang tersedia di repository dan hasil eksperimen
yang telah dicatat selama penelitian. Angka yang belum tersedia tidak dibuat
atau diperkirakan.

## A. Training utama/proposed

### A.1 Step 11 — 24 Oktober 2026

Konfigurasi utama: Proposed Hybrid ResNet-50 + MA-LDS, refined mask, balanced
class weighting, tanpa weighted sampler. Training menggunakan dua fase:
30 epoch freeze encoder dan 90 epoch full fine-tuning.

| Metrik test | Hasil |
|---|---:|
| Accuracy | **71,88%** |
| Cohen's Kappa | **0,5895** |
| Dice | **0,4786** |
| IoU | **0,3321** |

Jumlah test: 224 citra. Hasil ini berasal dari
`docs/24okt/test/metrics.json`.

### A.2 Final flowchart — 25 Oktober 2026

Konfigurasi masih menggunakan Proposed Hybrid ResNet-50 + MA-LDS, tetapi
weighted sampler digunakan bersama balanced class weighting.

| Metrik test | Hasil |
|---|---:|
| Accuracy | **56,25%** |
| Cohen's Kappa | **0,3906** |
| Dice | **0,5025** |
| IoU | **0,3536** |
| Macro-F1 | **0,5745** |

Jumlah test: 224 citra.

Perbandingan dengan Step 11:

- Accuracy turun dari 71,88% menjadi 56,25%.
- Kappa turun dari 0,5895 menjadi 0,3906.
- Dice naik dari 0,4786 menjadi 0,5025.
- IoU naik dari 0,3321 menjadi 0,3536.

Penurunan klasifikasi terutama terlihat pada recall level 1.

## B. Classification report final flowchart 25 Oktober

Confusion matrix berikut menggunakan baris = kelas asli dan kolom = kelas
prediksi.

| Asli \ Prediksi | Level 0 | Level 1 | Level 2 | Level 3 |
|---|---:|---:|---:|---:|
| Level 0 | 64 | 6 | 6 | 0 |
| Level 1 | 55 | 24 | 18 | 0 |
| Level 2 | 1 | 1 | 26 | 1 |
| Level 3 | 0 | 0 | 10 | 12 |

| Kelas | Recall |
|---|---:|
| Level 0 | 84,21% |
| Level 1 | 24,74% |
| Level 2 | 89,66% |
| Level 3 | 54,55% |

Interpretasi utama: model cukup baik mengenali level 0 dan level 2, tetapi
lemah pada level 1. Sebanyak 55 citra level 1 diprediksi sebagai level 0.

## C. Cross-validation

Hasil CV 25 Oktober berasal dari validation folds, bukan test set:

| Metrik | Mean ± SD |
|---|---:|
| Accuracy | **47,98% ± 2,67%** |
| Cohen's Kappa | **0,2958 ± 0,0341** |
| Dice | **0,4924 ± 0,0155** |
| IoU | **0,3457 ± 0,0135** |

Nilai CV tidak boleh dibandingkan langsung dengan accuracy test 71,88% atau
56,25% karena berasal dari evaluasi yang berbeda.

## D. Ablation study

Ablation yang tersedia hanya menghasilkan metrik validation karena
`skip_test_eval=True`. Oleh karena itu hasil berikut belum boleh disebut hasil
test.

Metrik diambil pada epoch dengan validation loss terendah:

| Varian | Epoch terbaik | Val. loss | Val. accuracy | Val. Kappa | Val. Dice | Val. IoU | Test |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | 96 | 1,4145 | 60,19% | 0,4458 | 0,5096 | 0,3590 | Tidak dievaluasi |
| Tanpa Spatial Attention | 111 | 0,9897 | 53,24% | 0,3532 | 0,5154 | 0,3623 | Tidak dievaluasi |
| Tanpa Dilated Convolution | 115 | 1,0684 | 52,78% | 0,3449 | 0,5204 | 0,3664 | Tidak dievaluasi |

Interpretasi sementara tidak boleh menyatakan varian terbaik secara final hanya
berdasarkan tiga file ini. Pada validation split yang tersedia, baseline memiliki
accuracy dan Kappa klasifikasi lebih tinggi, sedangkan varian tanpa dilated
convolution memiliki Dice dan IoU sedikit lebih tinggi. Ini belum membuktikan
bahwa penghapusan komponen lebih baik karena test tidak dievaluasi dan model
proposed lengkap belum ada dalam kumpulan file ini.

File ablation yang tersedia baru mencakup `baseline`, `no_attention`, dan
`no_dilation`. Hasil `no_amfm`, `lds`, `no_fusion`, dan `proposed` belum tersedia
di repository sehingga tidak boleh diisi dengan angka perkiraan.

## E. Eksperimen SMOTE

Folder SMOTE yang ditemukan di Google Drive:

```text
smote_feature_experiment/metrics.json
smote_feature_experiment_120epochs/metrics.json
smote_feature_experiment_120epochs_charts_v2/metrics.json
smote_regularization_v1/control/metrics.json
smote_regularization_v1/dropout_05/metrics.json
```

Isi metrik dari Google Drive belum tersalin ke repository ini, sehingga hasil
SMOTE belum dimasukkan ke tabel angka. SMOTE harus dilaporkan sebagai variasi
penanganan ketidakseimbangan kelas pada classifier head berbasis fitur beku,
bukan sebagai perubahan arsitektur utama.

## F. Eksperimen dropout dan hidden layer

Hasil validation yang telah dicatat:

| Konfigurasi | Validation loss | Validation accuracy |
|---|---:|---:|
| Hidden 64, dropout 0,3 | **0,5995** | **71,30%** |
| Hidden 64, dropout 0,5 | **0,6025** | **71,76%** |

Perbedaan kecil dan berasal dari satu seed. Dropout 0,5 belum menunjukkan
peningkatan performa yang konsisten. Hasil hidden layer 32 belum memiliki angka
final yang terdokumentasi.

## G. Eksperimen fine-tuning layer4

Eksperimen ini merupakan variasi tambahan dan bukan metode utama proposal.
Training memakai classifier head dan/atau `layer4`, AdamW, serta validation
split. Sebagian run sempat berjalan, tetapi keseluruhan kombinasi seed dan mode
belum selesai serta terdapat masalah kompatibilitas trainer/checkpoint.

Karena belum lengkap, hasil layer4 **tidak digunakan sebagai hasil final** dan
tidak digunakan untuk mendukung klaim novelty proposal.

## H. Kesimpulan sementara

Hasil utama yang paling lengkap dan terdokumentasi adalah training proposed
Hybrid ResNet-50 pada 24 dan 25 Oktober. Hasil 24 Oktober memiliki klasifikasi
test lebih baik, sedangkan hasil 25 Oktober memiliki Dice dan IoU sedikit lebih
baik tetapi klasifikasi lebih rendah.

Kesimpulan penelitian harus berfokus pada:

1. Kemampuan model melakukan segmentasi dan klasifikasi secara simultan.
2. Perbandingan model proposed dengan varian ablation.
3. Accuracy, Kappa, Macro-F1, recall tiap kelas, Dice, IoU, dan confusion matrix.
4. Keterbatasan performa pada level 1 dan ketidakseimbangan kelas.

SMOTE, dropout, hidden layer, dan fine-tuning layer4 dilaporkan sebagai
eksperimen tambahan, bukan sebagai inti novelty penelitian.
