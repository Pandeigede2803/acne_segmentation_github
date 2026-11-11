# Tujuan dan Hasil Penelitian Inti

## Tujuan utama

Penelitian ini mengembangkan model **hybrid multi-task** untuk melakukan dua tugas
secara bersamaan pada citra ACNE04-v2:

1. Segmentasi lesi jerawat.
2. Klasifikasi tingkat keparahan jerawat berdasarkan kriteria Hayashi
   (level 0–3).

Kontribusi utama yang diuji adalah pemanfaatan fitur morfologi dari cabang
segmentasi untuk membantu klasifikasi severity melalui **Adaptive
Morphology-Aware Fusion Module (AMFM)** dan **MA-LDS**.

## Hasil yang wajib dilaporkan

## Contoh pengisian berdasarkan hasil yang tersedia

Angka berikut berasal dari `docs/25okt/metrics.json` dan
`docs/25okt/test_class_predictions.csv`. Pastikan hasil ini memang checkpoint
yang akan digunakan sebagai hasil final sebelum dimasukkan ke tesis.

### 1. Performa segmentasi

- Dice Coefficient: **0,5025**
- Intersection over Union (IoU): **0,3536**
- Contoh visual mask target dan mask prediksi

Contoh kalimat laporan:

> Pada data uji, model memperoleh Dice Coefficient sebesar 0,5025 dan IoU
> sebesar 0,3536. Nilai tersebut menunjukkan tingkat kesesuaian antara mask
> hasil prediksi dan mask target pada tugas segmentasi.

Masukkan gambar perbandingan mask target dan prediksi dari folder hasil
training. Jangan membuat gambar contoh baru jika file visualnya belum tersedia.

### 2. Performa klasifikasi

- Accuracy: **0,5625 (56,25%)**
- Cohen's Kappa: **0,3906**
- Macro-F1: **0,5745 (57,45%)**

Recall setiap kelas:

| Kelas | Recall |
|---|---:|
| Level 0 / mild | 0,8421 (84,21%) |
| Level 1 / moderate | 0,2474 (24,74%) |
| Level 2 / severe | 0,8966 (89,66%) |
| Level 3 / very severe | 0,5455 (54,55%) |

Confusion matrix, dengan baris sebagai kelas asli dan kolom sebagai kelas
prediksi:

| Kelas asli \ Prediksi | Level 0 | Level 1 | Level 2 | Level 3 |
|---|---:|---:|---:|---:|
| Level 0 | 64 | 6 | 6 | 0 |
| Level 1 | 55 | 24 | 18 | 0 |
| Level 2 | 1 | 1 | 26 | 1 |
| Level 3 | 0 | 0 | 10 | 12 |

Contoh kalimat laporan:

> Pada data uji sebanyak 224 citra, model memperoleh accuracy sebesar 56,25%
> dan Cohen's Kappa sebesar 0,3906. Macro-F1 yang diperoleh adalah 57,45%.
> Recall tertinggi terdapat pada level 2 sebesar 89,66%, sedangkan recall
> terendah terdapat pada level 1 sebesar 24,74%. Kesalahan terbesar terlihat
> pada level 1 yang diprediksi sebagai level 0 sebanyak 55 citra. Oleh karena
> itu, performa model belum merata pada seluruh kelas severity dan accuracy
> tidak boleh digunakan sebagai satu-satunya dasar penilaian.

Kelas severity:

```text
Level 0: mild
Level 1: moderate
Level 2: severe
Level 3: very severe
```

### 3. Ablation study

Ablation digunakan untuk menguji apakah komponen utama benar-benar memberi
kontribusi:

| Model | Spatial Attention | AMFM | MA-LDS |
|---|---:|---:|---:|
| Baseline | Tidak | Tidak | Tidak |
| Varian 1 | Ya | Tidak | Tidak |
| Varian 2 | Ya | Tidak | Ya |
| Proposed | Ya | Ya | Ya |

Model proposed dibandingkan dengan varian tanpa AMFM dan/atau tanpa MA-LDS.
Tidak semua metrik harus meningkat. Jika peningkatan tidak konsisten, hasil
tetap dilaporkan secara jujur sebagai temuan eksperimen.

## Prosedur evaluasi

```text
Training set
    ↓
Validation set untuk memilih checkpoint terbaik
    ↓
Model dikunci
    ↓
Test set digunakan satu kali untuk evaluasi akhir
```

Test set tidak digunakan untuk memilih model atau tuning hyperparameter.

## Tabel hasil akhir

| Model | Accuracy | Kappa | Macro-F1 | Dice | IoU |
|---|---:|---:|---:|---:|---:|
| Baseline | ... | ... | ... | ... | ... |
| Tanpa AMFM | ... | ... | ... | ... | ... |
| Tanpa MA-LDS | ... | ... | ... | ... | ... |
| Proposed | ... | ... | ... | ... | ... |

## Batas interpretasi

Keberhasilan penelitian tidak ditentukan hanya oleh kenaikan accuracy.
Kesimpulan harus mempertimbangkan macro-F1, recall kelas minoritas, Dice, IoU,
confusion matrix, serta konsistensi hasil eksperimen.

SMOTE, perubahan dropout, perubahan ukuran hidden layer, dan fine-tuning
`layer4` merupakan **eksperimen tambahan**, bukan tujuan utama atau novelty
utama proposal. Eksperimen tersebut dilaporkan terpisah jika hasilnya tersedia.

## Rumusan tujuan singkat untuk laporan

> Penelitian ini bertujuan mengembangkan model hybrid multi-task yang mampu
> melakukan segmentasi lesi jerawat dan klasifikasi tingkat keparahan secara
> simultan. Penelitian juga menguji kontribusi informasi morfologi dari cabang
> segmentasi melalui AMFM dan MA-LDS terhadap performa klasifikasi severity pada
> dataset ACNE04-v2.
