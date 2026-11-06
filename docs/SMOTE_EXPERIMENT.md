# Eksperimen SMOTE pada Fitur Klasifikasi

Eksperimen ini membandingkan training hibrid biasa dengan varian SMOTE khusus klasifikasi. SMOTE tidak diterapkan pada piksel foto atau mask.

## Rancangan

1. Jalankan training hibrid biasa sampai tersedia `best_model.pt` dan `metrics.json`.
2. Gunakan checkpoint terbaik sebagai ekstraktor fitur yang dibekukan. Embedding gabungan dari cabang visual dan morfologi dihitung untuk train, validation, dan test.
3. Terapkan SMOTE hanya pada embedding train. Target jumlah fitur dibuat moderat: rata-rata geometrik antara jumlah kelas dan kelas train terbanyak. Kelas terbanyak tidak ditambah.
4. Latih ulang classifier head pada embedding train asli dan sintetis. Validation dipakai memilih epoch terbaik; test hanya dievaluasi sesudah pemilihan.
5. Segmentasi tetap memakai checkpoint dan foto-mask asli. SMOTE tidak menambah contoh segmentasi.

## Menjalankan

```bash
python scripts/train_smote_classifier_head.py \
  --trainer scripts/train_proposal_hybrid_resnet50_malds.py \
  --manifest PATH/refined_manifest.csv \
  --checkpoint PATH/best_model.pt \
  --metrics PATH/metrics.json \
  --output-dir PATH/smote_feature_experiment
```

Skrip menghasilkan `smote_classifier_head.pt` dan `metrics.json` dengan jumlah kelas sebelum/sesudah SMOTE serta akurasi dan kappa test.

## Batas interpretasi

Ini adalah eksperimen klasifikasi dua tahap dengan backbone dan cabang hibrid dibekukan, bukan training end-to-end yang identik dengan run biasa. Perbandingan mengukur dampak SMOTE pada classifier head di atas fitur checkpoint tersebut. Laporkan hasil sebagai varian SMOTE dan jangan menyebut fitur sintetis sebagai foto atau mask baru.

Sebelum menjalankan, pastikan manifest mengacu pada gambar ACNE04-v2 yang cocok dengan `Acne04-v2_annotations.json`, dan gunakan split yang sama pada baseline dan SMOTE. Jika ada identitas pasien, split harus dilakukan per pasien sebelum eksperimen.
