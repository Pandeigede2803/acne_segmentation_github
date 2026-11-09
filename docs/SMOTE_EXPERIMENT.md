# Eksperimen SMOTE pada Fitur Klasifikasi

Eksperimen ini membandingkan training hibrid biasa dengan varian SMOTE khusus klasifikasi. SMOTE tidak diterapkan pada piksel foto atau mask.

## Rancangan

1. Jalankan training hibrid biasa sampai tersedia `best_model.pt` dan `metrics.json`.
2. Gunakan checkpoint terbaik sebagai ekstraktor fitur yang dibekukan. Embedding gabungan dari cabang visual dan morfologi dihitung untuk train, validation, dan test.
3. Terapkan SMOTE hanya pada embedding train. Target jumlah fitur dibuat moderat: rata-rata geometrik antara jumlah kelas dan kelas train terbanyak. Kelas terbanyak tidak ditambah.
4. Latih ulang classifier head pada embedding train asli dan sintetis selama 120 epoch secara default. Setiap epoch, `train_acc` dihitung hanya pada embedding train asli dalam mode evaluasi (dropout nonaktif), bukan pada fitur sintetis SMOTE. Validation dipakai memilih epoch terbaik; test hanya dievaluasi sesudah pemilihan.
5. Segmentasi tetap memakai checkpoint dan foto-mask asli. SMOTE tidak menambah contoh segmentasi.

## Menjalankan

```bash
python scripts/train_smote_classifier_head.py \
  --trainer scripts/train_proposal_hybrid_resnet50_malds.py \
  --manifest PATH/refined_manifest.csv \
  --checkpoint PATH/best_model.pt \
  --metrics PATH/metrics.json \
  --output-dir PATH/smote_feature_experiment_120epochs \
  --epochs 120
```

Skrip membuat folder output saat mulai, mencetak progres ekstraksi fitur dan setiap epoch (termasuk `train_acc` dan `val_acc`), lalu menyimpan `smote_classifier_head.pt` pada validasi terbaik, `last_checkpoint.pt` setiap epoch, dan `history_partial.json` secara berkala. Riwayat pada checkpoint dan berkas JSON memuat `train_acc` setiap epoch. Setelah evaluasi selesai, `metrics.json` berisi jumlah kelas sebelum/sesudah SMOTE serta akurasi dan kappa test. Folder output contoh dan Step 23 notebook terpisah dari eksperimen 100 epoch; gunakan folder baru lagi bila ingin mempertahankan hasil run 120 epoch sebelumnya.

## Batas interpretasi

Ini adalah eksperimen klasifikasi dua tahap dengan backbone dan cabang hibrid dibekukan, bukan training end-to-end yang identik dengan run biasa. Perbandingan mengukur dampak SMOTE pada classifier head di atas fitur checkpoint tersebut. Laporkan hasil sebagai varian SMOTE dan jangan menyebut fitur sintetis sebagai foto atau mask baru.

Untuk menyimpulkan manfaat SMOTE, diperlukan pembanding classifier head tanpa SMOTE dengan fitur, split, dan pengaturan training yang sama. Tambahan epoch tidak menjamin akurasi test meningkat; perubahan kode ini belum menghasilkan angka baru sebelum training dijalankan.

Sebelum menjalankan, pastikan manifest mengacu pada gambar ACNE04-v2 yang cocok dengan `Acne04-v2_annotations.json`, dan gunakan split yang sama pada baseline dan SMOTE. Jika ada identitas pasien, split harus dilakukan per pasien sebelum eksperimen.
