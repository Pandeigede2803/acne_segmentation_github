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

## Visualisasi (Step 24 notebook)

Setelah training, jalankan Step 24 atau perintah berikut (memerlukan `matplotlib`):

```bash
python scripts/plot_smote_results.py --run-dir PATH/smote_feature_experiment_120epochs
```

Folder `charts/` memuat kurva akurasi dan cross-entropy loss train–validasi, distribusi kelas sebelum/sesudah SMOTE, confusion matrix test dalam jumlah dan proporsi per baris, serta grafik precision/recall/F1 per kelas. Nilai yang tidak terdefinisi karena tidak ada contoh/prediksi kelas diberi 0. CSV riwayat training, confusion matrix, dan laporan per kelas juga disimpan. Loss train, seperti akurasi train, dihitung pada fitur train asli dalam mode evaluasi; bukan loss optimisasi pada campuran fitur SMOTE.

Trainer terbaru menyimpan `test_class_predictions.csv` (indeks sampel mengikuti urutan baris test dalam manifest), confusion matrix, nama kelas, dan epoch terbaik di `metrics.json`. Evaluasi test memakai head dengan validation loss terbaik. Plotter juga menerima `history_partial.json` jika training belum selesai. Run lama yang tidak memiliki prediksi/confusion matrix atau metrik train tidak bisa menghasilkan chart tersebut; data yang hilang tidak direka ulang.

Script visualisasi perlu di-commit/push sebelum Step 24 dapat mengunduhnya dari GitHub. Grafik bukan bukti bahwa SMOTE lebih baik tanpa eksperimen pembanding yang terkontrol.

## Audit validasi dan regularisasi (Step 25–27)

- **Step 25** melakukan inference hanya pada split `val`, memakai backbone beku dan head terbaik yang sudah ada. Head lama dengan hidden 64 tetap didukung; dimensi head dibaca dari bobot. Jangan mengganti manifest/backbone dengan yang berbeda dari training head tersebut. Hasil meliputi confusion matrix, support/error rate/precision/recall/F1 per kelas, CSV seluruh prediksi dan kesalahan, pasangan severity yang tertukar, dan contoh foto. Kelas tanpa support memiliki error rate `null`; precision/recall/F1 yang tidak terdefinisi diisi 0. Contoh dipilih berdasarkan confidence per pasangan sehingga tidak representatif. Foto/path bersifat sensitif dan tidak perlu masuk Git.
- **Step 26** menyediakan kontrol (`--hidden-dim 64 --dropout 0.3`) dan percobaan dropout (`--hidden-dim 64 --dropout 0.5`). **Step 27** menyediakan percobaan ukuran head (`--hidden-dim 32 --dropout 0.3`). Semua flag eksekusi default `False`. Tidak ada perubahan default arsitektur Step 23.
- Semua run baru menggunakan seed tetap untuk inisialisasi dan training, fitur dari backbone/manifest yang sama, SMOTE yang sama, batch size dan jumlah epoch yang sama. Sebaiknya jalankan ulang kontrol: run lama belum menyetel semua RNG. Reproducibility lintas hardware/versi library tidak dijamin. Konfigurasi disimpan dalam metrics/checkpoint, dan trainer menolak folder output yang sudah berisi file agar tidak menimpa hasil.
- Eksperimen regularisasi memakai `--skip-test`: tidak mengekstrak fitur atau menghitung metrik test. Bandingkan pada epoch validation loss terbaik, bukan epoch test terbaik atau hanya gap train–val terkecil. Metrik test memang tidak tersedia pada output ini. Grafik akurasi/loss dan distribusi kelas tetap tersedia.
- Untuk menjalankan audit dari CLI, gunakan `python scripts/audit_smote_validation.py --trainer PATH/train_proposal_hybrid_resnet50_malds.py --manifest PATH/manifest.csv --checkpoint PATH/backbone/best_model.pt --metrics PATH/backbone/metrics.json --head PATH/run/smote_classifier_head.pt --output-dir PATH/run/validation_audit --device cpu`. Tidak ada optimizer/training atau akses gambar test. Dibutuhkan PyTorch, torchvision, Pillow, dan matplotlib seperti lingkungan notebook.

Dropout lebih besar atau head lebih kecil belum tentu memperbaiki generalisasi. Jangan menyimpulkan ada peningkatan sebelum eksperimen dijalankan; gunakan beberapa seed/CV untuk menilai kestabilan dan evaluasi test final setelah konfigurasi dikunci.

## Batas interpretasi

Ini adalah eksperimen klasifikasi dua tahap dengan backbone dan cabang hibrid dibekukan, bukan training end-to-end yang identik dengan run biasa. Perbandingan mengukur dampak SMOTE pada classifier head di atas fitur checkpoint tersebut. Laporkan hasil sebagai varian SMOTE dan jangan menyebut fitur sintetis sebagai foto atau mask baru.

Untuk menyimpulkan manfaat SMOTE, diperlukan pembanding classifier head tanpa SMOTE dengan fitur, split, dan pengaturan training yang sama. Tambahan epoch tidak menjamin akurasi test meningkat; perubahan kode ini belum menghasilkan angka baru sebelum training dijalankan.

Sebelum menjalankan, pastikan manifest mengacu pada gambar ACNE04-v2 yang cocok dengan `Acne04-v2_annotations.json`, dan gunakan split yang sama pada baseline dan SMOTE. Jika ada identitas pasien, split harus dilakukan per pasien sebelum eksperimen.
