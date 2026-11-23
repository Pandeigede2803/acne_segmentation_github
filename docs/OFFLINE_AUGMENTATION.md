# Augmentasi Offline Pasangan Foto-Mask

`scripts/augment_train_pairs_offline.py` membuat pasangan foto-mask tambahan untuk kelas train sebelum training. Karena split berasal dari manifest awal, hanya baris `train` yang di-augmentasi; validation dan test disalin tanpa perubahan. Rotasi dan flip yang sama diterapkan ke foto serta mask. Perubahan brightness/contrast hanya diterapkan ke foto.

Pada jalur notebook saat ini, `AUGMENT_TARGET_RATIO = 1.0`: setiap kelas train ditambah hingga sama dengan jumlah kelas train asli terbesar. Ini **augmentasi 100% terhadap target keseimbangan kelas**, bukan menggandakan semua citra dan bukan SMOTE. Nilai default CLI script jika dipanggil tanpa argumen dapat berbeda; notebook memberikan `--target-ratio` secara eksplisit.

Sumber wajib melalui audit JSON resmi ACNE04-v2: 1.204 citra unik dan 32.443 anotasi. Folder sumber boleh memiliki lebih banyak gambar, tetapi hanya nama pada JSON resmi yang boleh masuk manifest asli. Label severity mengikuti awalan nama file original, bukan dihitung ulang dari jumlah lesi v2. Hasil augmentasi berisi tambahan pasangan sintetis train, bukan tambahan pasien/citra asli independen.

Di Colab, jalankan Step 6 dan audit Step 8, lalu Step 10 dan Step 10a dengan `OFFLINE_AUGMENT = True`, `AUGMENT_TARGET_RATIO = 1.0`. Step 11 sekarang wajib menggunakan manifest augmented; tidak ada fallback ke data asli. Step 20 menggunakan train/validation dari manifest augmented yang sama. Jangan menjalankan script augmentasi pada manifest augmented sebagai input.

Cache preprocessing/augmentasi tanpa provenance yang sesuai ditolak. Gunakan folder baru untuk sumber/protokol berbeda dan pertahankan hasil lama. Lihat [panduan verifikasi dataset](ACNE04V2_DATASET_VALIDATION.md).

## Split train/test tanpa validasi + 1000/kelas — Step 31

`scripts/split_train_test_manifest.py` membagi ulang **manifest asli 1.204
citra** menjadi train/test saja (tanpa `val`), stratified per kelas dengan seed
42. Split lama diabaikan. `scripts/augment_train_pairs_offline.py` kemudian
menerima `--target-per-class 1000` untuk menambah tiap kelas train sampai 1.000
pasangan foto-mask; baris test disalin tanpa augmentasi.

Notebook dipisah menjadi dua langkah. **Step 31** hanya menyiapkan data:
preprocessing ulang pada **512×512** (`refine_masks_color.py`, letterbox, warna
asli, tanpa denoise), split **60/40** dan **70/30** tanpa `val`, lalu augmentasi
train sampai 1.000/kelas. Tanpa preprocessing ulang, training 512 hanya
meng-upscale citra 320. **Step 31b** baru menjalankan training proposed dengan
`--no-validation` dan batch 8. Semua output ditulis ke root baru
`output/experiments_512/`; run lama hanya dibaca sebagai sumber. Karena tidak
ada validation untuk memilih checkpoint, `last_checkpoint.pt` tetap ditulis tiap
epoch dan `best_model.pt` berisi **epoch akhir**; test dievaluasi sekali setelah
checkpoint akhir tersimpan. Ringkasan data ada di `split_aug_preparation.csv`;
ringkasan test di `split_aug_results.csv`. Angka Step 31b adalah run baru dan
tidak boleh dicampur dengan T1/T2/T3.

SMOTE tetap eksperimen terpisah pada fitur klasifikasi dan tidak diubah oleh jalur ini. Untuk perbandingan yang adil, bandingkan run biasa dan offline augmentation dengan split, checkpoint awal, konfigurasi, serta seed yang sama; validation/test tetap distribusi asli.
