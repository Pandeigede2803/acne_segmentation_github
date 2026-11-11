# Augmentasi Offline Pasangan Foto-Mask

`scripts/augment_train_pairs_offline.py` membuat pasangan foto-mask tambahan untuk kelas train sebelum training. Karena split berasal dari manifest awal, hanya baris `train` yang di-augmentasi; validation dan test disalin tanpa perubahan. Rotasi dan flip yang sama diterapkan ke foto serta mask. Perubahan brightness/contrast hanya diterapkan ke foto.

Pada jalur notebook saat ini, `AUGMENT_TARGET_RATIO = 1.0`: setiap kelas train ditambah hingga sama dengan jumlah kelas train asli terbesar. Ini **augmentasi 100% terhadap target keseimbangan kelas**, bukan menggandakan semua citra dan bukan SMOTE. Nilai default CLI script jika dipanggil tanpa argumen dapat berbeda; notebook memberikan `--target-ratio` secara eksplisit.

Sumber wajib melalui audit JSON resmi ACNE04-v2: 1.204 citra unik dan 32.443 anotasi. Folder sumber boleh memiliki lebih banyak gambar, tetapi hanya nama pada JSON resmi yang boleh masuk manifest asli. Label severity mengikuti awalan nama file original, bukan dihitung ulang dari jumlah lesi v2. Hasil augmentasi berisi tambahan pasangan sintetis train, bukan tambahan pasien/citra asli independen.

Di Colab, jalankan Step 6 dan audit Step 8, lalu Step 10 dan Step 10a dengan `OFFLINE_AUGMENT = True`, `AUGMENT_TARGET_RATIO = 1.0`. Step 11 sekarang wajib menggunakan manifest augmented; tidak ada fallback ke data asli. Step 20 menggunakan train/validation dari manifest augmented yang sama. Jangan menjalankan script augmentasi pada manifest augmented sebagai input.

Cache preprocessing/augmentasi tanpa provenance yang sesuai ditolak. Gunakan folder baru untuk sumber/protokol berbeda dan pertahankan hasil lama. Lihat [panduan verifikasi dataset](ACNE04V2_DATASET_VALIDATION.md).

SMOTE tetap eksperimen terpisah pada fitur klasifikasi dan tidak diubah oleh jalur ini. Untuk perbandingan yang adil, bandingkan run biasa dan offline augmentation dengan split, checkpoint awal, konfigurasi, serta seed yang sama; validation/test tetap distribusi asli.
