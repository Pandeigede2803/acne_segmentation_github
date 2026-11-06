# Augmentasi Offline Pasangan Foto-Mask

`scripts/augment_train_pairs_offline.py` membuat pasangan foto-mask tambahan untuk kelas train sebelum training. Karena split berasal dari manifest awal, hanya baris `train` yang di-augmentasi; validation dan test disalin tanpa perubahan. Rotasi dan flip yang sama diterapkan ke foto serta mask. Perubahan brightness/contrast hanya diterapkan ke foto.

Default target adalah 75% dari jumlah kelas train terbesar, dengan jumlah asli dipertahankan untuk kelas yang sudah melampaui target. Untuk hitungan train Oktober (347, 445, 130, 95), hasilnya sekitar 347, 445, 334, 334. Ini oversampling moderat, bukan SMOTE. Hasil disimpan dalam folder dan manifest terpisah di Drive.

Di Colab, jalankan Step 10, aktifkan `OFFLINE_AUGMENT = True` pada Step 10a, lalu jalankan training biasa. Training menggunakan manifest augmented; tanpa itu, manifest asli tetap dipakai. Jangan menjalankan ulang script pada manifest augmented sebagai input. Jika perlu membuat ulang, mulai lagi dari `refined_manifest.csv` asli.

SMOTE tetap eksperimen terpisah pada fitur klasifikasi dan tidak diubah oleh jalur ini. Untuk perbandingan yang adil, bandingkan run biasa dan offline augmentation dengan split, checkpoint awal, konfigurasi, serta seed yang sama; validation/test tetap distribusi asli.
