# Catatan perubahan kode dan rencana training

**Tanggal:** 24 Oktober 2026  
**Status:** Perubahan kode diterapkan; training ulang belum dijalankan.

## Perubahan kode yang sudah diterapkan

### Preprocessing dan resize

- Preprocessing warna agresif (equalize per kanal, denoise, sharpen) tidak lagi menjadi default. Mode tersebut tetap tersedia sebagai `legacy` untuk eksperimen pembanding.
- Resize default memakai `letterbox` agar foto tidak diregangkan ke persegi. Mode lama `stretch` tetap tersedia.
- Manifest baru mencatat `resize_mode` dan `preprocessing_mode` yang dipakai.
- Panduan Colab menggunakan preprocessing `none`, resize `letterbox`, dan folder output versi baru agar artefak lama tidak tertimpa.

### Training dan checkpoint

- Optimizer SGD dibuat satu kali untuk setiap fase training dan dipertahankan antar-epoch, sehingga momentum tidak direset tiap epoch.
- Checkpoint menyimpan state optimizer fase 1 dan 2 serta random state Python, PyTorch, dan CUDA untuk resume yang lebih konsisten.
- Resume dari checkpoint yang sudah mencapai batas epoch sekarang berhenti dengan pesan yang meminta kenaikan batas epoch atau penggunaan folder output baru.
- Checkpoint lama yang tidak menyimpan state optimizer masih bisa dimuat; momentum dimulai dari awal dan akan muncul peringatan.

### Normalisasi model

- Opsi `--input-normalization auto` menerapkan normalisasi ImageNet saat `--pretrained` digunakan.
- Inferensi membaca pilihan normalisasi dari `metrics.json` run agar input inferensi sama dengan input training. Run lama tanpa metadata tetap memakai mode `none`.

## Rencana sebelum training ulang

1. Sinkronkan script terbaru ke Colab.
2. Generate ulang mask dan manifest ke folder baru `refined_masks_output_letterbox_v2` dengan `--resize-mode letterbox --preprocessing-mode none`.
3. Tinjau beberapa visualisasi mask dari setiap kelas sebelum training.
4. Jalankan training baru ke folder `training_runs/proposal_hybrid_letterbox_v2`; jangan memakai `--resume` atau folder checkpoint lama.
5. Evaluasi validation set untuk pemilihan model, lalu catat metrik test dan confusion matrix setelah konfigurasi dipilih.
6. Bandingkan hasil run baru dengan baseline lama. Karena preprocessing berubah bersamaan dengan optimisasi dan normalisasi, perubahan skor belum bisa diatribusikan ke satu faktor saja. Untuk mengisolasi penyebab, jalankan ablation dengan satu perubahan pada satu waktu.

## Belum dilakukan

- Training ulang, evaluasi model baru, dan perbandingan metrik belum dijalankan.
- Mask dan label belum diaudit manual secara menyeluruh.
- Belum ada bukti eksperimental bahwa perubahan preprocessing meningkatkan hasil; itu masih hipotesis yang perlu diuji.
