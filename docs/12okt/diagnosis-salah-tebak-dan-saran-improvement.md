# Diagnosis salah tebak dan saran perbaikan model

## Tujuan

Dokumen ini merangkum pola salah tebak pada test set, membandingkan 12 foto asli dengan input hasil preprocessing, dan menyusun langkah eksperimen berikutnya. Ini adalah diagnosis awal; foto salah tebak tidak otomatis berarti label dataset salah.

Bahan yang diperiksa:

- [Log prediksi test](test-prediksi-model-12okt.md)
- [Manifest refinement](refined_manifest.csv)
- [Perbandingan 12 foto asli, preprocessing, dan overlay](perbandingan-12-foto-asli-preprocessing.jpg)
- [Script preprocessing dan refinement mask](../../scripts/refine_masks_color.py)
- [Script training hybrid](../../scripts/train_proposal_hybrid_resnet50_malds.py)

## Status perubahan kode

Perbaikan lanjutan sudah diterapkan pada script lokal setelah analisis awal ini: preprocessing warna agresif tidak lagi menjadi pilihan default, resize letterbox menjaga rasio aspek, optimizer dipertahankan per fase dan state-nya disimpan saat checkpoint, serta normalisasi ImageNet dipakai otomatis saat training dengan encoder pretrained. Inferensi membaca pengaturan normalisasi dari `metrics.json` run. Hasil historis di bawah tetap menggambarkan run lama; perubahan ini belum dilatih ulang atau dievaluasi.

## Ringkasan hasil

Model benar pada 136 dari 224 foto (accuracy 60,7%) dan salah pada 88 foto. Sebagian besar kesalahan terjadi pada level yang berdekatan. Kesalahan level 0↔1 sendiri berjumlah 51 foto, sekitar 58% dari semua kesalahan.

| Label sebenarnya | Jumlah foto test | Salah tebak | Pola utama |
|---|---:|---:|---|
| Level 0 | 76 | 23 | 20 menjadi level 1, 3 menjadi level 2 |
| Level 1 | 97 | 48 | 31 menjadi level 0, 16 menjadi level 2, 1 menjadi level 3 |
| Level 2 | 29 | 6 | 4 menjadi level 1, 2 menjadi level 3 |
| Level 3 | 22 | 11 | Semua salah tebak menjadi level 2 |
| **Total** | **224** | **88** | **84 dari 88 kesalahan hanya meleset satu level** |

Model cenderung merapatkan prediksi ke level 1 dan 2. Level 1 adalah kelas terbanyak pada data training, dan model jarang memprediksi level 3: hanya 14 prediksi level 3 untuk 22 contoh level 3 pada test. Ini perlu diperiksa bersama distribusi data dan kualitas label; class weighting balanced saja belum menyelesaikan kebingungan kelas.

## Apa yang terlihat dari perbandingan foto?

Pada 12 foto yang tersedia, foto input preprocessing tampak lebih terang dan kontras daripada foto asli. Tekstur dan bintik kemerahan lebih menonjol. Perubahan ini konsisten dengan kode yang:

1. Menyamakan histogram setiap kanal B, G, dan R secara terpisah.
2. Menerapkan median filter.
3. Mempertajam gambar dengan unsharp mask.
4. Mengubah gambar menjadi 320 × 320. Dengan crop ratio default 1.0, seluruh gambar dipakai lalu diskalakan ke persegi; foto non-persegi dapat berubah proporsi.

Perubahan warna akibat equalization per kanal berisiko mengubah rona kulit dan tingkat kemerahan. Penajaman juga dapat menonjolkan pori, bekas jerawat, noise, dan tekstur lain yang bukan lesi aktif. Dari 12 contoh, preprocessing patut diuji sebagai faktor yang mungkin memperburuk prediksi, tetapi belum terbukti menjadi satu-satunya penyebab salah tebak.

Mask refinement dibuat dari gambar sebelum preprocessing. Jadi, foto input dapat berubah tampilannya sementara target mask tetap berasal dari gambar warna asli. Perbedaan ini perlu dipertimbangkan terutama ketika mengevaluasi segmentasi.

## Sampel yang dibandingkan

Tabel berikut menunjukkan 12 sampel yang dipilih dari foto salah tebak dan tersedia dalam folder lokal. Nama label mengikuti awalan nama file.

| Label | Nama file | Prediksi model |
|---|---|---|
| Level 0 | `levle0_146.jpg`, `levle0_156.jpg`, `levle0_210.jpg`, `levle0_260.jpg` | Level 1 |
| Level 1 | `levle1_166.jpg`, `levle1_234.jpg`, `levle1_261.jpg`, `levle1_300.jpg`, `levle1_363.jpg` | Level 0 atau 2 |
| Level 2 | `levle2_39.jpg` | Level 3 |
| Level 3 | `levle3_42.jpg`, `levle3_88.jpg` | Level 2 |

Lembar perbandingan memperlihatkan foto asli, input preprocessing, dan overlay hasil segmentasi. Overlay berwarna membandingkan mask prediksi dengan mask referensi; overlay tersebut bukan peta tingkat keparahan jerawat.

## Temuan kode training pada run historis

- Pada versi kode yang menghasilkan run historis, optimizer SGD dibuat ulang di dalam loop setiap epoch. Akibatnya, momentum tidak diteruskan ke epoch berikutnya.
- Checkpoint run historis tidak memuat state optimizer. Resume juga tidak persis melanjutkan dinamika optimisasi sebelumnya.
- Pada versi kode run historis, tensor gambar hanya diskalakan ke rentang 0–1 walau encoder pretrained digunakan; normalisasi ImageNet belum diterapkan.
- Eksekusi 12 Oktober resume dari epoch 121 dengan batas 120 epoch, sehingga tidak menjalankan training baru; hanya evaluasi checkpoint.

## Saran langkah perbaikan

### 1. Uji preprocessing secara terkontrol

Buat eksperimen pembanding dengan split, seed, model, epoch, dan konfigurasi loss yang sama:

- **A:** gambar asli, hanya resize yang konsisten.
- **B:** pipeline preprocessing lama.
- **C:** resize dengan koreksi rasio aspek (padding/letterbox), tanpa equalization per kanal dan tanpa sharpening.

Bandingkan accuracy, macro F1, balanced accuracy, kappa, confusion matrix, Dice, dan IoU. Jalankan beberapa seed bila sumber daya memungkinkan. Jangan memilih pipeline hanya dari contoh visual; gunakan validation set, lalu laporkan test set sekali setelah pilihan dibuat.

### 2. Audit label pada kasus yang sering tertukar

Tinjau foto level 0↔1 terlebih dahulu, kemudian level 1↔2 dan level 2↔3. Minta penilai yang memahami pedoman label untuk menilai foto tanpa melihat prediksi model. Catat label lama, label hasil audit, alasan, dan tingkat keyakinan. Jangan mengubah label hanya karena model tidak setuju.

### 3. Audit mask secara terpisah

Pilih contoh mask dari tiap level dan variasi Dice rendah/sedang/tinggi. Tampilkan foto asli, mask lingkaran awal, mask refined, dan mask prediksi. Periksa apakah mask meliputi lesi yang relevan, tidak memasukkan terlalu banyak kulit normal, dan konsisten antar-foto. Klasifikasi severity dan kualitas segmentasi adalah dua masalah berbeda.

### 4. Benahi pipeline optimisasi dan preprocessing bersama

- Pertahankan satu optimizer per fase; simpan dan pulihkan `optimizer_state_dict` saat resume.
- Simpan seluruh konfigurasi eksperimen dan gunakan folder output baru untuk setiap eksperimen.
- Jika menggunakan pretrained ResNet50, terapkan normalisasi input yang sesuai dan identik saat training, validation, test, serta inferensi.
- Pastikan augmentasi brightness/contrast saat training tidak menggandakan efek equalization yang agresif.

### 5. Pertimbangkan sifat label yang berurutan

Level 0 sampai 3 memiliki urutan. Karena sebagian besar kesalahan hanya satu tingkat, bandingkan classifier ordinal atau loss yang memberi penalti sesuai jarak kelas dengan pendekatan 4-class biasa. Tetap gunakan metrik klasifikasi standar dan laporkan mean absolute error level sebagai metrik tambahan.

## Rencana eksperimen yang disarankan

1. Bekukan daftar split dan audit kebocoran identitas/foto duplikat antar-split.
2. Audit label dan mask pada subset terpilih; catat perubahan tanpa mengedit diam-diam.
3. Jalankan ablation A/B/C preprocessing dengan pipeline optimizer/checkpoint yang sudah diperbaiki.
4. Pilih konfigurasi berdasarkan validation set.
5. Evaluasi konfigurasi terpilih pada test set dan simpan checkpoint, config, seed, confusion matrix, prediksi per foto, serta metrik.

## Batas kesimpulan

Analisis visual preprocessing di sini menggunakan 12 sampel salah tebak yang tersedia, bukan seluruh dataset. Metrik test berasal dari satu run dan satu split. Belum ada eksperimen terkontrol yang membuktikan preprocessing menyebabkan penurunan akurasi; rekomendasi utama adalah mengujinya sebagai ablation yang adil.
