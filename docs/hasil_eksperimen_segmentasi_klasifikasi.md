# Hasil Eksperimen Segmentasi dan Klasifikasi

Dokumen ini merangkum hasil eksperimen yang telah dijalankan pada folder `projectsegmentasi` dengan fokus:

- dataset `ACNE04-v2`
- anotasi segmentasi berbasis lingkaran (`cx, cy, radius`)
- target klasifikasi severity `levle0` sampai `levle3`
- evaluasi:
  - `Accuracy`
  - `Cohen's Kappa`
  - `Dice`
  - `IoU`

---

## 1. Tujuan Eksperimen

Tujuan utama eksperimen adalah menguji apakah arsitektur hybrid:

```text
Preprocessing
-> Segmentasi lokal
-> Ekstraksi fitur morfologi
-> Fitur global
-> AMFM
-> Klasifikasi severity
```

dapat menghasilkan:

- segmentasi lesi yang baik
- dan sekaligus klasifikasi severity yang baik

Secara khusus, fokus pengamatan utama adalah:

- apakah cabang segmentasi benar-benar belajar
- apakah nilai `Dice` dan `IoU` dapat meningkat
- apakah perubahan setup membantu kontribusi segmentasi ke hasil akhir

---

## 2. Ringkasan Eksperimen yang Sudah Dicoba

Eksperimen yang telah dilakukan:

1. `Dilated U-Net baseline`
2. `Dilated U-Net + augmentasi`
3. `Dilated U-Net + image_size 320`
4. `Dilated U-Net + BCE + Dice Loss`
5. `ResNet-50 encoder + AMFM`

Catatan penting:
- pada hampir semua eksperimen, klasifikasi masih bisa belajar sebagian
- tetapi segmentasi sangat sulit membaik

---

## 3. Hasil Eksperimen Penting

### 3.1 Baseline awal (100 image)

Hasil test:

```text
loss = 1.2351
acc  = 0.4211
dice = 0.0282
iou  = 0.0172
```

Interpretasi:
- pipeline berhasil dijalankan sampai akhir
- klasifikasi berjalan, tetapi akurasi masih rendah
- segmentasi sangat lemah
- hasil segmentasi pada tahap ini masih berbasis target dari bounding box, sehingga belum presisi

---

### 3.2 ACNE04-v2 baseline (20 epoch)

Hasil penting yang terbaca:

- `train_acc` naik dari `0.4412` menjadi `0.7500`
- `train_kappa` naik menjadi `0.6009`
- `val_acc` akhir mencapai `0.5714`
- `val_kappa` akhir mencapai `0.3596`

Interpretasi:
- cabang klasifikasi belajar
- model mulai memahami severity secara global
- tetapi nilai segmentasi tetap sangat lemah

Makna:
- model lebih banyak belajar dari fitur global
- kontribusi segmentasi ke hasil akhir belum kuat

---

### 3.3 200 image + augmentasi train

Konfigurasi:
- `max_images = 200`
- augmentasi aktif
- rotasi `±15 derajat`
- horizontal flip

Hasil test:

```text
loss  = 1.7070
acc   = 0.5556
kappa = 0.1802
dice  = 0.0000
iou   = 0.0000
```

Interpretasi:
- akurasi klasifikasi membaik
- kappa mulai bernilai positif
- tetapi segmentasi tetap gagal

Makna:
- augmentasi membantu klasifikasi
- augmentasi belum cukup untuk menghidupkan segmentasi

---

### 3.4 200 image + augmentasi + image_size 320

Konfigurasi:
- `max_images = 200`
- augmentasi aktif
- `image_size = 320`

Hasil test:

```text
loss  = 1.3244
acc   = 0.4444
kappa = 0.1332
dice  = 0.0000
iou   = 0.0000
```

Interpretasi:
- klasifikasi masih berjalan, tetapi menurun dibanding eksperimen augmentasi sebelumnya
- segmentasi tetap tidak bergerak

Makna:
- menaikkan resolusi saja belum cukup
- masalah segmentasi tidak selesai hanya dengan memperbesar input

---

### 3.5 Dilated U-Net + BCE + Dice Loss

Eksperimen ini dibuat untuk menguji apakah penguatan loss segmentasi bisa membantu nilai `Dice` dan `IoU`.

Pada salah satu run sempat muncul:

```text
loss  = 3.3881
acc   = 0.5000
kappa = 0.1549
dice  = 0.0002
iou   = 0.0001
```

Interpretasi:
- ada sinyal awal bahwa cabang segmentasi mulai merespons
- `Dice` dan `IoU` tidak lagi nol total
- namun nilainya masih sangat kecil

Tetapi pada run berikutnya hasil test menjadi:

```text
loss  = 2.9002
acc   = 0.3889
kappa = 0.0000
dice  = 0.0000
iou   = 0.0000
```

Interpretasi:
- eksperimen ini belum stabil
- klasifikasi menurun
- segmentasi kembali nol

Makna:
- `BCE + Dice Loss` memberi arah yang menarik, tetapi belum konsisten
- perbaikan loss saja belum cukup untuk menghasilkan segmentasi yang stabil

---

### 3.6 ResNet-50 encoder + AMFM

Eksperimen ResNet-50 diuji sebagai pembanding backbone.

Temuan penting:
- hasil segmentasi tetap tidak menunjukkan perbaikan berarti
- `Dice` dan `IoU` tetap sangat rendah

Makna:
- masalah utama bukan semata-mata karena backbone
- mengganti encoder ke ResNet-50 belum otomatis menyelesaikan masalah segmentasi

---

## 4. Analisis Umum

Dari seluruh eksperimen, pola yang konsisten adalah:

### 4.1 Klasifikasi lebih mudah belajar

Pada banyak run:
- `accuracy` naik
- `kappa` kadang ikut naik

Ini menunjukkan bahwa model masih bisa:
- menangkap fitur global wajah
- memprediksi severity pada tingkat tertentu

### 4.2 Segmentasi sangat sulit belajar

Pada banyak run:
- `dice = 0.0000`
- `iou = 0.0000`

Ini mengindikasikan bahwa:
- model cenderung memprediksi mask hampir kosong
- atau mask prediksi tidak overlap secara berarti dengan ground truth

### 4.3 Kontribusi segmentasi ke hybrid belum optimal

Secara struktur, model hybrid memang sudah dibuat:

```text
Segmentasi lokal -> fitur morfologi -> AMFM -> klasifikasi
```

Namun secara praktis:
- cabang segmentasi belum cukup kuat
- sehingga fitur morfologi belum mampu memberi kontribusi yang nyata ke hasil akhir

Artinya:
- hybrid sudah ada secara desain
- tetapi kontribusi segmentasi secara nyata belum terbukti kuat

---

## 5. Dugaan Penyebab Kegagalan Segmentasi

Beberapa kemungkinan penyebab:

### 5.1 Lesi terlalu kecil

- jerawat menempati area yang sangat kecil dibanding wajah
- piksel background jauh lebih dominan
- model cenderung memilih prediksi background

### 5.2 Target lingkaran belum tentu natural

- anotasi ACNE04-v2 berbentuk lingkaran
- lesi asli tidak selalu berbentuk lingkaran sempurna
- bisa terjadi mismatch antara bentuk lesi visual dan target mask

### 5.3 Segmentasi kalah oleh klasifikasi

- model belajar dua tugas sekaligus
- klasifikasi severity tampak lebih mudah dipelajari
- akibatnya model lebih banyak mengoptimalkan cabang klasifikasi

### 5.4 Data masih terbatas

- eksperimen utama masih banyak dilakukan di `200 image`
- jumlah ini bisa belum cukup untuk segmentasi piksel-level

---

## 6. Kesimpulan Sementara

Kesimpulan utama dari hasil eksperimen:

1. Arsitektur hybrid dapat dijalankan end-to-end.
2. Cabang klasifikasi mampu belajar sebagian.
3. Cabang segmentasi belum menunjukkan performa yang memadai.
4. Nilai `Dice` dan `IoU` yang hampir selalu nol menunjukkan bahwa model belum berhasil mempelajari mask lesi secara stabil.
5. Penggantian backbone ke ResNet-50 belum menyelesaikan masalah segmentasi.
6. Penggunaan `BCE + Dice Loss` menunjukkan arah yang lebih menjanjikan, tetapi hasilnya masih belum konsisten.

---

## 7. Implikasi untuk Penelitian

Implikasi penting bagi paper:

- hasil saat ini belum cukup untuk menyatakan bahwa cabang segmentasi memberikan kontribusi kuat terhadap klasifikasi severity
- hasil lebih mendukung interpretasi bahwa model saat ini masih dominan bergantung pada fitur global
- untuk membuktikan manfaat hybrid secara lebih kuat, segmentasi harus terlebih dahulu dibuat lebih stabil

---

## 8. Arah Lanjutan yang Disarankan

Langkah yang paling masuk akal berikutnya:

1. menambah jumlah data eksperimen
2. mengecek visualisasi mask prediksi vs ground truth
3. menguji loss segmentasi yang lebih kuat secara lebih sistematis
4. membandingkan run dalam format tabel yang konsisten

Langkah yang sangat penting:

```text
Visualisasi ground truth mask vs predicted mask
```

Karena tanpa melihat mask secara langsung, sulit memastikan apakah model:
- benar-benar memprediksi kosong
- atau memprediksi area kecil yang salah posisi

---

## 9. Ringkasan Akhir

Secara sederhana:

```text
Klasifikasi: belajar sebagian
Segmentasi: belum stabil
Hybrid: sudah terbentuk, tetapi kontribusi segmentasi belum optimal
```

Dengan demikian, hasil eksperimen saat ini lebih tepat dibaca sebagai:

```text
proof-of-concept pipeline hybrid yang berhasil dijalankan,
namun cabang segmentasi masih memerlukan perbaikan signifikan
agar mampu memberi dampak nyata pada hasil akhir klasifikasi severity
```

