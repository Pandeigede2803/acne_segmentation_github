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

### 3.7 Update eksperimen Google Colab - 22 Juni 2026

Eksperimen terbaru dijalankan di Google Colab menggunakan seluruh dataset yang berhasil divalidasi.

Informasi dataset:

```text
Total gambar = 1457
Device       = cuda
```

Distribusi kelas severity:

| Kelas | Jumlah | Persentase |
|---|---:|---:|
| `levle0` | 497 | 34.1% |
| `levle1` | 637 | 43.7% |
| `levle2` | 186 | 12.8% |
| `levle3` | 137 | 9.4% |

Hasil validasi pada epoch 120:

```text
Val Dice = 0.4555
Val Acc  = 0.4815
```

Hasil test:

```text
loss  = 1.1503
acc   = 0.5000
kappa = 0.0652
dice  = 0.4515
iou   = 0.3082
```

Interpretasi:

- segmentasi menunjukkan peningkatan signifikan dibanding eksperimen sebelumnya
- nilai `Dice = 0.4515` dan `IoU = 0.3082` menunjukkan mask prediksi sudah memiliki overlap dengan ground truth
- model tidak lagi sepenuhnya gagal pada cabang segmentasi
- klasifikasi masih belum kuat karena `Accuracy = 0.5000` disertai `Kappa = 0.0652`
- nilai kappa yang sangat rendah menunjukkan prediksi severity belum memiliki kesepakatan yang kuat dengan label asli

Makna:

- penggunaan dataset yang lebih besar dan training lebih panjang membuat cabang segmentasi mulai belajar
- masalah utama mulai bergeser dari segmentasi yang gagal total menjadi klasifikasi severity yang belum stabil
- salah satu faktor kuat yang memengaruhi klasifikasi adalah ketidakseimbangan jumlah data antar level severity

Analisis imbalance:

- `levle0` dan `levle1` mendominasi dataset dengan total 1134 gambar
- `levle2` dan `levle3` hanya berjumlah 323 gambar
- `levle1` berjumlah sekitar 4.65 kali lebih banyak dibanding `levle3`
- kondisi ini dapat menyebabkan model bias ke kelas mayoritas, terutama `levle0` dan `levle1`

Dengan kondisi tersebut, accuracy 0.5000 belum cukup untuk menyatakan model klasifikasi sudah baik. Nilai `Cohen's Kappa = 0.0652` lebih menunjukkan bahwa model belum mampu membedakan seluruh level severity secara seimbang.

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

Namun pada update eksperimen Google Colab tanggal 22 Juni 2026, segmentasi mulai menunjukkan perbaikan:

```text
test_dice = 0.4515
test_iou  = 0.3082
```

Artinya, cabang segmentasi sudah mulai belajar dan tidak lagi berada pada kondisi gagal total. Meskipun demikian, performa segmentasi masih perlu ditingkatkan agar mask prediksi lebih presisi dan lebih konsisten.

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

### 5.5 Distribusi kelas severity tidak seimbang

Pada eksperimen Google Colab tanggal 22 Juni 2026, total dataset yang valid adalah 1457 gambar dengan distribusi:

```text
levle0 = 497
levle1 = 637
levle2 = 186
levle3 = 137
```

Distribusi ini tidak seimbang karena `levle0` dan `levle1` jauh lebih dominan dibanding `levle2` dan `levle3`.

Dampaknya:

- model lebih sering melihat contoh dari kelas mayoritas
- model berpotensi bias memprediksi `levle0` atau `levle1`
- accuracy bisa terlihat cukup, tetapi kappa tetap rendah
- kelas minoritas seperti `levle2` dan `levle3` lebih sulit dipelajari

---

## 6. Kesimpulan Sementara

Kesimpulan utama dari hasil eksperimen:

1. Arsitektur hybrid dapat dijalankan end-to-end.
2. Cabang klasifikasi mampu belajar sebagian.
3. Pada eksperimen awal, cabang segmentasi belum menunjukkan performa yang memadai.
4. Pada update Google Colab 22 Juni 2026, segmentasi mulai membaik dengan `Dice = 0.4515` dan `IoU = 0.3082`.
5. Klasifikasi severity masih lemah karena `Accuracy = 0.5000` tetapi `Cohen's Kappa = 0.0652`.
6. Ketidakseimbangan distribusi kelas severity menjadi dugaan kuat penyebab klasifikasi belum stabil.
7. Penggunaan backbone ResNet-50 dengan dilated decoder mulai menunjukkan arah yang lebih baik pada segmentasi, tetapi klasifikasi masih perlu diperbaiki.

---

## 7. Implikasi untuk Penelitian

Implikasi penting bagi paper:

- hasil terbaru menunjukkan bahwa cabang segmentasi mulai memberikan sinyal pembelajaran yang lebih baik
- nilai `Dice` dan `IoU` terbaru dapat digunakan sebagai bukti bahwa pipeline segmentasi sudah berkembang dibanding eksperimen awal
- hasil klasifikasi belum cukup kuat karena accuracy masih sedang dan kappa sangat rendah
- untuk membuktikan manfaat hybrid secara lebih kuat, perlu analisis tambahan terhadap confusion matrix dan distribusi prediksi per kelas
- imbalance kelas severity perlu dibahas sebagai faktor yang memengaruhi performa klasifikasi

---

## 8. Arah Lanjutan yang Disarankan

Langkah yang paling masuk akal berikutnya:

1. mengecek confusion matrix klasifikasi severity
2. mengecek apakah prediksi model bias ke `levle0` atau `levle1`
3. menggunakan class weighting atau weighted sampler untuk mengatasi imbalance
4. mengecek visualisasi mask prediksi vs ground truth
5. menguji loss segmentasi dan loss klasifikasi secara lebih sistematis
6. membandingkan run dalam format tabel yang konsisten

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
Segmentasi: mulai membaik pada eksperimen terbaru
Klasifikasi: masih lemah dan kemungkinan terpengaruh imbalance kelas
Hybrid: sudah berjalan, tetapi kontribusi segmentasi ke klasifikasi masih perlu dibuktikan lebih kuat
```

Dengan demikian, hasil eksperimen saat ini lebih tepat dibaca sebagai:

```text
pipeline hybrid yang berhasil dijalankan dan mulai menunjukkan peningkatan segmentasi,
namun klasifikasi severity masih perlu diperbaiki terutama karena distribusi kelas tidak seimbang
dan nilai Cohen's Kappa masih rendah
```
