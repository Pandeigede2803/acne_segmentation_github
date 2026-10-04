# Flow Chart Sederhana Hasil Penelitian

Dokumen ini menjelaskan alur penelitian segmentasi dan klasifikasi acne secara sederhana, tetapi tetap detail. Fokus utama penelitian ini adalah menguji model hybrid yang menggabungkan segmentasi lesi jerawat dan klasifikasi tingkat keparahan acne.

---

## 1. Flow Besar Penelitian

```mermaid
flowchart TD
    A[Dataset ACNE04-v2] --> B[Ambil gambar wajah]
    A --> C[Ambil anotasi lesi: cx, cy, radius]

    B --> D[Preprocessing gambar]
    C --> E[Buat mask biner lesi]

    D --> F[Split data: train, validation, test]
    E --> F

    F --> G[Training model hybrid]

    G --> H[Cabang segmentasi]
    G --> I[Cabang klasifikasi global]

    H --> J[Prediksi mask lesi]
    J --> K[Ekstraksi fitur morfologi: jumlah, area, distribusi lesi]

    I --> L[Ekstraksi fitur global wajah]

    K --> M[AMFM Fusion]
    L --> M

    M --> N[Classification Head]
    N --> O[Prediksi severity: level 0 - level 3]

    J --> P[Evaluasi segmentasi: Dice dan IoU]
    O --> Q[Evaluasi klasifikasi: Accuracy dan Cohen's Kappa]

    P --> R[Analisis hasil]
    Q --> R

    R --> S[Kesimpulan penelitian]
```

---

## 2. Penjelasan Flow Besar

Penelitian dimulai dari dataset `ACNE04-v2`. Dataset ini berisi dua informasi utama:

- gambar wajah pasien acne
- anotasi lesi jerawat berbentuk lingkaran

Anotasi lesi terdiri dari:

```text
cx     = posisi tengah lesi pada sumbu x
cy     = posisi tengah lesi pada sumbu y
radius = ukuran lingkaran lesi
```

Dari anotasi tersebut, dibuat mask segmentasi biner:

```text
0 = background
1 = area lesi jerawat
```

Setelah gambar dan mask siap, data dibagi menjadi:

```text
train set
validation set
test set
```

Data train digunakan untuk melatih model, data validation digunakan untuk memantau performa selama training, dan data test digunakan untuk mengevaluasi hasil akhir model.

---

## 3. Flow Preprocessing Gambar

```mermaid
flowchart TD
    A[Gambar wajah asli] --> B[Convert ke RGB]
    B --> C[Center crop opsional]
    C --> D[Histogram equalization]
    D --> E[Denoise]
    E --> F[Sharpen]
    F --> G[Resize ke image size]
    G --> H[Gambar siap masuk model]
```

Preprocessing dilakukan agar gambar lebih siap dipelajari oleh model. Tahap ini membantu menormalkan input, memperjelas tekstur, dan menyesuaikan ukuran gambar dengan kebutuhan model.

Pada eksperimen ini, preprocessing dilakukan secara `on-the-fly` saat training.

---

## 4. Flow Pembuatan Mask Segmentasi

```mermaid
flowchart TD
    A[Anotasi lingkaran: cx, cy, radius] --> B[Sesuaikan koordinat ke ukuran gambar target]
    B --> C[Buat area lingkaran]
    C --> D[Isi area lingkaran dengan nilai 1]
    D --> E[Isi area luar lingkaran dengan nilai 0]
    E --> F[Mask biner lesi]
```

Mask ini menjadi target pembelajaran untuk cabang segmentasi. Artinya, model dilatih agar bisa menghasilkan mask prediksi yang mirip dengan mask ground truth.

---

## 5. Flow Model Hybrid

```mermaid
flowchart TD
    A[Input gambar wajah] --> B[Encoder: Dilated U-Net atau ResNet-50]
    B --> C[Bottleneck Feature]
    C --> D[Spatial Attention]

    D --> E[Segmentation Decoder]
    E --> F[Prediksi mask lesi]

    F --> G[Ekstraksi fitur morfologi]
    G --> H[AMFM Fusion]

    D --> I[Ekstraksi fitur global wajah]
    I --> H

    H --> J[Classification Head]
    J --> K[Output severity acne: level 0 - level 3]
```

Model disebut hybrid karena menggabungkan dua jenis informasi:

| Komponen | Fungsi |
|---|---|
| Cabang segmentasi | Memprediksi area lesi jerawat |
| Fitur morfologi | Mengambil informasi jumlah, area, dan distribusi lesi |
| Fitur global wajah | Mengambil informasi kondisi wajah secara keseluruhan |
| AMFM Fusion | Menggabungkan fitur morfologi dan fitur global |
| Classification Head | Menentukan tingkat severity acne |

Secara konsep, model diharapkan tidak hanya melihat wajah secara umum, tetapi juga memperhatikan lokasi dan bentuk lesi jerawat.

---

## 6. Flow Evaluasi

```mermaid
flowchart TD
    A[Hasil training model] --> B[Prediksi mask lesi]
    A --> C[Prediksi severity acne]

    B --> D[Bandingkan dengan ground truth mask]
    D --> E[Hitung Dice]
    D --> F[Hitung IoU]

    C --> G[Bandingkan dengan label severity asli]
    G --> H[Hitung Accuracy]
    G --> I[Hitung Cohen's Kappa]

    E --> J[Analisis performa segmentasi]
    F --> J
    H --> K[Analisis performa klasifikasi]
    I --> K

    J --> L[Kesimpulan hasil eksperimen]
    K --> L
```

Metrik yang digunakan:

| Metrik | Digunakan untuk | Makna sederhana |
|---|---|---|
| Dice | Segmentasi | Mengukur kemiripan mask prediksi dengan mask asli |
| IoU | Segmentasi | Mengukur area overlap antara mask prediksi dan mask asli |
| Accuracy | Klasifikasi | Mengukur persentase prediksi severity yang benar |
| Cohen's Kappa | Klasifikasi | Mengukur kesepakatan prediksi dengan label asli dengan mempertimbangkan peluang tebak acak |

---

## 7. Flow Hasil Penelitian

```mermaid
flowchart TD
    A[Eksperimen dijalankan] --> B[Klasifikasi belajar sebagian]
    A --> C[Segmentasi sulit belajar]

    B --> D[Accuracy beberapa kali naik]
    B --> E[Kappa kadang bernilai positif]

    C --> F[Dice sering 0.0000]
    C --> G[IoU sering 0.0000]

    F --> H[Prediksi mask kemungkinan kosong atau tidak overlap]
    G --> H

    H --> I[Fitur morfologi menjadi lemah]
    I --> J[AMFM lebih bergantung pada fitur global]
    J --> K[Hybrid sudah berjalan, tetapi kontribusi segmentasi belum kuat]
```

Hasil eksperimen menunjukkan pola utama:

```text
Klasifikasi: masih bisa belajar sebagian
Segmentasi : belum stabil dan sering gagal
Hybrid     : sudah terbentuk, tetapi belum optimal
```

Klasifikasi masih bisa belajar karena model dapat memanfaatkan fitur global wajah. Namun, segmentasi sulit belajar karena area lesi sangat kecil dibanding background. Akibatnya, model cenderung memprediksi background atau mask kosong.

---

## 8. Interpretasi Hasil

Secara sederhana, hasil penelitian dapat dibaca seperti ini:

```mermaid
flowchart TD
    A[Pipeline berhasil dijalankan end-to-end] --> B[Model bisa melakukan training dan evaluasi]
    B --> C[Klasifikasi severity belajar sebagian]
    B --> D[Segmentasi lesi belum berhasil stabil]
    D --> E[Dice dan IoU sangat rendah]
    E --> F[Fitur morfologi belum kuat]
    F --> G[Kontribusi segmentasi ke klasifikasi belum terbukti besar]
```

Artinya, masalah utama penelitian saat ini bukan pada pipeline kode dasar. Pipeline sudah berjalan dari preprocessing, training, evaluasi, sampai penyimpanan hasil.

Masalah utama ada pada perilaku belajar model, khususnya cabang segmentasi.

---

## 9. Dugaan Penyebab Segmentasi Gagal

Beberapa penyebab yang paling mungkin:

1. Lesi jerawat terlalu kecil dibanding ukuran wajah.
2. Piksel background jauh lebih banyak daripada piksel lesi.
3. Model cenderung memilih solusi aman dengan memprediksi background.
4. Anotasi lingkaran mungkin kurang sesuai dengan bentuk asli lesi.
5. Cabang klasifikasi lebih mudah belajar dibanding cabang segmentasi.
6. Data eksperimen masih relatif terbatas untuk tugas segmentasi piksel-level.

---

## 10. Kesimpulan Sederhana

Kesimpulan utama penelitian:

```text
Model hybrid berhasil dibuat dan dijalankan.
Cabang klasifikasi mampu belajar sebagian.
Cabang segmentasi belum stabil.
Dice dan IoU yang sering nol menunjukkan model belum berhasil mempelajari mask lesi.
Karena segmentasi lemah, fitur morfologi belum memberi kontribusi besar.
```

Dengan demikian, hasil penelitian saat ini lebih tepat disebut sebagai:

```text
Proof-of-concept pipeline hybrid segmentasi dan klasifikasi acne yang berhasil berjalan,
tetapi cabang segmentasi masih perlu diperbaiki agar benar-benar membantu klasifikasi severity.
```

