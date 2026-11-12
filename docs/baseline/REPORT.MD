# Laporan Penelitian

**Fokus: Proposed Hybrid Multi-Task dengan Augmentasi Offline 100% dan Ablation Study**

> Dokumen ini hanya membahas training proposed augmented pada Step 11 dan
> ablation augmented pada Step 20, serta pembanding single-task pada Step 29
> di `acne_colab.ipynb`. Semua tabel hasil
> sengaja dikosongkan sampai artefak eksperimen tersebut tersedia dan diperiksa.
> Sel kosong bukan nilai nol. Dokumen ini tidak menyatakan training sudah selesai.

## 1. Rumusan masalah

1. Bagaimana merancang dan membangun kerangka kerja hibrida yang mengintegrasikan segmentasi lesi presisi berbasis Dilated UNet dengan mekanisme atensi spasial dan klasifikasi keparahan jerawat secara simultan dalam satu unified deep learning pipeline?
2. Bagaimana memanfaatkan representasi fitur dari tugas segmentasi (lokal) untuk meningkatkan performa tugas klasifikasi global (Deep Representation Learning) sehingga menghasilkan sistem yang lebih akurat dibandingkan pendekatan single-task?
3. Bagaimana mengevaluasi performa kerangka kerja yang diusulkan pada dataset ACNE04-v2 menggunakan metrik evaluasi standar dari domain computer vision dan pencitraan medis, serta membuktikan kontribusi setiap komponen arsitektur melalui ablation study?

## 2. Tujuan penelitian

Tujuan berikut mengacu pada proposal; ketercapaiannya akan dibahas setelah hasil tersedia.

1. Mengembangkan kerangka kerja hibrida Deep Learning yang mampu melakukan segmentasi lesi jerawat secara presisi (mengenali tipe dan lokasi lesi) sekaligus mengklasifikasikan tingkat keparahan berdasarkan kriteria Hayashi secara otomatis.
2. Mengeksplorasi dan mengoptimalkan mekanisme Deep Representation Learning yang memanfaatkan fitur representasi dari cabang segmentasi untuk memperkuat cabang klasifikasi global dalam arsitektur multi-task learning.
3. Mengevaluasi performa model pada dataset ACNE04-v2 menggunakan Accuracy, Cohen's Kappa, Dice Coefficient, dan IoU serta menguji kontribusi AMFM dan MA-LDS melalui ablation study.

**Batas implementasi:** mask yang dilatih saat ini bersifat biner (area lesi dan
background), bukan klasifikasi tipe papul/pustul/komedo. Baseline ablation tetap
multi-task; pembanding single-task murni disediakan terpisah pada Step 29.
Kedua batas ini perlu diperhatikan
saat menjawab tujuan dan rumusan masalah; jangan menyimpulkan klaim yang tidak diuji.

## 3. Ruang lingkup eksperimen

| Kelompok | Step | Fokus |
|---|---|---|
| Training utama | 11 | Model lengkap proposed + augmentasi offline 100% |
| Ablation | 20 | Enam varian + proposed pada data augmented dan protokol yang sebanding |
| Pembanding single-task | 29 | ResNet-50 klasifikasi murni, train/validation augmented yang sama |

Eksperimen lain tidak dimasukkan dalam laporan ini. Proposed pada Step 20
merupakan kontrol yang dilatih terpisah tanpa evaluasi test. Hasil validation
kontrol ini menjadi pembanding utama antar-varian ablation, bukan hasil test Step 11.

## 4. Dataset dan pembagian data

Dataset yang dituju adalah ACNE04-v2, subset dari gambar ACNE04 original.
[JSON resmi](https://github.com/AIpourlapeau/acne04v2/blob/5930801fc75fb1a7084c26c1db3095d3e42d1f55/Acne04-v2_annotations.json)
memuat **1.204 gambar unik** dan **32.443 anotasi lesi**. Berdasarkan awalan nama
file, distribusinya adalah level 0: 470, level 1: 579, level 2: 133, level 3: 22.
Label severity pada pipeline mengikuti awalan nama file original, **bukan hasil
perhitungan ulang jumlah lesi dari anotasi v2**.

Folder sumber boleh berisi lebih banyak file, misalnya 1.457 gambar. File di luar
daftar JSON v2 tidak dimasukkan ke manifest. Hitungan file folder tidak boleh
dilaporkan sebagai jumlah sampel penelitian v2. Jumlah tiap split pada tabel
berikut tetap harus diisi dari manifest aktual setelah audit, bukan perkiraan.

Notebook memverifikasi anotasi dengan commit resmi
`5930801fc75fb1a7084c26c1db3095d3e42d1f55` dan SHA-256
`ae03fc23fed0925c254dc4c4c9e55b539bd199b236151cb649857378a43539e9`.
Manifest asli wajib berisi tepat 1.204 nama unik yang sama dengan JSON resmi,
jumlah lesi per gambar harus cocok, dan seluruh file pasangan harus tersedia.
Anotasi XML original dan cache manifest tanpa provenance yang sesuai ditolak.

Pembagian train/validation/test ditetapkan **sebelum** augmentasi. Hanya train
yang ditambah. Validation dan test tetap memakai baris asli tanpa resampling.
Periksa duplikasi dan kebocoran subjek/sumber antar-split.

| Kelas | Train asli | Train setelah augmentasi 100% | Validation asli | Test asli |
|---|---:|---:|---:|---:|
| Level 0 / mild | | | | |
| Level 1 / moderate | | | | |
| Level 2 / severe | | | | |
| Level 3 / very severe | | | | |
| **Total** | | | | |

## 5. Preprocessing dan augmentasi offline 100%

### 5.1 Preprocessing

- Foto dipersiapkan menggunakan letterbox pada kanvas 320 × 320 sesuai
  konfigurasi notebook, tanpa equalization/sharpening pada mode `none`.
- Target mask dibuat dari anotasi lingkaran dan deteksi warna Lab relative
  redness serta HSV, dibatasi pada area anotasi dan dirapikan secara morfologi.
- Fallback circle mask digunakan jika hasil deteksi terlalu kecil.
- Lab/HSV dipakai untuk pembentukan target mask; ini berbeda dari peningkatan
  warna foto masukan. Dice/IoU mengukur kesesuaian terhadap target mask tersebut,
  bukan otomatis terhadap kontur manual yang divalidasi dokter.

Manifest yang sudah ada dapat dipakai kembali oleh notebook. Pastikan foto-mask
aktual sesuai konfigurasi yang dilaporkan, bukan hanya opsi pada cell.

### 5.2 Definisi augmentasi 100%

`AUGMENT_TARGET_RATIO = 1.0` berarti setiap kelas train ditambah hingga mencapai
**jumlah kelas train asli terbesar**, bukan menggandakan semua foto.
Sampel asli tetap dipertahankan dan pasangan sintetis hanya berasal dari train.

- Rotasi dan horizontal flip diterapkan berpasangan pada foto dan mask.
- Brightness/contrast hanya diterapkan pada foto.
- Validation/test tidak ditambah atau diubah.
- Semua varian ablation memakai manifest augmented yang sama, bukan membuat
  pasangan augmentasi offline baru untuk tiap varian.
- Augmentasi online pada trainer tetap aktif di train; ini tambahan terhadap
  augmentasi offline. Validation/test tidak memakai augmentasi online.

## 6. Training utama — Step 11

### 6.1 Model lengkap proposed

Alur model:

```text
Foto → Shared encoder ResNet-50 → Spatial Attention
                                ├─ Dilated decoder → Mask prediksi
                                │                    → Vektor morfologi
                                └─ Global pooling → Fitur global
Fitur morfologi + fitur global → AMFM → Klasifikasi severity
```

Vektor morfologi mencakup area, density, cx, cy, dan spread. MA-LDS membentuk
distribusi target klasifikasi berdasarkan morfologi target pada proses training.
Model mengoptimalkan loss segmentasi BCE + Dice, loss klasifikasi KL, dan
morphology consistency MSE.

### 6.2 Konfigurasi yang ditetapkan

Tabel ini adalah protokol, bukan hasil. Cocokkan kembali dengan konfigurasi
tersimpan saat mengisi hasil final.

| Parameter | Konfigurasi |
|---|---|
| Varian | `proposed` |
| Manifest | `AUGMENTED_MANIFEST` dari Step 10a |
| Target augmentasi offline | 1,0 × jumlah kelas train asli terbesar per kelas |
| Encoder | ResNet-50 pretrained, normalisasi ImageNet |
| Input / batch default | 320 × 320 / 16, mengikuti konfigurasi pusat |
| Fase training default | 30 epoch freeze + 90 epoch fine-tuning |
| Class weighting | `balanced`, dihitung dari data train yang digunakan |
| Weighted sampler | Tidak digunakan |
| Smoothing | MA-LDS |
| Seed | 42 |
| Pemilihan checkpoint | Total validation loss terendah |
| Evaluasi test | Setelah training; tidak digunakan untuk tuning |

Run baru harus memakai `RUN_ID` yang belum digunakan. Resume hanya untuk run
yang sama dengan sumber dan konfigurasi tetap. Identitas sumber dicatat dalam
`augmentation_training_provenance.json`; jangan menimpa hasil eksperimen lain.

### 6.3 Identitas run

| Informasi | Isian |
|---|---|
| Run ID / folder output | |
| Tanggal training | |
| Manifest dan SHA-256 | |
| Trainer dan SHA-256 | |
| Perangkat dan versi pustaka | |
| Status selesai | |
| Epoch checkpoint terbaik | |
| Lokasi `metrics.json` dan checkpoint | |

### 6.4 Hasil utama

Train dan validation diisi dari epoch checkpoint terbaik yang sama. Test diisi
dari evaluasi checkpoint tersebut, bukan dari epoch terakhir secara otomatis.

| Evaluasi | Epoch terbaik | Accuracy | Kappa | Dice | IoU | Total loss |
|---|---:|---:|---:|---:|---:|---:|
| Train | | | | | | |
| Validation | | | | | | |
| Test | | | | | | |

### 6.5 Hasil klasifikasi per kelas pada test

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Level 0 | | | | |
| Level 1 | | | | |
| Level 2 | | | | |
| Level 3 | | | | |
| Macro average | | | | |
| Weighted average | | | | |

Confusion matrix, visualisasi target vs prediksi mask, dan kurva training akan
ditambahkan dari artefak run yang sama setelah tersedia.

## 7. Ablation study augmented — Step 20

### 7.1 Protokol perbandingan

Step 20 membaca artefak run Step 11 yang telah tersimpan di Drive; training
utama tidak dijalankan ulang. Setelah runtime reset, path sumber, manifest,
konfigurasi, serta versi trainer dipulihkan dan diverifikasi sebelum ablation.

- Mengambil hanya train/validation dari manifest augmented Step 11.
- Menggunakan class weighting balanced, tanpa weighted sampler, dan konfigurasi
  input, batch, epoch, seed, serta sumber trainer yang sesuai Step 11.
- Tujuh model dilatih terpisah: enam varian dan satu proposed lengkap.
- Test tidak dievaluasi (`--skip-test-eval`).
- Setiap run dipilih berdasarkan total validation loss terendah.
- Folder ablation terpisah dari training utama dan hasil eksperimen lain.
  Varian selesai dibaca kembali, varian terputus di-resume; pengulangan dari
  awal harus memakai folder baru.
- Catatan protokol dan hash tersimpan dalam `ablation_provenance.json`.
  Pemeriksaan hash belum mencakup isi masing-masing foto/mask.

### 7.2 Definisi varian

| Varian | Spatial Attention | Dilation | Fusi untuk klasifikasi | Smoothing |
|---|---|---|---|---|
| baseline | Tidak | Tidak | Global saja | Tidak |
| no_attention | Tidak | Ya | AMFM | MA-LDS |
| no_dilation | Ya | Tidak | AMFM | MA-LDS |
| no_amfm | Ya | Ya | Concatenation | MA-LDS |
| lds | Ya | Ya | AMFM | LDS tetap |
| no_fusion | Ya | Ya | Global saja | MA-LDS |
| proposed | Ya | Ya | AMFM | MA-LDS |

Semua varian tetap memiliki cabang segmentasi dan shared encoder. `baseline`
mengubah beberapa komponen sekaligus sehingga bukan ablasi satu komponen.
`no_amfm` mengganti AMFM dengan concatenation dan mengubah dimensi input
classifier. `lds` adalah smoothing tetap pada implementasi lokal, bukan otomatis
reproduksi penuh metode referensi. MA-LDS yang tetap aktif pada `no_fusion`
berarti varian tersebut masih memakai informasi morfologi target saat training;
yang dihilangkan adalah fusi fitur morfologi prediksi untuk klasifikasi.

### 7.3 Identitas eksperimen ablation

| Informasi | Isian |
|---|---|
| Folder ablation | |
| Run Step 11 yang menjadi sumber protokol | |
| Manifest train/validation dan SHA-256 | |
| Trainer dan SHA-256 | |
| Status ketujuh varian | |
| Lokasi `ablation_results.csv` | |

### 7.4 Hasil ablation — validation

Semua angka harus berasal dari run augmented ini, pada epoch dengan total
validation loss minimum masing-masing varian.

| Varian | Epoch terbaik | Train accuracy | Val. accuracy | Val. Kappa | Val. Dice | Val. IoU | Val. loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | | | | | | | |
| no_attention | | | | | | | |
| no_dilation | | | | | | | |
| no_amfm | | | | | | | |
| lds | | | | | | | |
| no_fusion | | | | | | | |
| proposed | | | | | | | |

### 7.5 Rencana analisis

1. Bandingkan setiap varian terhadap proposed pada validation yang sama.
2. Bahas klasifikasi dan segmentasi secara terpisah; keduanya tidak harus meningkat
   bersamaan. Jangan membandingkan validation ablation dengan test training utama.
3. Gunakan loss untuk memilih checkpoint dalam run, bukan sebagai ukuran tunggal
   keunggulan antar-varian dengan distribusi target berbeda.
4. Laporkan arah perubahan apa adanya; komponen tidak harus memberi efek positif.
5. Satu seed belum membuktikan kestabilan atau signifikansi statistik.

**Pembahasan hasil:**


## 8. Pembanding single-task — Step 29 dan evaluasi test Step 30

ResNet-50 pretrained digunakan untuk klasifikasi severity saja, tanpa decoder,
Spatial Attention, AMFM, MA-LDS, mask input, dan loss segmentasi/morfologi.
Global pooling, proyeksi 128, serta classifier 64/dropout 0,3 mempertahankan
lebar cabang global proposed. Ini bukan varian `baseline` multi-task pada ablation.

Pembanding memakai train/validation dari manifest augmented yang sama, ImageNet
normalization, augmentasi online, class weighting balanced, tanpa weighted sampler,
seed dan anggaran dua fase yang sama dengan Step 11. Bobot awal berasal dari
ImageNet, bukan checkpoint proposed. Step 29 hanya mengevaluasi validation;
**Step 30 mengevaluasi checkpoint terpilih pada test yang sama dengan proposed,
tanpa training ulang atau pemilihan ulang epoch berdasarkan test**. Dice/IoU tidak berlaku
karena model tidak menghasilkan mask.

Checkpoint dipilih dengan validation CE berbobot terendah. Proposed dipilih
dengan total multi-task validation loss terendah. Jadi perbandingan ini menilai
sistem keseluruhan, bukan mengisolasi kontribusi segmentasi saja; ablation tetap
diperlukan. Kesamaan seed tidak menjamin inisialisasi head atau hasil bit-identik.

| Model | Epoch terbaik | Val. accuracy | Val. Kappa | Val. Macro-F1 |
|---|---:|---:|---:|---:|
| ResNet-50 single-task | | | | |
| Proposed augmented | | | | |

Perbandingan akhir pada test dilaporkan terpisah dari validation. Jumlah sampel
diambil dari manifest dan audit yang sama untuk kedua model; hasil test tidak
dibandingkan dengan angka validation. Model single-task dikunci berdasarkan
validation CE minimum sebelum test dijalankan.

| Model | Jumlah test | Test accuracy | Test Kappa | Test Macro-F1 |
|---|---:|---:|---:|---:|
| ResNet-50 single-task | | | | |
| Proposed augmented | | | | |

**Pembahasan hasil:**


## 9. Jawaban rumusan masalah dan ketercapaian tujuan

Bagian ini diisi setelah hasil training utama dan ablation augmented diverifikasi.
Keberhasilan implementasi dibedakan dari bukti peningkatan performa.

| Rumusan masalah | Bukti dari eksperimen augmented | Jawaban / keterbatasan |
|---|---|---|
| 1. Pipeline hybrid simultan | | |
| 2. Kontribusi fitur lokal terhadap klasifikasi | | |
| 3. Evaluasi dan kontribusi komponen | | |

Klaim dibanding single-task harus didukung hasil pembanding Step 29 setelah
training selesai; tabel ablation saja tidak memenuhi pembanding tersebut. Bukti manfaat
augmentasi dibanding tanpa augmentasi juga memerlukan kontrol tersendiri,
dan bukan kesimpulan otomatis dari eksperimen yang seluruhnya augmented.

## 10. Kesimpulan

**Diisi setelah hasil eksperimen augmented tersedia dan diperiksa.**
