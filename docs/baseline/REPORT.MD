# Laporan Penelitian

11 NOVEMBER 2026

Tujuan, Prosedur Evaluasi, dan Hasil Eksperimen Model Hybrid Multi-Task

## 1. Rumusan masalah
1. Bagaimana merancang dan membangun kerangka kerja hibrida yang mengintegrasikan segmentasi lesi presisi berbasis Dilated UNet dengan mekanisme atensi spasial dan klasifikasi keparahan jerawat secara simultan dalam satu unified deep learning pipeline?
2. Bagaimana memanfaatkan representasi fitur dari tugas segmentasi (lokal) untuk meningkatkan performa tugas klasifikasi global (Deep Representation Learning) sehingga menghasilkan sistem yang lebih akurat dibandingkan pendekatan single-task?
3. Bagaimana mengevaluasi performa kerangka kerja yang diusulkan pada dataset ACNE04-v2 menggunakan metrik evaluasi standar dari domain computer vision dan pencitraan medis, serta membuktikan kontribusi setiap komponen arsitektur melalui ablation study?

## 2. Tujuan utama
Penelitian ini memiliki tiga tujuan utama dalam pengembangan model hybrid multi-task untuk analisis citra ACNE04-v2:

1. Mengembangkan Kerangka Kerja Hibrida: Mengembangkan kerangka kerja hibrida Deep Learning yang mampu melakukan segmentasi lesi jerawat secara presisi (mengenali tipe dan lokasi lesi) sekaligus mengklasifikasikan tingkat keparahan berdasarkan kriteria Hayashi secara otomatis.
2. Eksplorasi Representation Learning: Mengeksplorasi dan mengoptimalkan mekanisme Deep Representation Learning yang memanfaatkan fitur representasi dari cabang segmentasi untuk memperkuat cabang klasifikasi global dalam arsitektur multi-task learning.
3. Evaluasi Performa & Ablation Study: Mengevaluasi performa model yang dikembangkan pada dataset ACNE04-v2 menggunakan metrik Akurasi, Cohen's Kappa, Dice Coefficient, dan Intersection over Union (IoU) serta membuktikan kontribusi komponen AMFM dan MA-LDS melalui ablation study.


## 3. Dataset dan konsistensi sumber

Proposal menyebut ACNE04-v2 dengan 1.204 citra dan 32.443 anotasi sirkular.
Perbedaan yang dijelaskan dalam proposal adalah anotasi bounding box pada
ACNE04 original dan anotasi sirkular pada v2. Identitas versi dataset aktual
harus ditelusuri dari arsip dan anotasi sumber, bukan nama folder saja.

**Ada ketidaksesuaian:** run Oktober menggunakan 1.457 citra, dengan split
1.017 train, 216 validation, dan 224 test, bukan 1.204 citra pada proposal.

| Kelas | Jumlah menurut proposal | Jumlah dalam catatan eksperimen 1.457 citra |
|---|---:|---:|
| Level 0 / mild | 470 | 497 |
| Level 1 / moderate | 579 | 637 |
| Level 2 / severe | 133 | 186 |
| Level 3 / very severe | 22 | 137 |
| **Total** | **1.204** | **1.457** |

Distribusi eksperimen tersebut berasal dari catatan terdahulu dan perlu
dicocokkan kembali dengan manifest tiap run. Laporan final perlu menjelaskan
asal tambahan citra, versi anotasi, sumber label severity, dan kriteria inklusi.
Prediksi label severity dataset bukan bukti validasi klinis independen.


## 4. Proses penelitian secara global

1. Membaca citra, anotasi lesi, dan label severity.
2. Membentuk circle mask dan refined mask berbasis Lab/HSV.
3. Menyiapkan pasangan foto-mask dan manifest train/validation/test.
4. Melatih shared encoder ResNet-50, Spatial Attention, dan dilated decoder.
5. Mengekstrak morfologi prediksi mask: area, density, cx, cy, dan spread.
6. Menggabungkan fitur global dan morfologi melalui AMFM untuk klasifikasi.
7. Mengoptimalkan loss BCE + Dice, KL dengan MA-LDS, dan morphology MSE.
8. Memilih checkpoint dengan total validation loss terendah.
9. Melakukan CV, ablation, dan evaluasi test sebagai rangkaian terpisah.

Test tidak digunakan untuk tuning. Train, validation, validation folds CV,
dan test mengukur subset berbeda dan harus dilaporkan terpisah.


## 5. Preprocessing

Perlu dibedakan pengolahan foto dan pembentukan mask:

- Notebook terbaru mengatur foto ke 320 × 320 dengan letterbox dan mode `none`,
  tanpa equalization/sharpening warna. Catatan penghapusan perubahan warna
  agresif belum membuktikan bahwa perubahan tersebut menyebabkan kenaikan
  accuracy; diperlukan eksperimen terkontrol.
- **Lab relative redness dan HSV tetap digunakan untuk membuat refined mask**
  di dalam anotasi lingkaran. Jika hasil deteksi terlalu kecil, digunakan fallback
  circle mask. Jadi, redness tidak dihapus dari seluruh pipeline.

Preprocessing dapat dilewati jika manifest sudah ada. Opsi notebook terbaru
belum membuktikan pengolahan yang dipakai run lama; artefak aktual perlu diperiksa.

Mask bersifat biner: background dan area lesi, bukan kategori papul/pustul/komedo.
Dengan demikian, tujuan “mengenali tipe lesi” belum dievaluasi. Dice/IoU mengukur
kesesuaian terhadap target mask hasil preprocessing, bukan langsung terhadap
kontur manual tervalidasi dokter.



## 6. Eksperimen 24 Oktober — Step 11

Model proposed Hybrid ResNet-50 + AMFM + MA-LDS menggunakan class weighting
balanced, tanpa weighted sampler. Training terdiri atas 30 epoch freeze encoder
dan 90 epoch full fine-tuning. Augmentasi online train meliputi rotasi, flip,
dan brightness/contrast; validation/test tanpa augmentasi.

Sumber: [`metrics.json`](../24okt/test/metrics.json) dan
[laporan Step 11](../laporan_penelitian_24_oktober_step11.md).

| Evaluasi | Epoch | Accuracy |
|---|---:|---:|
| Train pada checkpoint terbaik | 116 | 71,48% |
| Validation pada checkpoint terbaik | 116 | 67,13% |
| Test checkpoint terbaik, 224 citra | — | **71,88%** |

| Metrik test | Hasil |
|---|---:|
| Cohen's Kappa | 0,5895 |
| Dice | 0,4786 |
| IoU | 0,3321 |

Ini bukan eksperimen pertama seluruh penelitian; hasil terdahulu juga tercatat.
Identitas preprocessing dan versi script persis saat run belum seluruhnya terkunci.


## 7. Eksperimen augmentasi offline 100%

**Hasil numerik run ini belum teridentifikasi dari sumber yang diperiksa.**
Augmentasi online pada run biasa bukan bukti augmentasi offline 100%.

| Informasi yang diperlukan | Status |
|---|---|
| Folder run dan `metrics.json` | Belum teridentifikasi |
| Manifest augmented dan jumlah train per kelas | Belum terverifikasi |
| Definisi 100%: penggandaan data atau target keseimbangan kelas | Perlu dijelaskan |
| Epoch terbaik dan metrik validation/test | Belum tersedia untuk diisi |

Menurut [panduan augmentasi offline](../OFFLINE_AUGMENTATION.md), hanya train
yang ditambah; validation/test tidak diubah. Panduan itu menjelaskan target
moderat 75% kelas terbesar, bukan bukti protokol 100% aktual. SMOTE adalah
eksperimen berbeda pada fitur klasifikasi, bukan penambahan foto-mask.

## 8. Model lengkap flowchart — Step 21

Run yang dirujuk sebagai 25 Oktober memakai `proposed`: Spatial Attention,
dilated decoder, AMFM, dan MA-LDS aktif. Training 30 + 90 epoch memakai class
weighting balanced serta weighted sampler. Enam varian ablation bukan komponen
yang dijalankan bersamaan; setiap varian adalah model terpisah.

Sumber: [`metrics.json`](../25okt/metrics.json). Checkpoint terbaik berdasarkan
total validation loss adalah **epoch 115**, bukan epoch terakhir.

| Evaluasi | Accuracy | Kappa | Dice | IoU |
|---|---:|---:|---:|---:|
| Validation checkpoint terbaik | 49,54% | 0,2978 | 0,5090 | 0,3571 |
| Test checkpoint terbaik | 56,25% | 0,3906 | 0,5025 | 0,3536 |

Train accuracy pada epoch 115 adalah 78,66%. Macro-F1 test sebesar 0,5745.
Recall test level 0–3 berturut-turut 84,21%, 24,74%, 89,66%, dan 54,55%.
Kelemahan utama klasifikasi terlihat pada level 1.

Dibanding 24 Oktober, accuracy test turun 15,63 poin persentase sementara Dice
naik sekitar 0,0239. Ini belum membuktikan efek tunggal weighted sampler karena
versi script dan identitas persis data antar-run belum dikunci.

## 9. Ablation study lengkap — validation

Sumber enam varian: [log ablation](../ABALATION_COMPLETE.MD).
Proposed berasal dari history `docs/25okt/metrics.json`. Setiap baris menggunakan
epoch dengan **total validation loss terendah**, bukan accuracy tertinggi.

| Varian | Epoch | Train accuracy | Val. accuracy | Val. Kappa | Val. Dice | Val. IoU | Val. loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 96 | 80,14% | 60,19% | 0,4458 | 0,5096 | 0,3590 | 1,4145 |
| no_attention | 111 | 77,29% | 53,24% | 0,3532 | 0,5154 | 0,3623 | 0,9897 |
| no_dilation | 115 | 78,66% | 52,78% | 0,3449 | **0,5204** | **0,3664** | 1,0684 |
| no_amfm | 107 | 86,73% | 60,19% | 0,4231 | 0,5153 | 0,3614 | 1,0812 |
| lds | 117 | 74,04% | 50,00% | 0,3173 | 0,5025 | 0,3513 | 1,3134 |
| no_fusion | 112 | 83,97% | **63,89%** | **0,4831** | 0,5077 | 0,3565 | 0,9779 |
| **proposed** | **115** | **78,66%** | **49,54%** | **0,2978** | **0,5090** | **0,3571** | **1,0946** |

### Definisi varian menurut trainer lokal

| Varian | Attention | Dilation | Fusi untuk klasifikasi | Smoothing |
|---|---|---|---|---|
| baseline | Tidak | Tidak | Global saja | Tidak |
| no_attention | Tidak | Ya | AMFM | MA-LDS |
| no_dilation | Ya | Tidak | AMFM | MA-LDS |
| no_amfm | Ya | Ya | Concatenation | MA-LDS |
| lds | Ya | Ya | AMFM | LDS tetap |
| no_fusion | Ya | Ya | Global saja | MA-LDS |
| proposed | Ya | Ya | AMFM | MA-LDS |

Semua varian masih memiliki decoder segmentasi dan shared encoder. Baseline
bukan single-task murni. `no_amfm` juga mengubah dimensi input classifier akibat
concatenation. Varian `lds` memakai smoothing tetap pada implementasi lokal,
bukan otomatis reproduksi penuh metode Prokhorov dan Kalinin.

### Interpretasi dan batas perbandingan

- Accuracy `no_fusion` lebih tinggi sekitar **14,35 poin persentase** daripada
  proposed; `no_amfm` lebih tinggi sekitar **10,65 poin persentase**.
- Jika kondisi aktual setara, satu run ini belum mendukung manfaat AMFM/fusion
  untuk meningkatkan klasifikasi. Hasil negatif tetap temuan penelitian yang sah.
- Proposed memiliki accuracy sedikit lebih rendah daripada `lds`, tetapi Dice/IoU
  sedikit lebih tinggi. Manfaat MA-LDS belum konsisten lintas metrik.
- `no_dilation` memiliki Dice/IoU tertinggi pada tabel; selisih kecil belum
  membuktikan signifikansi statistik atau keunggulan umum.
- Notebook membentuk manifest ablation dari train/validation manifest filtered
  yang dipakai Step 21 dan mengatur protokol training yang sebanding. Kesamaan
  data serta versi trainer saat run aktual masih perlu diverifikasi.
- Loss antar-varian tidak otomatis setara karena target klasifikasi berbeda.
  Loss dipakai memilih checkpoint dalam tiap run, bukan dasar tunggal ranking.
- Jangan membandingkan validation ini langsung dengan test atau rata-rata CV.
  Satu seed belum membuktikan kestabilan statistik.

Rincian: [hasil ablation validation](../HASIL_ABLATION_VALIDASI.md).

## 10. Cross-validation

Lima fold dibentuk hanya dari 1.017 citra train. Validation asli dan test asli
tidak digunakan sebagai validation folds. Angka ini berbeda sumber evaluasinya
dari validation final pada tabel ablation dan metrik test.

| Metrik | Mean ± SD |
|---|---:|
| Accuracy | 47,98% ± 2,67% |
| Cohen's Kappa | 0,2958 ± 0,0341 |
| Dice | 0,4924 ± 0,0155 |
| IoU | 0,3457 ± 0,0135 |

Sumber: [laporan flowchart](../25okt/laporan_penelitian_25_oktober_flowchart.md).

## 11. Jawaban rumusan masalah dan ketercapaian tujuan

| Rumusan masalah | Jawaban berdasarkan bukti saat ini | Kekurangan |
|---|---|---|
| 1. Membangun pipeline hybrid simultan | Terjawab secara implementasi: satu model menghasilkan mask dan severity. | Klaim presisi dibatasi kualitas mask; klasifikasi tipe lesi belum dievaluasi. |
| 2. Fitur lokal meningkatkan klasifikasi dibanding single-task | Fusi diimplementasikan dan diuji, tetapi peningkatan klasifikasi belum didukung hasil ablation. | Pembanding single-task murni yang terkontrol belum tersedia. |
| 3. Evaluasi dan kontribusi komponen | Metrik, CV, enam varian, dan proposed sudah dilaporkan. | Konsistensi dataset/protokol aktual dan signifikansi kontribusi belum dibuktikan. |

Tujuan pengembangan kerangka hybrid tercapai secara teknis, tetapi bagian
“mengenali tipe lesi” belum tercapai. Eksplorasi representation learning sudah
dilakukan; klaim fitur lokal memperkuat klasifikasi belum terpenuhi. Evaluasi
telah dilaksanakan, tetapi tidak membuktikan semua komponen meningkatkan performa.

**Rumusan masalah terjawab tidak berarti hipotesis keunggulan metode terbukti.**
Tidak ada persentase pencapaian resmi yang dapat dihitung hanya dari accuracy.

## 12. Kesimpulan dan tindak lanjut

Kerangka hybrid segmentasi dan klasifikasi berhasil dikembangkan serta dievaluasi.
Model final menghasilkan accuracy test 56,25%, Kappa 0,3906, Dice 0,5025, dan
IoU 0,3536. Pada validation yang dilaporkan, proposed belum mengungguli model
tanpa fusion dalam klasifikasi. Laporan belum dapat menyatakan AMFM atau fitur
morfologi terbukti meningkatkan klasifikasi dibanding single-task.

Prioritas berikutnya:

1. Jelaskan perbedaan 1.204 vs 1.457 citra dan identitas versi dataset.
2. Verifikasi split, foto-mask, versi trainer, dan konfigurasi antar-run; periksa
   duplikasi serta kebocoran subjek/sumber.
3. Lengkapi hasil augmentasi offline 100% dari artefak run yang sesuai.
4. Siapkan pembanding single-task murni bila klaim tersebut dipertahankan.
5. Gunakan validation/CV untuk eksperimen lanjutan, bukan tuning pada test.

Tidak diperlukan training ulang hanya untuk melengkapi tabel ablation ini.

