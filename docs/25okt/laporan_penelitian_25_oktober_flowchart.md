# Laporan Penelitian :  25 Oktober 2026
## Jalur Flowchart Proposal (melewati Step 11 sampai 17)

## 1. Tujuan dan jalur yang dijalankan

Penelitian memakai foto ACNE04-v2 untuk dua tugas: segmentasi area jerawat menggunakan mask dan klasifikasi tingkat keparahan ke empat kelas (`levle0` sampai `levle3`). Pada 25 Oktober, jalur yang dibahas mengikuti tambahan flowchart proposal di notebook: setup dan preprocessing sampai Step 10, **melewati training legacy Step 11 sampai 17**, kemudian menjalankan Step 18 dan seterusnya.

Jadi, Step 11 sampai 17 bukan bagian dari run flowchart ini. Model final pada jalur flowchart dilatih tersendiri di Step 21 menggunakan manifest hasil Step 18. Cross-validation dan ablation adalah eksperimen pendukung yang terpisah dari training final.

## 2. Persiapan data sebelum masuk ke jalur flowchart (Step 1 sampai 10)

1. **Setup GPU:** Colab memeriksa CUDA dan menyiapkan GPU untuk pemrosesan model.
2. **Mount Drive:** notebook menghubungkan Google Drive untuk membaca dataset/script dan menyimpan output.
3. **Install dependencies:** PyTorch, torchvision, OpenCV, Pillow, NumPy, pandas, Matplotlib, scikit-learn, dan pustaka pendukung dipasang.
4. **Import pustaka:** fungsi untuk file, gambar, tabel, grafik, dan GPU dimuat ke sesi.
5. **Unduh dataset:** ACNE04-v2 diunduh dari folder Drive ke runtime Colab.
6. **Ekstrak dan deteksi input:** arsip diekstrak; folder gambar dan anotasi JSON/XML dicari otomatis.
7. **Siapkan scripts:** `acne_scripts.zip` diekstrak. Script preprocessing, trainer, dan inferensi dimasukkan ke Python path.
8. **Konfigurasi path:** lokasi gambar, anotasi, output dan parameter umum training ditentukan; path divalidasi dan distribusi kelas dihitung.
9. **Preview kelas:** contoh foto empat kelas ditampilkan untuk pemeriksaan visual dasar.
10. **Preprocessing mask:** anotasi lingkaran/bounding box dipakai membentuk circle mask. Mask disempurnakan dengan deteksi kemerahan relatif pada Lab dan warna merah/putih/kuning pada HSV, dibatasi pada area lingkaran, lalu dirapikan dengan operasi morfologi. Jika hasil deteksi terlalu kecil, circle mask dipakai sebagai fallback. Foto diubah menjadi kanvas 320 × 320 dengan letterbox sesuai konfigurasi, tanpa equalization warna atau sharpening pada mode `none`. Script membuat manifest yang menghubungkan foto, mask, label, split, serta status fallback.

Pada jalur flowchart, manifest tersebut menjadi input Step 18. Karena Step 10 tidak membuat ulang manifest jika file sudah tersedia, isi manifest aktual penting untuk memastikan preprocessing benar-benar sesuai opsi notebook.

## 3. Jalur flowchart proposal

### Step 18 :  Audit foto dan membuat manifest filtered

Notebook memeriksa dukungan trainer untuk varian eksperimen dan resume checkpoint. Jika script di runtime lama, notebook mengambil versi terbaru dari GitHub. Lalu notebook membaca `refined_manifest.csv`, membuat tabel review foto yang mencatat filename, label, split, lokasi gambar, jumlah anotasi lingkaran, dan fallback mask. Contact sheet dibuat agar foto serta overlay mask dapat diperiksa secara visual.

Kolom `include` menentukan apakah foto dipertahankan. Foto yang dikecualikan harus diberi alasan; kode memeriksa duplikasi nama, kesesuaian daftar dengan manifest, dan keberadaan semua kelas di tiap split. Foto yang lolos disimpan sebagai `refined_manifest_filtered.csv`. Catatan run menyebut semua foto mendapat `include=yes`, sehingga jumlah split tetap 1.017 train, 216 validation, dan 224 test (total 1.457).

### Step 19 :  5-fold cross-validation

Lima fold dibuat **hanya dari 1.017 foto train**. Foto dikelompokkan menurut kelas, diacak dengan seed 42, lalu dibagi bergiliran ke lima kelompok validation. Dalam setiap putaran, satu kelompok dipakai validation dan empat kelompok lainnya train. Validation asli dan test asli tidak dimasukkan ke fold; test evaluation dilewati pada semua fold.

Setiap fold menjalankan model `proposed` selama 120 epoch (30 freeze + 90 fine-tuning), ResNet-50 pretrained, class weighting balanced, dan weighted sampler. Hasil yang dibandingkan adalah validation loss, accuracy, kappa, Dice, dan IoU. Lima fold berarti lima training terpisah, bukan 600 epoch pada satu model.

| Metrik validation 5-fold | Rata-rata ± simpangan baku |
|---|---:|
| Accuracy | **47,98% ± 2,67%** |
| Cohen's kappa | **0,2958 ± 0,0341** |
| Dice segmentasi | **0,4924 ± 0,0155** |
| IoU segmentasi | **0,3457 ± 0,0135** |

![Hasil 5-fold cross-validation](download%20(8).png)

### Step 20 :  Ablation study

Ablation membandingkan komponen model dengan melatih enam varian pada manifest train/validation yang sama. Baris test dibuang dari manifest ablation, sehingga test set tidak dipakai memilih komponen. Setiap varian direncanakan untuk 120 epoch, class weighting balanced, weighted sampler aktif, dan seed 42. Kode menyimpan checkpoint per epoch dan ringkasan validation setelah tiap varian agar proses bisa dilanjutkan.

| Urutan | Varian | Perubahan yang diuji |
|---:|---|---|
| 1 | `baseline` | Menonaktifkan attention dan dilation, memakai global feature tanpa fusion AMFM dan tanpa MA-LDS smoothing; decoder segmentasi tetap dipakai. |
| 2 | `no_attention` | Menonaktifkan Spatial Attention. |
| 3 | `no_dilation` | Menghilangkan dilation pada blok decoder yang menggunakannya. |
| 4 | `no_amfm` | Mengganti fusion adaptif dengan concatenation fitur global dan morfologi. |
| 5 | `lds` | Memakai label distribution smoothing tetap, bukan smoothing yang menyesuaikan morfologi. |
| 6 | `no_fusion` | Menggunakan fitur global saja untuk klasifikasi. |

> Catatan status: berkas metrik lokal kini tersedia untuk `baseline`, `no_attention`, dan `no_dilation`. Angka berikut diambil pada epoch dengan validation loss terbaik untuk masing-masing varian. Metrik `no_amfm`, `lds`, dan `no_fusion` belum ditemukan, jadi hasil enam varian secara lengkap belum tersedia.

| Varian | Epoch terbaik | Train accuracy | Validation accuracy | Test |
|---|---:|---:|---:|---|
| `baseline` | 96 | 80,14% | 60,19% | Tidak dipakai |
| `no_attention` | 111 | 77,29% | 53,24% | Tidak dipakai |
| `no_dilation` | 115 | 78,66% | 52,78% | Tidak dipakai |

Train accuracy menunjukkan ketepatan pada foto latihan, sedangkan validation accuracy pada foto yang disisihkan untuk pemeriksaan. Angka train accuracy epoch 120 untuk `no_attention` adalah 76,60%; tabel menggunakan epoch terbaik (111), agar sejalan dengan pemilihan checkpoint berdasarkan validation loss.

### Step 21 :  Training final dan evaluasi test satu kali

Setelah audit, CV, dan pemilihan eksperimen, notebook menjalankan satu training final pada seluruh manifest filtered dengan split train/validation asli. Test digunakan untuk evaluasi final setelah training, bukan untuk CV atau ablation.

Konfigurasi yang tercatat di `metrics.json`:

- Varian `proposed` dengan Spatial Attention aktif; ResNet-50 pretrained dan normalisasi ImageNet.
- Train 1.017, validation 216, test 224; input 320 × 320, batch 16, seed 42, GPU CUDA.
- 30 epoch encoder dibekukan lalu 90 epoch fine-tuning seluruh model.
- Class weighting balanced **dan** weighted sampler aktif.
- Loss segmentasi BCE + Dice; klasifikasi MA-LDS berbasis KL; loss morfologi MSE.
- Augmentasi train: rotasi sampai 15°, flip horizontal 0,5, brightness/contrast sampai 0,15. Validation/test tanpa augmentasi.
- Checkpoint terbaik dipilih berdasarkan validation loss. Berkas lokal menunjukkan riwayat sampai epoch 120 dan hasil evaluasi test.

Hasil test dari checkpoint terbaik:

| Metrik test | Hasil |
|---|---:|
| Accuracy | **56,25%** (126/224) |
| Cohen's kappa | **0,3906** |
| Dice segmentasi | **0,5025** |
| IoU segmentasi | **0,3536** |

Pada epoch 120, train accuracy 80,33% dan validation accuracy 54,17%. Angka tersebut adalah metrik epoch terakhir; hasil test berasal dari checkpoint terpilih menurut validation loss, bukan berarti model mencapai 80,33% pada test.

![Kurva training final 120 epoch](training_curves_final.png)

### Step 22 :  Visualisasi dan analisis hasil

Cell ini tidak melatih ulang model. Cell membuat kurva train/validation, ringkasan validation CV, confusion matrix test, dan metrik per kelas dari checkpoint final serta prediksi yang tersimpan. Confusion matrix menunjukkan kesalahan antar tingkat severity. Pada hasil final, recall `levle1` 24,7% (24 dari 97 foto); kelas ini sering salah diprediksi. Dice/IoU test mengukur segmentasi, sedangkan confusion matrix dan accuracy mengukur klasifikasi.

![Confusion matrix test final](confusion_matrix_test.png)

## 4. Interpretasi singkat

Jalur flowchart menambahkan audit visual foto, 5-fold CV, dan ablation sebelum final run. CV memberi estimasi validation pada lima pembagian train; ablation dirancang menguji kontribusi komponen; final run menghasilkan satu evaluasi test. Hasil test final adalah accuracy 56,25%, kappa 0,3906, Dice 0,5025, dan IoU 0,3536. Jarak train accuracy epoch terakhir (80,33%) terhadap validation (54,17%) menunjukkan kemungkinan overfitting. Hasil ablation lengkap belum tersedia, sehingga kontribusi tiap komponen belum dapat disimpulkan.

## 5. Catatan tanggal dan sumber

Notebook menyimpan direktori eksperimen dengan nama `flowchart_experiments_2026-10-24` dan output final `2026-10-24_flowchart_proposed_filtered`. Laporan ini merujuk alur/hasil yang dibahas sebagai **25 Oktober**. File metrik, checkpoint, kurva, dan confusion matrix yang tersedia lokal diperbarui pada 6 November; ini adalah salinan/hasil terbaru dari run final yang sama menurut konfigurasi, bukan run dengan kode atau konfigurasi berbeda.

- Notebook dan kode Step 18 sampai 22: [acne_colab.ipynb](../../acne_colab.ipynb).
- Ringkasan hasil final dan evaluasi test: [hasil-final-training-flowchart-25-okt-2026.md](hasil-final-training-flowchart-25-okt-2026.md).
- Perbandingan hasil 24 dan 25 Oktober: [perbandingan-training-24-dan-25-oktober-2026.md](perbandingan-training-24-dan-25-oktober-2026.md).
- Konfigurasi serta riwayat final: [metrics.json](metrics.json).

---

**Draf untuk diperiksa:** silakan koreksi detail yang tidak sesuai dengan proses aktual 25 Oktober.
