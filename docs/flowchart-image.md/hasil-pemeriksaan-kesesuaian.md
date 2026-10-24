# Pemeriksaan Kesesuaian Flowchart dengan Eksekusi Colab

**Tanggal pemeriksaan:** 24 Oktober 2026  
**Status:** Notebook kini memiliki jalur eksperimen flowchart opsional (belum dijalankan). Sel lama tetap tersedia.

## Kesimpulan

Flowchart menggambarkan rancangan model dan alur eksperimen secara umum, tetapi belum sepenuhnya sama dengan pipeline yang dipakai Colab. Beberapa komponen cocok, sementara preprocessing, penyaringan data, evaluasi, dan eksperimen validasi perlu diperjelas agar diagram membedakan rancangan penelitian dari pekerjaan yang sudah benar-benar dijalankan.

## Flowchart 1 — Alur global penelitian

![Flowchart 1 — Alur global penelitian](<Screenshot 2026-10-24 at 15.21.09.png>)

### Bagian yang sesuai

- Pembagian data stratified 70/15/15 sesuai dengan manifest training yang diperiksa.
- Rancangan memakai ResNet-50, cabang segmentasi, cabang klasifikasi, AMFM, MA-LDS, serta evaluasi Dice/IoU dan accuracy/kappa.

### Bagian yang perlu diperjelas

- Implementasi memberi fitur bottleneck ke Spatial Attention sebelum fitur diteruskan ke decoder segmentasi maupun cabang klasifikasi. Diagram arsitektur global belum menampilkan Spatial Attention secara eksplisit.
- Decoder kode memakai skip connection bergaya U-Net dan dilated convolution pada beberapa blok. Sebutan “Dilated U-Net” merangkum idenya, tetapi deskripsi “decoder bergaya U-Net dengan blok konvolusi dilated” lebih tepat untuk implementasi ini.
- Diagram menyebut preprocessing sebagai *image enhancement*. Pada konfigurasi Colab versi baru, input dibuat dengan `preprocessing-mode none` dan `resize-mode letterbox`; equalization dan sharpening tidak dijalankan.
- Notebook kini menambahkan cell opsional untuk baseline/ablation, 5-fold CV, dan final run. Training belum dijalankan; flag eksekusinya default `False`.

## Flowchart 2 — Preprocessing dan pembuatan mask

![Flowchart 2 — Preprocessing dan pembuatan mask](<Screenshot 2026-10-24 at 15.21.12.png>)

### Bagian yang sesuai

- Script membaca anotasi lingkaran, membentuk circle mask, mendeteksi piksel lesi dengan Lab/HSV, membatasi hasil ke area circle, lalu menjalankan morphological closing dan opening.
- Jika area hasil refinement kurang dari 2% area circle, script memakai circle mask sebagai fallback.
- Mask dibuat sebelum augmentasi training. Augmentasi pasangan gambar-mask dijalankan saat training saja.

### Bagian yang perlu diperjelas

- Diagram menampilkan *image enhancement* sebagai langkah wajib. Pada run letterbox v2, preprocessing warna agresif dinonaktifkan; gambar hanya di-resize dengan letterbox tanpa equalization atau sharpening.
- Script mendukung anotasi JSON maupun VOC XML/folder XML, sehingga “Parse anotasi JSON” lebih tepat diganti menjadi “Baca anotasi JSON/XML”.
- Notebook kini menambahkan audit manual melalui CSV dan contact sheets foto dengan overlay refined mask. Keputusan include/exclude tetap harus dibuat manusia; pipeline tidak otomatis mengklasifikasikan kualitas foto.
- Diagram belum menunjukkan bahwa augmentasi hanya diterapkan ke data train, bukan validation/test.

## Flowchart 3 — Proses training

![Flowchart 3 — Proses training model hybrid](<Screenshot 2026-10-24 at 15.20.28.png>)

### Bagian yang sesuai

- Fase 1 membekukan encoder selama 30 epoch dengan learning rate `1e-3`; fase 2 membuka seluruh jaringan selama 90 epoch dengan learning rate `1e-4`.
- Kode memakai SGD dengan momentum 0,9, BCE + Dice untuk segmentasi, KL berbasis MA-LDS untuk klasifikasi, dan MSE morphology loss dengan bobot 0,2.
- Model terbaik dipilih berdasarkan validation loss terendah dan metrik validation dicatat per epoch.
- Pada kode terbaru, optimizer dipertahankan per fase dan state optimizer disimpan di checkpoint untuk resume.

### Bagian yang perlu diperjelas

- Urutan diagram menempatkan “Simpan best model” sebelum evaluasi validation. Secara aktual, validation dijalankan dahulu; checkpoint best diperbarui jika validation loss lebih rendah.
- Notebook memakai class-weighted loss dengan pilihan `balanced` atau `none`. Weighted sampler tidak aktif pada run yang tercatat. Bila flowchart dimaksudkan mewakili run ini, cantumkan konfigurasi tersebut.
- Notebook kini menambahkan 5-fold CV atas split train saja dan ablation yang melewati test set. Keduanya masih menunggu dijalankan.

## Pemeriksaan penyaringan data

Script refinement menggunakan gambar yang memiliki anotasi; gambar tanpa anotasi dilewati. Notebook kini membuat CSV audit dan contact sheets dengan mask overlay untuk foto yang tercantum di manifest. Operator perlu meninjau gambar dan mask lalu mengisi `include=yes/no`; exclusion memerlukan alasan. Tidak ada filter kualitas foto otomatis.

Karena itu, flowchart saat ini belum cukup untuk memastikan bahwa foto yang tidak sesuai kriteria penelitian sudah dibuang. Aturan inklusi/eksklusi perlu ditulis eksplisit dan daftar file yang lolos filter perlu disimpan sebelum mask dan split dibuat.

## Ketidaksesuaian hasil evaluasi yang perlu dituntaskan

Catatan hasil di [`docs/24okt/test/hasil_test.md`](../24okt/test/hasil_test.md) memuat dua angka accuracy yang berbeda untuk test set:

- Metrik test yang dicetak training: **0,7188**.
- Dari 224 baris prediksi inferensi: **90 benar dari 224**, yaitu **0,4018**. Confusion matrix yang disimpan juga menunjukkan tidak ada prediksi level 3.

Hasil inferensi ini belum konsisten dengan metrik training. Sebelum flowchart menyatakan evaluasi final selesai, pastikan checkpoint, manifest, preprocessing, dan normalisasi input pada inferensi sama dengan yang dipakai saat training, lalu cocokkan kembali confusion matrix dan metriknya.

## Ringkasan status

| Bagian | Status terhadap Colab |
|---|---|
| Refined mask Lab/HSV dan fallback 2% | Sesuai |
| Split stratified 70/15/15 | Sesuai dengan satu run yang diperiksa |
| Training dua fase dan komponen loss | Sebagian besar sesuai |
| Image enhancement | Tidak sesuai untuk konfigurasi letterbox v2; mode legacy masih tersedia |
| Audit foto dan mask manual | Cell ditambahkan; belum dilakukan |
| Weighted sampler | Tidak aktif pada run yang tercatat |
| 5-fold cross-validation | Cell ditambahkan; belum dijalankan |
| Ablation enam varian dan baseline LDS | Cell ditambahkan; belum dijalankan |
| Evaluasi inferensi final | Perlu dicocokkan dengan metrik training |

## File yang dibandingkan

- Notebook: [`acne_colab.ipynb`](../../acne_colab.ipynb)
- Script refinement: [`scripts/refine_masks_color.py`](../../scripts/refine_masks_color.py)
- Script training: [`scripts/train_proposal_hybrid_resnet50_malds.py`](../../scripts/train_proposal_hybrid_resnet50_malds.py)
- Script inferensi: [`scripts/infer_testing_images.py`](../../scripts/infer_testing_images.py)
- Hasil test: [`hasil_test.md`](../24okt/test/hasil_test.md)
