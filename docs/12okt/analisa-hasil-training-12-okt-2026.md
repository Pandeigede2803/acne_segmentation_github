# Analisis hasil training dan pengujian model — 12 Oktober 2026

## 1. Ringkasan temuan

Model hybrid ResNet50 + Spatial Attention + Decoder + AMFM + MA-LDS menunjukkan proses belajar yang berjalan: loss menurun, sedangkan Dice, accuracy, dan kappa meningkat. Namun, hasil akhirnya masih memiliki dua keterbatasan utama: ketepatan segmentasi terhadap mask referensi masih rendah dan klasifikasi sering tertukar antara tingkat keparahan yang berdekatan.

Eksekusi yang dicatat pada 12 Oktober **tidak menjalankan epoch training baru**. Program memuat checkpoint yang sudah mencapai epoch 120, mencoba melanjutkan dari epoch 121, lalu langsung melakukan evaluasi karena batas training tetap 120 epoch.

Perbandingan run menunjukkan bahwa konfigurasi balanced memiliki accuracy dan kappa lebih tinggi, tetapi Dice dan IoU sedikit lebih rendah. Pemeriksaan kode juga menemukan optimizer SGD dibuat ulang setiap epoch, sehingga state momentum tidak dipertahankan antar-epoch.

Laporan ini membedakan hasil yang langsung terlihat dari artefak, perilaku yang terverifikasi di kode lokal, dan dugaan penyebab yang masih perlu eksperimen. Kode lokal belum tentu identik dengan seluruh versi script yang dipakai pada training historis, karena log menyatakan script diperbarui dari GitHub.

## 2. Sumber dan batas analisis

Sumber hasil eksperimen:

- [Log training dan rekap metrik](hasil-training-12-okt-2026.md).
- [Log prediksi 224 gambar test](test-prediksi-model-12okt.md).
- [Grafik training](training-12-okt.png).
- [Confusion matrix](confusion-matrix.png).
- [Contoh overlay segmentasi](foto-contoh-test.png).

Kode yang diperiksa:

- [Training proposal hybrid](../../scripts/train_proposal_hybrid_resnet50_malds.py): loop training, resume, loss, ekstraksi morfologi, dan pemilihan checkpoint.
- [Inferensi test](../../scripts/infer_testing_images.py): threshold dan arti warna overlay.

Checkpoint, manifest lengkap, `metrics.json` asli, dan mask individual di Google Drive tidak tersedia sebagai bahan pemeriksaan langsung dalam folder ini. Karena itu, epoch checkpoint terbaik, distribusi Dice per gambar, kebocoran identitas antar-split, dan kualitas semua anotasi belum dapat dipastikan.

Metrik per kelas dalam laporan dihitung ulang dari 224 baris log prediksi. Hasil confusion matrix cocok dengan gambar yang disimpan. Nilai dari grafik dibaca secara visual dan ditulis sebagai perkiraan jika angka persisnya tidak tersedia di log.

## 3. Konfigurasi dan distribusi data

| Komponen | Nilai yang tercatat |
|---|---|
| Model | ResNet50 + Spatial Attention + Decoder + AMFM + MA-LDS |
| Manifest | `refined_manifest.csv` |
| Device | CUDA |
| Train / validation / test | 1.017 / 216 / 224 gambar |
| Phase 1 | Encoder freeze, 30 epoch, learning rate 0,001 |
| Phase 2 | Full fine-tuning, 90 epoch, learning rate 0,0001 |
| Total epoch | 120 |
| Batch size | 16 |
| Ukuran input | 320 × 320 |
| Class weighting | Balanced |
| Weighted sampler | Tidak digunakan |
| Direktori run | `2026-10-04_proposal_class_weighting_balanced` |

Distribusi kelas training:

| Kelas | Jumlah | Proporsi train | Bobot klasifikasi |
|---|---:|---:|---:|
| Level 0 | 347 | 34,1% | 0,7327 |
| Level 1 | 445 | 43,8% | 0,5713 |
| Level 2 | 130 | 12,8% | 1,9558 |
| Level 3 | 95 | 9,3% | 2,6763 |

Level 1 sekitar 4,7 kali lebih banyak daripada level 3. Bobot balanced menggunakan rumus `N_train / (4 × jumlah_kelas)`, sehingga kesalahan pada kelas minoritas memperoleh kontribusi loss lebih besar. Bobot ini diterapkan pada loss klasifikasi, bukan langsung pada loss segmentasi. Tanpa weighted sampler, kelas minoritas tetap lebih jarang muncul dalam batch sesuai distribusi data.

Level 0 adalah kategori tingkat keparahan terendah pada label dataset; contoh level 0 tetap memiliki mask lesi. Karena itu, level 0 tidak boleh otomatis ditafsirkan sebagai gambar tanpa jerawat.

## 4. Apa yang terjadi pada eksekusi 12 Oktober?

Urutan yang teramati pada log:

1. Script training diperbarui dari GitHub.
2. Checkpoint ditemukan di direktori run lama.
3. Model dan riwayat dimuat; program mencetak `Resume dari epoch 121`.
4. Program langsung mencetak `Evaluasi akhir test set` tanpa log epoch baru.
5. Evaluasi menghasilkan accuracy 0,6071, kappa 0,4346, Dice 0,4430, dan IoU 0,3016.

Kode lokal menggunakan pola berikut:

```python
total_epochs = args.phase1_epochs + args.phase2_epochs
start_epoch = int(checkpoint["epoch"]) + 1
for epoch in range(start_epoch, total_epochs + 1):
    ...
```

Untuk checkpoint epoch 120 dan total 120, rentangnya menjadi `range(121, 121)`, sehingga tidak memiliki iterasi. Setelah itu, program memuat `best_model.pt` untuk pengujian.

**Implikasi:** hasil ini adalah evaluasi ulang model run yang sudah selesai, bukan bukti peningkatan akibat training baru pada 12 Oktober. Grafik 120 epoch berasal dari riwayat sebelumnya. Download bobot pretrained juga tidak membuktikan training dimulai ulang; bobot model kemudian ditimpa oleh checkpoint.

Untuk melanjutkan training diperlukan batas total epoch yang lebih tinggi dan verifikasi konfigurasi checkpoint. Untuk eksperimen baru, gunakan direktori output baru agar hasil tidak bercampur dengan checkpoint lama.

## 5. Interpretasi grafik training

![Kurva training selama 120 epoch](training-12-okt.png)

### 5.1 Awal training: keluaran belum informatif

Pada sekitar 10–15 epoch pertama, Dice mendekati nol, accuracy berada sekitar 0,12–0,13, dan kappa mendekati nol. Ini menunjukkan keluaran awal belum membedakan kelas dengan baik dan mask biner pada threshold evaluasi belum memiliki overlap berarti.

Dice dihitung setelah probabilitas mask dibinerkan pada threshold 0,5. Oleh sebab itu, Dice mendekati nol tidak membuktikan semua fitur segmentasi berhenti belajar: probabilitas bisa berubah sebelum melewati threshold. Prediksi mask kosong dan konsentrasi prediksi pada satu kelas merupakan kemungkinan, tetapi perlu prediksi per-epoch untuk memastikannya.

### 5.2 Sekitar epoch 15–30: peningkatan cepat dengan fluktuasi

Accuracy dan Dice meningkat tajam. Validation Dice juga mengalami penurunan sesaat yang cukup besar. Peningkatan ini terjadi saat phase 1, karena encoder baru dijadwalkan unfreeze pada epoch 31. Dengan demikian, lonjakan awal tidak tepat jika langsung disebut akibat unfreeze encoder.

Fluktuasi menunjukkan keluaran validation belum stabil. Penyebabnya bisa mencakup perubahan probabilitas di sekitar threshold, komposisi data, atau dinamika optimisasi. Grafik saja tidak dapat menentukan penyebab spesifik.

### 5.3 Setelah epoch 30: kemajuan bertahap

Loss train dan validation terus turun. Dice train naik perlahan hingga sekitar 0,50; validation Dice bergerak naik dengan fluktuasi hingga 0,4668 pada epoch 120. Accuracy train mendekati 0,63, sedangkan validation accuracy terakhir 0,5694.

Model masih memperoleh perbaikan sampai akhir. Grafik tidak memperlihatkan validation loss memburuk terus-menerus ketika train loss menurun, sehingga tidak ada bukti kuat overfitting berat.

Tetap terdapat generalization gap: train lebih baik daripada validation. Akan tetapi, kualitas segmentasi pada train sendiri masih terbatas. Ini mengarahkan pemeriksaan pada target mask, optimisasi, resolusi, dan kemampuan representasi, selain regularisasi.

Menyebut keadaan ini sebagai underfitting yang sudah terbukti juga terlalu kuat: kualitas target dan definisi metrik dapat membatasi skor. Kesimpulan yang didukung adalah bahwa performa masih terbatas pada train maupun validation.

## 6. Hasil akhir dan perbandingan run

| Metrik test | Run sebelumnya | Run balanced | Perubahan |
|---|---:|---:|---:|
| Accuracy | 0,5000 | 0,6071 | +0,1071 atau +10,71 poin persentase |
| Cohen’s kappa | 0,0652 | 0,4346 | +0,3694 |
| Dice | 0,4515 | 0,4430 | −0,0085 |
| IoU | 0,3082 | 0,3016 | −0,0066 |

Accuracy balanced setara dengan **136 prediksi benar dari 224 gambar**, sehingga terdapat 88 kesalahan klasifikasi. Kappa yang meningkat menunjukkan kesesuaian klasifikasi setelah memperhitungkan kesesuaian yang diharapkan dari distribusi kelas menjadi lebih tinggi. Nilai kappa run lama yang mendekati nol mengindikasikan accuracy 50% saat itu belum disertai kesesuaian kelas yang kuat; pola prediksi run lama perlu diperiksa sebelum menyatakan terjadi collapse ke satu kelas.

Dice dan IoU mengukur overlap mask, bukan accuracy klasifikasi. Dice 0,4430 tidak berarti 44,3% lesi terdeteksi, dan IoU 0,3016 tidak berarti accuracy piksel 30,16%. Kode menghitung skor per gambar lalu merata-ratakannya, sehingga keduanya bukan satu skor global yang dihitung dari gabungan semua piksel.

Perbedaan segmentasi relatif kecil dan belum disertai pengulangan seed atau interval ketidakpastian. Karena itu, belum dapat disimpulkan penurunan tersebut konsisten secara statistik.

Rekap dua run juga belum membuktikan hubungan sebab-akibat class weighting. Untuk membuktikannya, konfigurasi lain, split, seed, preprocessing, versi kode, dan aturan pemilihan checkpoint harus disamakan.

Loss akhir yang tercatat:

| Komponen | Nilai |
|---|---:|
| Total loss | 1,1247 |
| Segmentation loss | 0,6538 |
| Classification loss | 0,4701 |
| Best validation loss | 1,1025 |

Kode lokal menambahkan komponen morphology loss berbobot selain dua loss utama. Nilai morphology loss tidak ditampilkan dalam rekap ini. Selisih total terhadap jumlah dua komponen sekitar 0,0008, tetapi nilai pastinya tidak dapat direkonstruksi dari angka yang sudah dibulatkan dan tanpa konfigurasi asli lengkap.

## 7. Analisis klasifikasi per kelas

![Confusion matrix test](confusion-matrix.png)

Baris menunjukkan kelas asli; kolom menunjukkan kelas prediksi.

| Kelas asli → prediksi | Level 0 | Level 1 | Level 2 | Level 3 | Total |
|---|---:|---:|---:|---:|---:|
| Level 0 | 53 | 20 | 3 | 0 | 76 |
| Level 1 | 31 | 49 | 16 | 1 | 97 |
| Level 2 | 0 | 4 | 23 | 2 | 29 |
| Level 3 | 0 | 0 | 11 | 11 | 22 |
| Total prediksi | 84 | 73 | 53 | 14 | 224 |

Metrik dihitung dari matrix di atas:

| Kelas | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Level 0 | 63,10% | 69,74% | 66,25% |
| Level 1 | 67,12% | 50,52% | 57,65% |
| Level 2 | 43,40% | 79,31% | 56,10% |
| Level 3 | 78,57% | 50,00% | 61,11% |

- **Macro F1:** 0,6028.
- **Balanced accuracy / macro recall:** 0,6239.
- **Accuracy:** 0,6071.

Precision menjawab seberapa banyak prediksi suatu kelas yang benar. Recall menjawab seberapa banyak anggota kelas asli yang berhasil ditemukan. Perbedaan keduanya penting untuk memahami level 2 dan 3.

### 7.1 Level 0 dan level 1 masih saling tertukar

Sebanyak 20 gambar level 0 diprediksi level 1, dan 31 gambar level 1 diprediksi level 0. Kedua arah ini menyumbang **51 dari 88 kesalahan**, sekitar 58,0% seluruh kesalahan.

Ini merupakan masalah terbesar secara jumlah. Model belum memiliki batas keputusan yang konsisten untuk kedua kelas. Kemiripan tampilan, variasi pencahayaan, informasi lesi yang tidak tertangkap mask, dan konsistensi label merupakan dugaan yang perlu diperiksa pada contoh salah.

### 7.2 Recall level 2 tinggi, tetapi precision rendah

Model menemukan 23 dari 29 gambar level 2, tetapi menghasilkan 53 prediksi level 2. Sebanyak 30 prediksi tersebut berasal dari kelas lain: 3 level 0, 16 level 1, dan 11 level 3.

Dengan demikian, level 2 bukan kelas yang sudah terselesaikan. Model banyak mengarahkan gambar dari kelas tetangga ke level 2, sehingga recall tinggi disertai precision hanya 43,40%.

### 7.3 Level 3 sering diprediksi terlalu ringan

Sebanyak 11 dari 22 gambar level 3 menjadi level 2. Precision level 3 cukup tinggi pada sampel ini, tetapi recall 50% berarti separuh level 3 tidak dikenali.

Jumlah test level 3 hanya 22 gambar. Satu perubahan prediksi mengubah recall sekitar 4,55 poin persentase; hasilnya perlu dibaca dengan memperhatikan kecilnya dukungan kelas.

### 7.4 Mayoritas kesalahan hanya berjarak satu level

Sebanyak **84 dari 88 kesalahan (95,45%)** berjarak satu level. Empat sisanya berjarak dua level. Ini menunjukkan pola urutan keparahan mulai tertangkap, tetapi batas antarlevel belum akurat.

Terdapat 46 prediksi lebih ringan dan 42 lebih berat daripada label asli. Jadi, kecenderungan meremehkan keparahan sangat jelas pada level 3, tetapi tidak tepat jika disebut sebagai bias besar yang seragam pada seluruh kelas.

### 7.5 Confidence keluaran belum cukup untuk menilai kalibrasi

Rata-rata probabilitas kelas terpilih yang tercetak pada log sekitar **0,5052**. Banyak keluaran berada di sekitar 0,4–0,6. Ini menunjukkan prediksi tidak selalu dominan pada satu kelas, tetapi belum membuktikan model terkalibrasi buruk atau baik.

MA-LDS memang menggunakan distribusi target yang memberikan massa probabilitas ke kelas tetangga. Confidence yang tidak mendekati 1 dapat berkaitan dengan smoothing tersebut. Pemeriksaan kalibrasi memerlukan probabilitas lengkap semua kelas, bukan hanya probabilitas kelas pemenang.

## 8. Analisis segmentasi dari overlay

![Contoh overlay test](foto-contoh-test.png)

Arti warna terverifikasi dari fungsi `make_overlay`:

| Warna | Makna terhadap mask referensi |
|---|---|
| Hijau | Prediksi foreground di luar foreground referensi |
| Merah | Foreground referensi yang tidak diprediksi |
| Kuning | Foreground prediksi dan referensi bertumpuk |

Gambar memperlihatkan daerah kuning, tetapi juga titik hijau dan merah. Model sudah menghasilkan lokasi foreground, namun masih memiliki ketidaksesuaian lokasi, luas, atau bentuk terhadap referensi. Ada kasus target terdeteksi sebagian dan prediksi tambahan di lokasi lain.

Contoh ini hanya menampilkan gambar level 0, sehingga belum dapat digunakan untuk menilai kualitas segmentasi seluruh tingkat keparahan. Perlu contoh level 1–3 serta gambar dengan Dice rendah, sedang, dan tinggi.

Secara evaluasi, hijau dihitung sebagai false positive dan merah sebagai false negative. Secara visual, status lesi sebenarnya tetap bergantung pada kelengkapan dan akurasi anotasi. Jika mask refinement menghilangkan lesi valid, prediksi yang tampak masuk akal dapat tetap dihitung salah.

Mask di contoh tampak berupa area kecil dan beberapa area mendekati bentuk lingkaran. Hal ini belum membuktikan kualitas anotasi salah. Audit perlu membandingkan foto asli, anotasi sebelum refinement, mask setelah refinement, dan probabilitas prediksi.

## 9. Temuan implementasi yang perlu diperiksa atau diperbaiki

### 9.1 Optimizer dibuat ulang setiap epoch

Dalam loop training, kode lokal membuat objek `torch.optim.SGD(..., momentum=0.9)` baru setiap epoch. Objek baru tidak memiliki momentum dari epoch sebelumnya.

Momentum tetap bekerja antar-batch di dalam satu epoch, tetapi hilang pada pergantian epoch. Ini merupakan kelemahan implementasi yang terverifikasi pada kode lokal; besarnya pengaruh terhadap skor historis belum diketahui.

Perbaikan yang disarankan:

1. Buat optimizer sekali untuk fase yang sama.
2. Pertahankan state optimizer antar-epoch.
3. Ubah learning rate lewat parameter group atau scheduler.
4. Jika optimizer perlu dibuat ulang saat unfreeze, lakukan di transisi fase yang jelas, bukan setiap epoch.

### 9.2 Checkpoint belum menyimpan state optimizer

Checkpoint menyimpan model, epoch, best validation loss, history, dan args, tetapi tidak menyimpan `optimizer_state_dict`. Resume melanjutkan bobot dan nomor epoch, tetapi tidak memulihkan momentum.

Untuk kelanjutan yang lebih konsisten, simpan dan muat state optimizer serta scheduler jika digunakan. State RNG juga diperlukan jika ingin reproduksi kelanjutan yang lebih ketat. Temuan ini tidak berarti evaluasi ulang 12 Oktober berubah karena momentum: pada eksekusi tersebut tidak ada update parameter.

### 9.3 Freeze parameter encoder belum membekukan statistik BatchNorm

`set_encoder_trainable(False)` hanya mengubah `requires_grad`. Ketika training berjalan, `model.train(True)` tetap mengaktifkan mode train pada BatchNorm encoder.

Akibatnya, walaupun parameter encoder tidak memperoleh gradient, running mean dan variance BatchNorm dapat berubah. Ini bisa menjadi pilihan fine-tuning yang disengaja, tetapi istilah freeze encoder perlu dijelaskan: parameter dibekukan, statistik BatchNorm belum tentu dibekukan.

Jika tujuan phase 1 adalah encoder sepenuhnya tetap, atur encoder ke mode eval setelah pemanggilan mode train pada model. Bandingkan kedua kebijakan menggunakan validation; belum ada bukti bahwa perubahan statistik BatchNorm adalah penyebab utama hasil saat ini.

### 9.4 Best checkpoint dipilih berdasarkan total validation loss

`best_model.pt` diperbarui ketika total validation loss menurun. Total ini menggabungkan loss segmentasi, klasifikasi, dan morfologi. Checkpoint terbaik menurut kriteria ini belum tentu memiliki Dice atau kappa tertinggi.

Validation Dice 0,4668 dan accuracy 0,5694 pada judul grafik adalah nilai epoch 120. Test mengevaluasi checkpoint dengan validation loss terbaik. Epoch terbaik tidak tercantum dalam rekap, sehingga kedua nilai tidak boleh dianggap berasal dari checkpoint identik tanpa pemeriksaan lebih lanjut.

Simpan checkpoint terpisah untuk best total loss, best Dice, dan best kappa, lalu tentukan aturan pemilihan utama dari validation sebelum pengujian final.

### 9.5 Fitur bernama density bukan jumlah atau kepadatan objek lesi

Kode menghitung `density` sebagai proporsi piksel dengan probabilitas di atas 0,5. Pada mask target biner, `area = mean(mask)` dan `density = mean(mask > 0.5)` memiliki nilai identik.

Artinya, dua fitur target tersebut redundan. Density saat ini tidak menghitung jumlah lesi, connected components, atau kepadatan objek per area wajah. Ini membatasi interpretasi morfologi sebagai informasi jumlah lesi.

Pada prediksi, density menggunakan hard threshold dan tidak memberikan gradient melalui operasi perbandingan. Komponen lain seperti soft area, centroid, dan spread tetap memiliki jalur gradient. Karena itu, tidak tepat menyatakan seluruh cabang morfologi tidak dapat belajar.

Perbaikan harus dimulai dari definisi fitur yang ingin direpresentasikan. Jika diperlukan jumlah objek, evaluasi fitur connected components untuk analisis atau estimator yang sesuai untuk training; jangan menganggap mengganti nama density saja memperbaiki kemampuan model.

### 9.6 Normalisasi input perlu diselaraskan dengan penggunaan pretrained encoder

Kode lokal mengonversi RGB ke rentang 0–1 tanpa transformasi mean/std tambahan. Log menunjukkan bobot pretrained ResNet50 diunduh. Ini merupakan hal yang perlu diaudit bersama konfigurasi preprocessing aktual dan versi kode run.

Belum dapat dipastikan bahwa normalisasi menjadi penyebab skor rendah. Eksperimen normalisasi harus dilakukan konsisten pada train, validation, dan inferensi, serta pada run baru agar checkpoint lama tidak dievaluasi dengan transformasi input yang berbeda.

## 10. Dugaan penyebab dan cara membuktikannya

| Dugaan | Bukti saat ini | Pemeriksaan yang diperlukan |
|---|---|---|
| Ketidakseimbangan kelas memengaruhi klasifikasi | Level 2–3 lebih sedikit; run balanced lebih baik pada accuracy/kappa | Ablation dengan split, seed, dan kode identik |
| Target segmentasi tidak lengkap atau kurang tepat | Menggunakan refined mask; overlay memiliki ketidaksesuaian | Audit anotasi asli dan refined mask pada sampel terpilih |
| Lesi kecil sulit terwakili pada input 320 | Foreground kecil tampak pada contoh | Bandingkan resolusi dan ukur ukuran lesi setelah resize |
| Optimisasi belum optimal | SGD dibuat ulang setiap epoch pada kode lokal | Bandingkan optimizer persisten dengan konfigurasi identik |
| Fitur morfologi kurang informatif | Area dan density target identik | Ablation global-only, morphology-only, dan fusion |
| Smoothing memengaruhi pemisahan kelas tetangga | MA-LDS memberi probabilitas target pada tetangga | Bandingkan smoothing dengan kontrol tetap |
| Ada kebocoran identitas antar-split | Belum ada bukti langsung | Audit subject ID, duplikasi, dan kemiripan gambar antar-split |

Daftar ini merupakan hipotesis kerja, bukan diagnosis yang semuanya sudah terbukti. Kebocoran data, khususnya, tidak boleh dinyatakan terjadi tanpa memeriksa manifest dan identitas subjek.

## 11. Rencana tindak lanjut berprioritas

### Prioritas 1 — Pastikan eksperimen dapat ditelusuri

- Simpan command training, seluruh args, versi kode atau commit, seed, manifest, dan epoch checkpoint terbaik.
- Bedakan run baru dari resume dan evaluasi ulang.
- Gunakan direktori run baru untuk perubahan konfigurasi.
- Tambahkan status yang jelas ketika checkpoint sudah mencapai batas epoch.
- Simpan state optimizer dan pertahankan momentum antar-epoch.

Tujuannya memastikan perbandingan berikutnya benar-benar membandingkan eksperimen yang diketahui, bukan hasil dari konfigurasi atau checkpoint yang tercampur.

### Prioritas 2 — Audit mask dan contoh salah

- Pilih sampel semua kelas dengan Dice rendah, sedang, dan tinggi.
- Tampilkan foto, mask sebelum refinement, mask setelah refinement, probabilitas, dan overlay.
- Periksa lesi hilang, area tambahan, alignment, ukuran objek setelah resize, dan mask kosong.
- Tinjau 31 kasus level 1 → 0 dan 11 kasus level 3 → 2.
- Verifikasi pemisahan identitas subjek antar-split jika informasi subjek tersedia.

Jika target tidak konsisten, perubahan arsitektur saja dapat tetap menghasilkan skor yang terbatas.

### Prioritas 3 — Ukur kelemahan dengan metrik yang sesuai

- Simpan Dice dan IoU per gambar serta ringkasan per kelas.
- Tambahkan precision dan recall foreground untuk membedakan prediksi berlebih dari lesi yang terlewat.
- Laporkan macro F1, balanced accuracy, dan dukungan tiap kelas.
- Simpan probabilitas semua kelas untuk analisis confidence dan kalibrasi.
- Catat epoch terbaik untuk setiap kriteria validation.
- Pisahkan evaluasi mask kosong jika ada, karena skor overlap dapat berubah tajam pada kasus tersebut.

### Prioritas 4 — Jalankan eksperimen terkontrol

| Eksperimen | Perubahan utama | Pertanyaan yang dijawab |
|---|---|---|
| Optimizer persisten | Pertahankan state antar-epoch | Apakah reset momentum membatasi training? |
| Audit normalisasi | Transformasi input konsisten dengan konfigurasi encoder | Apakah skala input memengaruhi hasil? |
| Class weighting | None versus balanced | Apakah weighting meningkatkan macro F1 dan recall minoritas? |
| Kebijakan BatchNorm | Statistik tetap versus diperbarui saat freeze | Kebijakan phase 1 mana lebih baik? |
| Threshold mask | Pilih threshold pada validation | Apakah 0,5 sesuai untuk tradeoff foreground? |
| Resolusi input | 320 versus resolusi lebih tinggi sesuai memori | Apakah detail lesi kecil menjadi batas utama? |
| Cabang morfologi | Global-only versus fusion | Apakah morfologi memberi kontribusi tambahan? |
| MA-LDS | Smoothing terkontrol versus pembanding | Apakah smoothing membantu urutan kelas atau mengaburkan batas? |

Ubah satu faktor utama per perbandingan, pertahankan split, dan ulangi seed untuk konfigurasi yang menjanjikan. Pilih threshold dan checkpoint menggunakan validation, lalu gunakan test sebagai evaluasi akhir. Mengoptimalkan threshold langsung pada test akan membuat hasil test terlalu optimistis.

Weighted sampler dapat diuji setelah baseline terkontrol tersedia. Menggabungkan sampler dan class weighting sekaligus mengubah dua mekanisme penyeimbangan, sehingga perlu dibandingkan secara terpisah sebelum menyimpulkan keduanya diperlukan.

### Prioritas 5 — Pertimbangkan tambahan epoch berdasarkan validation

Kurva masih menunjukkan kemajuan, sehingga tambahan epoch layak diuji setelah masalah optimizer dan pelacakan run ditangani. Namun, tambahan epoch tidak otomatis mengatasi anotasi, fitur morfologi, atau resolusi.

Gunakan aturan penghentian berdasarkan validation dan laporkan hasilnya. Jika validation berhenti membaik sementara train terus naik, evaluasi generalisasi sebelum memperpanjang training lagi.

## 12. Kesimpulan untuk pelaporan penelitian

Pada pengujian 224 gambar, konfigurasi balanced mencapai accuracy 60,71%, Cohen’s kappa 0,4346, Dice 0,4430, dan IoU 0,3016. Dibandingkan rekap run sebelumnya, klasifikasi meningkat, tetapi segmentasi tidak menunjukkan peningkatan. Sebanyak 95,45% kesalahan klasifikasi berjarak satu tingkat keparahan, dengan kebingungan utama antara level 0–1 serta level 2–3.

Kurva menunjukkan model belajar dan belum memberikan bukti overfitting berat, tetapi kemampuan segmentasi masih terbatas bahkan pada training set. Eksekusi 12 Oktober merupakan evaluasi ulang checkpoint yang sudah mencapai 120 epoch. Pemeriksaan kode lokal mengidentifikasi reset optimizer setiap epoch, resume tanpa state optimizer, dan fitur morfologi target yang sebagian redundan sebagai hal yang perlu ditangani atau diuji.

Hasil ini belum cukup untuk menyatakan arsitektur hybrid gagal, class weighting merupakan satu-satunya penyebab perbaikan, atau penambahan epoch akan menyelesaikan masalah. Langkah paling kuat berikutnya adalah memperbaiki keterlacakan dan optimisasi, mengaudit mask, lalu melakukan ablation terkontrol berdasarkan validation.
