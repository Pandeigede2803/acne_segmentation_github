# Step 34 --- Annotation-Guided Mask Refinement V2

## 1. Objective

Audit dan perbaiki kualitas pseudo-ground-truth segmentation mask
ACNE04-v2 sebelum retraining. Pertahankan Step 31--33 dan seluruh
artefak baseline. Jangan mengubah ResNet-50, Dilated UNet, Spatial
Attention, AMFM, MA-LDS, atau hyperparameter training pada tahap ini.

Repository: https://github.com/Pandeigede2803/acne_segmentation_github

## 2. Evidence dan diagnosis

Run Step 31b raw, 512×512, split 60/40, 120 epoch: training Dice 0.7528
vs test 0.4153; IoU 0.6172 vs 0.2810; accuracy 99.58% vs 72.05%; macro
F1 99.58% vs 55.55%.

Visualisasi sembilan sampel Level 3 memperlihatkan delapan kesalahan:
enam Level 2 dan dua Level 1. Satu sampel benar. Ini bukan estimasi
recall seluruh kelas karena sampel dipilih secara khusus.

  Sampel       Observasi visual                                Prediksi
  ------------ ----------------------------------------------- ----------
  levle3_101   Sebagian area lesi tidak tercakup               Level 2
  levle3_106   Area kemerahan pipi tidak sepenuhnya tercakup   Level 2
  levle3_112   Wajah kecil relatif terhadap kanvas             Level 2
  levle3_133   Cakupan mask tampak terbatas                    Level 1
  levle3_39    Mask menyerupai lingkaran anotasi               Level 2
  levle3_53    Kemerahan luas, mask relatif sedikit            Level 1
  levle3_77    Sebagian area anotasi tampak menyusut           Level 2
  levle3_94    Mask lesi berdekatan tampak menyatu             Level 2
  levle3_117   Mask tersebar pada wajah                        Level 3

Overlay hijau adalah refined target mask, bukan predicted segmentation
mask. Observasi ini bersifat kualitatif; kemerahan bukan bukti pasti
lesi aktif. Anotasi awal juga mungkin tidak lengkap, sehingga kualitas
refinement dan kualitas anotasi harus diperiksa terpisah.

## 3. Source code yang perlu diperiksa

-   `acne_colab.ipynb`: Step 31, 31b, 31c, 31d, 32, 33.
-   `scripts/refine_masks_color.py`: `detect_lesion_mask()` (64--110),
    `refine_mask_with_red()` (167--191), pembuatan mask (540--567).
-   `scripts/train_proposal_hybrid_resnet50_malds.py`:
    `mask_morphology()` (147--160), `morphology_from_mask()` (314--326),
    fusion (344--356), `make_malds_targets()` (377--390).
-   `scripts/face_skin_utils.py`: utilitas face/skin region.

Step 31 menggunakan `--preprocessing-mode none` untuk input RGB, tetapi
tetap melakukan refinement target mask. Step 33 skinfilter adalah varian
terpisah, bukan penyebab yang sudah terbukti pada run raw.

### Kelemahan V1

``` python
skin_pixels = a_channel[circle_mask == 0]
threshold = skin_mean + lab_sigma * skin_std
red_mask = detect_lesion_mask(bgr_image, circle_mask, lab_sigma)
refined = cv2.bitwise_and(red_mask, circle_mask)
```

-   Baseline global dapat tercemar background, rambut, pakaian dan lesi
    lain.
-   Intersection membatasi deteksi pada anotasi awal; tidak dapat
    memulihkan lesi yang tidak dianotasi secara sistematis.
-   Closing/opening kernel 5×5 berpotensi menggabungkan atau menghapus
    lesi kecil.
-   Fallback global pada rasio area \< 0.02 tidak menjamin setiap
    anotasi tetap terwakili.
-   Morphological features belum memiliki lesion count; area dan density
    target biner identik. Ini dicatat untuk eksperimen terpisah, bukan
    perubahan Step 34.
-   MA-LDS menggunakan area, density dan spread dari target mask; AMFM
    menggunakan fitur dari predicted mask. Kualitas mask dapat
    memengaruhi keduanya, tetapi hubungan kausal belum terbukti.

## 4. Proposed pipeline V2

1.  Gunakan gambar asli dan transformasi letterbox 512 yang sama dengan
    V1. Pertahankan koordinat anotasi individual.
2.  Bentuk face/skin region sebagai bantuan pemilihan baseline, bukan
    hard gate yang menghapus anotasi. Jika deteksi wajah gagal, gunakan
    neighborhood anotasi dan catat fallback.
3.  Untuk setiap lingkaran anotasi, ambil piksel kulit sekitar dalam
    local annulus, keluarkan seluruh anotasi lain, padding, dan piksel
    nonkulit. Gunakan statistik robust (median dan MAD dengan floor)
    atau local mean/std sebagai varian terpisah; catat metode dan jumlah
    piksel valid.
4.  Terapkan local adaptive Lab redness, pertahankan HSV merah/pustule
    sebagai kandidat tambahan. Batasi kandidat pada anotasi yang
    bersangkutan. Jangan membuat label baru dari kemerahan di luar
    anotasi.
5.  Terapkan morphology per anotasi dengan kernel kecil atau tanpa
    morphology. Pastikan hasil akhir tetap dalam support anotasi.
6.  Jika kandidat terlalu kecil atau kosong, gunakan fallback circle per
    anotasi; catat fallback dan alasannya. Fallback adalah proxy kasar,
    bukan mask lesi terverifikasi.
7.  Gabungkan hasil per anotasi menjadi binary mask. Simpan RGB input
    tanpa perubahan warna tambahan.

## 5. CPU-only experiment matrix

  Parameter    V1       V2 candidate
  ------------ -------- --------------------------
  Baseline     Global   Local surrounding skin
  Lab sigma    1.2      0.5, 0.8, 1.2
  Morphology   5×5      1×1 (identity), 3×3, 5×5
  Fallback     Global   Per annotation
  Resolution   512      512

Mulai dari circle-only control, V1 reproducibility control, lalu local
baseline dengan sigma 1.2/kernel 5; lanjutkan variasi sigma/kernel pada
subset development. Ubah satu faktor setiap kali jika memungkinkan.
Threshold fallback per anotasi harus eksplisit; gunakan 0.02 sebagai
starting point, bukan nilai optimal. Jangan memilih parameter
berdasarkan test set.

## 6. Quantitative audit

Untuk setiap gambar dan kelas, laporkan: - Jumlah anotasi asli, jumlah
anotasi yang masih memiliki piksel mask, jumlah anotasi hilang,
retention rate. - Per-anotasi retained area ratio = area(mask ∩
circle_i) / area(circle_i); distribusi median, IQR, minimum. - Rasio
luas refined mask terhadap circle union, jumlah fallback, jumlah
komponen terhubung (diagnostik, bukan jumlah lesi pasti), jumlah
komponen kecil yang hilang, area di luar anotasi. - Perubahan area,
density, centroid dan spread yang masuk ke target morfologi. - IoU V1/V2
terhadap circle mask sebagai ukuran kesesuaian dengan proxy anotasi,
bukan akurasi klinis. - Jika tersedia, evaluasi terhadap subset mask
manual yang independen menggunakan Dice/IoU dan lesion-level recall.

Jangan mengoptimalkan retention saja: circle fallback dapat membuat
retention tinggi tanpa memperbaiki ketepatan batas lesi. Laporkan
kualitas batas dan false-positive secara terpisah jika anotasi manual
tersedia.

## 7. Deliverables dan implementasi agent

Buat script baru `scripts/refine_masks_color_v2.py` dan cell notebook
Step 34 dengan `RUN_MASK_REFINEMENT_V2 = False` secara default. Gunakan
folder output baru, misalnya `output/experiments_512_mask_v2/`, jangan
overwrite baseline.

Artefak: - `audit_per_image.csv`, `audit_per_annotation.csv`,
`audit_per_class.csv` - `comparison_grid.png`: original RGB \| circle \|
V1 \| V2; tambahan local redness map dan fallback overlay bila perlu. -
`audit_report.md`, `refinement_config.json`, `provenance.json`, manifest
V2 dengan path yang valid. - Mask PNG biner 0/255, ukuran 512×512.

Audit sembilan sampel yang sudah ditampilkan untuk diagnosis saja. Untuk
pemilihan parameter gunakan subset stratified dari original training
(sebelum augmentation); pisahkan validation dari training jika nanti
retraining. Pertahankan membership test 40% dan seed 42. Jangan augment
validation/test. Jangan gunakan hasil test untuk tuning; karena test
sebelumnya sudah dilihat, dokumentasikan keterbatasan ini dan gunakan
holdout independen jika memungkinkan untuk konfirmasi final.

Unit tests: anotasi kecil, anotasi overlap, wajah tidak terdeteksi,
annulus kosong, padding, mask kosong, koordinat letterbox, determinisme,
dan tidak ada modifikasi file baseline. Validasi jumlah baris dan
mapping label. Catat hash script, anotasi, manifest dan konfigurasi.

## 8. Acceptance gate dan next action

-   V1 dapat direproduksi atau perbedaannya dijelaskan melalui
    provenance.
-   V2 tidak menghapus anotasi secara diam-diam; fallback dilaporkan.
-   Visualisasi dan statistik per kelas tersedia, terutama Level 3.
-   Tidak ada perubahan arsitektur atau training dan tidak ada
    penggunaan GPU wajib.
-   Agent berhenti setelah audit dan meminta persetujuan sebelum
    retraining.

Jika V2 menunjukkan kualitas target yang lebih baik pada development
set, rancang eksperimen terkontrol V1 vs V2 dengan split, seed,
augmentation, arsitektur, epoch budget dan checkpoint selection yang
sama. Gunakan validation macro F1, recall Level 3, Kappa, Dice/IoU;
evaluasi test hanya setelah konfigurasi dikunci. Perubahan lesion count
atau MA-LDS dilakukan sebagai ablation terpisah.

## 9. Limitations

Diagnosis didasarkan pada source code yang telah diperiksa dan overlay
yang diberikan, bukan inspeksi langsung file mask individual atau ground
truth pixel-level manual. Nilai sigma, kernel, fallback, nama script dan
folder V2 adalah usulan implementasi, bukan parameter optimal yang
terbukti. Perbaikan mask belum terbukti meningkatkan klasifikasi. Tidak
ada retraining atau perubahan repository yang dilakukan dalam penyusunan
dokumen ini.
