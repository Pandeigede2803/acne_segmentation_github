# Step 20 — Ablation setelah Runtime Reset

**Tidak perlu menjalankan training Step 11 lagi.** Yang diperlukan adalah
hasilnya di Drive: `metrics.json`, provenance, manifest asli/augmented, foto
preprocessed, dan mask train/validation.

## Menjalankan

1. Buka notebook terbaru, aktifkan GPU.
2. Langsung buka Step 20. Path sumber v4 sudah diisi pada `ABLATION_SOURCE_RUN`;
   edit hanya jika memakai run sumber lain.
3. Sediakan `prepare_augmented_ablation.py` dan `train_single_task_resnet50.py`
   terbaru di `output/ablation_scripts`, atau publish sebelum unduhan pertama.
   File kedua hanya menyediakan helper pemulihan; tidak melatih single-task.
   Jika helper sudah ada di cache Step 29, notebook menyalinnya otomatis.
4. Ubah `RUN_ABLATIONS=True`, jalankan cell. Drive di-mount bila diperlukan.
5. Varian yang selesai dilewati; varian terputus di-resume. Untuk mengulang
   ablation dari awal, ganti suffix `ABLATION_DIR`, bukan folder sumber Step 11.

Notebook memulihkan ukuran input, batch, epoch, learning rate, seed, bobot loss,
dan augmentasi online dari konfigurasi Step 11. Class weighting balanced, tanpa
weighted sampler; train/validation dari manifest augmented yang sama. Test tidak
dievaluasi saat training; evaluasi test terpisah tersedia di Step 20b.
Enam varian dan proposed kontrol merupakan training ablation terpisah,
bukan pemanggilan ulang training utama Step 11.

## Setelah file /content hilang

Tidak perlu variabel `OUTPUT_A`, `v2_audit`, `IMAGE_DIR`, JSON anotasi mentah,
atau script refinement di runtime lama. Bukti audit v2 historis dan hash manifest
diverifikasi dari provenance yang tersimpan. Foto preprocessed dan mask Drive
tetap harus ada. Isi pikselnya belum diperiksa melalui hash.

Trainer hybrid yang digunakan harus memiliki SHA-256 identik dengan trainer
Step 11. Cache/unduhan yang berbeda ditolak. Jika versi GitHub sudah berubah,
pulihkan salinan trainer asli ke `output/ablation_scripts`; **jangan mengulang
training Step 11 hanya untuk menghilangkan error** atau melewati pemeriksaan.

Cache/provenance ablation yang sudah sesuai dipertahankan. Perubahan input,
konfigurasi, atau versi trainer tidak boleh dimasukkan ke checkpoint yang sama.

## Evaluasi hasil — Step 20a

Setelah ketujuh varian selesai (termasuk `proposed` kontrol ablation), jalankan
cell **Step 20a** pada notebook. Bisa dilakukan setelah runtime reset tanpa
mengulang training: sesuaikan `EVAL_ABLATION_DIR` dengan folder Step 20 di Drive.
Sediakan `evaluate_ablation_validation.py` terbaru di `output/evaluation_scripts`
atau publish sebelum unduhan pertama. Cell menolak run yang belum lengkap,
manifest yang berubah, dan metrik test. Output CSV serta grafik ditulis ke
`<ABLATION_DIR>/validation_evaluation/`; checkpoint training tidak diubah.

Pemilihan checkpoint tiap varian berdasarkan **validation loss minimum**.
Tabel membandingkan accuracy, Kappa, Dice, dan IoU dengan kontrol `proposed`
yang dilatih di folder ablation yang sama, bukan run Step 11. Delta accuracy
dalam poin persentase. Seluruh angka adalah **validation**, bukan test; satu
seed belum membuktikan signifikansi statistik. Loss antar-objektif berbeda
tidak boleh dijadikan dasar pemeringkatan varian.

## Evaluasi hasil — Step 20b (test)

Jalankan cell **Step 20b** setelah Step 20a. Cell memakai tujuh checkpoint
`best_model.pt` yang sama dan satu manifest
`refined_manifest_offline_augmented_100pct.csv` (split test 184 citra).
Manifest wajib memiliki SHA-256 identik dengan manifest aug100 pada provenance
sumber Step 11; manifest lain ditolak. Threshold segmentasi 0,5, tanpa tuning.
Output `ablation_test_results.csv` berisi kolom `variant, acc, kappa, macro_f1,
dice, iou`, ditulis di folder ablation. Angka ini adalah test run ablation;
jangan mencampurnya dengan test run utama Step 11 dalam satu tabel. Bila
keduanya dibandingkan, cantumkan label run di tabel.
