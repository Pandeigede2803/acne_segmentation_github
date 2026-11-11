# Pembanding ResNet-50 Single-Task

Step **29** di `acne_colab.ipynb` ditambahkan di akhir agar step lama tidak berubah.
Bisa langsung dijalankan sesudah setup Step 10a/11 atau ablation Step 20;
tidak perlu training eksperimen tambahan pada Step 21–28.

## Tujuan

Menyediakan pembanding klasifikasi murni untuk rumusan masalah nomor 2 dan M1
proposal. Model: ResNet-50 ImageNet → GAP → Linear 2048→128 → Linear 128→64 →
ReLU → Dropout 0,3 → Linear 64→4. Tidak ada decoder, Spatial Attention, AMFM,
MA-LDS, loss segmentasi, atau loss morfologi. Dataset trainer hanya membuka foto
train/validation, bukan mask atau foto test.

## Protokol

- Manifest sumber augmented 100% sama dengan Step 11, diaudit sebagai subset v2.
- Manifest single-task hanya menyimpan foto/label train dan validation.
- SGD momentum 0,9; default 30 epoch encoder freeze (LR 1e-3), 90 fine-tuning
  (LR 1e-4), seed 42, batch/input dan augmentasi online sesuai Step 11.
- Class weighting balanced dihitung dari train; tanpa weighted sampler.
- Loss CE per sampel dikalikan class weight kemudian dirata-ratakan, sebanding
  dengan reduksi loss klasifikasi hybrid ketika target berupa hard label.
- Model terbaik dipilih dari validation CE berbobot minimum. Proposed memakai
  total multi-task validation loss; jangan membandingkan nilai kedua loss secara
  langsung atau mengklaim perbandingan ini mengisolasi efek segmentasi saja.
- Tidak memakai checkpoint proposed. Pretrained encoder berasal dari ImageNet.
- Satu seed dan anggaran epoch sama tidak menjamin hasil bit-identik atau
  signifikansi statistik. Backbone tanpa decoder lebih ringan, tetapi durasi
  training harus diukur, bukan diasumsikan.

## Menjalankan

1. Gunakan notebook terbaru. Step 29 dapat dijalankan setelah runtime reset
   tanpa menjalankan cell Step 11 lagi: `ST_SOURCE_RUN` menunjuk folder run
   proposed v4 di Drive. Edit satu path ini jika memakai run sumber berbeda.
   Drive akan di-mount jika diperlukan saat training diaktifkan.
2. Unggah `scripts/train_single_task_resnet50.py` terbaru ke path `ST_SCRIPT`
   (default `output/single_task_scripts/train_single_task_resnet50.py` di Drive)
   atau publish ke GitHub sebelum unduhan pertama. Script lama yang belum punya
   helper pemulihan akan ditolak dengan pesan khusus. Perubahan lokal tidak
   otomatis memperbarui file di Drive/GitHub.
3. Jangan jalankan bersamaan dengan training proposed pada GPU yang sama.
4. Aktifkan `RUN_SINGLE_TASK = True` di Step 29.
5. Jika terputus, gunakan `ST_SOURCE_RUN` dan `ST_BUNDLE` yang sama lalu jalankan ulang.
   Checkpoint akhir epoch otomatis di-resume; epoch terputus dapat diulang.
6. Untuk training baru, ganti suffix `ST_BUNDLE`; jangan hapus hasil lama.

Checkpoint menyimpan model, kedua optimizer, RNG Python/PyTorch/CUDA, history,
state terbaik, dan prediksi validation terbaik. Penulisan checkpoint/JSON
memakai temporary file lalu replace. Hanya muat checkpoint buatan sendiri yang
dipercaya. Identitas manifest/script dicatat, tetapi isi piksel foto tidak di-hash.

## Hasil

Pemulihan menggunakan provenance audit v2 yang telah disimpan saat Step 11.
Manifest asli dan augmented di Drive harus tetap cocok hash-nya; keutuhan
baris asli, validation/test yang tidak berubah, keseimbangan kelas, serta
keberadaan foto preprocessed train/validation diperiksa ulang. File mentah,
mask, JSON anotasi, dan trainer hybrid lama di `/content` tidak dibutuhkan oleh
classifier ini dan tidak diunduh ulang untuk memulai training single-task.
Identitas historisnya tetap tersimpan dalam provenance sumber; ini bukan audit
ulang piksel/anotasi mentah. Provenance tanpa bukti audit v2 resmi ditolak.

Di folder `ST_RUN`: `best_model.pt`, `last_checkpoint.pt`, `history_partial.json`,
`metrics.json`, dan `validation_predictions.csv`. Log di `ST_BUNDLE/training.log`.
Ringkasan mencakup accuracy, Kappa, Macro-F1, recall/precision/F1 tiap kelas,
dan confusion matrix validation. **Dice/IoU tidak berlaku**, bukan nol.

Cell ringkasan membandingkan validation single-task dengan validation checkpoint
terbaik Step 11 bila tersedia. Tidak melakukan training atau evaluasi test.
Tabel laporan tetap kosong sampai hasil run baru tersedia.
