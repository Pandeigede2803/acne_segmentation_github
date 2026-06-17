# Flow Eksperimen ACNE04-v2

Dokumen ini menjelaskan alur eksperimen yang sedang dipakai pada folder `projectsegmentasi`.

Fokus eksperimen:
- dataset: `ACNE04-v2`
- anotasi segmentasi: lingkaran `cx, cy, radius`
- target klasifikasi: severity `levle0` s.d. `levle3`
- arsitektur pembanding:
  - `Dilated U-Net + Spatial Attention + AMFM`
  - `Dilated U-Net + BCE + Dice Loss`
  - `ResNet-50 Encoder + Spatial Attention + AMFM`

---

## 1. Sumber Data

```text
ACNE04-v2
|
+-- image folder
|   /datasetacne04/acne_1024/small_1024
|
+-- annotation json
    /acne04v2/Acne04-v2_annotations.json
```

Isi anotasi:
- `file_name`
- `image_id`
- `coordinates = [cx, cy]`
- `radius`
- `class_name = acne`

Label severity diambil dari nama file:

```text
levle0_xxx.jpg -> severity 0
levle1_xxx.jpg -> severity 1
levle2_xxx.jpg -> severity 2
levle3_xxx.jpg -> severity 3
```

---

## 2. Flow Besar Eksperimen

```text
RAW IMAGE + CIRCULAR ANNOTATION
              |
              v
      Load metadata image
              |
              v
    Match image dengan lingkaran lesi
              |
              v
  Stratified subset (mis. 200 / 400 image)
              |
              v
   Stratified split train / val / test
              |
              v
        Preprocessing image
              |
              v
   Bangun mask biner dari lingkaran
              |
              v
           Train model
              |
              v
      Evaluasi segmentasi + klasifikasi
              |
              v
 Simpan best model + metrics + visualisasi
```

---

## 3. Flow Preprocessing

```text
Input image
   |
   v
Convert RGB
   |
   v
Center crop (opsional)
   |
   v
Histogram equalization
   |
   v
Denoise
   |
   v
Sharpen
   |
   v
Resize ke image_size
   |
   v
Output image siap model
```

Catatan:
- default `crop_ratio = 1.0`, jadi tidak memotong area agresif
- preprocessing dilakukan `on-the-fly` saat training, bukan disimpan manual ke folder baru

---

## 4. Flow Pembuatan Mask Segmentasi

```text
Anotasi lingkaran (cx, cy, radius)
           |
           v
Hitung posisi lingkaran pada ukuran target
           |
           v
Isi area lingkaran = 1
Luar lingkaran = 0
           |
           v
Binary mask hitam-putih
```

Makna mask:
- `0` = background
- `1` = lesi

---

## 5. Flow Model Dilated U-Net + AMFM

```text
Input image
   |
   v
Dilated U-Net Encoder
   |
   v
Bottleneck feature
   |
   v
Spatial Attention
   |
   +------------------------------+
   |                              |
   v                              v
Segmentation Decoder         Global Feature
   |                              |
   v                              |
Mask Lesi Prediksi                |
   |                              |
   v                              |
Fitur Morfologi ------------------+
   |
   v
AMFM Fusion
   |
   v
Classification Head
   |
   v
Severity Hayashi
```

Penjelasan:
- cabang kiri menghasilkan `mask lesi`
- dari mask itu dihitung `fitur morfologi`
- cabang kanan mengambil `fitur global`
- keduanya digabung di `AMFM`
- hasil fusion dipakai untuk klasifikasi severity

---

## 6. Flow Model Dilated U-Net + BCE + Dice Loss

Arsitektur sama dengan flow sebelumnya.

Perbedaannya ada pada bagian loss segmentasi:

```text
Mask prediksi
   |
   +--> BCE Loss
   |
   +--> Dice Loss
          |
          v
  Segmentation Loss lebih kuat
```

Tujuan:
- mendorong model lebih serius belajar segmentasi
- tidak hanya fokus ke background

---

## 7. Flow Model ResNet-50 + AMFM

```text
Input image
   |
   v
ResNet-50 Encoder
   |
   v
Bottleneck feature
   |
   v
Spatial Attention
   |
   +------------------------------+
   |                              |
   v                              v
Segmentation Decoder         Global Feature
   |                              |
   v                              |
Mask Lesi Prediksi                |
   |                              |
   v                              |
Fitur Morfologi ------------------+
   |
   v
AMFM Fusion
   |
   v
Classification Head
   |
   v
Severity Hayashi
```

Perbedaan utama dengan versi Dilated U-Net:
- encoder awal diganti `ResNet-50`
- tujuan pembanding: melihat apakah encoder yang lebih kuat membantu segmentasi dan klasifikasi

---

## 8. Flow Augmentasi

Augmentasi hanya diterapkan pada `train set`.

```text
Image train + mask train
        |
        v
Rotasi acak (-15 s/d +15 derajat)
        |
        v
Horizontal flip acak
        |
        v
Masuk ke training
```

Catatan penting:
- augmentasi gambar dan mask dilakukan bersama
- supaya posisi lesi tetap sinkron

---

## 9. Flow Training

```text
Train set
   |
   v
Batch 1 -> forward -> hitung loss -> update bobot
Batch 2 -> forward -> hitung loss -> update bobot
Batch 3 -> forward -> hitung loss -> update bobot
...
Semua batch selesai
   |
   v
1 epoch selesai
```

Kalau `epochs = 20`, artinya:

```text
Seluruh data train dipelajari 20 kali
```

---

## 10. Komponen Loss

### 10.1 Versi baseline

```text
Total Loss
   |
   +-- Segmentation Loss (BCE)
   +-- Classification Loss
   +-- Morphology Loss
```

### 10.2 Versi pembanding

```text
Total Loss
   |
   +-- Segmentation Loss
   |      |
   |      +-- BCE
   |      +-- Dice Loss
   |
   +-- Classification Loss
   +-- Morphology Loss
```

---

## 11. Evaluasi

Evaluasi dibagi dua:

### 11.1 Evaluasi segmentasi

```text
Predicted mask vs Ground truth mask
        |
        +-- Dice
        +-- IoU
```

### 11.2 Evaluasi klasifikasi

```text
Predicted severity vs Ground truth severity
        |
        +-- Accuracy
        +-- Cohen's Kappa
```

Loss yang juga dicatat:
- `loss`
- `seg_loss`
- `cls_loss`
- `morph_loss`

---

## 12. File Script yang Dipakai

### 12.1 Baseline Dilated U-Net

```text
projectsegmentasi/train_baseline_dilated_unet.py
```

### 12.2 Dilated U-Net + BCE + Dice Loss

```text
projectsegmentasi/train_baseline_dilated_unet_bce_dice.py
```

### 12.3 ResNet-50 + AMFM

```text
projectsegmentasi/train_resnet50_amfm.py
```

### 12.4 Visualisasi metrics

```text
projectsegmentasi/visualize_baseline_metrics.py
```

---

## 13. Output Hasil

```text
projectsegmentasi/baseline_runs/
|
+-- dilated_unet_acne04v2/
|   +-- best_model.pt
|   +-- metrics.json
|   +-- visualization/
|
+-- dilated_unet_acne04v2_bce_dice/
|   +-- best_model.pt
|   +-- metrics.json
|
+-- resnet50_amfm_acne04v2/
    +-- best_model.pt
    +-- metrics.json
```

Makna file:
- `best_model.pt` = bobot model terbaik
- `metrics.json` = angka hasil training/evaluasi
- `visualization/` = grafik dan laporan visual

---

## 14. Ringkasan Logika Penelitian

```text
Segmentasi lokal
    ->
ekstraksi morfologi lesi
    ->
digabung dengan fitur global wajah
    ->
dipakai untuk klasifikasi severity
```

Inti ide hybrid:
- bukan hanya menebak severity dari wajah secara global
- tetapi mencoba memakai bukti lokal lesi untuk memperkuat keputusan akhir

