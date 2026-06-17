# Problem Model Hybrid Saat Ini

Dokumen ini merangkum problem utama dari eksperimen model hybrid segmentasi + klasifikasi yang sudah dijalankan sampai 26 Mei 2026.

---

## 1. Ringkasan Kondisi

Model yang sudah diuji:

- `Dilated U-Net + AMFM`
- `Dilated U-Net + BCE + Dice Loss`
- `ResNet-50 + AMFM`
- `Hybrid staged dari code_dari_claude`

Semua pipeline utama sudah berhasil dijalankan:

- preprocessing
- training
- evaluasi
- logging ke `metrics.json`
- logging ke `experiment_results.csv`
- penyimpanan output per stage

Jadi problem sekarang **bukan lagi di engineering pipeline**, tetapi di **perilaku belajar model**, khususnya pada cabang segmentasi.

---

## 2. Problem Utama

### 2.1 Segmentasi tetap gagal belajar

Gejala yang paling konsisten:

- `Dice = 0.0000`
- `IoU = 0.0000`

Ini terjadi berulang pada berbagai variasi:

- backbone berbeda
- image size berbeda
- augmentasi aktif
- preprocessing dipisah
- training staged

Maknanya:

- model belum berhasil menghasilkan mask lesi yang overlap dengan ground truth
- kemungkinan prediksi mask hampir kosong atau tidak relevan

---

### 2.2 Klasifikasi masih bisa belajar sebagian

Meskipun segmentasi gagal, klasifikasi masih bisa bergerak:

- `accuracy` beberapa kali berada di kisaran `0.38 - 0.55`
- `kappa` kadang kecil positif, tetapi sering kembali mendekati `0`

Maknanya:

- model masih bisa memanfaatkan fitur global wajah
- tapi kontribusi segmentasi ke hasil akhir belum nyata

Jadi sistem hybrid saat ini secara praktik lebih mirip:

```text
global classifier dengan segmentasi lemah
```

bukan:

```text
segmentasi kuat yang memperkuat klasifikasi
```

---

### 2.3 Pergantian backbone tidak menyelesaikan masalah

Eksperimen dengan `ResNet-50` juga tidak memperbaiki `Dice/IoU`.

Maknanya:

- masalah utama bukan semata-mata di backbone
- mengganti encoder tidak otomatis membuat segmentasi hidup

---

### 2.4 Resolusi lebih besar juga belum cukup

Eksperimen `image_size = 320` tidak memberi perubahan berarti pada segmentasi.

Maknanya:

- masalah bukan hanya karena `224` terlalu kecil
- resolusi lebih besar saja belum cukup memaksa model belajar mask

---

### 2.5 Augmentasi tidak cukup mengatasi masalah

Augmentasi yang sudah dicoba:

- rotasi `±15 derajat`
- horizontal flip

Hasil:

- klasifikasi kadang sedikit terbantu
- segmentasi tetap nol

Maknanya:

- augmentasi bukan solusi utama untuk bottleneck saat ini

---

### 2.6 BCE + Dice Loss sempat memberi sinyal, tetapi belum stabil

Eksperimen `BCE + Dice Loss` sempat menghasilkan:

- `dice` kecil di atas nol
- `iou` kecil di atas nol

Tetapi hasil ini tidak konsisten pada run lain.

Maknanya:

- perubahan loss bergerak ke arah yang lebih benar
- tetapi belum cukup stabil untuk menyelesaikan problem segmentasi

---

## 3. Dugaan Akar Masalah

Beberapa dugaan paling kuat:

### 3.1 Model memilih solusi “mask kosong”

Kemungkinan besar model menemukan solusi aman:

- prediksi hampir semua background
- loss total tetap bisa turun
- klasifikasi tetap bisa belajar dari fitur global

Akibatnya:

- `Dice/IoU` runtuh ke nol

---

### 3.2 Ketidakseimbangan piksel

Pada mask segmentasi:

- background jauh lebih banyak
- area lesi sangat kecil

Akibatnya:

- model lebih mudah menebak background
- objek lesi tidak benar-benar dipelajari

---

### 3.3 Target lingkaran mungkin kurang natural

Ground truth dibuat dari:

- `cx`
- `cy`
- `radius`

Masalahnya:

- lesi asli tidak selalu berbentuk lingkaran sempurna
- jadi target segmentasi bisa kurang cocok dengan penampilan visual lesi

---

### 3.4 Segmentasi kalah dari cabang klasifikasi

Karena model multitask:

- satu cabang belajar segmentasi
- satu cabang belajar klasifikasi

Kalau klasifikasi lebih mudah dipelajari, model bisa:

- mengoptimalkan fitur global
- mengabaikan pembelajaran mask secara serius

---

## 4. Bukti Bahwa Problem Sudah Bukan di Pipeline

Hal-hal berikut **sudah berhasil**:

- preprocessing terpisah disimpan ke disk
- mask hitam-putih dibuat dan disimpan
- training staged berhasil jalan
- output per stage berhasil disimpan
- CSV hasil eksperimen berhasil dicatat

Artinya:

```text
problem utama sekarang ada pada learning behavior model,
bukan lagi pada implementasi dasar pipeline.
```

---

## 5. Folder Penting untuk Audit Lanjutan

### 5.1 Hasil training hybrid staged

[hybrid_resnet50_staged](/Users/macbookprom1/Kuliah%20s2/dataset-acne/projectsegmentasi/code_dari_claude/runs/hybrid_resnet50_staged:1)

Isi penting:

- `best_model.pt`
- `metrics.json`
- `stage_outputs_test/`

### 5.2 Output per stage

[stage_outputs_test](/Users/macbookprom1/Kuliah%20s2/dataset-acne/projectsegmentasi/code_dari_claude/runs/hybrid_resnet50_staged/stage_outputs_test:1)

Yang perlu dicek besok:

- `pred_mask.png`
- `gt_mask.png`
- `stage_summary.csv`
- `stage_outputs.pt`

Tujuan audit:

- cek apakah `pred_mask` benar-benar kosong
- cek apakah attention map hidup atau mati
- cek apakah fitur morfologi (`v_morph`) punya variasi atau tidak

---

## 6. Kesimpulan Sementara

Kesimpulan paling jujur saat ini:

1. Pipeline hybrid sudah berhasil diimplementasikan end-to-end.
2. Cabang klasifikasi dapat belajar sebagian.
3. Cabang segmentasi masih gagal belajar secara stabil.
4. Backbone, augmentasi, dan resolusi belum menyelesaikan masalah segmentasi.
5. Problem utama kemungkinan besar ada pada objective segmentasi, ketidakseimbangan piksel, dan sifat target mask.

---

## 7. Fokus Lanjutan Besok

Prioritas analisis berikutnya:

1. buka `pred_mask.png` dan `gt_mask.png`
2. pastikan apakah prediksi mask kosong
3. cek `stage_summary.csv`
4. evaluasi apakah perlu:
   - loss segmentasi yang lebih kuat
   - weighting loss berbeda
   - perubahan target mask
   - strategi threshold / calibration

---

## 8. Ringkasan Satu Kalimat

```text
Model hybrid sudah jalan, tetapi cabang segmentasi belum benar-benar belajar,
sehingga kontribusi segmentasi ke klasifikasi akhir masih sangat lemah.
```

