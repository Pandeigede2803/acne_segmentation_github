# Acne Lesion Segmentation & Severity Classification

Penelitian S2 — Hybrid pipeline untuk segmentasi lesi jerawat dan klasifikasi tingkat keparahan menggunakan dataset **ACNE04-v2**.

---

## Daftar Isi

1. [Gambaran Umum](#1-gambaran-umum)
2. [Dataset](#2-dataset)
3. [Struktur Direktori](#3-struktur-direktori)
4. [Arsitektur Model](#4-arsitektur-model)
5. [Pipeline Eksperimen](#5-pipeline-eksperimen)
6. [Instalasi — Lokal](#6-instalasi)
7. [Menjalankan di Google Colab](#7-menjalankan-di-google-colab)
8. [Cara Menjalankan — Lokal](#8-cara-menjalankan--lokal)
9. [Hasil Eksperimen](#9-hasil-eksperimen)
10. [Analisis & Temuan](#10-analisis--temuan)
11. [Arah Pengembangan Lanjutan](#11-arah-pengembangan-lanjutan)

---

## 1. Gambaran Umum

Project ini mengimplementasikan arsitektur **hybrid multitask** yang secara simultan melakukan:

- **Segmentasi** — memprediksi mask piksel-level area lesi jerawat
- **Klasifikasi** — menentukan tingkat keparahan (severity) berdasarkan skala Hayashi (levle0–levle3)

Ide inti:

```
Segmentasi lokal lesi
    → ekstraksi fitur morfologi (jumlah, ukuran, distribusi lesi)
    → digabung dengan fitur global wajah via AMFM
    → klasifikasi severity yang lebih informatif
```

Hipotesis utama: dengan memanfaatkan informasi lokal lesi (bukan hanya fitur global wajah), model dapat membuat keputusan klasifikasi yang lebih kuat dan dapat diinterpretasikan.

---

## 2. Dataset

### ACNE04-v2

| Properti | Detail |
|---|---|
| Jumlah gambar beranotasi | **1204 gambar** |
| Resolusi asli | 1024 × 1024 px (`small_1024`) |
| Format anotasi | JSON — titik pusat lingkaran `(cx, cy, radius)` |
| Label severity | `levle0` s.d. `levle3` |

### Lokasi Data

```text
/datasetacne04/acne_1024/small_1024/      ← folder gambar
/acne04v2/Acne04-v2_annotations.json     ← anotasi JSON
```

### Format Nama File

Severity diambil langsung dari nama file:

```text
levle0_xxx.jpg  →  severity 0  (ringan)
levle1_xxx.jpg  →  severity 1  (sedang)
levle2_xxx.jpg  →  severity 2  (berat)
levle3_xxx.jpg  →  severity 3  (sangat berat)
```

### Distribusi Kelas

| Kelas | Jumlah Gambar | Proporsi |
|---|---|---|
| levle0 | 470 | 39.0% |
| levle1 | 579 | 48.1% |
| levle2 | 133 | 11.0% |
| levle3 | 22  | 1.8%  |
| **Total** | **1204** | **100%** |

> **Catatan:** Dataset sangat imbalanced. `levle3` hanya 22 gambar, menjadi bottleneck utama klasifikasi.

### Split Data

| Split | Rasio |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Split dilakukan **stratified per kelas** untuk menjaga distribusi di setiap subset.

---

## 3. Struktur Direktori

```text
projectsegmentasi/
│
├── README.md                               ← dokumen ini
│
├── ── SCRIPT PREPROCESSING ──
├── preprocess_acne04v2_dataset.py          ← preprocessing pipeline utama (semua data)
├── preprocess_sample_100_images.py         ← preprocessing subset 100 gambar
├── refine_masks_color.py                   ← generate refined mask berbasis warna
├── check_annotations.py                    ← audit jumlah gambar beranotasi per level
│
├── ── SCRIPT TRAINING ──
├── train_baseline_dilated_unet.py          ← baseline Dilated U-Net + AMFM (BCE loss)
├── train_baseline_dilated_unet_bce_dice.py ← Dilated U-Net + BCE + Dice Loss
├── train_preprocessed_dilated_unet_bce_dice.py  ← versi dengan input preprocessed
├── train_resnet50_amfm.py                  ← ResNet-50 encoder + AMFM
├── train_proposal_hybrid_resnet50_malds.py ← proposal model + MA-LDS loss
│
├── ── SCRIPT INFERENCE & VISUALISASI ──
├── infer_testing_images.py                 ← inferensi test set + simpan mask prediksi
├── visualize_baseline_metrics.py           ← plot kurva training & metrik
├── sample_classification_subset.py         ← sampling stratified subset
│
├── ── KODE DARI CLAUDE ──
├── code_dari_claude/
│   ├── train_hybrid_resnet50_staged.py     ← training staged (phase1 frozen + phase2 finetune)
│   ├── preprocess_hybrid_dataset.py        ← preprocessing untuk hybrid pipeline
│   └── runs/
│       ├── hybrid_resnet50_staged/
│       │   ├── best_model.pt
│       │   └── metrics.json
│       └── preprocessed_400/
│           ├── manifest.csv
│           └── images/
│
├── ── OUTPUT PREPROCESSING ──
├── acne04v2_preprocessed_all/
│   └── manifest.csv
├── refined_masks_output/
│   ├── refined_manifest.csv                ← manifest utama untuk training terbaru
│   ├── refined_masks/                      ← mask ground truth berbasis warna
│   ├── preprocessed_images/               ← gambar setelah equalize+denoise+sharpen
│   └── visualizations/                    ← overlay 3-panel untuk audit visual
├── preprocessed_100_images/
├── sample_100_images/
│
├── ── OUTPUT TRAINING ──
├── baseline_runs/
│   ├── experiment_results.csv              ← rekap semua run dalam satu CSV
│   ├── dilated_unet_acne04v2/
│   │   ├── best_model.pt
│   │   ├── metrics.json
│   │   └── visualization/
│   ├── dilated_unet_acne04v2_bce_dice/
│   │   ├── best_model.pt
│   │   └── metrics.json
│   ├── resnet50_amfm_acne04v2/
│   │   ├── best_model.pt
│   │   └── metrics.json
│   └── proposal_hybrid_resnet50_malds/
│       ├── best_model.pt
│       └── metrics.json
│
├── testing_predictions/                    ← output inferensi test set
├── sample_100_manifest.csv
└── preprocessed_100_manifest.csv
```

---

## 4. Arsitektur Model

### 4.1 Komponen Utama

#### Dilated U-Net Encoder

- Encoder berbasis U-Net dengan **dilated convolutions** (atrous convolutions)
- Dilated conv memperluas receptive field tanpa mengurangi resolusi
- Cocok untuk menangkap lesi kecil dengan konteks area sekitar yang lebih luas

#### Spatial Attention Module

- Diterapkan setelah bottleneck encoder
- Membantu model fokus ke area yang relevan (area lesi)

#### Segmentation Decoder

- Upsampling bertahap dengan skip connections dari encoder
- Output: **binary mask** (1 = lesi, 0 = background)

#### Morphological Feature Extraction

- Dari mask prediksi, dihitung fitur morfologi:
  - total area lesi
  - jumlah kontur lesi
  - distribusi ukuran lesi
- Fitur ini merepresentasikan "seberapa parah" secara kuantitatif

#### AMFM — Adaptive Morphology-aware Fusion Module

- Menggabungkan dua aliran informasi:
  - **Fitur morfologi** (dari mask lesi)
  - **Fitur global** (dari bottleneck encoder)
- Hasil fusi digunakan untuk klasifikasi akhir

#### Classification Head

- Fully connected layers di atas output AMFM
- Prediksi severity: 4 kelas (levle0–levle3)
- Loss: Cross-entropy / Label Smoothing (tergantung varian)

### 4.2 Diagram Alur Arsitektur

```
Input Image (3 × H × W)
        │
        ▼
Dilated U-Net Encoder
        │
        ▼
Bottleneck Features
        │
        ▼
Spatial Attention
        │
   ┌────┴────────────────────┐
   │                         │
   ▼                         ▼
Segmentation Decoder    Global Features
   │                         │
   ▼                         │
Binary Mask Prediksi          │
   │                         │
   ▼                         │
Fitur Morfologi ─────────────┤
                              │
                              ▼
                         AMFM Fusion
                              │
                              ▼
                     Classification Head
                              │
                              ▼
                    Severity (levle0–levle3)
```

### 4.3 Varian Model yang Diuji

| Model | Encoder | Loss Segmentasi | Loss Klasifikasi |
|---|---|---|---|
| Baseline Dilated U-Net | Dilated U-Net | BCE | Cross-Entropy |
| Dilated U-Net + Dice | Dilated U-Net | BCE + Dice | Cross-Entropy |
| ResNet-50 + AMFM | ResNet-50 | BCE | Cross-Entropy |
| Proposal MA-LDS | ResNet-50 | BCE + Dice | MA-LDS + KL |
| Hybrid Staged | ResNet-50 | Focal Loss | Cross-Entropy |

### 4.4 Training Staged (Hybrid Staged)

Strategi dua fase untuk menghindari conflict antara backbone pretrained dan head baru:

```
Phase 1 (freeze backbone):
  - ResNet-50 encoder dibekukan
  - Hanya decoder, attention, AMFM, classifier yang dilatih
  - LR: 1e-3 | Epochs: 15

Phase 2 (full finetune):
  - Seluruh network dibuka (unfreeze)
  - LR lebih kecil: 1e-4 | Epochs: 15
```

---

## 5. Pipeline Eksperimen

### 5.1 Preprocessing Image

```
Gambar original (RGB)
    │
    ▼
Histogram Equalization (per channel R, G, B)
    │
    ▼
Denoising (Median Filter, kernel 3×3)
    │
    ▼
Sharpening (Unsharp Mask, radius=1.6, amount=140%)
    │
    ▼
Resize ke image_size (default: 320×320)
```

Preprocessing disimpan sekali ke disk (`preprocessed_images/`) dan tidak diulang saat training (efisiensi I/O).

### 5.2 Pembuatan Mask Ground Truth

Dua pendekatan mask yang telah diuji:

#### A. Circle Mask (awal — geometris)

```
Anotasi (cx, cy, radius)
    │
    ▼
Isi area lingkaran = 1
Area luar = 0
    │
    ▼
Binary mask (murni geometris)
```

Masalah: tidak mengikuti batas visual lesi, memasukkan kulit normal di sekitar jerawat.

#### B. Refined Color Mask (terbaru — berbasis warna)

```
Gambar original (warna asli)
    │
    ├──► Lab relative redness  (PRIMER — robust kulit gelap)
    ├──► HSV merah absolut     (merah cerah)
    └──► HSV putih/kuning      (kepala pustule bernanah)
    │
    ▼
Gabungkan ketiga channel deteksi
    │
    ▼
Interseksi dengan circle annotation (hanya piksel di dalam radius)
    │
    ▼
Refined mask (lebih natural, mengikuti batas visual lesi)
```

Fallback: jika area merah < 2% dari circle → gunakan circle mask (terjadi pada komedo yang tidak merah).

### 5.3 Augmentasi

Diterapkan hanya pada train set. Image dan mask diaugmentasi **secara bersamaan** agar posisi lesi tetap sinkron.

| Augmentasi | Parameter |
|---|---|
| Rotasi acak | ±15 derajat |
| Horizontal flip | probabilitas 0.5 |
| Brightness jitter | ±15% |
| Contrast jitter | ±15% |

### 5.4 Fungsi Loss

#### Segmentation Loss

| Varian | Formula |
|---|---|
| BCE | `−[y·log(p) + (1−y)·log(1−p)]` |
| BCE + Dice | `λ·BCE + (1−λ)·(1 − 2·TP/(2·TP+FP+FN))` |
| Focal Loss | `−α·(1−p)^γ · log(p)` — lebih kuat untuk class imbalance piksel |

#### Classification Loss

| Varian | Keterangan |
|---|---|
| Cross-Entropy | standar |
| MA-LDS + KL | label smoothing adaptif berbasis morfologi, dipandu KL divergence |

#### Total Loss

```
Total Loss = λ_seg · L_seg + λ_cls · L_cls + λ_morph · L_morph
```

Default weights:
- `λ_seg = 0.6` (dinaikan dari 0.4 → 0.6 untuk mendorong segmentasi)
- `λ_cls = 0.4`
- `λ_morph = 0.2`

### 5.5 Metrik Evaluasi

| Metrik | Task | Keterangan |
|---|---|---|
| **Dice** | Segmentasi | `2·TP / (2·TP + FP + FN)` |
| **IoU** | Segmentasi | `TP / (TP + FP + FN)` |
| **Accuracy** | Klasifikasi | proporsi prediksi benar |
| **Cohen's Kappa** | Klasifikasi | koreksi agreement terhadap kebetulan |

---

## 6. Instalasi

### Prasyarat

- Python 3.10+
- pip / venv

### Setup Environment

```bash
# Buat virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependensi
pip install torch torchvision pillow opencv-python numpy
```

### Verifikasi

```bash
python3 -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

> Di Apple Silicon (M1/M2/M3), gunakan device `mps` untuk akselerasi.

---

## 7. Menjalankan di Google Colab

> **Rekomendasi untuk training** — laptop terlalu berat? Gunakan GPU gratis T4 di Google Colab.  
> Notebook sudah tersedia di: `projectsegmentasi/acne_segmentasi_colab.ipynb`

### 7.1 Persiapan Satu Kali (di Laptop)

**Step 1 — Zip dataset dan script dari terminal:**

```bash
cd "/Users/macbookprom1/Kuliah s2/dataset-acne"

# Zip dataset (gambar + anotasi)
zip -r acne_dataset.zip \
  datasetacne04/acne_1024/small_1024/ \
  acne04v2/Acne04-v2_annotations.json

# Zip semua script Python
zip -r acne_scripts.zip \
  projectsegmentasi/*.py \
  projectsegmentasi/code_dari_claude/*.py
```

**Step 2 — Buat folder di Google Drive:**

```
MyDrive/
└── acne_project/          ← buat folder ini di Drive
    ├── acne_dataset.zip   ← upload file ini
    └── acne_scripts.zip   ← upload file ini
```

Cara upload: buka [drive.google.com](https://drive.google.com) → drag & drop kedua file zip ke folder `acne_project/`.

**Step 3 — Upload notebook ke Colab:**

1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Klik `File → Upload notebook`
3. Pilih file `projectsegmentasi/acne_segmentasi_colab.ipynb`

---

### 7.2 Aktifkan GPU di Colab

Setiap kali membuka notebook baru:

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

Verifikasi GPU aktif dengan menjalankan cell pertama notebook. Output yang benar:

```
CUDA available : True
GPU name       : Tesla T4
VRAM           : 15.0 GB
```

---

### 7.3 Struktur Folder di Google Drive (Setelah Setup)

```
MyDrive/acne_project/
├── acne_dataset.zip
├── acne_scripts.zip
└── output/                          ← dibuat otomatis oleh notebook
    ├── refined_masks_output/
    │   ├── refined_manifest.csv
    │   ├── refined_masks/
    │   ├── preprocessed_images/
    │   └── visualizations/
    ├── training_runs/
    │   └── proposal_hybrid_resnet50_malds/
    │       ├── best_model.pt        ← disimpan otomatis ke Drive
    │       ├── metrics.json
    │       └── last_checkpoint.pt   ← untuk resume jika sesi putus
    └── testing_predictions/
```

> **Penting:** Output training langsung disimpan ke Drive sehingga tidak hilang saat sesi Colab berakhir.

---

### 7.4 Urutan Menjalankan Notebook

Jalankan cell-cell berikut secara berurutan:

| Cell | Fungsi | Waktu |
|---|---|---|
| **Setup GPU & Mount Drive** | Cek GPU + koneksi ke Drive | ~30 detik |
| **Install Dependencies** | Install torch, opencv, dll | ~1 menit |
| **Unzip Dataset** | Ekstrak dataset ke `/content/` | ~3–5 menit |
| **Konfigurasi Path** | Set semua path + validasi | ~5 detik |
| **Generate Refined Mask** | Buat mask dari warna lesi | ~5–15 menit |
| **Training (Option A)** | Train model proposal + MA-LDS | ~2–4 jam (T4) |
| **Inferensi** | Prediksi test set + simpan mask | ~5 menit |
| **Visualisasi** | Plot confusion matrix + kurva | ~1 menit |

---

### 7.5 Perbedaan Setting Lokal vs Colab

| Parameter | Lokal (Mac) | Colab (T4 GPU) |
|---|---|---|
| `--device` | `mps` | `cuda` |
| `--batch-size` | `8` | `16` |
| `--num-workers` | `0` | `4` |
| `--image-size` | `224` | `320` |
| Output dir | folder lokal | `/content/drive/MyDrive/acne_project/output/` |

---

### 7.6 Resume Training Jika Sesi Putus

Sesi Colab gratis bisa putus setelah ~12 jam. Untuk melanjutkan:

1. Buka notebook lagi
2. Jalankan ulang cell **Setup GPU**, **Mount Drive**, **Install Dependencies**, **Konfigurasi Path**
3. Langsung jalankan cell training dengan flag `--resume`:

```python
!python "{TRAIN_SCRIPT_A}" \
  --manifest "{MANIFEST_A}" \
  --output-dir "{OUTPUT_A}" \
  --image-size 320 \
  --batch-size 16 \
  --phase1-epochs 30 \
  --phase2-epochs 90 \
  --device cuda \
  --pretrained \
  --weighted-sampler \
  --resume          # ← tambahkan flag ini
```

Model akan dilanjutkan dari epoch terakhir yang tersimpan di `last_checkpoint.pt`.

---

### 7.7 Connect VS Code ke Google Colab (Opsional)

Ingin coding di VS Code tapi tetap pakai GPU Colab? Ada dua cara:

---

#### Opsi A — SSH Tunnel via ngrok (Rekomendasi)

VS Code di laptop terhubung langsung ke runtime GPU Colab via SSH.

**Langkah 1 — Daftar ngrok (gratis):**
1. Buka [dashboard.ngrok.com](https://dashboard.ngrok.com) → Sign up
2. Salin **Auth Token** dari menu "Your Authtoken"

**Langkah 2 — Jalankan cell ini di Colab (sebelum cell lain):**

```python
# Setup SSH server di Colab
!apt-get install -qq openssh-server -y
!echo 'root:colabpassword' | chpasswd
!echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
!service ssh restart

# Buka tunnel
!pip install pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("ISI_TOKEN_NGROK_KAMU_DI_SINI")
tunnel = ngrok.connect(22, "tcp")
print("✅ SSH Address:", tunnel.public_url)
# Contoh output: tcp://4.tcp.ngrok.io:12345
```

**Langkah 3 — Hubungkan VS Code:**

1. Install extension **Remote - SSH** di VS Code (dari Microsoft)
2. Tekan `Cmd+Shift+P` (Mac) atau `Ctrl+Shift+P` (Windows)
3. Pilih: `Remote-SSH: Connect to Host...`
4. Masukkan address dari output cell di atas, contoh:
   ```
   root@4.tcp.ngrok.io -p 12345
   ```
5. Pilih OS: **Linux**
6. Masukkan password: `colabpassword`
7. VS Code akan terbuka dengan koneksi ke Colab — GPU sudah aktif!

**Langkah 4 — Buka folder project di VS Code:**

Setelah terhubung, buka terminal VS Code (`Ctrl+` `` ` ``) lalu:

```bash
# Mount Drive dulu di Colab (jalankan di cell Colab browser)
# lalu di terminal VS Code:
cd /content/acne_project/projectsegmentasi
```

> **Catatan:** Tunnel ngrok gratis punya batas koneksi. Kalau putus, jalankan ulang cell ngrok dan reconnect VS Code.

---

#### Opsi B — Edit Notebook di VS Code, Training di Browser Colab

Cara lebih sederhana tanpa setup SSH:

1. Install extension **Jupyter** di VS Code
2. Buka file `acne_segmentasi_colab.ipynb` langsung di VS Code
3. Edit cell-cell notebook di VS Code (lebih nyaman dari browser)
4. Simpan perubahan → upload ulang ke Colab → jalankan training di browser

Cocok untuk: edit konfigurasi parameter, baca hasil, debugging ringan.

---

#### Perbandingan Kedua Opsi

| | Opsi A (SSH Tunnel) | Opsi B (Edit Lokal) |
|---|---|---|
| GPU Colab | Ya, langsung | Tidak (harus pindah ke browser) |
| VS Code penuh | Ya | Ya |
| Setup | ~5 menit | Tidak perlu |
| Debugging realtime | Ya | Tidak |
| Cocok untuk | Training & development aktif | Edit parameter saja |

---

### 7.8 Troubleshooting Colab

| Masalah | Solusi |
|---|---|
| `CUDA not available` | Runtime → Change runtime type → GPU |
| `CUDA out of memory` | Kurangi `batch-size` ke `8` atau `image-size` ke `224` |
| Dataset tidak ditemukan | Pastikan zip sudah diunzip: cek cell "Unzip Dataset" |
| Sesi putus saat training | Gunakan flag `--resume`, checkpoint ada di Drive |
| Training sangat lambat | Naikkan `num-workers` ke `4`, atau upgrade ke Colab Pro |
| `ModuleNotFoundError` | Jalankan ulang cell "Install Dependencies" dan "sys.path" |

---

## 8. Cara Menjalankan — Lokal

### Step 1 — Generate Refined Mask (semua data)

```bash
.venv/bin/python projectsegmentasi/refine_masks_color.py \
  --apply-preprocessing \
  --max-images 0 \
  --output-dir projectsegmentasi/refined_masks_output
```

Argumen:
- `--max-images 0` → proses semua gambar (1204)
- `--apply-preprocessing` → simpan gambar preprocessed sekaligus
- `--output-dir` → folder output

Output yang dihasilkan:
- `refined_manifest.csv`
- `refined_masks/` — mask ground truth baru
- `preprocessed_images/` — gambar siap training
- `visualizations/` — overlay 3-panel untuk audit visual

### Step 2 — Training Baseline Dilated U-Net

```bash
.venv/bin/python projectsegmentasi/train_baseline_dilated_unet.py \
  --epochs 20 \
  --batch-size 8 \
  --image-size 320 \
  --augment-train
```

### Step 3 — Training Proposal Hybrid ResNet50 + MA-LDS

```bash
.venv/bin/python projectsegmentasi/train_proposal_hybrid_resnet50_malds.py \
  --manifest projectsegmentasi/acne04v2_preprocessed_all/manifest.csv \
  --phase1-epochs 30 \
  --phase2-epochs 90 \
  --pretrained \
  --weighted-sampler
```

Argumen penting:
- `--pretrained` → gunakan bobot ImageNet untuk ResNet-50
- `--weighted-sampler` → oversample kelas minoritas (levle3)
- `--phase1-epochs` / `--phase2-epochs` → jumlah epoch per fase

### Step 4 — Training Hybrid Staged (terbaru)

```bash
.venv/bin/python projectsegmentasi/code_dari_claude/train_hybrid_resnet50_staged.py \
  --manifest projectsegmentasi/refined_masks_output/refined_manifest.csv \
  --augment-train \
  --pretrained-backbone \
  --epochs-phase1 20 \
  --epochs-phase2 20 \
  --batch-size 16 \
  --focal-alpha 0.75 \
  --focal-gamma 2.0
```

### Step 5 — Inferensi Test Set

```bash
.venv/bin/python projectsegmentasi/infer_testing_images.py \
  --manifest projectsegmentasi/refined_masks_output/refined_manifest.csv \
  --checkpoint projectsegmentasi/baseline_runs/dilated_unet_acne04v2_bce_dice/best_model.pt \
  --output-dir projectsegmentasi/testing_predictions/run_x \
  --split test \
  --image-size 320 \
  --threshold 0.5
```

### Step 6 — Visualisasi Metrik

```bash
.venv/bin/python projectsegmentasi/visualize_baseline_metrics.py
```

---

## 9. Hasil Eksperimen

### 9.1 Rekap Semua Run

| Eksperimen | Encoder | Data | Dice | IoU | Acc | Kappa |
|---|---|---|---|---|---|---|
| Baseline circle mask (100 img) | Dilated U-Net | 100 | 0.0282 | 0.0172 | 0.421 | — |
| ACNE04-v2 baseline (20 epoch) | Dilated U-Net | ~full | ≈0.000 | ≈0.000 | 0.571 | 0.360 |
| + Augmentasi (200 img) | Dilated U-Net | 200 | 0.000 | 0.000 | 0.556 | 0.180 |
| + image_size 320 (200 img) | Dilated U-Net | 200 | 0.000 | 0.000 | 0.444 | 0.133 |
| + BCE + Dice Loss | Dilated U-Net | 200 | 0.0002 | 0.0001 | 0.500 | 0.155 |
| ResNet-50 + AMFM | ResNet-50 | 200 | ≈0.000 | ≈0.000 | 0.500 | 0.066 |
| Preprocessed all + BCE+Dice | Dilated U-Net | ~full | **0.0839** | **0.0463** | **0.598** | 0.001 |
| Refined mask + Focal Loss (172 img) | ResNet-50 | 172 | 0.0133 | 0.0068 | 0.286 | 0.000 |
| **Hybrid Staged (full, 30 epoch)** | **ResNet-50** | **~full** | **0.0133** | **0.0068** | **0.286** | **0.000** |

### 9.2 Run Terbaik per Metrik

| Metrik | Model | Nilai |
|---|---|---|
| Dice tertinggi | Dilated U-Net Preprocessed + BCE+Dice | **0.0839** |
| IoU tertinggi | Dilated U-Net Preprocessed + BCE+Dice | **0.0463** |
| Accuracy tertinggi | Dilated U-Net Preprocessed + BCE+Dice | **0.598** |
| Kappa tertinggi | Dilated U-Net baseline (full) | **0.360** |

### 9.3 Detail Run Hybrid Staged (Terbaru)

Konfigurasi:
- Manifest: `refined_manifest.csv` (refined color mask)
- Backbone: ResNet-50 pretrained (ImageNet)
- Fase 1: 15 epoch, LR=1e-3 (backbone frozen)
- Fase 2: 15 epoch, LR=1e-4 (full finetune)
- Batch size: 8
- Device: Apple MPS

```
Test Metrics:
  loss     : 0.4101
  seg_loss : 0.5077
  cls_loss : 0.2638
  dice     : 0.0133
  iou      : 0.0068
  acc      : 0.2857
  kappa    : 0.0000
```

Tren training (Phase 1):
- Dice train konsisten naik dari **0.0102** (epoch 1) → **0.0146** (epoch 15)
- Val Dice naik dari **0.0074** → **0.0151**
- Klasifikasi belum stabil (kappa sering 0 di val)

---

## 10. Analisis & Temuan

### 10.1 Pola Konsisten

| Cabang | Status | Detail |
|---|---|---|
| Segmentasi | Belum stabil | Dice masih sangat kecil (<0.1), tidak konsisten |
| Klasifikasi | Belajar sebagian | Accuracy bisa sampai 0.60, Kappa masih rendah |
| Hybrid AMFM | Proof-of-concept | Pipeline jalan, kontribusi segmentasi belum kuat |

### 10.2 Root Cause Segmentasi Lemah

1. **Imbalanced pixel** — lesi sangat kecil dibanding background, model cenderung prediksi mask kosong
2. **Target lingkaran kurang natural** — anotasi geometris (circle) tidak mengikuti batas visual lesi yang sesungguhnya
3. **Competing objective** — klasifikasi lebih mudah dipelajari; model mengoptimalkan fitur global, mengabaikan mask
4. **Data terbatas** — terutama untuk levle3 (22 gambar), sulit belajar di pixel level

### 10.3 Solusi yang Sudah Dicoba

| Solusi | Dampak pada Segmentasi |
|---|---|
| Augmentasi (rotasi, flip) | Tidak signifikan |
| Resolusi lebih besar (320) | Tidak signifikan |
| BCE + Dice Loss | Ada sinyal kecil, belum konsisten |
| ResNet-50 backbone | Tidak membantu |
| Refined color mask | Sinyal lebih kuat (Dice dari 0 → 0.013) |
| Focal Loss | Membantu imbalance piksel |
| Staged training | Backbone finetune lebih stabil |

### 10.4 Implikasi untuk Paper

- Pipeline hybrid end-to-end sudah berhasil diimplementasikan
- Kontribusi segmentasi ke klasifikasi **belum terbukti** secara empiris
- Hasil saat ini lebih tepat dibaca sebagai *global classifier dengan segmentasi lemah*
- Untuk klaim hybrid yang kuat, segmentasi harus memiliki Dice > 0.3 minimal

---

## 11. Arah Pengembangan Lanjutan

### Prioritas Tinggi

1. **Visualisasi mask prediksi vs ground truth** — audit apakah prediksi benar-benar kosong atau salah posisi
2. **Training dengan semua 1204 gambar + refined mask + Focal Loss** — uji apakah sinyal Dice bisa bertahan
3. **WeightedRandomSampler** — oversample levle2 dan levle3

### Prioritas Sedang

4. **Naikkan `focal_alpha` ke 0.85** jika Dice stagnan setelah full-data training
5. **Cek `stage_summary.csv` dan `pred_mask.png`** di folder `stage_outputs_test/`
6. **Posforcing segmentasi** — pretrain segmentasi saja dulu, baru latih joint

### Target Metrik Realistis (dengan 1204 data)

| Metrik | Target Minimum |
|---|---|
| Dice | > 0.05 |
| IoU | > 0.03 |
| Accuracy | > 0.40 |
| Kappa | > 0.10 |

---

## File Kunci

| File | Fungsi |
|---|---|
| `acne_segmentasi_colab.ipynb` | Notebook Google Colab — pipeline lengkap siap pakai |
| `refine_masks_color.py` | Generate refined mask + preprocessing |
| `train_proposal_hybrid_resnet50_malds.py` | Model proposal terlengkap (MA-LDS) |
| `code_dari_claude/train_hybrid_resnet50_staged.py` | Training staged utama |
| `infer_testing_images.py` | Inferensi + simpan mask prediksi |
| `refined_masks_output/refined_manifest.csv` | Manifest training terbaru |
| `baseline_runs/experiment_results.csv` | Rekap semua run eksperimen |
| `hasil_eksperimen_segmentasi_klasifikasi.md` | Analisis detail per eksperimen |
| `progress_26_mei_2026.md` | Progress & rencana terbaru |
| `problem_model_hybrid_saat_ini.md` | Identifikasi root cause masalah |

---

## Referensi Teknis

- **ACNE04-v2** — dataset anotasi segmentasi jerawat berbasis lingkaran
- **Dilated U-Net** — U-Net dengan atrous convolutions untuk receptive field lebih luas
- **AMFM** — Adaptive Morphology-aware Feature Fusion Module
- **Focal Loss** — `FL(p) = −α(1−p)^γ log(p)`, robust untuk extreme class imbalance
- **MA-LDS** — Morphology-Aware Label Distribution Smoothing untuk regularisasi klasifikasi
- **Cohen's Kappa** — metrik agreement yang memperhitungkan kebetulan, lebih adil untuk imbalanced data
- **Hayashi Scale** — skala severity jerawat 4 tingkat yang dipakai sebagai target klasifikasi

file acne https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ

import kagglehub

# Download latest version
path = kagglehub.dataset_download("karmagames/acne04-v2")

print("Path to dataset files:", path)


https://github.com/AIpourlapeau/acne04v2