# Progress & Rencana Lanjutan — 26 Mei 2026

## Apa yang Sudah Dikerjakan Hari Ini

### 1. Identifikasi Root Cause Segmentasi Gagal
- Ground truth mask sebelumnya dibuat dari lingkaran geometris murni (`cx, cy, radius`)
- Mask tidak mencerminkan penampakan visual lesi → model bingung
- Jerawat bernanah (pustule) tidak tertangkap karena kepala putih/kuning
- Jerawat merah gelap di kulit cokelat tidak tertangkap karena HSV threshold absolut

### 2. Script Baru: `refine_masks_color.py`
**Lokasi:** `projectsegmentasi/refine_masks_color.py`

Pendekatan deteksi lesi:
- **Lab relative redness** (PRIMER) — threshold relatif terhadap tone kulit per gambar, robust untuk kulit gelap
- **HSV merah absolut** — untuk merah cerah
- **HSV putih/kuning** — untuk kepala pustule bernanah
- Interseksi dengan circle annotation → hanya piksel lesi yang masuk

Output yang dihasilkan:
- `refined_masks/` — mask ground truth baru
- `preprocessed_images/` — gambar preprocessed (equalize + denoise + sharpen)
- `visualizations/` — overlay mask di atas gambar asli (3 panel)
- `refined_manifest.csv` — manifest lengkap dengan split, label_index, semua path

### 3. Update Training Script
**Lokasi:** `projectsegmentasi/code_dari_claude/train_hybrid_resnet50_staged.py`

Perubahan:
- Baca `refined_manifest.csv` (format baru) — tidak perlu baca annotations.json lagi
- Gunakan `preprocessed_path` jika tersedia, fallback ke `image_path` + on-the-fly preprocessing
- **Focal Loss** menggantikan BCE biasa — lebih robust untuk extreme pixel imbalance
- lambda_seg dinaikkan dari 0.4 → **0.6** (segmentasi lebih diprioritaskan)
- lambda_cls turun dari 0.6 → **0.4**

### 4. Hasil Eksperimen Hari Ini

| Eksperimen | Dice | IoU | Acc | Kappa |
|---|---|---|---|---|
| Baseline (circle mask + BCE) | 0.0000 | 0.0000 | 0.38–0.55 | ~0 |
| Refined mask + pos_weight=10, 80 data | 0.0014 | — | 0.25 | 0 |
| Refined mask + Focal Loss, 172 data | **0.0133** | **0.0068** | 0.2857 | 0 |

**Segmentasi mulai belajar** — Dice naik 10x dari 0 ke 0.0133.
Kappa masih 0 karena data terlalu sedikit (172 gambar, ~6 per kelas di test set).

---

## Yang Harus Dilakukan Besok

### Step 1 — Generate Mask untuk Semua Data (1204 gambar)
```bash
.venv/bin/python projectsegmentasi/refine_masks_color.py \
  --apply-preprocessing \
  --max-images 0 \
  --output-dir projectsegmentasi/refined_masks_output
```

Estimasi waktu: beberapa menit.
Hasil: `refined_manifest.csv` dengan ~1204 gambar, test set ~180 gambar (~45 per kelas).

### Step 2 — Training Ulang dengan Data Lengkap
```bash
.venv/bin/python projectsegmentasi/code_dari_claude/train_hybrid_resnet50_staged.py \
  --augment-train \
  --pretrained-backbone \
  --epochs-phase1 20 \
  --epochs-phase2 20 \
  --batch-size 16 \
  --focal-alpha 0.75 \
  --focal-gamma 2.0
```

### Step 3 — Evaluasi Hasil
Target yang realistis dengan 1204 data:

| Metrik | Target |
|---|---|
| Dice | > 0.05 |
| IoU | > 0.03 |
| Acc | > 0.40 |
| Kappa | > 0.10 |

Kalau Dice masih stagnan setelah training:
- Cek `pred_mask.png` di `stage_outputs_test/`
- Naikkan `--focal-alpha` ke 0.85
- Cek kualitas mask di `visualizations/` untuk kelas dengan performa buruk

---

## Catatan Teknis Penting

### Kenapa Tidak Preprocessing di Training Script
Preprocessing (equalize + denoise + sharpen) dilakukan **sekali saja** saat generate mask dan disimpan ke `preprocessed_images/`. Tidak diulang saat training untuk efisiensi.

### Fallback Mask
13–59 gambar menggunakan **fallback ke circle mask** (area merah < 2% circle). Ini terjadi pada jerawat yang warnanya tidak cukup merah (misal komedo). Masih acceptable.

### Distribusi Data per Level
```
levle0: 470 gambar beranotasi
levle1: 579 gambar beranotasi
levle2: 133 gambar beranotasi
levle3:  22 gambar beranotasi  ← paling sedikit, bottleneck
Total : 1204 gambar
```

Imbalance kelas sudah dikompensasi oleh Focal Loss (`focal_alpha=0.75`).

### File Penting
| File | Fungsi |
|---|---|
| `projectsegmentasi/refine_masks_color.py` | Generate refined mask + preprocessing |
| `projectsegmentasi/check_annotations.py` | Cek jumlah gambar beranotasi per level |
| `code_dari_claude/train_hybrid_resnet50_staged.py` | Training script utama |
| `refined_masks_output/refined_manifest.csv` | Manifest training terbaru |
| `refined_masks_output/visualizations/` | Hasil visualisasi mask per level |
