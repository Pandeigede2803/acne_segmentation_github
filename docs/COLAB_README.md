# Panduan Google Colab — Acne Segmentation Project

Pipeline segmentasi jerawat ACNE04-v2 di Google Colab dari nol.

---

## Yang Perlu Diupload dari Laptop

> **Hanya 5 file Python (~130 KB total.)**
> Dataset dan anotasi sudah tersedia online — tidak perlu upload dari laptop.

| File | Dari mana |
|---|---|
| Gambar dataset (1204 jpg) | Kaggle / Google Drive — download langsung di Colab |
| `Acne04-v2_annotations.json` | GitHub — clone langsung di Colab |
| `train_baseline_dilated_unet.py` | Upload dari laptop ke Drive |
| `train_proposal_hybrid_resnet50_malds.py` | Upload dari laptop ke Drive |
| `refine_masks_color.py` | Upload dari laptop ke Drive |
| `infer_testing_images.py` | Upload dari laptop ke Drive |
| `code_dari_claude/train_hybrid_resnet50_staged.py` | Upload dari laptop ke Drive |

---

## Persiapan — Upload Script ke Drive (Lakukan Sekali)

Zip 5 file script dari terminal laptop:

```bash
cd "/Users/macbookprom1/Kuliah s2/dataset-acne"

zip acne_scripts.zip \
  projectsegmentasi/train_baseline_dilated_unet.py \
  projectsegmentasi/train_proposal_hybrid_resnet50_malds.py \
  projectsegmentasi/refine_masks_color.py \
  projectsegmentasi/infer_testing_images.py \
  projectsegmentasi/code_dari_claude/train_hybrid_resnet50_staged.py
```

Upload `acne_scripts.zip` ke Google Drive → folder `MyDrive/acne_project/`.

---

## Step 1 — Buka Colab & Aktifkan GPU

1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Buat notebook baru: **File → New notebook**
3. Aktifkan GPU: **Runtime → Change runtime type → T4 GPU → Save**

Verifikasi:

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "tidak aktif")
```

Output yang benar:
```
CUDA: True
GPU : Tesla T4
```

---

## Step 2 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive terhubung")
```

---

## Step 3 — Install Dependencies

```python
%%capture
!pip install torch torchvision pillow opencv-python-headless \
             numpy matplotlib pandas tqdm kagglehub gdown
print("✅ Semua library terinstall")
```

---

## Step 4 — Download Dataset (Pilih Salah Satu)

### Option A — Dari Kaggle (Rekomendasi, paling lengkap)

```python
import kagglehub

path = kagglehub.dataset_download("karmagames/acne04-v2")
print("✅ Dataset tersimpan di:", path)

# Lihat isi folder
import os
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 2:
        for f in files[:5]:
            print(f'{indent}  {f}')
```

Simpan path ini, akan dipakai di Step 6:

```python
KAGGLE_PATH = path  # simpan untuk dipakai nanti
```

---

### Option B — Dari Google Drive (download folder)

```python
import gdown

# Download folder dataset
gdown.download_folder(
    "https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ",
    output="/content/acne_dataset/",
    quiet=False
)
print("✅ Dataset selesai didownload")
!ls /content/acne_dataset/
```

---

## Step 5 — Clone Anotasi v2 dari GitHub

```python
!git clone https://github.com/AIpourlapeau/acne04v2 /content/acne04v2

# Verifikasi
import json
annot_path = "/content/acne04v2/Acne04-v2_annotations.json"
with open(annot_path) as f:
    data = json.load(f)

imgs   = data.get("images", [])
annots = data.get("annotations", [])
print(f"✅ Anotasi berhasil di-clone")
print(f"   Jumlah gambar  : {len(imgs)}")
print(f"   Jumlah anotasi : {len(annots)}")
```

Output yang benar:
```
✅ Anotasi berhasil di-clone
   Jumlah gambar  : 1204
   Jumlah anotasi : 32443
```

---

## Step 6 — Ekstrak Script dari Drive

```python
!unzip -q "/content/drive/MyDrive/acne_project/acne_scripts.zip" \
       -d "/content/acne_project/"

# Verifikasi
import os
scripts = [
    "/content/acne_project/projectsegmentasi/train_baseline_dilated_unet.py",
    "/content/acne_project/projectsegmentasi/refine_masks_color.py",
    "/content/acne_project/projectsegmentasi/train_proposal_hybrid_resnet50_malds.py",
    "/content/acne_project/projectsegmentasi/infer_testing_images.py",
    "/content/acne_project/projectsegmentasi/code_dari_claude/train_hybrid_resnet50_staged.py",
]
for s in scripts:
    status = "✅" if os.path.exists(s) else "❌"
    print(f"{status} {os.path.basename(s)}")
```

---

## Step 7 — Konfigurasi Path

> **Jalankan cell ini setiap kali membuka sesi Colab baru** (setelah Step 2–6).

```python
import sys
from pathlib import Path

# ── Sesuaikan IMAGE_DIR dengan hasil download di Step 4 ────────────
# Jika pakai Option A (Kaggle):
IMAGE_DIR = Path(KAGGLE_PATH) / "all_1024"   # sesuaikan subfolder
# Jika pakai Option B (Google Drive):
# IMAGE_DIR = Path("/content/acne_dataset/all_1024")

ANNOT_JSON  = Path("/content/acne04v2/Acne04-v2_annotations.json")
SCRIPTS_DIR = Path("/content/acne_project/projectsegmentasi")

# ── Output — simpan ke Drive agar tidak hilang saat sesi berakhir ──
OUTPUT_DIR  = Path("/content/drive/MyDrive/acne_project/output")
MASK_DIR    = OUTPUT_DIR / "refined_masks_output"
TRAIN_DIR   = OUTPUT_DIR / "training_runs"
INFER_DIR   = OUTPUT_DIR / "predictions"

for d in [MASK_DIR, TRAIN_DIR, INFER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Python path ──────────────────────────────────────────────────────
for p in [str(SCRIPTS_DIR), str(SCRIPTS_DIR / "code_dari_claude")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Validasi ─────────────────────────────────────────────────────────
print("=== KONFIGURASI ===")
print(f"Gambar      : {IMAGE_DIR}")
print(f"Anotasi     : {ANNOT_JSON}")
print(f"Scripts     : {SCRIPTS_DIR}")
print(f"Output      : {OUTPUT_DIR}")
print(f"Device      : {DEVICE}")
print()

errors = []
if not IMAGE_DIR.exists():   errors.append(f"❌ Folder gambar tidak ada: {IMAGE_DIR}")
if not ANNOT_JSON.exists():  errors.append(f"❌ JSON anotasi tidak ada: {ANNOT_JSON}")
if not SCRIPTS_DIR.exists(): errors.append(f"❌ Scripts tidak ada: {SCRIPTS_DIR}")

if errors:
    for e in errors: print(e)
else:
    n = len(list(IMAGE_DIR.glob("*.jpg")))
    print(f"✅ Semua path valid | {n} gambar ditemukan")
```

---

## Step 8 — Generate Refined Mask

Buat mask ground truth berbasis deteksi warna lesi.
**Jalankan sekali saja** — hasil disimpan ke Drive, tidak perlu diulang.

```python
REFINED_MANIFEST = MASK_DIR / "refined_manifest.csv"

!python "{SCRIPTS_DIR}/refine_masks_color.py" \
  --image-dir       "{IMAGE_DIR}" \
  --annotation-json "{ANNOT_JSON}" \
  --output-dir      "{MASK_DIR}" \
  --apply-preprocessing \
  --max-images 0

# Verifikasi
import pandas as pd
if REFINED_MANIFEST.exists():
    df = pd.read_csv(REFINED_MANIFEST)
    print(f"\n✅ Manifest dibuat: {len(df)} baris")
    print(df['split'].value_counts().to_string())
else:
    print("❌ refined_manifest.csv tidak ditemukan. Cek error di atas.")
```

Output yang benar:
```
✅ Manifest dibuat: 1204 baris
split
train    842
val      181
test     181
```

---

## Step 9 — Training

> Estimasi waktu: **2–3 jam** di GPU T4 untuk 30+90 epoch.
> Checkpoint disimpan ke Drive tiap epoch — aman kalau sesi putus.

```python
OUTPUT_RUN = TRAIN_DIR / "proposal_hybrid_v1"

!python "{SCRIPTS_DIR}/train_proposal_hybrid_resnet50_malds.py" \
  --manifest        "{REFINED_MANIFEST}" \
  --output-dir      "{OUTPUT_RUN}" \
  --image-size      320 \
  --batch-size      16 \
  --phase1-epochs   30 \
  --phase2-epochs   90 \
  --device          cuda \
  --pretrained \
  --weighted-sampler \
  --num-workers     4 \
  --seed            42
```

### Monitor progress (buka di tab Colab terpisah):

```python
import json, matplotlib.pyplot as plt

metrics_path = OUTPUT_RUN / "metrics.json"
if not metrics_path.exists():
    print("Training belum mulai.")
else:
    with open(metrics_path) as f:
        data = json.load(f)
    h = data.get("history", [])
    if h:
        epochs     = [x["epoch"] for x in h]
        train_dice = [x.get("train_dice", 0) for x in h]
        val_dice   = [x.get("val_dice", 0) for x in h]
        train_acc  = [x.get("train_acc", 0) for x in h]
        val_acc    = [x.get("val_acc", 0) for x in h]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, train_dice, label="Train Dice", color="green")
        ax1.plot(epochs, val_dice,   label="Val Dice",   color="red")
        ax1.set_title("Dice Score"); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(epochs, train_acc, label="Train Acc",  color="steelblue")
        ax2.plot(epochs, val_acc,   label="Val Acc",    color="orange")
        ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

        last = h[-1]
        plt.suptitle(
            f"Epoch {last['epoch']} | Val Dice: {last.get('val_dice',0):.4f} "
            f"| Val Acc: {last.get('val_acc',0):.4f}"
        )
        plt.tight_layout(); plt.show()
```

---

## Step 10 — Resume Jika Sesi Colab Putus

1. Jalankan ulang Step 1 (GPU) → Step 2 (Drive) → Step 3 (install) → Step 5 (clone GitHub) → Step 6 (unzip script) → Step 7 (path)
2. Jalankan cell ini:

```python
!python "{SCRIPTS_DIR}/train_proposal_hybrid_resnet50_malds.py" \
  --manifest        "{REFINED_MANIFEST}" \
  --output-dir      "{OUTPUT_RUN}" \
  --image-size      320 \
  --batch-size      16 \
  --phase1-epochs   30 \
  --phase2-epochs   90 \
  --device          cuda \
  --pretrained \
  --weighted-sampler \
  --num-workers     4 \
  --resume
```

> Flag `--resume` memuat `last_checkpoint.pt` dari `OUTPUT_RUN/` dan melanjutkan dari epoch terakhir.

---

## Step 11 — Lihat Hasil Test Metrics

```python
metrics_file = OUTPUT_RUN / "metrics.json"
if not metrics_file.exists():
    print("Training belum selesai.")
else:
    with open(metrics_file) as f:
        data = json.load(f)
    tm = data.get("test_metrics", {})
    print("=" * 40)
    print("     HASIL TEST SET")
    print("=" * 40)
    print(f"  Accuracy    : {tm.get('acc',   0):.4f}")
    print(f"  Kappa       : {tm.get('kappa', 0):.4f}")
    print(f"  Dice Score  : {tm.get('dice',  0):.4f}")
    print(f"  IoU         : {tm.get('iou',   0):.4f}")
    print("-" * 40)
    print(f"  Seg Loss    : {tm.get('seg_loss', 0):.4f}")
    print(f"  Cls Loss    : {tm.get('cls_loss', 0):.4f}")
    print("=" * 40)
    print(f"  Best Val Loss: {data.get('best_val_loss', 0):.4f}")
```

---

## Step 12 — Inferensi & Visualisasi

```python
CHECKPOINT = OUTPUT_RUN / "best_model.pt"
INFER_OUT  = INFER_DIR / "proposal_hybrid_v1"

if not CHECKPOINT.exists():
    print("❌ Training belum selesai.")
else:
    # Inferensi
    !python "{SCRIPTS_DIR}/infer_testing_images.py" \
      --manifest    "{REFINED_MANIFEST}" \
      --checkpoint  "{CHECKPOINT}" \
      --output-dir  "{INFER_OUT}" \
      --split       test \
      --image-size  320 \
      --threshold   0.5 \
      --device      cuda

    # Tampilkan 8 contoh mask prediksi
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    pred_files = sorted(INFER_OUT.glob("*.png"))[:8]
    if pred_files:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, pf in zip(axes.flatten(), pred_files):
            ax.imshow(PILImage.open(pf))
            ax.set_title(pf.stem[:20], fontsize=8)
            ax.axis("off")
        plt.suptitle("Predicted Segmentation Masks — Test Set")
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "predicted_masks.png"), dpi=100)
        plt.show()
```

---

## Ringkasan Sesi Baru Colab

Setiap kali membuka Colab baru, cukup jalankan urutan ini (±5 menit):

```
Step 1  → Aktifkan GPU
Step 2  → Mount Drive
Step 3  → Install dependencies
Step 5  → Clone GitHub (anotasi)
Step 6  → Unzip scripts dari Drive
Step 7  → Konfigurasi path
```

Lalu lanjut ke Step 9 dengan `--resume` jika training belum selesai,
atau Step 11–12 jika training sudah selesai.

---

## Struktur Output di Drive

```
MyDrive/acne_project/
├── acne_scripts.zip                          ← upload sekali dari laptop
└── output/
    ├── refined_masks_output/
    │   ├── refined_manifest.csv              ← manifest training
    │   ├── refined_masks/                    ← mask ground truth
    │   ├── preprocessed_images/              ← gambar siap input model
    │   └── visualizations/                   ← overlay 3-panel
    ├── training_runs/
    │   └── proposal_hybrid_v1/
    │       ├── best_model.pt                 ← model terbaik
    │       ├── last_checkpoint.pt            ← untuk resume
    │       └── metrics.json                  ← history training
    └── predictions/
        └── proposal_hybrid_v1/
            └── *.png                         ← mask prediksi test set
```

---

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `CUDA not available` | Runtime → Change runtime type → GPU |
| `CUDA out of memory` | Kurangi `--batch-size` ke `8` |
| Kaggle download error | Tambahkan Kaggle API key: `kagglehub.login()` |
| `gdown` folder error | Pastikan link Drive bersifat publik (Anyone with link) |
| GitHub clone gagal | Cek koneksi internet Colab, coba ulang cell |
| Script tidak ditemukan | Jalankan ulang Step 6 (unzip dari Drive) |
| Sesi putus saat training | Jalankan Step 1-2-3-5-6-7 lalu training dengan `--resume` |
| Dice masih 0 | Naikkan `--seg-weight` ke `1.5`, atau `--weighted-sampler` |

---

## Sumber Dataset

| Sumber | Link | Isi |
|---|---|---|
| Kaggle | `karmagames/acne04-v2` | Gambar + anotasi lengkap |
| Google Drive | [folder link](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ) | Gambar |
| GitHub | [AIpourlapeau/acne04v2](https://github.com/AIpourlapeau/acne04v2) | Anotasi v2 JSON |
