# Panduan Google Colab — Acne Segmentation Project

Pipeline segmentasi jerawat ACNE04-v2 di Google Colab dari nol.

---

## Yang Perlu Disiapkan

Dataset diambil langsung dari folder Google Drive publik:

```text
https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ
```

Folder Drive tersebut berisi archive dataset, misalnya:

```text
Classification.tar  ← gambar dataset
Detection.tar       ← annotation XML VOC
```

Yang perlu kamu upload sendiri hanya script project ke `MyDrive/acne_project/`.

| File | Dari mana |
|---|---|
| `acne_scripts.zip` | Berisi semua script Python project |

---

## Persiapan — Upload Script ke Drive (Lakukan Sekali)

Zip script dari terminal laptop:

```bash
cd "/Users/macbookprom1/Kuliah s2/dataset-acne"

zip acne_scripts.zip \
  projectsegmentasi/train_baseline_dilated_unet.py \
  projectsegmentasi/train_proposal_hybrid_resnet50_malds.py \
  projectsegmentasi/refine_masks_color.py \
  projectsegmentasi/infer_testing_images.py \
  projectsegmentasi/code_dari_claude/train_hybrid_resnet50_staged.py
```

Upload file ini ke Google Drive:

```text
MyDrive/acne_project/
└── acne_scripts.zip
```

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
             numpy matplotlib pandas tqdm scikit-learn gdown
print("✅ Semua library terinstall")
```

## Step 4 — Download Dataset dari Folder Google Drive

```python
from pathlib import Path
import gdown

DOWNLOAD_DIR = Path("/content/acne_images")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DRIVE_FOLDER = "https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ"

gdown.download_folder(
    DATASET_DRIVE_FOLDER,
    output=str(DOWNLOAD_DIR),
    quiet=False,
    use_cookies=False,
)

print("✅ Dataset selesai didownload ke", DOWNLOAD_DIR)
```

Auto-detect folder gambar dan annotation JSON/XML:

```python
def find_image_dir(root):
    best_dir, best_count = None, 0
    for d in root.rglob("*"):
        if d.is_dir():
            count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.JPG"))) + \
                    len(list(d.glob("*.jpeg"))) + len(list(d.glob("*.JPEG")))
            if count > best_count:
                best_dir, best_count = d, count
    return best_dir

def find_annotation_path(root):
    json_candidates = list(root.rglob("*.json"))
    for c in json_candidates:
        if "annot" in c.name.lower() or "acne04" in c.name.lower():
            return c
    if json_candidates:
        return json_candidates[0]

    xml_dirs = []
    for d in root.rglob("*"):
        if d.is_dir():
            n_xml = len(list(d.glob("*.xml")))
            if n_xml > 0:
                xml_dirs.append((n_xml, d))
    if xml_dirs:
        xml_dirs.sort(reverse=True)
        return xml_dirs[0][1]
    return None

IMAGE_DIR = find_image_dir(DOWNLOAD_DIR)
ANNOT_JSON = find_annotation_path(DOWNLOAD_DIR)

print("Image dir :", IMAGE_DIR)
print("Annotation:", ANNOT_JSON)
```

---

## Step 5 — Ekstrak Script dari Drive

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

## Step 6 — Konfigurasi Path

> **Jalankan cell ini setiap kali membuka sesi Colab baru** setelah Drive, dependencies, dataset, dan script siap.

```python
import sys
from pathlib import Path

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

## Step 7 — Generate Refined Mask

Buat mask ground truth berbasis deteksi warna lesi.
**Jalankan sekali saja** — hasil disimpan ke Drive, tidak perlu diulang.

```python
REFINED_MANIFEST = MASK_DIR / "refined_manifest.csv"

!python "{SCRIPTS_DIR}/refine_masks_color.py" \
  --image-dir       "{IMAGE_DIR}" \
  --annotations-json "{ANNOT_JSON}" \
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

## Step 8 — Training

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

## Step 9 — Resume Jika Sesi Colab Putus

1. Jalankan ulang Step 1 (GPU) → Step 2 (Drive) → Step 3 (install) → Step 4 (download dataset) → Step 5 (script dari Drive) → Step 6 (path)
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

## Step 10 — Lihat Hasil Test Metrics

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

## Step 11 — Inferensi & Visualisasi

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
Step 4  → Download dataset dari folder Drive
Step 5  → Ekstrak scripts dari Drive
Step 6  → Konfigurasi path
```

Lalu lanjut ke Step 9 dengan `--resume` jika training belum selesai,
atau Step 10–11 jika training sudah selesai.

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
| Dataset tidak terdownload | Pastikan link folder Drive bisa diakses publik, lalu jalankan ulang Step 4 |
| Script tidak ditemukan | Jalankan ulang Step 5 (ekstrak script dari Drive) |
| Sesi putus saat training | Jalankan Step 1-2-3-4-5-6 lalu training dengan `--resume` |
| Dice masih 0 | Naikkan `--seg-weight` ke `1.5`, atau `--weighted-sampler` |

---

## Sumber Dataset

| Sumber | Link | Isi |
|---|---|---|
| Google Drive folder | `18yJcHXhzOv7H89t-Lda6phheAicLqMuZ` | Gambar + anotasi JSON |
