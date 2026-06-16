# Acne Lesion Segmentation & Severity Classification

Hybrid pipeline untuk segmentasi lesi jerawat dan klasifikasi severity menggunakan **ACNE04-v2**.

> Jalankan di Google Colab — buka notebook [`acne_colab.ipynb`](acne_colab.ipynb)

---

## Arsitektur

```
Input Image
    ↓
ResNet-50 Encoder + Dilated Decoder
    ↓
Spatial Attention
    ↙            ↘
Segmentation     Global Features
Decoder               │
    ↓                 │
Binary Mask           │
    ↓                 │
Morphological ────────┘
Features
    ↓
AMFM Fusion
    ↓
Classification Head → Severity (levle0–levle3)
```

Model: **ResNet-50 + Dilated U-Net Decoder + AMFM + MA-LDS Loss**  
Training: dua fase — backbone frozen (Phase 1) → full finetune (Phase 2)

---

## Dataset

| Sumber | Link | Isi |
|---|---|---|
| Kaggle | [`karmagames/acne04-v2`](https://www.kaggle.com/datasets/karmagames/acne04-v2) | Gambar + anotasi |
| Google Drive | [folder](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ) | Gambar |
| GitHub | [AIpourlapeau/acne04v2](https://github.com/AIpourlapeau/acne04v2) | Anotasi v2 JSON |

- **1204 gambar** wajah resolusi 1024×1024
- **32.443 anotasi** lingkaran lesi (`cx, cy, radius`)
- 4 kelas severity: `levle0` (ringan) → `levle3` (sangat berat)

---

## Struktur Repo

```
acne_segmentation_github/
├── README.md
├── acne_colab.ipynb              ← notebook utama — jalankan di Colab
├── .gitignore
└── scripts/
    ├── train_baseline_dilated_unet.py         ← base module (diimport script lain)
    ├── train_proposal_hybrid_resnet50_malds.py ← training utama
    ├── refine_masks_color.py                  ← generate refined mask
    ├── infer_testing_images.py                ← inferensi test set
    └── code_dari_claude/
        └── train_hybrid_resnet50_staged.py    ← training staged (alternatif)
```

---

## Cara Pakai di Google Colab

1. Buka [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub** → masukkan URL repo ini
3. Pilih `acne_colab.ipynb`
4. **Runtime → Change runtime type → T4 GPU**
5. Jalankan cell dari atas ke bawah

Lihat panduan lengkap di [`COLAB_GUIDE.md`](COLAB_GUIDE.md).

---

## Cara Update Script (Push & Pull)

```bash
# Di laptop — edit script, lalu push
git add scripts/
git commit -m "update: ..."
git push

# Di Colab — pull perubahan terbaru
!git -C /content/acne-seg pull
```

---

## Hasil Eksperimen

| Model | Dice | IoU | Accuracy | Kappa |
|---|---|---|---|---|
| Dilated U-Net baseline | ≈0.000 | ≈0.000 | 0.571 | 0.360 |
| + BCE + Dice Loss | 0.0002 | 0.0001 | 0.500 | 0.155 |
| Preprocessed + BCE+Dice | **0.084** | **0.046** | **0.598** | 0.001 |
| ResNet-50 + AMFM | ≈0.000 | ≈0.000 | 0.500 | 0.066 |
| Hybrid Staged (terbaru) | 0.013 | 0.007 | 0.286 | 0.000 |

---

## Referensi

- Wu et al., *Joint Acne Image Grading and Counting via Label Distribution Learning*, ICCV 2019
- Gazeau et al., *AcneAI: A New Acne Severity Assessment Method*, MICCAI 2024
