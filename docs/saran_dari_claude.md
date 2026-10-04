# Improvement Plan — Model Hybrid Segmentasi & Klasifikasi Acne
**Peneliti:** Pande I Gede Sudiahna | **Program:** S2 Sistem Informasi, STIKOM Bali  
**Status saat ini:** Val Accuracy 48.15% | Val Kappa 0.05 | Val Dice 0.4555  
**Target:** MAE < 2.93 (lampaui Prokhorov 2024) | Kappa > 0.55 | Accuracy 75–82%

---

## Kondisi Masalah Saat Ini

| Masalah | Indikator | Dampak |
|---|---|---|
| Class imbalance parah | Val Kappa ~0.05 meski Acc 48% | Model hanya hafal kelas mayoritas |
| MA-LDS belum aktif penuh | Loss masih CrossEntropy biasa | Novelty utama belum berkontribusi |
| Segmentasi belum stabil | Dice 0.45, kadang masih 0.0 | Fitur morfologi ke AMFM lemah |
| Split data tidak stratified | Gap train-val besar | Distribusi kelas tidak merata di val |
| Focal loss belum dipakai | BCE standar untuk segmentasi | Piksel lesi kecil kalah dominansi background |

---

## Prioritas 1 — Stratified Split + Weighted Loss
> **Dampak perkiraan: Kappa 0.05 → 0.35–0.45**  
> Harus dilakukan pertama sebelum improvement lain apapun.

### 1.1 Stratified Split
Pastikan proporsi setiap kelas severity (0–3) sama di train, val, dan test set.

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    images, labels,
    test_size=0.2,
    stratify=labels,   # <-- kunci utama
    random_state=42
)
```

**Kenapa penting:** Kalau split tidak stratified, val set bisa kebetulan penuh kelas 0 saja. Model yang selalu jawab "level 0" otomatis dapat accuracy tinggi tanpa belajar apapun — ini yang menyebabkan Kappa flat di 0.05.

### 1.2 Weighted Loss untuk Klasifikasi
Beri bobot lebih besar ke kelas minoritas (level 2 dan 3).

```python
from sklearn.utils.class_weight import compute_class_weight
import torch

weights = compute_class_weight(
    'balanced',
    classes=[0, 1, 2, 3],
    y=train_labels
)
class_weights = torch.FloatTensor(weights).to(device)

criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
```

### 1.3 WeightedRandomSampler (opsional, kombinasikan dengan weighted loss)
Paksa dataloader ambil kelas minoritas lebih sering saat training.

```python
from torch.utils.data import WeightedRandomSampler

sample_weights = [weights[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Hapus shuffle=True jika pakai sampler
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

---

## Prioritas 2 — Aktifkan MA-LDS Sepenuhnya
> **Dampak perkiraan: MAE turun 0.3–0.8 | Ini novelty utama penelitian**  
> Berdasarkan grafik training, loss masih menggunakan CrossEntropy biasa — MA-LDS belum aktif.

### 2.1 Ganti ke KL Divergence Loss
```python
criterion_cls = nn.KLDivLoss(reduction='batchmean')
```

### 2.2 Generate Distribusi Label Berbasis Morfologi (MA-LDS)
Radius rata-rata lesi mempengaruhi sigma distribusi Gaussian — lesi lebih besar atau lebih banyak berarti lebih uncertain di batas kelas.

```python
import numpy as np

def generate_morph_aware_distribution(label, avg_radius, num_classes=4):
    """
    label      : kelas severity ground truth (0-3)
    avg_radius : rata-rata radius lesi dari prediksi segmentasi
    """
    # Radius besar = lebih uncertain = sigma lebih lebar
    sigma = 0.5 + (avg_radius / 50.0)
    
    dist = np.exp(
        -0.5 * ((np.arange(num_classes) - label) / sigma) ** 2
    )
    return dist / dist.sum()   # normalisasi jadi probabilitas
```

### 2.3 Hitung Loss dengan Distribusi Morfologi
```python
# Dalam training loop
morph_dist = generate_morph_aware_distribution(
    label=y_true,
    avg_radius=avg_radius_from_segmentation
)
morph_dist_tensor = torch.FloatTensor(morph_dist).to(device)

log_pred = F.log_softmax(cls_output, dim=-1)
loss_cls = criterion_cls(log_pred, morph_dist_tensor)
```

**Kenapa ini kuat:** Prokhorov 2024 pakai LDS standar tanpa informasi morfologi. MA-LDS kamu memasukkan radius eksplisit dari anotasi ACNE04-v2 ke dalam distribusi label — ini yang membedakan dan menjadi kontribusi nyata tesismu.

---

## Prioritas 3 — Focal Loss untuk Segmentasi
> **Dampak perkiraan: Dice 0.45 → 0.55–0.65**  
> Mengatasi dominasi background piksel yang menyebabkan segmentasi sering prediksi mask kosong.

### 3.1 Implementasi Focal Loss
```python
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
```

### 3.2 Kombinasikan Dice Loss + Focal Loss
Jangan pakai BCE biasa — kombinasi ini jauh lebih efektif untuk segmentasi area kecil.

```python
focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

def segmentation_loss(pred_mask, gt_mask):
    dice = dice_loss(pred_mask, gt_mask)
    focal = focal_loss_fn(pred_mask, gt_mask)
    return 0.5 * dice + 0.5 * focal   # bobot bisa dituning
```

**Kenapa penting:** Lesi jerawat sangat kecil dibanding area wajah. Dengan BCE biasa, model cukup prediksi "semua background" dan loss-nya sudah rendah. Focal Loss memaksa model memperhatikan piksel lesi yang sulit.

---

## Prioritas 4 — Augmentasi Selektif Kelas Minoritas
> **Dampak perkiraan: Accuracy +3–5% | Mengurangi overfitting klasifikasi**

```python
from torchvision import transforms

augment_minority = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),
])

# Terapkan hanya untuk label severity 2 dan 3
def get_transform(label):
    if label in [2, 3]:   # kelas minoritas
        return augment_minority
    return transform_standar
```

---

## Prioritas 5 — Tuning Kecil yang Impactful
> **Dampak perkiraan: Accuracy +2–4% tambahan**

### 5.1 Learning Rate Scheduler — Cosine Annealing
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=90,        # sesuai epoch phase 2
    eta_min=1e-6
)
```

### 5.2 Label Smoothing Ringan
Membantu karena batas antar kelas acne severity memang ambigu secara visual.

```python
criterion_cls = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)
```

### 5.3 Test-Time Augmentation (TTA)
Boost accuracy 2–3% tanpa mengubah arsitektur apapun.

```python
def tta_predict(model, image, n_aug=5):
    preds = []
    for _ in range(n_aug):
        aug_img = random_augment(image)
        with torch.no_grad():
            preds.append(model(aug_img))
    return torch.stack(preds).mean(dim=0)
```

### 5.4 Gradient Clipping
Mencegah exploding gradient yang bisa destabilisasi training phase 2.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Urutan Implementasi yang Disarankan

```
Minggu 1:  Prioritas 1 — Stratified split + Weighted loss
           → Cek apakah Val Kappa naik ke >0.30

Minggu 2:  Prioritas 2 — Aktifkan MA-LDS penuh dengan KL Divergence
           → Cek apakah MAE mulai turun di bawah 3.0

Minggu 3:  Prioritas 3 — Ganti segmentation loss ke Dice + Focal
           → Cek apakah Dice naik ke >0.55

Minggu 4:  Prioritas 4 + 5 — Augmentasi + tuning kecil
           → Cek apakah Accuracy naik ke >70%
```

**Jangan implementasi semua sekaligus** — sulit mengetahui mana yang efektif jika semua diganti bersamaan.

---

## Proyeksi Performa Setelah Improvement

| Tahap | Accuracy | MAE | Kappa | Dice |
|---|---|---|---|---|
| Sekarang (baseline) | 48% | ~3.8 | 0.05 | 0.45 |
| Setelah Prioritas 1 | ~62–68% | ~3.5 | ~0.35–0.45 | 0.45 |
| Setelah Prioritas 2 | ~70–76% | ~2.5–2.8 | ~0.50–0.58 | 0.45 |
| Setelah Prioritas 3 | ~70–76% | ~2.3–2.6 | ~0.50–0.58 | ~0.58 |
| Setelah Prioritas 4+5 | ~75–82% | ~2.0–2.5 | ~0.55–0.65 | ~0.60 |
| **Target Prokhorov 2024** | **>84%** | **<2.93** | — | — |

---

## Klaim yang Bisa Dipertahankan di Sidang Tesis

Jika MAE < 2.93 dan Kappa > 0.55 tercapai:

> *"Model hybrid MA-LDS yang diusulkan berhasil melampaui baseline Prokhorov & Kalinin (2024) dalam metrik MAE, dengan membuktikan bahwa integrasi fitur morfologi lesi dari cabang segmentasi ke dalam proses label distribution smoothing memberikan kontribusi positif terhadap stabilitas prediksi severity acne."*

Ini klaim yang **jujur, terukur, dan cukup kuat untuk tesis S2.**

---

*Dokumen ini dibuat berdasarkan analisis hasil training epoch 120 (Val Dice: 0.4555, Val Acc: 0.4815) dan proposal tesis Pande I Gede Sudiahna, STIKOM Bali 2026.*