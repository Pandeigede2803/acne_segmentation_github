# Eksperimen Class Weighting di Google Colab

Tanggal dokumentasi: **4 Oktober 2026** (Asia/Makassar).
Status: script siap diuji; hasil eksperimen baru belum tersedia.

## Urutan yang Harus Dijalankan

1. Buka `acne_colab.ipynb` di Google Colab dan pilih runtime GPU.
2. Jalankan cell setup: dependency, mount Drive, penyediaan script, dan konfigurasi path.
3. Ganti file `train_proposal_hybrid_resnet50_malds.py` di folder `SCRIPTS_DIR`
   dengan versi terbaru dari repo ini. Lakukan setelah proses ekstraksi ZIP
   agar script terbaru tidak tertimpa script lama.
4. Jalankan konfigurasi path dan validasi dataset. Pastikan `TRAIN_SCRIPT_A`,
   `REFINED_MANIFEST`, dan `TRAIN_DIR` mengarah ke lokasi yang benar.
5. Gunakan `refined_manifest.csv` yang sama untuk kedua run. Jika belum ada,
   jalankan cell Generate Refined Mask (bagian 10) sekali. Pastikan path gambar
   dan mask di manifest tersedia di sesi Colab saat ini.
6. Jalankan cell Cek Kesiapan di bawah.
7. Jalankan Cell Training dengan `CLASS_WEIGHTING = "none"` sampai selesai.
8. Ubah hanya `CLASS_WEIGHTING = "balanced"`, lalu jalankan Cell Training lagi.
9. Jalankan Cell Perbandingan Hasil setelah kedua training selesai.

Gunakan cell training dalam dokumen ini sebagai pengganti bagian 11 notebook.
Training otomatis melakukan validasi setiap epoch dan evaluasi test setelah
memuat model terbaik; tidak perlu menjalankan script test terpisah untuk
mendapatkan empat metrik utama.

## Cell Cek Kesiapan

```python
import subprocess
import sys
import torch

assert torch.cuda.is_available(), "Aktifkan runtime GPU di Colab."
assert TRAIN_SCRIPT_A.exists(), f"Script tidak ditemukan: {TRAIN_SCRIPT_A}"
assert REFINED_MANIFEST.exists(), f"Manifest tidak ditemukan: {REFINED_MANIFEST}"
help_result = subprocess.run(
    [sys.executable, str(TRAIN_SCRIPT_A), "--help"],
    check=True, capture_output=True, text=True,
)
assert "--class-weighting" in help_result.stdout, "Perbarui script training di Colab."
print("GPU:", torch.cuda.get_device_name(0))
print("Script:", TRAIN_SCRIPT_A)
print("Manifest:", REFINED_MANIFEST)
print("Output:", TRAIN_DIR)
```

## Konfigurasi Eksperimen

| Parameter | Nilai |
|---|---|
| Tanggal/ID eksperimen | `2026-10-04` |
| Model | Proposal Hybrid ResNet-50 + AMFM + MA-LDS |
| Phase 1 | 30 epoch, backbone dibekukan |
| Phase 2 | 90 epoch, fine-tuning penuh |
| Image size | 320 |
| Batch size | 16, mengikuti konfigurasi notebook |
| Seed | 42 |
| Class weighting | `none`, kemudian `balanced` |
| Weighted sampler | Tidak diaktifkan pada kedua run |
| Pemilihan checkpoint | `val_loss` terkecil |

Jika GPU kehabisan memori, gunakan batch size 8 untuk kedua run sejak awal.
Pastikan variabel konfigurasi notebook sesuai tabel sebelum mulai.

Gunakan versi terbaru `scripts/train_proposal_hybrid_resnet50_malds.py`
di folder `SCRIPTS_DIR` Colab. Jika memakai ZIP lama, perbarui script tersebut
sebelum menjalankan cell berikut. Jalankan cell setup dan preprocessing pada
`acne_colab.ipynb` terlebih dahulu, kemudian gunakan cell di bawah sebagai
pengganti cell training Option A.

Script proposal sebelumnya sudah memakai class weighting secara otomatis.
Cell training lama juga memakai weighted sampler. Eksperimen berikut menguji
class weighting saja, tanpa sampler, agar efeknya bisa dibandingkan.

Bobot dihitung hanya dari split train:

```text
weight[c] = jumlah_train / (4 * jumlah_train_kelas_c)
```

Bobot mengalikan KL loss klasifikasi per sampel berdasarkan kelas asli.
Segmentation loss, morphology loss, arsitektur, dan split tetap sama.
Kappa sekarang dihitung dari seluruh prediksi dalam satu split, sehingga
angka Kappa dari script lama yang memakai rata-rata per batch perlu dihitung
ulang untuk perbandingan yang konsisten.

## Cell Training

```python
import subprocess
import sys

# Jalankan dua kali: "none" untuk kontrol, lalu "balanced" untuk weighting.
EXPERIMENT_DATE = "2026-10-04"
CLASS_WEIGHTING = "none"
RUN_DIR = TRAIN_DIR / f"{EXPERIMENT_DATE}_proposal_class_weighting_{CLASS_WEIGHTING}"

RESUME = (RUN_DIR / "last_checkpoint.pt").exists()

cmd = [
    sys.executable, str(TRAIN_SCRIPT_A),
    "--manifest", str(REFINED_MANIFEST),
    "--output-dir", str(RUN_DIR),
    "--class-weighting", CLASS_WEIGHTING,
    "--image-size", str(IMAGE_SIZE),
    "--batch-size", str(BATCH_SIZE),
    "--phase1-epochs", str(PHASE1_EPOCHS),
    "--phase2-epochs", str(PHASE2_EPOCHS),
    "--device", DEVICE,
    "--pretrained",
    "--num-workers", str(NUM_WORKERS),
    "--seed", "42",
]
if RESUME:
    cmd.append("--resume")
    print("Melanjutkan dari checkpoint terakhir.")
subprocess.run(cmd, check=True)
```

Gunakan manifest, jumlah epoch, batch size, dan seed yang sama pada kedua run.
Mulai dari awal untuk setiap run. Model terbaik tetap dipilih berdasarkan
`val_loss`, seperti setup sebelumnya. Jika run terputus, jalankan ulang cell
dengan mode yang sama untuk melanjutkan checkpoint secara otomatis. Pastikan
proses training sebelumnya sudah berhenti. Untuk mengulang eksperimen dari
awal, gunakan ID tanggal atau folder output baru. Resume pada script saat ini
memuat bobot, history, dan nomor epoch; state optimizer tidak dipulihkan.
Hindari memilih konfigurasi berdasarkan
hasil test; gunakan validasi untuk pemilihan konfigurasi.

## Cell Perbandingan Hasil

```python
import json
import pandas as pd

comparison = []
EXPERIMENT_DATE = "2026-10-04"
for mode in ["none", "balanced"]:
    path = TRAIN_DIR / f"{EXPERIMENT_DATE}_proposal_class_weighting_{mode}" / "metrics.json"
    if not path.exists():
        print(f"Belum tersedia: {path}")
        continue
    data = json.loads(path.read_text())
    test = data["test_metrics"]
    comparison.append({
        "class_weighting": mode,
        "train_counts": data["config"]["train_class_counts"],
        "class_weights": data["config"]["class_weights"],
        "best_val_loss": data["best_val_loss"],
        "test_accuracy": test["acc"],
        "test_kappa": test["kappa"],
        "test_dice": test["dice"],
        "test_iou": test["iou"],
    })
display(pd.DataFrame(comparison))
if comparison:
    pd.DataFrame(comparison).to_csv(
        TRAIN_DIR / f"{EXPERIMENT_DATE}_class_weighting_comparison.csv", index=False,
    )
```

Hasil lengkap dan bobot tersimpan di `metrics.json` pada masing-masing folder
run. Weighting belum tentu menaikkan accuracy keseluruhan; periksa juga Kappa
dan confusion matrix melalui cell inference notebook untuk melihat kelas
minoritas. Arahkan checkpoint inference ke `RUN_DIR / "best_model.pt"`.

## Lokasi Hasil

```text
TRAIN_DIR/
  2026-10-04_proposal_class_weighting_none/
    best_model.pt
    last_checkpoint.pt
    metrics.json
  2026-10-04_proposal_class_weighting_balanced/
    best_model.pt
    last_checkpoint.pt
    metrics.json
  2026-10-04_class_weighting_comparison.csv
```

Laporkan Accuracy, Cohen's Kappa, Dice, dan IoU dari kedua run. Periksa apakah
weighting membantu klasifikasi sekaligus mempertahankan kualitas segmentasi.
Total validation loss menggunakan bobot klasifikasi berbeda antar run,
sehingga nilainya tidak dapat langsung dipakai untuk menyatakan run mana
lebih baik. Gunakan metrik validasi klasifikasi untuk menilai konfigurasi.
