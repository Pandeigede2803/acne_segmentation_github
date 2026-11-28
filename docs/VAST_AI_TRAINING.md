# Menjalankan Step 42–48 di Vast.ai (tanpa Colab)

Pipeline ini tidak bergantung pada Colab. Notebook hanya membungkus pemanggilan
script CLI, jadi di Vast.ai cukup jalankan script yang sama lewat SSH/tmux.
Yang perlu diganti hanya **penyimpanan** dan **cara menjalankan**.

---

## Arsitektur yang disarankan

```
Laptop/Mac                 Vast.ai instance (GPU)                Persistent storage
-----------                ----------------------                ------------------
Step 42 (sudah)  ───────►  sync data masuk  ──────────────►      rclone remote (Drive/S3)
                           Step 43A/43B/45/44/46 (CPU)
                           Step 47 training (GPU)  ──sync tiap 10 menit──►
                           Step 48 evaluasi
```

- **Step 42–46 CPU-only**: bisa di Vast (CPU instance murah) atau di Mac.
- **Step 47 GPU**: Vast RTX 4090 / A100 / L40S.
- **Step 48**: CPU/GPU, setelah training selesai.
- Jangan pernah menyimpan satu-satunya salinan hasil di disk instance. Instance
  bisa dihentikan/preempted kapan saja.

---

## 1. Pilih instance

- Image: template PyTorch + CUDA (mis. `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`
  atau template Vast yang setara).
- GPU: RTX 4090 (paling murah untuk 512px ResNet50), A100/L40S jika ingin lebih cepat.
- Disk: minimal **60 GB** (dataset + checkpoint + cache).
- Aktifkan **persistent volume** bila tersedia, atau andalkan rclone.

Perkiraan kasar Step 47 (977 baris train, batch 4, 50 epoch ≈ 12.250 update):
**1,5–4 jam** di 4090. Angka aktual tergantung GPU, dataloader, dan augmentasi.
Cek harga instance saat itu; jangan mengandalkan estimasi ini untuk budgeting.

---

## 2. Bootstrap instance

Clone dulu (sekali), lalu jalankan bootstrap:

```bash
git clone https://github.com/Pandeigede2803/acne_segmentation_github.git /workspace/acne_segmentation_github
cd /workspace/acne_segmentation_github

export OUTPUT_ROOT=/workspace/acne_project/output
bash scripts/vast/bootstrap.sh
```

Script melakukan:

- install `rclone`, `rsync`, `tmux`,
- clone repo,
- install `opencv-python-headless`, `pandas`, `matplotlib`,
- membuat symlink `/content/drive/MyDrive/acne_project/output → $OUTPUT_ROOT`
  supaya path absolut di manifest tetap valid,
- cek `nvidia-smi` + torch CUDA.

**Kenapa symlink?** Manifest yang dihasilkan pipeline berisi path
`/content/drive/MyDrive/acne_project/output/...`. Dengan symlink, semua script
bisa dipakai tanpa menulis ulang manifest.

---

## 3. Siapkan akses storage

### Opsi A — rclone ke Google Drive (disarankan)

Di laptop (yang sudah punya `rclone.conf`):

```bash
scp ~/.config/rclone/rclone.conf root@<host>:~/.config/rclone/rclone.conf
```

Di instance:

```bash
rclone lsf gdrive:acne_project/output | head
```

Lalu tarik hanya yang dibutuhkan:

```bash
rclone copy gdrive:acne_project/output/experiments_512_mask_v2/recovery_v1 \
  /workspace/acne_project/output/experiments_512_mask_v2/recovery_v1 -P

rclone copy gdrive:acne_project/output/experiments_512/split_aug1000_img512_v1/split_60_40 \
  /workspace/acne_project/output/experiments_512/split_aug1000_img512_v1/split_60_40 -P
```

Untuk Step 45 kamu juga butuh preprocessed + mask V2:

```bash
rclone copy gdrive:acne_project/output/experiments_512_mask_v2/step36_full_v2 \
  /workspace/acne_project/output/experiments_512_mask_v2/step36_full_v2 -P
```

### Opsi B — upload langsung

```bash
# dari laptop
rsync -avP /path/lokal/recovery_v1 root@<host>:/workspace/acne_project/output/experiments_512_mask_v2/
```

Hindari mengunggah seluruh Drive; cukup folder yang dipakai.

---

## 4. Jalankan Step 43–46 (CPU)

Gunakan `tmux` agar aman saat SSH putus:

```bash
tmux new -s recovery
cd /workspace/acne_segmentation_github
export PY=/usr/bin/python3   # sesuaikan dengan image
```

Step 43A:

```bash
$PY scripts/detect_near_duplicates.py \
  --manifest /workspace/acne_project/output/experiments_512_mask_v2/recovery_v1/step42_exact_duplicates/clean_original_manifest.csv \
  --output-dir /workspace/acne_project/output/experiments_512_mask_v2/recovery_v1/step43_near_duplicates
```

Isi `near_duplicate_review.csv` (bisa dari laptop setelah disinkronkan), lalu 43B:

```bash
$PY scripts/build_grouped_split.py \
  --manifest .../step42_exact_duplicates/clean_original_manifest.csv \
  --exact-groups .../step42_exact_duplicates/exact_duplicate_groups.csv \
  --near-review .../step43_near_duplicates/near_duplicate_review.csv \
  --output-dir .../step43_grouped_split
```

Step 45:

```bash
$PY scripts/build_controlled_augmentation.py \
  --manifest .../step43_grouped_split/split_manifest.csv \
  --output-dir .../step45_controlled_aug
```

Step 44:

```bash
$PY scripts/audit_augmentation_visual.py \
  --augmented-manifest /workspace/acne_project/output/experiments_512/split_aug1000_img512_v1/split_60_40/manifest_aug.csv \
  --output-dir .../step44_visual_audit
```

Step 46 (gate):

```bash
$PY scripts/audit_class_weighting.py \
  --manifest .../step45_controlled_aug/controlled_aug_manifest.csv \
  --weight-manifest .../step43_grouped_split/split_manifest.csv \
  --clean-manifest .../step42_exact_duplicates/clean_original_manifest.csv \
  --split-integrity .../step43_grouped_split/split_integrity.json \
  --visual-report .../step44_visual_audit/augmentation_quality_report.md \
  --controlled-integrity .../step45_controlled_aug/controlled_aug_integrity.json \
  --output-dir .../step46_weighting
```

Berhenti di sini. Periksa `pretraining_go_no_go.md`.

---

## 5. Step 47 — training GPU

Setelah gate PASS dan disetujui:

```bash
mkdir -p /workspace/acne_project/output/training_runs/recovery_E2_controlled300_01

export MANIFEST=/workspace/acne_project/output/experiments_512_mask_v2/recovery_v1/step45_controlled_aug/controlled_aug_manifest.csv
export RUN_ID=recovery_E2_controlled300_01
export MAX_OPTIMIZER_STEPS=7700   # ganti dengan reference_updates dari Step 46

nohup bash scripts/vast/run_training.sh > train.log 2>&1 &
```

Atau langsung tanpa wrapper:

```bash
nohup python scripts/train_proposal_hybrid_resnet50_malds.py \
  --manifest "$MANIFEST" \
  --output-dir "/workspace/acne_project/output/training_runs/$RUN_ID" \
  --image-size 512 --batch-size 4 \
  --phase1-epochs 20 --phase2-epochs 30 \
  --pretrained --num-workers 4 --device cuda \
  --class-weighting none \
  --checkpoint-metric val_macro_f1 \
  --early-stopping-patience 5 \
  --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" \
  --skip-test-eval \
  > train.log 2>&1 &
```

Kalau instance preempted:

```bash
# instance baru: sync output lama dulu, lalu
RESUME=1 nohup bash scripts/vast/run_training.sh >> train.log 2>&1 &
```

`--resume` melanjutkan dari `last_checkpoint.pt` di output-dir yang sama.

---

## 6. Sinkronisasi hasil selama training

Jalankan di tmux terpisah:

```bash
export REMOTE=gdrive:acne_project/output
export LOCAL=/workspace/acne_project/output
export INTERVAL=600
bash scripts/vast/sync_outputs.sh
```

Script menyalin seluruh output ke remote setiap 10 menit. Jangan pakai
`rclone sync --delete`; gunakan `copy` agar hasil di remote tidak terhapus jika
file lokal sementara tidak lengkap.

Minimal yang harus terselamatkan jika instance mati:

- `last_checkpoint.pt` (resume),
- `best_model.pt` (checkpoint terpilih),
- `metrics.json` / `history_partial.json`,
- `epoch_metrics.csv`,
- `training_console.log`.

---

## 7. Step 48 — evaluasi

```bash
python scripts/infer_testing_images.py \
  --manifest .../step45_controlled_aug/controlled_aug_manifest.csv \
  --checkpoint .../training_runs/recovery_E2_controlled300_01/best_model.pt \
  --output-dir .../recovery_v1/step48_test_infer \
  --split test --image-size 512 --model-type proposal

python scripts/evaluate_final_experiment.py \
  --predictions .../recovery_v1/step48_test_infer/testing_predictions.csv \
  --metrics .../training_runs/recovery_E2_controlled300_01/metrics.json \
  --manifest .../step45_controlled_aug/controlled_aug_manifest.csv \
  --output-dir .../recovery_v1/step48_final_report
```

Setelah selesai, sync sekali lagi lalu matikan instance.

---

## 8. Catatan reproducibility

Simpan di folder run:

```bash
nvidia-smi > run_env.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)" >> run_env.txt
pip freeze >> run_env.txt
git -C /workspace/acne_segmentation_github rev-parse HEAD >> run_env.txt
```

Tanpa itu, hasil di Vast tidak bisa dibandingkan secara adil dengan run Colab.

---

## 9. Hal yang mudah salah

- **Path absolut manifest**: tanpa symlink `/content/drive/...`, script akan
  gagal menemukan foto/mask. Buat symlink atau sed path-nya.
- **rclone mount**: jangan mount Drive sebagai working directory training.
  Tulis ke disk lokal, sync periodik. Mount bisa membuat I/O lambat dan file
  "terlihat" tetapi belum terunggah.
- **Instance spot**: selalu siapkan resume + sync. Jangan menunggu training
  selesai baru upload.
- **Versi dependency**: image berbeda = hasil bisa berbeda. Catat versi.
- **Test**: jangan evaluasi test berkali-kali untuk memilih konfigurasi; aturan
  yang sama seperti di Colab tetap berlaku.
- **Jangan commit `rclone.conf`** atau token ke repo.

---

## 10. Ringkas: apa yang berubah dari Colab

| Colab | Vast.ai |
|---|---|
| `drive.mount()` | rclone copy / symlink |
| cell notebook | perintah CLI + `tmux` |
| output langsung ke Drive | disk lokal + sync periodik |
| sesi mati = runtime reset | instance mati = disk hilang (kecuali volume) |
| GPU T4 gratis | GPU sewa per jam |

Kode pipeline, guard, dan aturan test **tidak berubah**.
