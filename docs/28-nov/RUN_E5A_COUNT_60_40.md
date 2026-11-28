# RUN E5a — Hybrid AMFM + Counting Head

**Protokol:** 60/40 tanpa validation · controlled augmentation (balance 300/class, cap 7×)
· model hibrida AMFM · auxiliary counting head · budget referensi 50 epoch.

Dokumen ini khusus untuk satu run: **E5a**. Jangan mencampur dengan E4 (class
weighting balanced) atau E5b (count-primary 65 bin/LDS).

---

## 1. Definisi run

| Komponen | Nilai |
|---|---|
| Run ID | `recovery_E5a_controlled300_cap7_01` |
| Model | `ProposalHybridResNet50`, `variant=proposed` (AMFM aktif) |
| Tugas | segmentasi + grading + morfologi + **regresi jumlah lesi** |
| Target count | `nnew_count` (NNEW), fallback `n_circles` hanya bila kolom tidak ada |
| Split | 60% train / 0% val / 40% test |
| Augmentasi | controlled: target 300/class, cap 7×, seed 42 |
| Class weighting | `none` (uniform) |
| Epoch | phase1 20 + phase2 30 = 50 (batas atas; budget update bisa menghentikan lebih awal) |
| Checkpoint | epoch terakhir pada budget tetap; tanpa early stopping |
| Test | dibuka sekali di Step 48, setelah checkpoint terkunci |

Yang **berbeda dari E2** hanya satu hal: `--count-head` aktif. Split,
augmentasi, seed, batch, arsitektur, dan budget harus identik.

Catatan istilah: "augmentasi balance" di runbook ini berarti **controlled
augmentation Step 45** (target 300/class, cap 7×). Itu bukan
`class_weighting='balanced'`; yang terakhir adalah E4 dan tidak dicampur di sini
supaya efek count head bisa dipisahkan.

---

## 2. Prasyarat

E5a tidak boleh dimulai sebelum:

1. Manifest sumber sudah memakai label NNEW dan punya kolom `nnew_count`.
2. Step 42 dijalankan ulang dari manifest baru (CPU).
3. Step 43A/43B/45/46 selesai untuk split 60/40 dan gate Step 46 **PASS**.
4. E2 sudah dijalankan atau dijadwalkan dengan budget yang sama.

Kalau `nnew_count` tidak ada, trainer jatuh ke `n_circles` v2. Itu membuat label
NNEW dan target count v2 bertentangan—jangan lanjut.

---

## 3. Check ulang konfigurasi (sudah diverifikasi)

| Hal | Status |
|---|---|
| Step 43B default `0.60 / 0.0 / 0.40` | OK |
| Step 45 default `target=300`, `cap=7`, `seed=42` | OK |
| Step 46 default `batch=4`, `budget_epochs=50` | OK |
| Step 47 default `phase1=20`, `phase2=30` | OK |
| `--count-head`, `--count-weight`, `--count-loss` tersedia | OK |
| Trainer memprioritaskan `nnew_count` | OK |
| Resume menolak checkpoint dengan konfigurasi count head berbeda | OK |
| Inference mencatat `pred_count`, `pred_count_grade`, `count_mae`, `count_bias` | OK |
| Unit test | **204 OK, 5 skipped** |

Catatan: training tidak bisa di-dry-run di mesin lokal karena `torch` tidak
terpasang. Verifikasi runtime dilakukan oleh gate Step 46 dan pengecekan
`metrics.json` setelah run.

---

## 4. Urutan eksekusi

### 4.1 Patch manifest sumber (di Colab, sebelum Step 42)

```bash
python scripts/build_ground_truth_manifest.py \
  --classification-dir /content/acne_images/Classification \
  --annotations /content/acne_images/ACNE04/Classification/Acne04-v2_annotations.json \
  --image-dir /content/acne_images/Classification/JPEGImages \
  --out-dir /content/drive/MyDrive/acne_project/output/ground_truth \
  --audit-per-grade 15

python scripts/apply_ground_truth_labels.py \
  --manifest /content/drive/MyDrive/acne_project/output/experiments_512_mask_v2/.../refined_manifest_v2.csv \
  --ground-truth /content/drive/MyDrive/acne_project/output/ground_truth/ground_truth_manifest_nnew.csv \
  --output /content/drive/MyDrive/acne_project/output/experiments_512_mask_v2/.../refined_manifest_v2_nnew.csv
```

Sesuaikan path dengan lokasi Drive yang dipakai notebook. Script menulis
manifest baru; manifest asli tidak diubah.

### 4.2 Step 42 — exact duplicates (CPU)

```python
STEP42_SOURCE_MANIFEST = FULL_MASK_BASE / 'refined_manifest_v2_nnew.csv'
STEP42_OUTPUT = RECOVERY_ROOT / 'step42_exact_duplicates_nnew'
STEP42_CANONICAL_RULE = 'first_filename'
RUN_STEP42 = True
```

Setelah selesai, review `label_mask_conflicts.csv` dan
`duplicate_review_grid.png`. Jangan lanjut bila ada konflik baru yang belum
diputuskan.

### 4.3 Step 43A — near duplicates (CPU)

```python
STEP42_OUTPUT = RECOVERY_ROOT / 'step42_exact_duplicates_nnew'
STEP43_NEAR_OUTPUT = RECOVERY_ROOT / 'step43_near_duplicates_e5a'
RUN_STEP43_NEAR = True
```

Isi `near_duplicate_review.csv` seperti biasa. Output ini menjadi input Step 43B.

### 4.4 Step 43B — split 60/40 tanpa validation

```python
STEP42_OUTPUT = RECOVERY_ROOT / 'step42_exact_duplicates_nnew'
STEP43_NEAR_OUTPUT = RECOVERY_ROOT / 'step43_near_duplicates_e5a'
STEP43_SPLIT_OUTPUT = RECOVERY_ROOT / 'step43_grouped_split_60_40_e5a'
STEP43_TRAIN_RATIO, STEP43_VAL_RATIO, STEP43_TEST_RATIO = 0.60, 0.0, 0.40
RUN_STEP43_SPLIT = True
```

Gate: `split_integrity.json` harus `PASS`, tidak ada grup yang menyeberang
train/test, dan keempat kelas ada di train dan test.

### 4.5 Step 45 — controlled augmentation

```python
STEP43_SPLIT_OUTPUT = RECOVERY_ROOT / 'step43_grouped_split_60_40_e5a'
STEP45_OUTPUT = RECOVERY_ROOT / 'step45_controlled_aug_60_40_e5a'
STEP45_TARGET_PER_CLASS = 300
STEP45_CAP = 7
RUN_STEP45_CONTROLLED = True
```

Arti "balance": target 300 per kelas, tetapi dibatasi cap 7× dari jumlah asli.
Level 3 tidak akan mencapai 300 kalau citra aslinya sedikit (mis. 14 → 98).
Itu memang desainnya, bukan kegagalan.

Gate: `controlled_aug_integrity.json` harus `PASS` dan tidak ada sintetis di
val/test.

### 4.6 Step 46 — weight, budget, gate

```python
STEP45_OUTPUT = RECOVERY_ROOT / 'step45_controlled_aug_60_40_e5a'
STEP43_SPLIT_MANIFEST = RECOVERY_ROOT / 'step43_grouped_split_60_40_e5a' / 'split_manifest.csv'
STEP46_OUTPUT = RECOVERY_ROOT / 'step46_weighting_60_40_e5a'
STEP46_BATCH_SIZE = 4
STEP46_BUDGET_EPOCHS = 50
RUN_STEP46_WEIGHTING = True
```

Gate wajib **PASS** (`ready_for_gpu: true`). Kalau BLOCKED, perbaiki artefak;
jangan memaksa training.

---

## 5. Arti "epoch 50" dan budget update

Step 46 menghitung:

```
reference_updates = ceil(reference_train_size / batch_size) * 50
```

`reference_train_size` default = jumlah train **sebelum augmentasi** dari split
Step 43B. Step 47 memakai angka `reference_updates` itu sebagai
`--max-optimizer-steps`.

Konsekuensinya:

- E1 (manifest tanpa augmentasi) ≈ 50 epoch.
- E2/E5a (manifest controlled augmentation) **lebih sedikit dari 50 epoch**,
  karena satu epoch berisi lebih banyak baris.

Ilustrasi dengan angka lama 60/20/20 (angka aktual 60/40 bisa berbeda):

| Ukuran | Nilai |
|---|---:|
| Train original | 697 |
| Batch | 4 |
| Update referensi | `ceil(697/4) × 50 = 8.750` |
| Train setelah controlled aug | ±1.031 |
| Update/epoch | 258 |
| Epoch aktual E5a | `ceil(8.750/258) ≈ 34` |

Jadi "50 epoch" adalah **budget referensi**, bukan 50 pass atas data
augmentasi. Jangan mengubah `STEP47_MAX_OPTIMIZER_STEPS` menjadi
`updates_per_epoch × 50`; itu akan memberi E5a update 47% lebih banyak daripada
E1/E2 dan merusak perbandingan.

Nilai aktual ada di:

- `step46_metrics.json` → `budget.reference_updates`
- `metrics.json` Step 47 → `optimizer_steps_total`, `epochs_completed`,
  `stopped_reason`

Kalau `stopped_reason = max_optimizer_steps`, itu **benar**, bukan training
terpotong.

---

## 6. Step 47 — E5a

```python
STEP45_OUTPUT = RECOVERY_ROOT / 'step45_controlled_aug_60_40_e5a'
STEP47_MANIFEST_OVERRIDE = STEP45_OUTPUT / 'controlled_aug_manifest.csv'
STEP47_MANIFEST = STEP47_MANIFEST_OVERRIDE
STEP47_RUN_ID = 'recovery_E5a_controlled300_cap7_01'
STEP47_BATCH_SIZE = 4
STEP47_PHASE1_EPOCHS = 20
STEP47_PHASE2_EPOCHS = 30
STEP47_CLASS_WEIGHTING = 'none'
STEP47_COUNT_HEAD = True
STEP47_COUNT_WEIGHT = 0.1
STEP47_USE_VALIDATION = False
STEP47_MAX_OPTIMIZER_STEPS = None   # ambil dari gate Step 46
STEP46_APPROVED = True              # hanya setelah gate PASS dan disetujui
RUN_STEP47_TRAINING = True
```

Yang harus muncul di `metrics.json`:

```json
{
  "config": {"count_head": true, "count_weight": 0.1, "count_loss": "smooth_l1"},
  "optimizer_steps_total": 8750,
  "stopped_reason": "max_optimizer_steps",
  "checkpoint_selection": "final_epoch_no_validation"
}
```

Angka `optimizer_steps_total` harus sama dengan `reference_updates` Step 46.
Flag `no_validation` tersimpan di checkpoint, bukan di `metrics.json`; mode
tanpa validation ditandai oleh `checkpoint_selection = final_epoch_no_validation`.

Jangan mengubah `STEP47_COUNT_WEIGHT` setelah melihat hasil test. Kalau mau
mengeksplorasi bobot lain, itu run terpisah dengan keputusan sebelum test.

---

## 7. Step 48 — evaluasi test sekali

Karena Step 48 punya folder default bersama, gunakan folder baru:

```python
STEP47_RUN_ID = 'recovery_E5a_controlled300_cap7_01'
STEP47_OUTPUT = TRAIN_DIR / STEP47_RUN_ID
STEP47_MANIFEST = RECOVERY_ROOT / 'step45_controlled_aug_60_40_e5a' / 'controlled_aug_manifest.csv'
STEP48_INFER_OUTPUT = RECOVERY_ROOT / 'step48_test_infer_e5a'
STEP48_REPORT_OUTPUT = RECOVERY_ROOT / 'step48_final_report_e5a'
STEP48_TITLE = 'E5a — hybrid AMFM + count head, 60/40'
STEP48_COMPARISONS = ['E2=/path/ke/step48_final_report_e2/test_metrics.json']
RUN_STEP48_EVAL = True
```

`testing_predictions.csv` sekarang memuat:

| Kolom | Arti |
|---|---|
| `pred_count` | jumlah lesi prediksi |
| `pred_count_grade` | grade dari count prediksi (Hayashi) |
| `true_count` | `nnew_count` dari manifest |
| `count_error` | `pred_count - true_count` |
| `count_mae` / `count_bias` | ringkasan di `test_metrics.json` |

Jangan menghapus atau menimpa folder hasil test. Test hanya dibuka sekali.

---

## 8. Metrik yang wajib dilaporkan

| Metrik | Kenapa |
|---|---|
| Macro F1 | metrik utama; tidak didominasi kelas mayoritas |
| Recall L2 dan L3 | fokus masalah |
| Confusion matrix | melihat arah kesalahan |
| Accuracy | sekunder, mudah menyesatkan |
| `count_mae`, `count_bias` | apakah counter benar-benar belajar |
| `pred_count_grade` vs `pred_class` | apakah jalur count lebih baik |
| CI 95% | L3 test sangat kecil |

### Cara membaca hasil

| Pola | Arti | Tindakan |
|---|---|---|
| L3 recall naik, `count_bias` L3 kecil | Count head membantu | Lanjut, laporkan |
| L3 recall stagnan, `count_bias` L3 negatif besar | Model undercount | E5b: 65-bin + LDS |
| L3 recall stagnan, count akurat | Bottleneck di class head | Pertimbangkan E4/ordinal loss |
| `pred_count_grade` jauh lebih baik | Count layak jadi jalur prediksi | Fusion harus pre-registered |
| L2 turun, L3 naik | Trade-off | Laporkan, jangan sembunyikan |

Dengan L3 test sekitar 8 citra pada 60/40, satu citra = 12,5% recall. Jangan
mengklaim kemenangan dari selisih satu citra.

---

## 9. Guardrail

1. Jangan menjalankan E5a tanpa E2; tidak ada kontrol.
2. Jangan memakai `n_circles` v2 bila `nnew_count` tersedia.
3. Jangan mencampur `class_weighting='balanced'` (itu E4) dalam run ini.
4. Jangan memilih checkpoint atau bobot count berdasarkan test.
5. Jangan menimpa folder Step 42/43/45/46/47/48; pakai suffix baru.
6. Jangan mengubah split setelah Step 46 PASS.
7. Jangan mengevaluasi test lebih dari sekali per run.

---

## 10. Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `count_mae` kosong | `nnew_count` tidak ada dan `n_circles` kosong | Patch manifest, cek kolom |
| `count_head: false` di metrics | Flag tidak terbaca | Pastikan `STEP47_COUNT_HEAD=True` dan run ID baru |
| `FileExistsError` folder | Folder sudah terisi | Pakai run ID/folder baru |
| Gate Step 46 BLOCKED | Artefak belum lengkap | Baca `pretraining_go_no_go.md` |
| Epoch berhenti di 34, bukan 50 | Budget update tercapai | Normal; cek `stopped_reason` |
| `val` masih ada di manifest | Step 43B memakai val_ratio > 0 | Set `STEP43_VAL_RATIO=0.0`, folder baru |
| Inference error `count_head` | Checkpoint beda konfigurasi | Jalankan ulang dari checkpoint E5a yang benar |
| `ImportError: lds_targets` | Script inference butuh modul target LDS | Notebook Step 48 fetch otomatis; CLI: taruh `lds_targets.py` di folder script |

---

## 11. Checklist

- [ ] Manifest NNEW + `nnew_count` dibuat, 34 label berubah terverifikasi.
- [ ] Step 42 baru selesai; konflik label direview.
- [ ] Step 43A baru selesai; review near-duplicate diisi.
- [ ] Step 43B 60/40 PASS, tanpa val.
- [ ] Step 45 controlled aug PASS, tidak ada sintetis di test.
- [ ] Step 46 PASS, `reference_updates` tercatat.
- [ ] E2 selesai dengan budget yang sama.
- [ ] E5a `count_head=true`, `optimizer_steps_total == reference_updates`.
- [ ] Step 48 folder baru, test dibuka sekali.
- [ ] Laporan memuat Macro F1, recall L2/L3, CI, `count_mae`, `count_bias`.
