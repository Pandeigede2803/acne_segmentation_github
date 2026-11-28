# Step 42–48 — Dataset Recovery & Controlled Experiment

Implementasi dari protokol `docs/28-nov/step42_48_dataset_recovery_experiment.md`.

**Status harian 28 Nov 2026:** [`docs/28-nov/2026-11-28_project_status.md`](28-nov/2026-11-28_project_status.md)

Step 42–46 **CPU only**; Step 47 (training) baru boleh berjalan setelah gate
Step 46 PASS dan persetujuan eksplisit. Step 48 hanya evaluasi checkpoint terkunci.

---

## Status implementasi

| Step | Pekerjaan | Script | Output utama | Guard |
|---|---|---|---|---|
| 42 | Dedup exact + konflik label | `scripts/detect_exact_duplicates.py` | `clean_original_manifest.csv`, `quarantine_manifest.csv` | konflik label dikarantina; tidak ada file dihapus |
| 43A | Near-duplicate (dHash/pHash + flip) | `scripts/detect_near_duplicates.py` | `near_duplicate_review.csv` | semua kandidat `pending` sampai direview |
| 43B | Split ulang berbasis grup | `scripts/build_grouped_split.py` | `split_manifest.csv`, `split_integrity.json` | `status=PASS`; group tidak boleh lintas split |
| 44 | Visual audit augmentasi L2/L3 | `scripts/audit_augmentation_visual.py` | grid L2/L3/kontrol, `visual_audit_scores.csv` | grading fail/suspect/pass |
| 45 | Controlled augmentation 300/cap 7× | `scripts/build_controlled_augmentation.py` | `controlled_aug_manifest.csv`, `augmentation_lineage.csv` | `status=PASS`; maks 6 sintetis/sumber |
| 46 | Audit class weight + desain E1–E4 | `scripts/audit_class_weighting.py` | `experiment_matrix.yaml`, `training_budget.md` | `pretraining_go_no_go.md` PASS/BLOCKED |
| 47 | Training terkontrol | `scripts/train_proposal_hybrid_resnet50_malds.py` | `epoch_metrics.csv`, `best_model.pt` | val Macro F1; test tidak dibuka |
| 48 | Evaluasi final | `scripts/evaluate_final_experiment.py` | `test_metrics.json`, confusion matrix | test dievaluasi sekali |

Semua helper murni (grup, alokasi, metrik) berada di `scripts/recovery_common.py`
dan diuji tanpa numpy/PIL/torch di `tests/test_recovery_steps_42_48.py`.

---

## Alur di notebook

Cell baru ada di akhir `acne_colab.ipynb`: **42, 43A, 43B, 44, 45, 46, 47, 48**.
Semua toggle default `False`.

```
42  → 43A (review manual) → 43B → 44 → 45 → 46 (STOP)
                                              ↓ approval
                                             47 → 48
```

Folder kerja default:

```
/content/drive/MyDrive/acne_project/output/experiments_512_mask_v2/recovery_v1/
  step42_exact_duplicates/
  step43_near_duplicates/
  step43_grouped_split/
  step44_visual_audit/
  step45_controlled_aug/
  step46_weighting/
  step48_test_infer/
  step48_final_report/
```

Setiap cell menulis `pipeline_log` (`step_42.json`, …) dan console log di
`recovery_v1/`. Folder output yang sudah berisi file **ditolak**; gunakan suffix baru.

---

## Keputusan protokol yang diterapkan

1. **Step 42**
   - Hash byte (`bytes_sha256`) dan hash piksel hasil decode (`pixel_sha256`).
   - Grup duplikat deterministik `dup_0001…`; satu representasi canonical.
   - Konflik label → karantina. Mask berbeda dicatat `mask_variant` (default tidak
     dikarantina; aktifkan `--quarantine-mask-conflicts` bila diinginkan).
   - Tidak ada penghapusan file; rekonsiliasi jumlah per kelas ada di laporan.

2. **Step 43**
   - dHash 64-bit + pHash 64-bit (DCT), termasuk varian horizontal flip.
   - Kandidat ≠ keputusan: hanya baris `status=confirmed` di review CSV yang dipakai.
   - Split per grup (exact + near-dup + opsional patient ID), stratifikasi per kelas.
   - Default terbaru: **60/40 tanpa validation** (`--val-ratio 0`). Mode ini dipakai
     untuk protokol fixed-budget: checkpoint = epoch akhir, tidak ada pemilihan dari
     validation. Mode 60/20/20 tetap tersedia dengan `--val-ratio 0.2`.
   - `split_integrity.json` memeriksa group disjoint, kelas lengkap, dan mencatat
     keterbatasan patient ID / test reuse.

3. **Step 44**
   - Memakai dataset augmented yang sudah ada (tidak generate baru).
   - Level 3: semua original; Level 2: 30 original; kontrol L0/L1: 10/kelas.
   - Grid 4 panel + statistik seluruh sintetis (area ratio, mask kosong, reuse,
     duplikat output, intensitas).
   - Grading bukan penilaian klinis.

4. **Step 45**
   - `target_c = max(n_c, min(300, 7 × n_c))`.
   - Alokasi sintetis merata; selisih maksimal satu; maksimum **6** sintetis/sumber.
   - Lineage per sampel (angle, flip, brightness, contrast, area before/after).
   - Test/val disalin tanpa perubahan; integritas diverifikasi sebelum lanjut.

5. **Step 46**
   - Membaca implementasi trainer: `total/(4·n_c)`, bobot per sampel, KL reduction,
     soft target MA-LDS.
   - Bobot balanced dihitung dari **train original** (bukan distribusi augmented),
     sesuai aturan E4.
   - Matrix E1–E4 + budget optimizer updates (fixed-update vs fixed-epoch).
   - Gate PASS hanya jika artefak 42–45 lengkap; selain itu BLOCKED.

6. **Step 47**
   - Checkpoint dipilih dari **val Macro F1**, tie-break val loss
     (`--checkpoint-metric val_macro_f1`). Default lama `val_loss` tetap tersedia.
   - `--early-stopping-patience` dihitung per validation interval setelah unfreeze.
   - `--max-optimizer-steps` menghentikan training setelah budget update tercapai
     (berhenti di akhir epoch; overshoot maksimal satu epoch dan dicatat).
   - `epoch_metrics.csv` berisi seluruh history, termasuk per-class
     precision/recall/F1/support.
   - `--skip-test-eval` wajib; cell menolak `metrics.json` yang memuat test metrics.

7. **Step 48**
   - Inference test memakai `best_model.pt` dari Step 47, sekali.
   - Metrik: macro F1 (primary), balanced accuracy, accuracy, weighted F1,
     Cohen kappa, ordinal MAE, under/over-estimation.
   - Confusion matrix raw + row-normalized; Wilson CI per kelas; cluster bootstrap
     bila `group_id` tersedia.
   - Perbandingan run lama hanya `historical reference`.

---

## Perubahan pada trainer

`scripts/train_proposal_hybrid_resnet50_malds.py`:

- `--checkpoint-metric {val_loss,val_macro_f1}` (default `val_loss`, backward compatible)
- `--early-stopping-patience N` (0 = nonaktif)
- `--max-optimizer-steps N` (0 = nonaktif)
- `metrics.json` menambah: `checkpoint_metric`, `best_metric_value`,
  `best_val_macro_f1`, `best_val_dice`, `optimizer_steps_total`,
  `epochs_completed`, `early_stopped`, `stopped_reason`.
- `checkpoint_selection` bernilai `max_val_macro_f1` bila metrik itu dipilih.
- `epoch_metrics.csv` ditulis setiap epoch.
- `inference_run_guard.py` menghormati `checkpoint_metric` saat memverifikasi history.

Step 40 (jalur 1000/class) tidak diubah dan tetap memakai `min_val_loss` karena
tidak mengirim `--checkpoint-metric`.

---

## Menjalankan tanpa notebook

> **Penting:** script Step 42–46 dan 48 mengimpor `scripts/recovery_common.py`.
> File itu harus berada di folder yang sama. Notebook mengunduhnya otomatis;
> untuk pemakaian CLI, salin juga `recovery_common.py` ke folder script.

```bash
# 42
python scripts/detect_exact_duplicates.py \
  --manifest /path/refined_manifest_v2.csv --output-dir /path/step42

# 43A
python scripts/detect_near_duplicates.py \
  --manifest /path/step42/clean_original_manifest.csv --output-dir /path/step43a
# isi status=confirmed di near_duplicate_review.csv

# 43B
python scripts/build_grouped_split.py \
  --manifest /path/step42/clean_original_manifest.csv \
  --exact-groups /path/step42/exact_duplicate_groups.csv \
  --near-review /path/step43a/near_duplicate_review.csv \
  --output-dir /path/step43b

# 44 (dataset augmented yang sudah ada)
python scripts/audit_augmentation_visual.py \
  --augmented-manifest /path/manifest_final_aug.csv --output-dir /path/step44

# 45
python scripts/build_controlled_augmentation.py \
  --manifest /path/step43b/split_manifest.csv --output-dir /path/step45

# 46
python scripts/audit_class_weighting.py \
  --manifest /path/step45/controlled_aug_manifest.csv \
  --weight-manifest /path/step43b/split_manifest.csv \
  --clean-manifest /path/step42/clean_original_manifest.csv \
  --split-integrity /path/step43b/split_integrity.json \
  --visual-report /path/step44/augmentation_quality_report.md \
  --controlled-integrity /path/step45/controlled_aug_integrity.json \
  --output-dir /path/step46

# 47 (hanya setelah approval)
python scripts/train_proposal_hybrid_resnet50_malds.py \
  --manifest /path/step45/controlled_aug_manifest.csv \
  --output-dir /path/training_runs/recovery_E2_01 \
  --image-size 512 --batch-size 4 --phase1-epochs 20 --phase2-epochs 30 \
  --pretrained --class-weighting none \
  --checkpoint-metric val_macro_f1 \
  --early-stopping-patience 5 --max-optimizer-steps 7700 \
  --skip-test-eval

# 48
python scripts/infer_testing_images.py \
  --manifest /path/step45/controlled_aug_manifest.csv \
  --checkpoint /path/training_runs/recovery_E2_01/best_model.pt \
  --output-dir /path/step48_infer --split test --image-size 512 \
  --model-type proposal
python scripts/evaluate_final_experiment.py \
  --predictions /path/step48_infer/testing_predictions.csv \
  --metrics /path/training_runs/recovery_E2_01/metrics.json \
  --manifest /path/step45/controlled_aug_manifest.csv \
  --output-dir /path/step48_report
```

---

## Aturan yang tidak boleh dilanggar

- Test tidak dipakai untuk memilih varian mask, augmentasi, class weight,
  epoch, checkpoint, atau seed.
- `best_model.pt` Step 47 dipilih dari validation; Step 48 tidak mengubahnya.
- Jangan menimpa folder hasil lama; semua step menolak output non-kosong.
- Jangan mengubah transformasi dan oversampling sekaligus dalam satu eksperimen.
- Tanpa patient ID dan tanpa test eksternal, hasil tetap **evaluasi internal**.
- Target Macro F1 0,70 / recall Level 3 0,75 bersifat aspiratif, bukan syarat validitas.

---

## Cek status dan auto-skip

Sebelum mulai, jalankan cell **41b — Cek status Step 42–48**. Cell itu hanya
membaca folder hasil dan menampilkan:

| Kolom | Arti |
|---|---|
| `DONE` | artefak completion ada |
| `INCOMPLETE` | folder ada tetapi artefak belum lengkap |
| `MISSING` | belum dijalankan |

Cell ini juga mencetak rekomendasi step berikutnya.

Cell Step 42–48 **otomatis skip** bila hasil sudah ada **dan** inputnya sama
(hash manifest/kebijakan dibandingkan dengan provenance). Kalau input berubah,
cell menolak memakai folder lama dan meminta suffix baru (`_v2`, `_strict`, …).

Yang diverifikasi otomatis:

| Step | Dibandingkan |
|---|---|
| 42 | hash manifest + canonical rule + kebijakan karantina |
| 43A | hash manifest bersih |
| 43B | hash manifest bersih + hash review near-duplicate (bila tercatat) |
| 44 | hash manifest augmented |
| 45 | hash split manifest + target + cap + seed |
| 46 | hash manifest controlled + hash weight manifest (bila tercatat) |
| 47 | keberadaan `metrics.json` |
| 48 | keberadaan `test_metrics.json` |

## Menjalankan ulang sebuah step

Setiap step menolak menimpa folder hasil. Mengubah **opsi** (mis.
`STEP42_QUARANTINE_MASK_CONFLICTS`) tidak otomatis mengganti folder output.

Aturan:

1. Satu folder = satu keputusan/kebijakan. Jangan hapus hasil lama.
2. Ganti variabel folder, bukan hanya toggle. Contoh Step 42:

   ```python
   STEP42_OUTPUT = RECOVERY_ROOT / 'step42_exact_duplicates_strict'
   STEP42_QUARANTINE_MASK_CONFLICTS = True
   RUN_STEP42 = True
   ```

3. Step berikutnya memakai hasil terakhir selama cell dijalankan di sesi yang
   sama (variabel `STEP42_OUTPUT`/`STEP42_CLEAN_MANIFEST` ikut diperbarui).
4. Kalau folder sengaja ingin dipakai ulang (mis. resume training), itu hanya
   untuk Step 47 lewat CLI `--resume`; di notebook gunakan run ID baru.

Daftar variabel folder:

| Step | Variabel |
|---|---|
| 42 | `STEP42_OUTPUT` |
| 43A | `STEP43_NEAR_OUTPUT` |
| 43B | `STEP43_SPLIT_OUTPUT` |
| 44 | `STEP44_OUTPUT` |
| 45 | `STEP45_OUTPUT` |
| 46 | `STEP46_OUTPUT` |
| 47 | `STEP47_RUN_ID` (folder = `TRAIN_DIR / STEP47_RUN_ID`) |
| 48 | `STEP48_INFER_OUTPUT`, `STEP48_REPORT_OUTPUT` |

Cell Step 42–48 sekarang memeriksa folder **sebelum** menjalankan script dan
menampilkan pesan yang menyebut variabel mana yang harus diganti.

## E5 — counting head (tugas bantu ala Prokhorov)

Model awal tidak punya regresi jumlah lesi. Padahal `n_circles` sudah tersedia di
manifest. Prokhorov & Kalinin memakai penghitungan lesi sebagai tugas bantu untuk
grading; itu komponen yang belum ada di pipeline ini.

Implementasi:

```bash
python scripts/train_proposal_hybrid_resnet50_malds.py ... \
  --count-head --count-weight 0.1 --count-loss smooth_l1
```

- Target: `log1p(n_circles)`; loss SmoothL1 (atau MSE).
- Head regresi dari fitur fused (setelah AMFM).
- Metrik tambahan: `count_loss`, `count_mae` per epoch.
- Checkpoint inference mendeteksi `count_head.*` otomatis, jadi Step 48 tidak
  perlu flag tambahan.
- MA-LDS tidak diubah. Gaussian LDS ala Prokhorov belum diimplementasikan karena
  pemetaan jumlah lesi → skor keparahan harus mengikuti method paper, bukan ditebak.

Urutan eksperimen:

1. E1/E2 tanpa count head (baseline bersih).
2. **E5 = E2 + count head**, split/seed/budget sama.
3. Ablation AMFM (on/off) setelah itu.

Batasan: baris sintetis mewarisi `n_circles` dari citra sumber, padahal rotasi bisa
memotong lesi. Target count untuk data augmentasi karena itu mengandung noise.

## Mode tanpa validation (60/40)

Jika `--val-ratio 0`:

- manifest hanya berisi `train` dan `test`;
- Step 45/46 tetap berjalan (val opsional);
- Step 47 memakai `--no-validation`: `best_model.pt` = epoch terakhir pada budget
  optimizer update yang ditetapkan di muka;
- tidak ada early stopping dan tidak ada pemilihan checkpoint dari validation;
- perbandingan E1/E2 tidak boleh memilih pemenang berdasarkan test. Tetapkan
  E2 sebagai primary sebelum melihat hasil, dan laporkan E1 sebagai kontrol.

## Troubleshooting

- `ModuleNotFoundError: No module named 'recovery_common'`
  → `recovery_common.py` belum ada di `SCRIPTS_DIR`/folder script. Upload file
  itu (atau jalankan ulang cell yang mengunduhnya), lalu ulangi step.
- `FileExistsError: ... sudah berisi file`
  → ganti nama folder output (mis. `step42_exact_duplicates_v2`); jangan hapus
  hasil lama.
- `Split tidak PASS`
  → baca `split_integrity.json`; biasanya ada grup lintas split atau kelas
  hilang. Selesaikan duplikat/konflik dulu.

## Keterbatasan yang harus dilaporkan

1. Tidak ada patient ID resmi → independensi antar pasien tidak terbukti.
2. Near-duplicate hanya yang lolos review; false negative mungkin ada.
3. Split baru dari dataset yang sama bukan test eksternal; test lama sudah dipakai
   berulang kali.
4. Level 3 support kecil; interval kepercayaan lebar.
5. Area mask bukan pengganti jumlah lesi; grading visual memerlukan reviewer.
6. `area == density` pada mask biner; belum diubah.
