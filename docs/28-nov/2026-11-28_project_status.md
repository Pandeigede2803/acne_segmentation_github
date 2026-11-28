# Status Proyek — 28 November 2026

**Tanggal:** 2026-11-28  
**Commit:** `f1dbced`  
**Scope:** audit dataset, implementasi Step 42–48, keputusan protokol training
recovery ACNE04-v2.

Dokumen ini merangkum apa yang dikerjakan hari ini, angka yang sudah diperoleh,
keputusan yang diambil, bug yang ditemukan, dan langkah berikutnya. Dokumen
protokol lengkap ada di `docs/28-nov/step42_48_dataset_recovery_experiment.md`;
implementasi ada di `docs/RECOVERY_STEP42_48.md`.

---

## 1. Ringkasan eksekutif

Hari ini fokusnya bukan mengejar akurasi, tetapi memperbaiki validitas dataset
dan menyiapkan eksperimen yang bisa dipertanggungjawabkan.

Yang selesai:

1. Audit Step 34–41 dan kuantifikasi masalah augmentasi 1000/kelas.
2. Implementasi Step 42–48 (dedup, near-duplicate, split grouped, visual audit,
   controlled augmentation, audit class weight, training terkontrol, evaluasi final).
3. Penambahan cell status + auto-skip agar step yang sudah selesai tidak
   dijalankan ulang dan tidak gagal karena folder output sudah ada.
4. Perbaikan beberapa bug notebook (logging, variabel hilang, guard).
5. Keputusan protokol baru: **60/40 tanpa validation**, checkpoint = epoch akhir
   pada budget update tetap.

Yang belum:

- Rerun Step 43B/45/46 dengan split 60/40.
- Training E2 dan E1.
- Evaluasi test final.

---

## 2. Temuan audit utama

Sumber angka: `docs/AUDIT_ACNE_V2_STEP34_PLUS.md` dan
`docs/audit_step34/audit_report.md`.

| Temuan | Angka |
|---|---|
| Duplikasi sintetis Level 3 (skema lama) | median ~88 salinan/citra, maks 103 |
| Effective sample size Level 3 (skema lama) | 10,9 dari 1000 baris |
| Augmentasi sintetis total (skema lama) | 3.387 dari 4.000 baris (84,7%) |
| Retensi area mask setelah rotasi | median 1,000; P1 0,944–0,987 |
| Lesi terpotong total | 1,0–5,8% tergantung level |
| MA-LDS epsilon Level 2 vs 3 | AUC 0,784; 54,5% Level 3 di bawah P95 Level 2 |
| `area` vs `density` pada mask biner | identik |

Kesimpulan: augmentasi 1000/kelas tidak menambah informasi pasien, hanya
mengulang sedikit citra. Ini kandidat utama penyebab lemahnya recall Level 3.

### Sumber label severity

- Prefix nama file (`levle0`–`levle3`) berbeda dari label NNEW pada **34 file v2**
  (42 file pada seluruh ACNE04 1.457 citra).
- Label NNEW cocok 100% dengan grade Hayashi dari count NNEW; prefix tidak cocok
  pada 34/34 file tersebut.
- `n_circles` v2 bukan count NNEW: median 19 vs 6, dan grade dari `n_circles`
  hanya cocok dengan label NNEW pada 363/1.204 citra.
- Rincian: `docs/28-nov/label_nnew_vs_filename.md`; daftar tinjauan manual:
  `docs/28-nov/label_mismatch_34_review.csv`.
- **Pipeline Step 42 memakai prefix nama file sebagai label**
  (`level_from_filename` di `refine_masks_color.py`), jadi 34 file ikut salah
  label. Patch manifest dengan `scripts/apply_ground_truth_labels.py` dan
  jalankan ulang Step 42 sebelum Step 43B.
- Ground truth deterministik: `scripts/build_ground_truth_manifest.py`
  (GT-A = label/count NNEW, mask = circle v2). Review prioritas 47 file; audit
  sampel 60 citra (15 per grade).

---

## 3. Implementasi Step 42–48

| Step | Script | Output utama |
|---|---|---|
| 42 | `detect_exact_duplicates.py` | `clean_original_manifest.csv`, `quarantine_manifest.csv` |
| 43A | `detect_near_duplicates.py` | `near_duplicate_review.csv` |
| 43B | `build_grouped_split.py` | `split_manifest.csv`, `split_integrity.json` |
| 44 | `audit_augmentation_visual.py` | grid L2/L3/kontrol, `visual_audit_scores.csv` |
| 45 | `build_controlled_augmentation.py` | `controlled_aug_manifest.csv`, lineage |
| 46 | `audit_class_weighting.py` | `experiment_matrix.yaml`, `training_budget.md` |
| 47 | `train_proposal_hybrid_resnet50_malds.py` | `epoch_metrics.csv`, `best_model.pt` |
| 48 | `evaluate_final_experiment.py` | `test_metrics.json`, confusion matrix |

Tambahan E5: `--count-head` (regresi `n_circles`, target `log1p`, loss SmoothL1).
Inference mendeteksi head ini otomatis dari checkpoint. Gaussian LDS belum
diimplementasikan; MA-LDS tetap dipakai.

Helper murni: `scripts/recovery_common.py`.  
Prompt operator: `docs/AGENT_PROMPT_STEP42_48.md`.

Perubahan trainer:

- `--checkpoint-metric val_loss|val_macro_f1`
- `--early-stopping-patience`
- `--max-optimizer-steps`
- per-class precision/recall/F1 di history
- `epoch_metrics.csv`
- `--no-validation` tetap tersedia; `best_epoch` sekarang benar walau budget
  berhenti sebelum total epoch.

---

## 4. Hasil run hari ini

### Step 42 — exact duplicates

| Metrik | Nilai |
|---|---:|
| Baris sebelum | 1.204 |
| Baris sesudah | 1.159 |
| Grup duplikat | 36 |
| Baris berlebih | 36 |
| Konflik label | 9 grup (18 baris dikarantina) |
| Grup lintas train–test | 16 |
| Grup dengan mask berbeda | 36 |

Level 3 tidak terdampak duplikasi (tetap 22 citra).

Catatan: 36 grup lebih banyak dari audit sebelumnya (17) karena deteksi hash
piksel menangkap duplikat hasil re-encode.

### Step 43 — split

Split 60/20/20 (dengan validation) yang sempat dipakai:

| Split | L0 | L1 | L2 | L3 | Total |
|---|---:|---:|---:|---:|---:|
| train | 274 | 333 | 76 | 14 | 697 |
| val | 91 | 111 | 25 | 4 | 231 |
| test | 91 | 111 | 25 | 4 | 231 |

Status integritas: PASS (group disjoint, semua kelas ada).

### Step 44 — visual audit augmentasi lama

| Metrik | Nilai |
|---|---:|
| pass | 3.368 (99,4%) |
| suspect | 16 (0,47%) |
| fail | 3 (0,09%) |
| area ratio median | 1,000 |
| area ratio P1 / P99 | 0,971 / 1,016 |
| area ratio min / max | 0,848 / 1,200 |

Kesimpulan: transformasi lama layak dipakai ulang; 3 fail terlalu sedikit untuk
membatalkan. Audit ini berlaku untuk transformasinya, bukan sampel controlled.

### Step 45 — controlled augmentation

Formula: `target_c = max(n_c, min(300, 7 × n_c))`, maksimum 6 sintetis per
original. Status integritas: PASS.

### Step 46 — gate

Status: **PASS**.

Bobot balanced dari train original (split 60/20/20):

| Level | Train | Bobot | Kontribusi |
|---|---:|---:|---:|
| 0 | 274 | 0,6359 | 0,25 |
| 1 | 333 | 0,5233 | 0,25 |
| 2 | 76 | 2,2928 | 0,25 |
| 3 | 14 | 12,4464 | 0,25 |

Matrix E1–E4 (split 60/20/20):

| Run | Train | Augmentasi | Weight | Update/epoch | Epoch budget |
|---|---:|---|---|---:|---:|
| E1 | 697 | none | uniform | 175 | 50 |
| E2 | 1.031 | controlled 300/cap7 | uniform | 258 | 34 |
| E3 | 4.000 | 1000/kelas | uniform | 1.000 | 9 |
| E4 | 1.031 | controlled 300/cap7 | balanced | 258 | 34 |

Referensi budget: 8.750 optimizer updates.

---

## 5. Keputusan protokol terbaru

Setelah melihat validation Level 3 hanya 4 citra (60/20/20), diputuskan:

**Gunakan 60/40 tanpa validation.**

Konsekuensi:

1. Split hanya `train` dan `test`; test sekitar 40%.
2. Checkpoint = **epoch akhir** pada budget update yang ditetapkan di muka.
3. Tidak ada early stopping dan tidak ada pemilihan checkpoint dari validation.
4. E1 vs E2 tidak boleh dipilih pemenangnya berdasarkan test. E2 ditetapkan
   sebagai primary sebelum hasil dilihat; E1 sebagai kontrol.
5. Step 43B/45/46 harus dijalankan ulang ke folder baru karena split berubah.
6. Step 42, 43A, dan 44 tidak perlu diulang.

Dukungan teknis yang ditambahkan:

- `assign_grouped_splits` menerima `val_ratio=0`.
- `build_grouped_split.py` menerima `--val-ratio 0` dan hanya mewajibkan kelas
  lengkap di train/test.
- `build_controlled_augmentation.py` menerima manifest train/test tanpa val.
- Notebook Step 47 punya `STEP47_USE_VALIDATION = False` → `--no-validation`.
- Trainer memperbaiki `best_epoch` saat budget berhenti sebelum total epoch.

---

## 6. Bug yang ditemukan dan diperbaiki

| Lokasi | Masalah | Perbaikan |
|---|---|---|
| Step 42–48 | folder output sudah ada → `FileExistsError` berulang | auto-skip berbasis hash input + pesan yang menyebut variabel |
| Step 44 | `STEP44_OUTPUT` dan parameter sampling hilang → `NameError` | variabel dikembalikan |
| Step 45 | `pipeline_log(..., status=...)` bentrok dengan parameter | diganti `integrity_status` |
| Step 43B | jalur `failed` memakai `status=` | diganti `integrity_status` |
| Step 40a | bug laten `status=TRAIN_STATUS` | diganti `train_status` |
| Step 46 | guard memblokir metrics lama tanpa hash weight | hash weight opsional |
| Trainer | `best_epoch` salah saat `--no-validation` + budget stop | pakai epoch terakhir history |
| Notebook | status/auto-skip belum ada | cell 41b + guard di tiap step |

Test suite: **175 OK, 5 skipped**, termasuk:
- eksekusi semua cell 42–48 dengan toggle OFF,
- larangan `pipeline_log(..., status=/step=...)`,
- unit test alokasi, split grup, metrik, Wilson CI.

---

## 7. Status saat ini

| Item | Status |
|---|---|
| Audit Step 34–41 | selesai |
| Step 42 | selesai (1.159 baris) |
| Step 43A | selesai |
| Step 43B (60/20/20) | selesai, tetapi akan diganti |
| Step 44 | selesai (3368/16/3) |
| Step 45 (60/20/20) | selesai, tetapi akan diganti |
| Step 46 (60/20/20) | PASS, tetapi akan diganti |
| Step 43B/45/46 (60/40) | **belum dijalankan** |
| Step 47 E2 / E1 | **belum** |
| Step 48 | **belum** |

---

## 8. Langkah berikutnya

1. Patch manifest sumber dengan label NNEW:
   `scripts/apply_ground_truth_labels.py` → `refined_manifest_v2_nnew.csv`.
2. Arahkan `STEP42_SOURCE_MANIFEST` ke manifest baru dan **jalankan ulang Step 42**
   (CPU). Output Step 42 lama tidak boleh dipakai karena input label berubah.
3. Step 43B ke `step43_grouped_split_60_40` dengan `val_ratio=0`, `test_ratio=0.4`.
4. Step 45 ke `step45_controlled_aug_60_40`.
5. Step 46 ke `step46_weighting_60_40`; pastikan gate PASS.
6. Step 47 E2 (`USE_VALIDATION=False`, uniform, run ID baru).
7. Step 47 E1 (manifest split tanpa augmentasi, run ID baru).
8. Step 48 evaluasi test sekali per run, setelah checkpoint terkunci.
9. E5 (opsional): E2 + counting head, split/seed/budget sama, untuk menguji
   komponen Prokhorov yang belum ada di model.

Untuk training di luar Colab: `docs/VAST_AI_TRAINING.md`.

---

## 9. Batasan yang wajib dilaporkan

1. Tidak ada patient ID resmi → independensi antar pasien tidak terbukti.
2. Near-duplicate hanya yang lolos review; false negative mungkin ada.
3. Split baru dari dataset yang sama bukan test eksternal; test lama sudah
   dipakai berulang kali.
4. Tanpa validation: checkpoint bukan hasil seleksi, melainkan epoch akhir pada
   budget tetap.
5. Level 3 sangat kecil; interval kepercayaan lebar.
6. Augmentasi tidak menambah pasien; 13–14 citra Level 3 tetap 13–14 citra.
7. `area == density` pada mask biner; belum diubah.

---

## 10. Referensi

- Protokol: `docs/28-nov/step42_48_dataset_recovery_experiment.md`
- Runbook E5a (AMFM + count head, 60/40, 50-epoch budget):
  `docs/28-nov/RUN_E5A_COUNT_60_40.md`
- Implementasi: `docs/RECOVERY_STEP42_48.md`
- Prompt operator: `docs/AGENT_PROMPT_STEP42_48.md`
- Audit dataset: `docs/AUDIT_ACNE_V2_STEP34_PLUS.md`
- Hasil audit otomatis: `docs/audit_step34/audit_report.md`
- Training Vast.ai: `docs/VAST_AI_TRAINING.md`
- Pipeline Step 34–41: `docs/MASK_VALIDATION_STEP34_41.md`
