# Pipeline Validasi Mask & Training Final — Step 34–41

**Versi:** 2026-11-27  
**Konteks:** ACNE04-v2, 1.204 citra, 32.443 anotasi sirkular  
**Prinsip:** validasi target mask dan dataset **sebelum** training; test tidak dipakai untuk memilih apa pun.  
**Agent prompt:** [`docs/AGENT_PROMPT_STEP34_41.md`](AGENT_PROMPT_STEP34_41.md)

---

## 1. Masalah yang diselesaikan

Diagnosis awal menunjukkan banyak citra Level 3 gagal diklasifikasikan. Visualisasi
target mask memperlihatkan area kemerahan/lesi yang tidak seluruhnya tercakup
refined mask. Karena itu urutan kerja diubah:

1. Audit kualitas **target mask** (Circle vs V1 vs V2) tanpa training.
2. Ukur statistik dan fitur morfologi terhadap severity.
3. Tinjau kasus visual secara terarah.
4. Kunci varian mask dengan alasan yang dapat diaudit.
5. Siapkan data training final: validation + augmentasi.
6. Baru training (≤50 epoch), lalu evaluasi test sekali setelah checkpoint dikunci.

Tujuan utamanya bukan mengejar angka setinggi mungkin, tetapi memastikan model
belajar dari target dan distribusi data yang dapat dipertanggungjawabkan.

---

## 2. Goals per tahap

| Tahap | Goal | Bukan goal |
|---|---|---|
| Step 34 | Menyiapkan audit mask V1 vs V2 pada subset pilot | Training |
| Step 35 | Statistik awal Circle/V1/V2 pada 12 citra pilot | Menentukan pemenang dari 12 citra |
| Step 36 | Statistik seluruh 1.204 citra, train dan test terpisah | Memilih parameter dari test |
| Step 37 | Mengukur asosiasi fitur morfologi dengan severity | Menyimpulkan kausalitas |
| Step 38 | Tinjauan visual kasus disagreement/fallback | Mengubah mask otomatis demi angka |
| Step 39 | Mengunci varian mask + manifest final | Training |
| Step 39a | Membuat validation dan augmentasi train 1000/kelas | Mengevaluasi test |
| Step 40 | Training final ≤50 epoch memakai data train-ready | Membuka test |
| Step 40a | Menampilkan tabel training train/val | Training ulang |
| Step 41 | Evaluasi test final pada checkpoint terkunci | Tuning dari test |

---

## 3. Peta Step 34–41

| Step | Cell | Fungsi | Output utama |
|---|---|---|---|
| 34 | Konfigurasi | Parameter, path, helper | — |
| 34a | Validasi input | Cek Step 31, anotasi SHA-256, folder foto, script, provenance | `MASK_V2_COMMAND` |
| 34b | Jalankan pilot | V1/V2 pada 12 citra train | `step34_v2_sigma1.2_k1/` |
| 34c | Baca pilot | Tabel, grid 4 panel, report | — |
| 35 | Statistik pilot | Circle vs V1 vs V2 pada pilot | `step35_stats_pilot_v2/` |
| 36 | Statistik penuh | Generate V2 1.204 citra + statistik train/test | `step36_full_v2/`, `step36_stats_train/`, `step36_stats_test/` |
| 37 | Morfologi vs severity | Boxplot + korelasi fitur train | `morphology_by_severity.png` |
| 38 | Tinjauan visual | Kasus disagreement V1–V2 | `comparison_grid.png` |
| 39 | Finalisasi | Kunci varian mask + manifest final | `dataset_final_<variant>/manifest_final.csv` |
| 39a | Data training siap | Split val 15% + augmentasi train 1000/kelas | `dataset_final_<variant>_train_ready/` |
| 40 | Training final | ≤50 epoch, validasi aktif | `training_runs/<RUN_ID>/` |
| 40a | Tabel training | Baca `metrics.json`/history | — |
| 41 | Evaluasi test | Inference checkpoint final | `test_eval/` |

---

## 4. Detail tahap

### Step 34–34c — Audit pilot V1 vs V2

**Tujuan:** melihat langsung apakah refined mask menutupi lesi atau justru
menyusut menjadi circle anotasi.

Empat panel:

| Panel | Isi |
|---|---|
| `input 512` | Foto letterbox 512, warna asli |
| `circle (anotasi)` | Circle union anotasi resmi |
| `V1 global baseline` | Reproduksi `refine_masks_color.py` Step 31 |
| `V2 local annulus` | Baseline warna lokal per anotasi |

Konfigurasi pilot default:

- 12 citra train, 3 per level, seed 42.
- V2: `sigma=1.2`, `kernel=1×1`, `annulus=3×`, `min-valid-pixels=10`,
  `min-candidate-ratio=0.02`.
- V1 kontrol: `sigma=1.2`, `kernel=5×5`, `min-ratio=0.02`.

**Guard:** tidak menimpa output lama; anotasi v2 diunduh ulang dengan verifikasi
SHA-256 bila `/content` hilang.

### Step 35 — Statistik pilot

Script: `scripts/evaluate_mask_variants.py`.

Output:

- `metrics_per_image.csv`
- `metrics_per_annotation.csv`
- `metrics_per_class.csv`
- `morphology_severity_correlation.csv`
- `audit_report.md`
- `comparison_grid.png`

Metrik penting:

- IoU/retention terhadap **circle anotasi** = kesesuaian dengan pseudo-annotation,
  bukan akurasi klinis.
- Fallback = V2 memakai circle karena kandidat terlalu kecil.
- Anotasi di luar gambar dihitung terpisah.

### Step 36 — Statistik penuh 1.204 citra

Alur:

1. Generate V1/V2 untuk semua citra pada split 60/40, tanpa `--split`,
   `--per-level 0`, `--no-overlays`.
2. Evaluasi **train** dan **test** ke folder berbeda.
3. Test hanya dilaporkan; keputusan tetap dari train/validation.

Output:

- `step36_full_v2/refined_manifest_v2.csv`
- `step36_stats_train/`
- `step36_stats_test/`

### Step 37 — Morfologi vs severity

Fitur: luas relatif, rasio terhadap circle, jumlah komponen, sebaran, compactness,
centroid. Keluaran: `morphology_by_severity.png` + tabel Pearson/Spearman/eta².

**Interpretasi:** korelasi/eta² hanya asosiasi pada dataset ini. Bukan kausalitas,
bukan validasi klinis, dan tidak boleh dipakai untuk mengubah label.

### Step 38 — Tinjauan visual

Menampilkan kasus dengan selisih IoU/area V1–V2 terbesar dan IoU V2 terendah.
Tujuannya memeriksa apakah perbedaan statistik masuk akal secara visual.

### Step 39 — Finalisasi varian

Script: `scripts/finalize_mask_dataset.py`.

Input: manifest V2 penuh dari Step 36.  
Pilihan: `circle`, `v1`, atau `v2`.  
Wajib: `FINAL_DECISION_NOTE` yang menjelaskan alasan.

Output:

- `manifest_final.csv` (kolom `refined_mask` diarahkan ke varian terpilih)
- `final_counts.csv`
- `finalization.json`
- `finalization_report.md`

Mask tidak disalin/diubah; folder lama tidak ditimpa.

### Step 39a — Validation + augmentasi (WAJIB untuk training final)

Masalah yang diperbaiki: manifest final Step 39 masih **raw 60/40**. Level 3
hanya sekitar 13 citra train, sehingga model bias ke kelas mayoritas.

Alur Step 39a:

1. Split 15% train → `val`, stratified per kelas.
2. Augmentasi **hanya train** sampai 1000 citra/kelas memakai mask varian terpilih.
3. Test disalin tanpa perubahan.

Output: `dataset_final_<variant>_train_ready/`

- `manifest_train_val.csv`
- `manifest_final_aug.csv`
- `aug_pairs/`
- `split_validation.json`

**Guard Step 40:** menolak manifest tanpa `val` dan menolak train yang belum
mencapai target 1000/kelas.

### Step 40 — Training final ≤50 epoch

Konfigurasi default:

- 20 epoch phase 1 (encoder frozen) + 30 epoch phase 2 = **50 epoch**.
- Image 512, batch 4.
- Validation aktif; checkpoint dipilih dari validation loss.
- `--skip-test-eval`: test tidak dibuka.

Output: `training_runs/<RUN_ID>/`

- `best_model.pt`
- `last_checkpoint.pt`
- `metrics.json`
- `history_partial.json`

### Step 40a — Tabel training

Membaca `metrics.json`; jika belum ada, mencoba `history_partial.json` atau
`last_checkpoint.pt`. Otomatis mencari run terbaru di `TRAIN_DIR`.

Menampilkan train dan val loss/accuracy/Kappa/Dice/IoU, `checkpoint_selection`,
dan `best_epoch`.

### Step 41 — Evaluasi test final

Script: `infer_testing_images.py` + arsitektur trainer versi terbaru.

Output:

- `test_eval/summary.json`
- `testing_predictions.csv`
- `classification_report.csv`
- `confusion_matrix.png`
- `pred_masks/`, `overlays/`

Menampilkan accuracy, Kappa, mean Dice/IoU, classification report per kelas,
confusion matrix, dan daftar Level 3 yang salah.

**Guard:** tidak training, tidak mengubah checkpoint, tidak memilih parameter
dari test.

---

## 5. Struktur output di Drive

```text
output/experiments_512_mask_v2/
├── step34_v2_sigma1.2_k1/          # pilot V1/V2
├── step35_stats_pilot_v2/
├── step36_full_v2/                 # 1.204 mask V1/V2
├── step36_stats_train/
├── step36_stats_test/
├── dataset_final_v2/               # keputusan Step 39
│   ├── manifest_final.csv
│   ├── finalization.json
│   └── finalization_report.md
└── dataset_final_v2_train_ready/   # Step 39a
    ├── manifest_train_val.csv
    ├── manifest_final_aug.csv
    └── aug_pairs/

output/training_runs/<FINAL_RUN_ID>/
├── best_model.pt
├── metrics.json
├── history_partial.json
└── test_eval/                      # Step 41
```


### 5.1 Logging dan reporting untuk AI agent

Setiap step menulis log terstruktur lewat helper `pipeline_log()` di cell konfigurasi Step 34:

```text
output/experiments_512_mask_v2/pipeline_logs/
├── pipeline_steps.jsonl   # append-only, semua event
├── step_35.json           # event terakhir Step 35
├── step_36.json
├── ...
└── step_41.json
```

Format satu event:

```json
{
  "timestamp": "2026-11-27T...",
  "step": "40",
  "status": "started|success|failed",
  "run_id": "final_mask_50ep_02",
  "manifest": "...",
  "output_dir": "...",
  "command": ["python", "..."],
  "metrics": {"best_epoch": 50, "final_train": {}}
}
```

Aturan:

- `started` ditulis sebelum proses berjalan.
- `success` hanya ditulis jika step selesai.
- `failed` ditulis untuk training/test yang menangkap `CalledProcessError`.
- Jika `started` tidak diikuti `success`/`failed`, step terputus; agent harus melaporkan itu sebagai kegagalan/ketidaklengkapan.

Console log tambahan:

- `training_runs/<RUN_ID>/training_console.log` — seluruh stdout/stderr Step 40.
- `test_eval/test_console.log` — seluruh stdout/stderr Step 41.

Contoh membaca rekap untuk laporan agent:

```python
import json
from pathlib import Path

logs = Path('/content/drive/MyDrive/acne_project/output/experiments_512_mask_v2/pipeline_logs/pipeline_steps.jsonl')
events = [json.loads(line) for line in logs.read_text().splitlines() if line.strip()]
latest = {}
for event in events:
    latest[event['step']] = event
for step in sorted(latest):
    print(step, latest[step]['status'], latest[step]['timestamp'])
```

Gunakan log ini untuk mengisi laporan otomatis, bukan menebak dari output cell.

---

## 6. Aturan metodologis

1. **Test lock.** Test tidak dipakai memilih varian mask, augmentasi, epoch, atau
   checkpoint. Test hanya dievaluasi sekali setelah keputusan dikunci.
2. **Pseudo-mask, bukan ground truth klinis.** Circle adalah proxy anotasi.
   Kemerahan belum tentu lesi aktif.
3. **Augmentasi wajib untuk training final.** Hanya train yang diaugmentasi;
   val/test tidak.
4. **Validation wajib.** Tanpa val, `best_model.pt` hanya epoch terakhir dan
   tidak ada dasar pemilihan checkpoint.
5. **Provenance.** Simpan hash manifest, script, konfigurasi, dan alasan keputusan.
6. **Jangan menimpa hasil lama.** Gunakan folder/run ID baru.

---

## 7. Temuan pilot (12 citra train)

Dari `step34`/`step35` pilot:

| Kelas | Anotasi | Fallback V2 | Median IoU V1* | Median IoU V2* |
|---|---:|---:|---:|---:|
| Level 3 | 462 | 13 (2,8%) | 0,891 | 0,904 |
| Level 2 | 127 | 1 | 0,960 | 0,974 |
| Level 1 | 54 | 4 | 0,825 | 0,916 |
| Level 0 | 65 | 1 | 0,776 | 0,950 |

\*IoU terhadap circle anotasi, bukan akurasi klinis.

Catatan:

- 19 fallback dari 708 anotasi pilot.
- Median retention V2 = 1,0; artinya separuh anotasi non-fallback tidak menyusut
  dari circle. Ini belum membuktikan batas mask lebih tepat.
- Pilot hanya 12 citra; bukan dasar final.
- Pada pemeriksaan JSON penuh, ditemukan 49 anotasi di luar batas gambar pada
  `levle2_130.jpg`. Audit penuh harus menghitung kasus ini terpisah.

---

## 8. Pelajaran dari run training raw

Run `final_mask_50ep_01` (sebelum Step 39a) memakai manifest raw 60/40:

- 50 epoch, tanpa validation.
- Train accuracy ~0,606; Kappa ~0,384; macro F1 ~0,604; Dice ~0,487; IoU ~0,334.
- Level 3 hanya ~13 citra train.

Hasil ini **bukan bukti V2 gagal** dan tidak boleh dipakai sebagai hasil final.
Penyebab utamanya: tidak ada augmentasi, tidak ada validation, dan distribusi
kelas sangat timpang. Step 39a + Step 40 baru adalah protokol yang benar.

---

## 9. Cara menjalankan end-to-end

1. Setup Step 1–8, mount Drive.
2. **34 → 34a → 34b → 34c** (pilot).
3. **35** (statistik pilot).
4. **36** (full 1.204; train/test terpisah).
5. **37 → 38** (morfologi + visual).
6. **39** (kunci varian + alasan).
7. **39a** (val 15% + augmentasi 1000/kelas).
8. **40** dengan `FINAL_TRAIN_RUN_ID` baru.
9. **40a** (tabel train/val).
10. **41** (test sekali, setelah checkpoint dikunci).

---

## 10. Troubleshooting

| Masalah | Penyebab | Solusi |
|---|---|---|
| `DRIVE_ACCESS_MODE: None` | Step 4 belum jalan | Jalankan Step 4, mount Drive |
| `metrics.json` tidak ada | Path/run ID beda atau training belum selesai | Step 40a auto-detect; cek `history_partial.json`/`last_checkpoint.pt` |
| Exit code 2 saat training | Trainer lama tidak mengenal flag | Step 40 force-download trainer + cek flag |
| Exit code 1 saat test | Infer/arsitektur tidak sinkron dengan checkpoint | Step 41 force-download infer + trainer + baseline |
| Train tidak seimbang | Step 39a dilewati | Jalankan 39a; Step 40 akan menolak raw |
| Folder output sudah ada | Menimpa hasil lama | Gunakan run ID/folder baru |

---

## 11. Checklist sebelum klaim hasil

- [ ] Step 36 selesai pada seluruh 1.204 citra.
- [ ] Statistik train dan test dipisah.
- [ ] Varian mask dipilih dari train/validation, bukan test.
- [ ] `FINAL_DECISION_NOTE` terisi.
- [ ] Step 39a menghasilkan train 1000/kelas + val.
- [ ] Training ≤50 epoch dan checkpoint dipilih dari val.
- [ ] Test hanya dievaluasi sekali.
- [ ] Laporkan Level 3 recall/confusion matrix, bukan hanya accuracy.
- [ ] Sebutkan batasan pseudo-mask dan anotasi.

---

## 12. File terkait

| File | Fungsi |
|---|---|
| `acne_colab.ipynb` | Step 34–41 |
| `scripts/refine_masks_color_v2.py` | V1/V2 mask + audit pilot/full |
| `scripts/evaluate_mask_variants.py` | Statistik varian + morfologi + grid |
| `scripts/finalize_mask_dataset.py` | Manifest final |
| `scripts/split_train_validation.py` | Split val stratified |
| `scripts/augment_train_pairs_offline.py` | Augmentasi train |
| `scripts/infer_testing_images.py` | Evaluasi test |
| `scripts/train_proposal_hybrid_resnet50_malds.py` | Trainer |
| `tests/test_mask_refinement_v2.py` | Test audit mask |
| `tests/test_mask_pipeline_steps.py` | Test Step 34–41 + logging |
| `docs/AGENT_PROMPT_STEP34_41.md` | Prompt siap pakai untuk AI agent |
| `output/.../pipeline_logs/` | Log terstruktur per step |

---

## 13. Batasan

- Audit ini tidak memvalidasi diagnosis klinis, identitas pasien, atau kebocoran subjek.
- Circle anotasi bukan ground truth piksel tingkat klinis.
- IoU terhadap circle bukan ukuran akurasi lesi sebenarnya.
- Korelasi morfologi–severity bukan kausalitas.
- Augmentasi sintetis bukan citra/pasien baru.
- Test set sudah pernah dilihat pada eksperimen sebelumnya; dokumentasikan
  keterbatasan ini dan gunakan holdout independen bila memungkinkan.
