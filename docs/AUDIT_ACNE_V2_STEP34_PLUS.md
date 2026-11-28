# Audit Eksperimen ACNE Severity V2 — Dataset, Augmentasi, MA-LDS, Training

**Tanggal audit:** 2026-11-28  
**Scope:** Step 34–41 (mask V2, statistik, morfologi, finalisasi, training final, test)  
**Metode:** audit kode + replay deterministik split/augmentasi dari anotasi resmi,
lalu pengukuran geometri pada circle mask 512. Angka yang tidak dapat diverifikasi
dari artefak lokal ditandai **PERLU ARTEFAK**.

**Hasil pengukuran terbaru:** [`docs/audit_step34/audit_report.md`](audit_step34/audit_report.md),
dihasilkan oleh `scripts/audit_step34_dataset_augmentation.py`.

---

## Ringkasan eksekutif

Temuan paling penting:

1. **Run `final_mask_50ep_01` bukan run final yang sah** untuk klaim hasil:
   memakai manifest raw 60/40, tanpa augmentasi, tanpa validation. `best_model.pt`
   = epoch terakhir, bukan best validation.
2. **Augmentasi 1000/kelas sangat timpang per citra asli.** Terverifikasi dari
   anotasi resmi: Level 3 punya 11 citra asli, median **88 salinan** per citra,
   maksimum **103 salinan**, dan effective sample size hanya **10,9**. Level 2
   median 14 salinan, ESS 63,3.
3. **Tidak ada eksperimen “tanpa augmentasi ×1000”** di kode. Yang ada: raw,
   aug100 (Step 11), aug1000 (Step 31b/39a), dan ablation Step 20. Konfigurasinya
   tidak identik, jadi tidak bisa dibandingkan sebagai kontrol murni.
4. **Morphological target dan MA-LDS dihitung dari mask setelah augmentasi**
   (online/offline), bukan dari citra original.
5. **`area` dan `density` identik untuk mask biner.** Pada target MA-LDS,
   `density` tidak menambah informasi independen; epsilon efektif
   `0.05 + 2.0·area + 0.2·spread`.
6. **Split dilakukan sebelum augmentasi** dan hanya train yang diaugmentasi,
   tetapi **tidak ada patient ID**, sehingga kebocoran antar-split per pasien
   tidak dapat dikesampingkan.
7. **Rotasi ±15° memotong sebagian lesi.** Pada replay, 26–38% sampel sintetis
   kehilangan ≥0,1% area mask; 1,0–3,4% (Level 0/1/3) sampai 5,8% (Level 2)
   sampel kehilangan minimal satu anotasi secara total.
8. **MA-LDS tidak memisahkan Level 2 vs Level 3 sebaik klaim pilot.**
   Epsilon Level 3 mean 0,151 vs Level 2 0,096, tetapi AUC hanya **0,784** dan
   54,5% citra Level 3 berada di bawah P95 Level 2.
9. **Step 40a hanya menampilkan 10 epoch terakhir**, bukan seluruh history.
   Untuk audit epoch-by-epoch, baca `metrics.json`/`history_partial.json`.

**Gerbang sebelum training final:** putuskan ulang strategi augmentasi Level 2/3,
tambahkan audit kuantitatif area/lesi setelah rotasi, dan pastikan run ID serta
manifest training-ready yang dipakai.

---

## 1. Dataset dan augmentasi

### 1. Berapa jumlah citra original training setiap severity sebelum augmentasi?

Angka di bawah adalah **replay deterministik** dari `split_train_test_manifest.py`
(60/40, seed 42) dan `split_train_validation.py` (15% validation, seed 42) pada
1.204 citra resmi. Hasilnya cocok dengan aturan kode.

- Split 60/40:
  - Level 0: 282 train / 188 test
  - Level 1: 347 / 232
  - Level 2: 79 / 54
  - Level 3: 13 / 9
  - **Total train 721, test 483**
- Setelah Step 39a memisahkan 15% train menjadi validation:
  - Train: **240 / 295 / 67 / 11** = **613**
  - Validation: **42 / 52 / 12 / 2** = **108**
  - Test tetap **483**
- Total setelah augmentasi: **4000** baris train (1000/kelas).

Catatan: angka ini terverifikasi dari isi JSON resmi (1.204 gambar, 32.443 anotasi).
Nama file val spesifik tetap harus dibaca dari `split_validation.json` pada run
aktual; urutannya deterministik selama manifest dasar memakai urutan nama yang sama.

### 2. Transformasi augmentasi dan parameternya

**Offline augmentation** (`scripts/augment_train_pairs_offline.py`), hanya train:

| Transformasi | Parameter | Probabilitas |
|---|---|---|
| Rotasi | uniform `[-15°, +15°]`, image bilinear, mask nearest, fill hitam | selalu |
| Horizontal flip | — | 0.5 |
| Brightness | faktor uniform `[0.85, 1.15]` | selalu (jika >0) |
| Contrast | faktor uniform `[0.85, 1.15]` | selalu (jika >0) |
| JPEG quality | 95 untuk image; mask PNG lossless | selalu |

**Online augmentation saat training** (`PreprocessedAcneDataset.__getitem__`):

| Transformasi | Parameter | Probabilitas |
|---|---|---|
| Rotasi | uniform `[-15°, +15°]` | setiap epoch, train |
| Horizontal flip | 0.5 | setiap epoch, train |
| Brightness | `[0.85, 1.15]` | setiap epoch, train |
| Contrast | `[0.85, 1.15]` | setiap epoch, train |

Tidak ada crop dan tidak ada scaling acak pada augmentasi offline/online.
Scaling hanya terjadi saat preprocessing awal ke 512 letterbox.

### 3. Apakah image dan mask mendapat transformasi geometris identik?

**Ya.**

- Offline: satu `angle` dan satu `flip` yang sama diterapkan ke image dan mask.
- Online: `augment_pair()` menerapkan `rotate()` dan `flip()` yang sama ke
  keduanya; color enhancement hanya pada image.
- Perbedaan interpolasi: image bilinear, mask nearest. Koordinat transformasi
  sama, tetapi batas mask bisa bergeser ≤1 piksel.

### 4. Apakah cropping/rotation/scaling menghilangkan lesi atau mengubah luas?

- **Cropping:** tidak ada. Preprocessing memakai `crop_ratio=1.0`.
- **Scaling:** letterbox ke 512, bukan crop; rasio aspek dipertahankan.
- **Rotation:** dapat memotong area dekat sudut/border karena kanvas tidak
  diperluas (`expand=False`) dan fill hitam. Mask juga dirotasi dengan fill 0.
- **Flip:** tidak mengubah luas, hanya posisi.
- **Brightness/contrast:** tidak mengubah geometri, tetapi dapat mengubah
  deteksi warna pada mask V1/V2 jika augmentasi diterapkan sebelum mask creation
  (di pipeline ini mask sudah dibuat sebelum augmentasi, jadi target aman).

**Hasil pengukuran replay (circle mask anotasi, 512 letterbox, 3.387 sampel sintetis):**

| Level | Median retensi | P1 | Min | Sampel area turun ≥0,1% | Turun ≥1% | Turun ≥5% | Ada lesi hilang total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,0000 | 0,9444 | 0,8657 | 35,4% | 8,0% | 1,3% | 1,0% |
| 1 | 1,0000 | 0,9669 | 0,8805 | 37,9% | 7,0% | 0,4% | 3,4% |
| 2 | 1,0000 | 0,9704 | 0,9162 | 33,9% | 5,7% | 0,3% | 5,8% |
| 3 | 1,0000 | 0,9870 | 0,9779 | 25,6% | 2,7% | 0,0% | 1,0% |

Jadi rotasi bukan masalah besar untuk mayoritas sampel, tetapi tetap ada
ekor distribusi yang kehilangan area/lesi, terutama di Level 2. Ini diukur pada
circle anotasi; mask V1/V2 dapat sedikit berbeda karena bentuknya hasil refinement.

### 5. Apakah penggandaan setiap citra original merata dalam kelas?

**Tidak dijamin.** Offline augmenter memilih sumber dengan
`rng.choice(source_rows)` **with replacement** untuk setiap sampel sintetis.
Replay deterministik (seed 42) memberi:

| Kelas | Train asli | Sintetis ditambah | Total | Median salinan | Max salinan | Sumber tanpa salinan | ESS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Level 0 | 240 | 760 | 1000 | 3 | 11 | 8 | 202,0 |
| Level 1 | 295 | 705 | 1000 | 2 | 7 | 27 | 246,1 |
| Level 2 | 67 | 933 | 1000 | 14 | 28 | 0 | 63,3 |
| Level 3 | 11 | 989 | 1000 | 88 | 103 | 0 | **10,9** |

ESS = (Σn)²/Σn², dengan n = jumlah baris per citra asli (asli + salinan).
ESS Level 3 ≈ 10,9 berarti model efektif hanya melihat sekitar 11 citra
independen meskipun ada 1000 baris.

Total train final 4000; **84,7% sintetis**, dan Level 3 **98,9% sintetis**.
Satu citra Level 3 menyumbang hingga **10,4%** dari seluruh baris kelasnya.
Ini bukan distribusi citra independen dan berisiko membuat model menghafal
sedikit citra Level 3.

### 6. Apakah split dilakukan sebelum augmentasi dan bebas leakage?

- **Split sebelum augmentasi: ya.** Step 31 membuat split train/test dari 1.204
  citra asli; Step 39a memisahkan validation dari train; augmentasi hanya train.
- **Test/validation tidak diaugmentasi.**
- **Leakage augmented copies:** tidak ada copy train yang masuk val/test.
- **Leakage pasien:** tidak dapat dikesampingkan karena tidak ada patient ID.
  Split berbasis filename; citra dari pasien yang sama bisa berada di train dan
  test.
- **PERLU ARTEFAK:** cek overlap nama pasien jika metadata tersedia; jika tidak,
  dokumentasikan sebagai batasan.

---

## 2. Morphological features dan MA-LDS

### 7. Distribusi area, density, centroid X/Y, spread sebelum/sesudah augmentasi

Fitur dihitung oleh `mask_morphology()`:

- `area` = mean mask (0–1)
- `density` = proporsi piksel > 0.5
- `centroid_x`, `centroid_y` = pusat massa ternormalisasi
- `spread` = rata-rata deviasi absolut X/Y dari centroid

**Hasil replay per kelas (median, circle mask 512):**

| Level | Area asli (P5–P95) | Area sintetis | Spread asli (P5–P95) | Spread sintetis |
|---|---:|---:|---:|---:|
| 0 | 0,0015 (0,0002–0,0047) | 0,0015 | 0,1306 (0,0545–0,1909) | 0,1301 |
| 1 | 0,0030 (0,0005–0,0107) | 0,0031 | 0,1283 (0,0463–0,1943) | 0,1280 |
| 2 | 0,0067 (0,0010–0,0318) | 0,0069 | 0,1380 (0,0533–0,2012) | 0,1391 |
| 3 | 0,0245 (0,0028–0,1229) | 0,0244 | 0,1420 (0,0603–0,1944) | 0,1419 |

Karena `density == area` untuk mask biner, kolom density tidak ditampilkan
terpisah. Distribusi sintetis hampir sama dengan asli karena augmentasi hanya
merotasi/flip dan memilih ulang sumber (tidak mengubah bentuk lesi).

Catatan: ini fitur dari circle anotasi. Fitur dari mask V1/V2 aktual dihitung
ulang oleh `evaluate_mask_variants.py` di Step 35/36; gunakan output itu untuk
angka final berbasis V2.

### 8. Apakah fitur dihitung ulang setelah augmentasi?

**Ya.** `PreprocessedAcneDataset.__getitem__`:

1. membuka image+mask,
2. menerapkan augmentasi online (jika train),
3. mengubah mask ke tensor,
4. memanggil `mask_morphology(mask_tensor)` pada mask **setelah** augmentasi.

Untuk baris offline, mask yang dibaca adalah mask sintetis hasil rotasi/flip.
Jadi fitur target selalu dihitung dari mask yang dipakai model, bukan nilai
cache dari citra original.

### 9. Distribusi epsilon MA-LDS per severity

Formula di `make_malds_targets()`:

```text
eps = clamp(base_eps + 1.5*area + 0.5*density + 0.2*spread, base_eps, max_eps)
```

Default: `base_eps=0.05`, `max_eps=0.35`.

Temuan penting: untuk mask biner, `area == density` (keduanya proporsi piksel
aktif). Jadi secara efektif:

```text
eps = base_eps + 2.0*area + 0.2*spread
```

`density` tidak menambah sinyal independen pada target biner. Pada predicted
mask (sigmoid), area dan density berbeda, tetapi target MA-LDS memakai mask biner.

**Hasil replay penuh (semua 613 citra train, epsilon sintetis dari 3.387 sampel):**

| Level | Epsilon asli mean | Epsilon asli P5–P95 | Epsilon sintetis mean | Epsilon sintetis P5–P95 | Sintetis kena clip 0,35 |
|---|---:|---|---:|---|---:|
| 0 | 0,0793 | 0,0623–0,0946 | 0,0795 | 0,0628–0,0947 | 0,00% |
| 1 | 0,0833 | 0,0618–0,1055 | 0,0839 | 0,0636–0,1060 | 0,00% |
| 2 | 0,0957 | 0,0643–0,1494 | 0,0954 | 0,0639–0,1465 | 0,00% |
| 3 | 0,1506 | 0,0676–0,2779 | 0,1514 | 0,0669–0,3500 | 9,71% |

Perhatikan Level 0 dan Level 1 hampir tidak terbedakan oleh epsilon
(0,079 vs 0,083). Ukuran lesi (area) baru benar-benar menonjol di Level 3.

### 10. Apakah MA-LDS masih membedakan Level 2 dan Level 3?

Pada pilot 12 citra, Level 3 tampak terpisah jelas (0,224 vs 0,092). Pada replay
penuh, pemisahannya jauh lebih lemah:

- AUC Level 3 vs Level 2 berdasarkan epsilon: **0,784**
- Level 3 minimum 0,0668; Level 2 maksimum 0,1672; P95 Level 2 0,1494
- **54,5% citra Level 3** memiliki epsilon ≤ P95 Level 2
- 9,71% sampel sintetis Level 3 kena clip di 0,35

Artinya target MA-LDS masih memberi sinyal ukuran, tetapi tidak cukup untuk
mengklaim pemisahan Level 2 vs Level 3 yang bersih. Ini perlu diuji dengan
distribusi pada mask V2 aktual, bukan hanya circle anotasi.

---

## 3. Training dan evaluation

### 11. Apakah eksperimen tanpa augmentasi ×1000 dan dengan augmentasi ×1000 memakai konfigurasi identik?

**Tidak ada eksperimen “tanpa augmentasi ×1000” di kode/repo.** Yang ada:

| Run/eksperimen | Image | Batch | Epoch | Split/val | Augmentasi | Seed |
|---|---:|---:|---|---|---|---:|
| Step 11 proposed aug100 v4 | 320 | 16 | 30+90 | train/val/test | offline target rasio 1.0 | 42 |
| Step 31b aug1000 | 512 | 8 | mengikuti source (30+90) | 60/40 tanpa val | offline 1000/kelas | 42 |
| Final raw `final_mask_50ep_01` | 512 | 4 | 20+30 | 60/40 tanpa val | **tidak ada** | 42 |
| Final Step 39a/40 (baru) | 512 | 4 | 20+30 default | train/val/test | offline 1000/kelas | 42 |

Semua memakai seed 42 dan arsitektur proposed, tetapi **image size, batch, epoch,
split, validation, dan augmentasi berbeda**. Perbandingan raw vs aug1000 saat ini
**confounded**. Untuk klaim “efek augmentasi”, semua faktor selain augmentasi
harus disamakan.

### 12. Checkpoint final atau best validation? Bagaimana history?

- Trainer menyimpan `best_model.pt` berdasarkan **validation loss minimum** jika
  ada validation.
- Jika `--no-validation`, setelah loop `best_epoch = total_epochs` dan
  `best_model.pt` = epoch terakhir. Run `final_mask_50ep_01` termasuk kasus ini.
- `metrics.json` menyimpan `history` per epoch: `train_*` dan, jika ada validation,
  `val_*`.
- Step 40a hanya menampilkan 10 epoch terakhir. Audit penuh harus membaca
  `history`/`history_partial.json`, bukan hanya tail.

**PERLU ARTEFAK:** untuk setiap run, laporkan:
- `checkpoint_selection`, `best_epoch`, `best_val_loss`
- kurva train/val per epoch
- metrik val terbaik dan test final (test hanya sekali, setelah lock)

Cek cepat sebuah run (jalankan di Colab setelah Step 40):

```python
import json
from pathlib import Path
run = Path('/content/drive/MyDrive/acne_project/output/experiments_512_mask_v2/final_runs/<RUN_ID>')
m = json.loads((run / 'metrics.json').read_text())
print('selection :', m.get('checkpoint_selection'))
print('best_epoch:', m.get('best_epoch'), 'best_val_loss:', m.get('best_val_loss'))
history = m['history']
for row in history[-5:]:
    print(row['epoch'], row['train_loss'], row.get('val_loss'),
          row.get('val_acc'), row.get('val_kappa'))
best = min(history, key=lambda row: row['val_loss']) if any('val_loss' in row for row in history) else None
print('argmin val_loss:', best['epoch'] if best else 'no validation')
```

Jika `checkpoint_selection == 'final_epoch_no_validation'`, run itu tidak layak
dipakai sebagai hasil final.

---

## 4. Rekomendasi audit sebelum lanjut

1. **Jangan latih ulang Level 3 dengan ~88× duplikasi per citra.** Pilih strategi:
   - batasi synthetic per original (mis. maks 5–10),
   - target kelas lebih rendah (mis. 300–500),
   - atau online augmentation + class weighting/weighted sampler,
   - atau gabungkan oversampling terbatas dengan augmentasi.
2. **Rotasi sudah diukur** (bagian 1.4): mayoritas aman, tetapi 25–38% sampel
   kehilangan ≥0,1% area dan 1–6% kehilangan minimal satu anotasi. Jika ingin
   lebih ketat, pakai `expand=True` atau fill dengan border refleksi.
3. **Distribusi fitur morfologi dan epsilon hampir tidak berubah** setelah
   augmentasi (bagian 2.7/2.9). Yang perlu dihitung berikutnya adalah distribusi
   pada **mask V1/V2 aktual**, bukan circle anotasi.
4. **Audit kebocoran pasien** jika metadata tersedia; jika tidak, tulis sebagai
   batasan.
5. **Samakan konfigurasi** untuk perbandingan aug vs no-aug: split, seed, image
   size, batch, epoch, LR, loss weight, model.
6. **Jangan memakai test** untuk memilih strategi augmentasi atau epoch.
7. **Gunakan `metrics.json` penuh** untuk laporan, bukan hanya Step 40a.

---

## 5. Artefak yang perlu dikumpulkan

- `manifest_train_val.csv`, `manifest_final_aug.csv`
- `split_validation.json`
- mask V1/V2 aktual untuk sampel audit (untuk fitur non-circle)
- `metrics.json`/`history_partial.json` setiap run
- `pipeline_logs/pipeline_steps.jsonl`
- `training_console.log`, `test_eval/test_console.log`

Tanpa artefak ini, pertanyaan 12 dan verifikasi fitur pada mask V2 masih belum
lengkap.

---

## 6. Cara mereproduksi angka di dokumen ini

```bash
python scripts/audit_step34_dataset_augmentation.py \
  --annotations-json /path/to/Acne04-v2_annotations.json \
  --image-dir /path/to/JPEGImages \
  --output-dir docs/audit_step34
```

- Skrip memverifikasi SHA-256 anotasi resmi
  (`ae03fc23fed0925c254dc4c4c9e55b539bd199b236151cb649857378a43539e9`).
  Salinan lokal yang di-serialize ulang akan ditolak; unduh file resmi dari
  commit `5930801fc75fb1a7084c26c1db3095d3e42d1f55`.
- Split dan augmentasi direplay dengan urutan RNG yang sama seperti skrip
  pipeline (seed 42).
- Geometri diukur pada circle mask anotasi di 512px letterbox; rotasi/flip
  memakai PIL `rotate`/`transpose` yang sama dengan augmenter.
- Output: `docs/audit_step34/audit_metrics.json` dan
  `docs/audit_step34/audit_report.md`.
- Skrip tidak menulis ke `output/`, tidak training, dan tidak menyentuh test.
