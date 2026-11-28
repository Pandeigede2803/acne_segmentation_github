# Label NNEW vs Prefix Nama File pada ACNE04 / Acne04-v2

Dokumen ini menjawab satu pertanyaan: **apa arti "nama `levle1` tapi label NNEW 0"**, berapa
file yang terkena, dan mana yang sebaiknya dipakai saat training.

Semua angka di bawah dihitung dari file yang ada di repo ini. Perintah reproduksinya ada di
bagian akhir.

## 1. Ringkasan

- Setiap file ACNE04 punya **dua sumber label**: angka pada nama file (`levle0`–`levle3`) dan
  kolom label pada daftar NNEW (`NNEW_trainval_*.txt`, `NNEW_test_*.txt`).
- Untuk **34 file** di subset Acne04-v2 (1.204 citra), keduanya **tidak sama**.
- Pada seluruh ACNE04 (1.457 citra), jumlahnya **42 file**. Angka 34 adalah subset v2; ini
  koreksi dari angka 42 yang sebelumnya disebut untuk v2.
- Di 34 file itu, label NNEW **selalu** sama dengan hasil hitung ulang dari jumlah lesi memakai
  skala Hayashi. Prefix nama file **tidak pernah** cocok dengan hitungan tersebut.
- Artinya: label NNEW bukan opini kedua yang independen. Label itu **diturunkan dari count**.
  Prefix nama file tampaknya sisa pengelompokan lama dan tidak diperbarui untuk file-file ini.

## 2. Cara membaca "nama `levle1` tapi label NNEW 0"

Ambil satu baris dari `NNEW_trainval_0.txt`:

```
levle1_151.jpg 0 4
```

Tiga kolomnya:

| Kolom | Nilai | Arti |
|---|---:|---|
| Nama file | `levle1_151.jpg` | Di namanya tertulis `levle1`, jadi dulu file ini dianggap Level 1 |
| Label | `0` | Daftar NNEW memberi label severity **0** |
| Count | `4` | Daftar NNEW menghitung **4 lesi** |

Karena 4 lesi masuk rentang Level 0 (1–5 lesi), label `0` konsisten dengan count `4`.
Yang tidak konsisten adalah angka `1` pada nama file.

Jadi "pola `levle1` → label NNEW 0, 17 file" berarti: **ada 17 file yang namanya diawali
`levle1`, tetapi baris NNEW-nya memberi label 0**.

## 3. Tabel pola (subset v2, 1.204 citra)

| Prefix nama | Label NNEW | Jumlah file |
|---|---:|---:|
| `levle1` | 0 | 17 |
| `levle2` | 1 | 14 |
| `levle3` | 2 | 3 |
| **Total** | | **34** |

Tidak ada pola `levle0` → 1, `levle1` → 2, atau `levle3` → 3 yang berbeda di subset v2.

Untuk perbandingan, pada **seluruh 1.457 citra ACNE04** ada 42 file:

| Prefix nama | Label NNEW | Jumlah file |
|---|---:|---:|
| `levle1` | 0 | 17 |
| `levle2` | 1 | 14 |
| `levle3` | 2 | 8 |
| `levle1` | 2 | 2 |
| `levle0` | 1 | 1 |
| **Total** | | **42** |

Tambahan 8 file `levle3` → 2 itu termasuk 253 citra yang dibuang dari v2.

## 4. Bukti: label NNEW konsisten dengan count

Skala Hayashi yang dipakai:

| Jumlah lesi | Grade |
|---|---:|
| 1–5 | 0 |
| 6–20 | 1 |
| 21–50 | 2 |
| 51+ | 3 |

Hasil pengecekan:

| Pemeriksaan | Hasil |
|---|---|
| Label NNEW == grade dari count NNEW | **1.204 / 1.204** |
| Pada 34 file bermasalah: grade count == label NNEW | **34 / 34** |
| Pada 34 file bermasalah: grade count == prefix nama | **0 / 34** |

Karena label NNEW 100% cocok dengan count-nya, label NNEW dapat direproduksi dari kolom count.
Prefix nama file tidak dapat direproduksi dari count mana pun.

## 5. Sebaran label

Untuk 1.204 citra v2:

| Grade | Dari label NNEW | Dari prefix nama file |
|---|---:|---:|
| 0 | 487 | 470 |
| 1 | 576 | 579 |
| 2 | 122 | 133 |
| 3 | 19 | 22 |

Perbedaannya kecil secara total, tetapi menyentuh kelas 2 dan 3 yang justru paling langka.
Karena itu 34 file ini penting untuk recall Level 2/Level 3, bukan untuk accuracy keseluruhan.

## 6. Jebakan: `n_circles` v2 bukan count NNEW

Acne04-v2 punya anotasi circle sendiri (32.443 objek). Jumlahnya **tidak sama** dengan count
NNEW (18.983 bounding box):

| Ukuran | Nilai |
|---|---:|
| Median count NNEW | 6 |
| Median `n_circles` v2 | 19 |
| Maksimum `n_circles` v2 | 434 |
| Grade dari `n_circles` v2 == label NNEW | **363 / 1.204** |

Kalau `n_circles` v2 dipetakan ke Hayashi, distribusinya menjadi 123 / 522 / 400 / 159.
Itu sangat berbeda dari label NNEW (487 / 576 / 122 / 19).

Kesimpulan: **jangan memakai `n_circles` v2 sebagai target count atau target severity** untuk
eksperimen yang mengikuti Prokhorov. Anotasi v2 menandai lesi yang jauh lebih banyak (termasuk
lesi kecil), sehingga protokol hitungnya berbeda.

## 7. Count yang berubah antar fold

Ada **18 file v2** yang count NNEW-nya berbeda antar fold, walaupun labelnya tetap 3:

```
levle3_101.jpg: count 56, 56, 60, 64, 65
levle3_102.jpg: count 55, 57, 59, 61, 65
...
```

Semuanya `levle3` dengan label 3, dan semua count berada di 51–65 sehingga grade-nya tetap 3.
Jadi untuk klasifikasi 4 kelas, label file-file ini stabil. Untuk regresi count, targetnya
mengandung noise dan perlu dicatat sebagai batasan.

## 8. Implikasi untuk tujuan menaikkan akurasi

1. **Pilih satu sumber label dan pakai konsisten.** Untuk track yang mengikuti Prokhorov,
   gunakan label + count NNEW. Jangan mencampur prefix nama file sebagai label.
2. **Jangan "memperbaiki" 34 file hanya karena prefix-nya berbeda.** Label NNEW konsisten
   dengan count; prefix tidak. Mengganti label ke prefix akan membuat label bertentangan
   dengan count yang dipakai untuk loss LDS.
3. **Verifikasi manual tetap dilakukan**, tetapi hasilnya jangan dipakai untuk memilih model
   berdasarkan test. Verifikasi menghasilkan aturan label, bukan skor.
4. **Laporkan metrik per kelas.** Dengan 19 citra Level 3, accuracy bisa terlihat tinggi
   sementara recall Level 3 rendah. Laporkan Macro F1, recall L2/L3, dan interval kepercayaan.
5. **Untuk perbandingan AMFM**, baseline dan AMFM harus memakai label, count, split,
   transformasi, dan budget training yang sama. Satu-satunya perbedaan adalah AMFM.

## 9. Daftar 34 file untuk tinjauan manual

File CSV: `docs/28-nov/label_mismatch_34_review.csv`

Kolom: `file_name`, `prefix_label`, `nnew_label`, `nnew_count`,
`hayashi_grade_from_count`, `v2_circles`.

| No | File | Prefix | Label NNEW | Count NNEW | Grade count | v2 circles |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `levle1_151.jpg` | 1 | 0 | 4 | 0 | 6 |
| 2 | `levle1_159.jpg` | 1 | 0 | 3 | 0 | 5 |
| 3 | `levle1_181.jpg` | 1 | 0 | 1 | 0 | 9 |
| 4 | `levle1_198.jpg` | 1 | 0 | 1 | 0 | 35 |
| 5 | `levle1_278.jpg` | 1 | 0 | 4 | 0 | 25 |
| 6 | `levle1_279.jpg` | 1 | 0 | 4 | 0 | 38 |
| 7 | `levle1_343.jpg` | 1 | 0 | 4 | 0 | 1 |
| 8 | `levle1_344.jpg` | 1 | 0 | 4 | 0 | 14 |
| 9 | `levle1_360.jpg` | 1 | 0 | 4 | 0 | 7 |
| 10 | `levle1_368.jpg` | 1 | 0 | 4 | 0 | 11 |
| 11 | `levle1_379.jpg` | 1 | 0 | 3 | 0 | 15 |
| 12 | `levle1_402.jpg` | 1 | 0 | 2 | 0 | 48 |
| 13 | `levle1_413.jpg` | 1 | 0 | 4 | 0 | 19 |
| 14 | `levle1_457.jpg` | 1 | 0 | 4 | 0 | 22 |
| 15 | `levle1_463.jpg` | 1 | 0 | 4 | 0 | 36 |
| 16 | `levle1_544.jpg` | 1 | 0 | 5 | 0 | 27 |
| 17 | `levle1_559.jpg` | 1 | 0 | 1 | 0 | 2 |
| 18 | `levle2_105.jpg` | 2 | 1 | 11 | 1 | 51 |
| 19 | `levle2_106.jpg` | 2 | 1 | 17 | 1 | 111 |
| 20 | `levle2_114.jpg` | 2 | 1 | 11 | 1 | 46 |
| 21 | `levle2_115.jpg` | 2 | 1 | 18 | 1 | 33 |
| 22 | `levle2_117.jpg` | 2 | 1 | 7 | 1 | 23 |
| 23 | `levle2_118.jpg` | 2 | 1 | 14 | 1 | 56 |
| 24 | `levle2_119.jpg` | 2 | 1 | 18 | 1 | 86 |
| 25 | `levle2_128.jpg` | 2 | 1 | 14 | 1 | 79 |
| 26 | `levle2_160.jpg` | 2 | 1 | 20 | 1 | 62 |
| 27 | `levle2_62.jpg` | 2 | 1 | 14 | 1 | 4 |
| 28 | `levle2_64.jpg` | 2 | 1 | 18 | 1 | 21 |
| 29 | `levle2_68.jpg` | 2 | 1 | 17 | 1 | 25 |
| 30 | `levle2_86.jpg` | 2 | 1 | 11 | 1 | 9 |
| 31 | `levle2_99.jpg` | 2 | 1 | 11 | 1 | 24 |
| 32 | `levle3_112.jpg` | 3 | 2 | 22 | 2 | 87 |
| 33 | `levle3_39.jpg` | 3 | 2 | 26 | 2 | 50 |
| 34 | `levle3_90.jpg` | 3 | 2 | 38 | 2 | 232 |

Cara memakai tabel ini: buka citranya, hitung lesi secara manual tanpa melihat kolom label,
lalu catat apakah hasilnya lebih dekat ke prefix atau ke label NNEW. Lakukan tanpa melihat
prediksi model.

## 10. Batasan

- Belum ada adjudikasi manusia independen. Dokumen ini hanya menunjukkan **sumber mana yang
  konsisten secara internal**, bukan membuktikan prefix salah secara klinis.
- Label NNEW dan count NNEW berasal dari sumber yang sama, jadi kecocokan 100% bukan bukti
  independen bahwa label itu benar. Bisa saja count-nya yang salah dan label mengikuti count.
  Yang bisa disimpulkan: label NNEW dapat direproduksi, prefix tidak.
- Anotasi circle v2 memakai protokol berbeda (32.443 objek vs 18.983 bbox). Karena itu
  perbandingan count di tabel hanya untuk konteks, bukan target training.
- 18 file punya count tidak stabil antar fold. Label 4 kelasnya tetap stabil.

## 11. Ground truth operasional

Ambiguitas di dataset ini terdiri dari dua hal yang berbeda:

1. **Ambiguitas definisi** — sumber mana yang dipakai. Ini bisa diselesaikan
   secara deterministik dengan menetapkan satu kebijakan.
2. **Ambiguitas pengukuran** — apakah count/label itu benar secara klinis. Ini
   tidak bisa diselesaikan otomatis; harus diestimasi lewat audit sampel.

Jangan mencampur keduanya.

### GT-A: kompatibel Prokhorov (direkomendasikan)

| Komponen | Sumber |
|---|---|
| Severity | label NNEW |
| Count training | count NNEW pada fold masing-masing |
| Count deskriptif | median count antar fold |
| Mask segmentasi | `circle_masks/` dari anotasi resmi v2 |
| Split | 5 fold NNEW, dibatasi 1.204 citra |

Label severity stabil di semua fold (1.204/1.204), jadi tidak ada kebocoran
label dari fold lain. Count per fold dipakai apa adanya karena 18 file punya
count berbeda antar fold.

### GT-B: native v2

| Komponen | Sumber |
|---|---|
| Severity | Hayashi dari `n_circles` v2 |
| Count | `n_circles` v2 |
| Mask | circle v2 yang sama |

GT-B konsisten secara internal, tetapi distribusinya 123 / 522 / 400 / 159 dan
tidak sebanding dengan Prokhorov. Jangan mencampur GT-A dan GT-B dalam satu
eksperimen.

### Manifest deterministik

```bash
python scripts/build_ground_truth_manifest.py \
  --out-dir output/ground_truth \
  --audit-per-grade 15
```

Output:

| File | Isi |
|---|---|
| `ground_truth_manifest_nnew.csv` | 6.020 baris (1.204 file × 5 fold) |
| `ground_truth_review_nnew.csv` | 47 file prioritas tinjauan |
| `label_audit_sample_nnew.csv` | 60 citra acak berstrata (15 per grade) |
| `ground_truth_summary_nnew.json` | ringkasan angka |

Di Colab, sesuaikan `--classification-dir`, `--annotations`, dan `--image-dir`
dengan lokasi `/content/...` yang dipakai notebook.

47 file prioritas = 34 prefix mismatch + 14 count outlier − 1 irisan. Outlier
didefinisikan sebagai `|z| > 3` dari log-ratio `n_circles` v2 vs count NNEW,
dihitung di dalam grade yang sama. Ini triase, bukan label baru.

### Integrasi ke pipeline Step 42

Manifest sumber Step 42 (`refined_manifest_v2.csv`) memberi label dari prefix
nama file, karena `refine_masks_color.py` memakai `level_from_filename`. Agar
GT-A dipakai, patch manifest itu lebih dulu:

```bash
python scripts/apply_ground_truth_labels.py \
  --manifest /path/refined_manifest_v2.csv \
  --output /path/refined_manifest_v2_nnew.csv
```

Script menulis manifest baru (asli tidak diubah), mengganti `label_name` dan
`label_index`, serta menambahkan `nnew_count` dan `gt_source=nnew`.

Setelah itu:

1. arahkan `STEP42_SOURCE_MANIFEST` ke manifest baru;
2. **jalankan ulang Step 42** (CPU, bukan training) karena input label berubah;
3. lanjutkan Step 43B → 45 → 46 → 47.

Jangan memakai output Step 42 yang lama untuk label baru: dedup dan konflik
label di Step 42 ikut berubah.

Trainer counting head memakai `nnew_count` bila ada, dan baru jatuh ke
`n_circles` bila kolom itu tidak ada.

### Audit sampel untuk estimasi noise

`label_audit_sample_nnew.csv` berisi 60 citra: 15 per grade (grade 3 hanya punya
19 citra). Langkah yang benar:

1. Hitung lesi manual tanpa melihat kolom label dan tanpa melihat prediksi model.
2. Isi `manual_count`, `manual_grade`, `rater`, `notes`.
3. Hitung agreement: weighted kappa untuk grade, ICC/MAE untuk count.
4. Laporkan sebagai estimasi label noise, bukan sebagai cara memilih checkpoint.

Kalau audit menunjukkan NNEW sistematis salah, perbaikan label menjadi
eksperimen terpisah dengan protokol dan test yang sama. Jangan mengganti label
berdasarkan skor test.

### Yang tidak boleh dilakukan

- Membuat ground truth dari prediksi model (pseudo-label) lalu mengevaluasi
  model itu pada label tersebut.
- Memilih sumber label berdasarkan accuracy test.
- Memasukkan `n_circles` v2 ke Hayashi lalu menyebutnya label NNEW.
- Mencampur GT-A dan GT-B dalam satu run.

## 12. Reproduksi

```bash
python scripts/audit_label_sources.py \
  --out-csv docs/28-nov/label_mismatch_34_review.csv
```

Script ini read-only. Sumber yang dibaca:

- `datasetacne04/Classification/NNEW_trainval_*.txt`
- `datasetacne04/Classification/NNEW_test_*.txt`
- `../acne04v2/Acne04-v2_annotations.json`

Aturan Hayashi diambil dari `scripts/lds_targets.py` supaya sama dengan target LDS yang
dipakai di training.
