# Audit Step 34+ — Split, Augmentasi, Geometri Mask, MA-LDS

Dihasilkan otomatis oleh `scripts/audit_step34_dataset_augmentation.py`.
Angka split/augmentasi adalah replay deterministik dari skrip pipeline
(seed 42, target 1000/kelas). Geometri diukur pada circle mask anotasi resmi yang dirasterisasi ke 512px letterbox.

## 1. Jumlah per split/kelas

| Split | Level 0 | Level 1 | Level 2 | Level 3 | Total |
|---|---:|---:|---:|---:|---:|
| train | 240 | 295 | 67 | 11 | 613 |
| val | 42 | 52 | 12 | 2 | 108 |
| test | 188 | 232 | 54 | 9 | 483 |

## 2. Penggandaan sintetis per citra asli

| Level | Asli | Sintetis | Total | Median salinan | Max salinan | Sumber tanpa salinan | Sumber >50 salinan | Max share 1 sumber | ESS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 240 | 760 | 1000 | 3.0 | 11 | 8 | 0 | 0.0120 | 202.0 |
| 1 | 295 | 705 | 1000 | 2.0 | 7 | 27 | 0 | 0.0080 | 246.1 |
| 2 | 67 | 933 | 1000 | 14.0 | 28 | 0 | 0 | 0.0290 | 63.3 |
| 3 | 11 | 989 | 1000 | 88.0 | 103 | 0 | 11 | 0.1040 | 10.9 |

Total baris train setelah augmentasi: **4000**, sintetis **3387** (84.7%).

ESS (effective sample size) = (Σn)²/Σn², dengan n = jumlah baris per citra asli.
ESS jauh di bawah jumlah baris berarti model melihat citra yang sama berulang kali.

## 3. Retensi area mask dan lesi hilang setelah rotasi/flip offline

| Level | Median | P1 | P5 | Min | <0,999 | <0,99 | <0,95 | Sampel dengan lesi terpotong total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.9444 | 0.9861 | 0.8657 | 35.39% | 8.03% | 1.32% | 1.05% |
| 1 | 1.0000 | 0.9669 | 0.9874 | 0.8805 | 37.87% | 6.95% | 0.43% | 3.40% |
| 2 | 1.0000 | 0.9704 | 0.9889 | 0.9162 | 33.87% | 5.68% | 0.32% | 5.79% |
| 3 | 1.0000 | 0.9870 | 0.9937 | 0.9779 | 25.58% | 2.73% | 0.00% | 1.01% |

Retensi = area(mask setelah rotasi+flip) / area(mask sumber). Nilai <1 berarti
sebagian lesi terpotong oleh kanvas rotasi (fill hitam, tanpa expand).
"Lesi terpotong total" = pusat+radius anotasi berada seluruhnya di luar kanvas
setelah rotasi/flip. Retensi bisa sedikit >1 karena pembulatan rasterisasi.

## 4. Fitur morfologi per kelas (median)

| Level | area asli (P5–P95) | area sintetis | spread asli (P5–P95) | spread sintetis |
|---|---:|---:|---:|---:|
| 0 | 0.0015 (0.0002–0.0047) | 0.0015 | 0.1306 (0.0545–0.1909) | 0.1301 |
| 1 | 0.0030 (0.0005–0.0107) | 0.0031 | 0.1283 (0.0463–0.1943) | 0.1280 |
| 2 | 0.0067 (0.0010–0.0318) | 0.0069 | 0.1380 (0.0533–0.2012) | 0.1391 |
| 3 | 0.0245 (0.0028–0.1229) | 0.0244 | 0.1420 (0.0603–0.1944) | 0.1419 |

Untuk mask biner, `density` identik dengan `area`; karena itu epsilon efektif
= `0.05 + 2.0·area + 0.2·spread`.

## 5. Distribusi epsilon MA-LDS per kelas

| Level | Epsilon asli (mean) | Epsilon asli (P5–P95) | Epsilon sintetis (mean) | Epsilon sintetis (P5–P95) | Sintetis kena clip 0,35 |
|---|---:|---|---:|---|---:|
| 0 | 0.0793 | 0.0623–0.0946 | 0.0795 | 0.0628–0.0947 | 0.00% |
| 1 | 0.0833 | 0.0618–0.1055 | 0.0839 | 0.0636–0.1060 | 0.00% |
| 2 | 0.0957 | 0.0643–0.1494 | 0.0954 | 0.0639–0.1465 | 0.00% |
| 3 | 0.1506 | 0.0676–0.2779 | 0.1514 | 0.0669–0.3500 | 9.71% |

AUC Level 3 vs Level 2 (epsilon): **0.7843**.
Level 3 minimum 0.0668; Level 2 maksimum 0.1672; P95 Level 2 0.1494. 54.5% citra Level 3 berada di bawah P95 Level 2.

## 6. Catatan metode

- Split dan pemilihan sumber augmentasi direplay dengan urutan RNG yang sama seperti
  `augment_train_pairs_offline.py` (choice, rotasi, flip, brightness, contrast).
- Geometri memakai circle mask anotasi, bukan mask V1/V2 hasil refinement warna.
  Rotasi/flip diterapkan pada mask yang sama; jadi retensi area di sini adalah
  batas geometris murni, bukan hasil akhir V2.
- Fitur morfologi dan epsilon dihitung dengan rumus trainer yang sama
  (`area`, `density`, `centroid`, `spread`; eps = base + 1.5·area + 0.5·density + 0.2·spread).
- Test tidak dipakai untuk menghitung apa pun di laporan ini.
