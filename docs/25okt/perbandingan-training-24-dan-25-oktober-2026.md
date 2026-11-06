# Perbandingan Training 24 dan 25 Oktober 2026

**Dibuat:** 25 Oktober 2026  
**Tujuan:** Membandingkan hasil Step 11 pada 24 Oktober dengan final flowchart run pada 25 Oktober, serta mencatat perbedaan konfigurasi yang dapat menjelaskan perubahan metrik.

## Perbandingan konfigurasi

| Aspek | 24 Oktober — Step 11 | 25 Oktober — Final flowchart | Catatan |
|---|---|---|---|
| Model | Proposed Hybrid ResNet-50 + MA-LDS | Proposed Hybrid ResNet-50 + MA-LDS (`proposed`, attention aktif) | Nama varian sama menurut konfigurasi; metrik run final terambil dari `docs/25okt/metrics.json`. |
| Manifest | `refined_manifest.csv` | `refined_manifest_filtered.csv` | Review Step 18 otomatis memberi `yes` ke semua foto. Jumlah data yang dilaporkan cocok, tetapi kesamaan berkas manifest belum diverifikasi byte per byte. |
| Train / validation / test | 1.017 / 216 / 224 | 1.017 / 216 / 224 | Jumlah split sama. Distribusi train final: 347 / 445 / 130 / 95 untuk `levle0`–`levle3`. |
| Epoch | 30 + 90 = 120 | 30 + 90 = 120 | Final run bukan satu model yang dilatih selama 600 epoch. CV adalah lima training terpisah; final tetap 120 epoch. |
| Batch / ukuran gambar | 16 / 320 px | 16 / 320 px | Sama menurut konfigurasi notebook. |
| Bobot kelas | `balanced` | `balanced` | Sama. |
| Weighted sampler | **Tidak digunakan** | **Digunakan** | Perbedaan konfigurasi training yang terlihat jelas dari perintah notebook. |
| Seed | 42 | 42 | Sama menurut perintah notebook. |
| Metrik test dari ringkasan trainer 24 Okt | Accuracy **71,88%**, kappa **0,5895** | Accuracy **56,25%**, kappa **0,3906** pada `metrics.json` | Angka final memakai berkas hasil terbaru yang tersedia secara lokal. |
| Accuracy dihitung dari confusion matrix yang benar | **71,88%** (161/224) | **56,25%** (126/224) | Turun **15,63 poin persentase** atau 35 prediksi benar pada test set. |
| Metrik segmentasi test | Dice **0,4786**, IoU **0,3321** | Dice **0,5025**, IoU **0,3536** | Keduanya tercatat sebagai metrik test; Dice/IoU validation CV tetap harus dibedakan. |
| Evaluasi CV | Tidak ada pada laporan Step 11 ini | Akurasi **47,98% ± 2,67%**, kappa **0,2958 ± 0,0341**, Dice **0,4924 ± 0,0155**, IoU **0,3457 ± 0,0135** | Angka 25 Oktober ini berasal dari validation folds, bukan test. |

## Perbandingan dua confusion matrix

Matrix 24 Oktober yang benar untuk Step 11 adalah `docs/24okt/test/download (12).png`. Baris menunjukkan label sebenarnya dan kolom menunjukkan prediksi.

| Label sebenarnya | 24 Okt: benar | 24 Okt: recall | 25 Okt: benar | 25 Okt: recall |
|---|---:|---:|---:|---:|
| `levle0` | 61/76 | 80,3% | 64/76 | 84,2% |
| `levle1` | 63/97 | 64,9% | 24/97 | 24,7% |
| `levle2` | 22/29 | 75,9% | 26/29 | 89,7% |
| `levle3` | 15/22 | 68,2% | 12/22 | 54,5% |
| **Total benar** | **161/224** | **71,88% accuracy** | **126/224** | **56,25% accuracy** |

Perubahan paling besar ada pada `levle1`: benar turun dari 63 foto menjadi 24, yaitu **39 foto lebih sedikit**. Sebaliknya, `levle2` membaik dari 22 menjadi 26 foto benar dan `levle0` dari 61 menjadi 64. `levle3` turun dari 15 menjadi 12.

Distribusi semua prediksi juga berubah. Matrix 24 Oktober memprediksi kelas `levle0`, `levle1`, `levle2`, dan `levle3` masing-masing sebanyak 85, 80, 40, dan 19 foto. Matrix 25 Oktober masing-masing sebanyak 120, 31, 60, dan 13 foto. Jadi model 25 Oktober jauh lebih jarang memilih `levle1` dan lebih sering memilih `levle0` serta `levle2`.

Matrix `docs/24okt/test/download (3).png` yang sempat dibahas sebelumnya berasal dari hasil inferensi lain: diagonalnya berjumlah 90/224 (40,18%) dan cocok dengan metrik `testing_predictions.csv` di catatan lama. Matrix itu **bukan** matrix yang benar untuk membandingkan Step 11. Matrix `download (12).png` cocok dengan akurasi 71,88% dari trainer.

## Perbandingan hasil test per kelas — 25 Oktober

| Kelas | Precision | Recall | F1-score | Jumlah foto test |
|---|---:|---:|---:|---:|
| `levle0` | 0,533 | 0,842 | 0,653 | 76 |
| `levle1` | 0,774 | 0,247 | 0,375 | 97 |
| `levle2` | 0,433 | 0,897 | 0,584 | 29 |
| `levle3` | 0,923 | 0,545 | 0,686 | 22 |
| **Macro F1** | **0,666** | **0,633** | **0,575** | **224** |
| **Weighted F1** | **0,663** | **0,563** | **0,527** | **224** |

Hasil 25 Oktober menunjukkan recall `levle1` rendah: sekitar 24 dari 97 foto kelas tersebut dikenali dengan benar. Model juga memiliki recall tinggi tetapi precision rendah pada `levle2`, sehingga cukup banyak foto dari kelas lain ikut diprediksi sebagai `levle2`. Ini menunjukkan kesalahan tidak tersebar rata antar kelas.

## Kenapa skor bisa turun?

Penyebab pastinya belum dapat dipastikan hanya dari dua skor test. Perbedaan paling jelas pada kode Step 11 dan final run adalah penggunaan weighted sampler:

- **Step 11:** balanced class weighting saja.
- **Final flowchart:** balanced class weighting ditambah weighted sampler.

Keduanya sama-sama memengaruhi kontribusi kelas saat training. Memakai keduanya dapat mengubah keseimbangan prediksi dan berpotensi terlalu mengoreksi kelas tertentu. Pola precision/recall 25 Oktober memperlihatkan prediksi `levle1` jarang, sementara `levle2` cukup sering dipilih. Ini membuat weighted sampler menjadi **dugaan yang perlu diuji**, bukan penyebab yang sudah terbukti.

Jumlah split final sama dengan jumlah yang tercatat sebelumnya dan filter otomatis mempertahankan semua foto. Karena itu, penghapusan foto oleh audit bukan penjelasan yang tampak paling mungkin. Namun hash/isi manifest Step 11 tidak tersedia untuk memastikan setiap nama file sama.

Versi/commit script yang dipakai kedua run juga tidak tercatat di laporan. Notebook Step 11 dapat mengambil script dari GitHub bila script lokal belum mendukung opsi eksperimen; final flowchart juga memastikan script mendukung eksperimen. Kesamaan arsitektur dan detail implementasi antar-run belum dapat dipastikan tanpa menyimpan hash atau salinan script yang benar-benar dipakai.

## Batas perbandingan

1. Matrix `download (12).png` dan ringkasan trainer 24 Oktober sama-sama menghasilkan 71,88%. Matrix `download (3).png` adalah hasil inferensi lain dan tidak dipakai dalam perbandingan Step 11.
2. Skor 47,98% CV pada 25 Oktober adalah rata-rata validation fold. Itu bukan skor test dan tidak boleh dibandingkan langsung dengan accuracy test Step 11.
3. Metrik test final terbaru tersedia di `docs/25okt/metrics.json`: kappa 0,3906, Dice 0,5025, dan IoU 0,3536. Berkas lokal bertanggal modifikasi 6 November 2026 pukul 12:06.
4. Karena hasil test sudah digunakan untuk melihat perbedaan ini, jangan gunakan test berulang kali untuk memilih konfigurasi berikutnya. Gunakan validation/CV untuk perbandingan eksperimen berikut.

## Kesimpulan

Berdasarkan matrix yang benar, hasil test turun dari **71,88% pada 24 Oktober** menjadi **56,25% pada 25 Oktober**. Penurunan terutama datang dari recall `levle1` yang jatuh, bersamaan dengan pergeseran prediksi ke `levle0` dan `levle2`. Final flowchart menambahkan weighted sampler di atas class weighting; perubahan distribusi prediksi membuat sampler faktor kandidat yang kuat untuk diuji, tetapi belum membuktikan sebab tunggal karena versi script dan identitas persis manifest belum dikunci.

Perlu diingat bahwa waktu keseluruhan eksperimen 25 Oktober lebih panjang karena mencakup lima training CV dan satu final training. Final model sendiri tetap dilatih 120 epoch; CV tidak menambahkan epoch ke checkpoint final.
