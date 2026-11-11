# Hasil Ablation Study — Validation

## Sumber dan protokol

Enam varian telah selesai menurut `docs/ABALATION_COMPLETE.MD`.
Angka berikut berasal dari output `metrics.json` Google Drive yang dikirim
pengguna. Setiap baris memakai epoch dengan **total validation loss terendah**,
bukan epoch dengan accuracy tertinggi.

Model lengkap `proposed` ditambahkan dari salinan lokal
[`docs/25okt/metrics.json`](25okt/metrics.json), dengan `config.output_dir`
menunjuk `training_runs/2026-10-24_flowchart_proposed_filtered` (Step 21).
Baris yang dipilih adalah epoch 115, dengan `val_loss` minimum
1,0945889243373164, sesuai `best_val_loss` yang tersimpan. Penambahan ini hanya
membaca hasil training yang sudah tersedia, tanpa training ulang.

Manifest: `flowchart_experiments_2026-10-24/ablation/train_val_manifest.csv`.
Semua angka di sini adalah **validation**, bukan test. Waktu delapan detik pada
log ringkasan adalah waktu membaca hasil tersimpan, bukan durasi training enam
model.

Kode notebook membentuk manifest ablation dari baris train/validation pada
manifest filtered yang juga digunakan Step 21. Perintah keduanya mengatur
class weighting balanced, weighted sampler, seed 42, serta jumlah epoch yang
sama. Ini menunjukkan protokol yang dimaksud sebanding; identitas isi manifest,
foto/mask, dan versi trainer saat run aktual tetap perlu diverifikasi sebelum
selisih hasil diatribusikan pada komponen arsitektur saja.

## Hasil

| Varian | Epoch terbaik | Train accuracy | Val. accuracy | Val. Kappa | Val. Dice | Val. IoU | Val. loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 96 | 80,14% | 60,19% | 0,4458 | 0,5096 | 0,3590 | 1,4145 |
| no_attention | 111 | 77,29% | 53,24% | 0,3532 | 0,5154 | 0,3623 | 0,9897 |
| no_dilation | 115 | 78,66% | 52,78% | 0,3449 | 0,5204 | 0,3664 | 1,0684 |
| no_amfm | 107 | 86,73% | 60,19% | 0,4231 | 0,5153 | 0,3614 | 1,0812 |
| lds | 117 | 74,04% | 50,00% | 0,3173 | 0,5025 | 0,3513 | 1,3134 |
| no_fusion | 112 | 83,97% | **63,89%** | **0,4831** | 0,5077 | 0,3565 | 0,9779 |
| **proposed** | **115** | **78,66%** | **49,54%** | **0,2978** | **0,5090** | **0,3571** | **1,0946** |

## Interpretasi

- **Klasifikasi:** `no_fusion` memiliki validation accuracy dan Kappa tertinggi
   di antara tujuh konfigurasi yang dilaporkan. Selisih accuracy terhadap baseline
  sekitar **3,70 poin persentase**.
- **Segmentasi:** `no_dilation` memiliki Dice dan IoU tertinggi di antara enam
   konfigurasi tersebut. Selisih Dice terhadap baseline sekitar **0,0109**.
- **Pembanding model lengkap:** accuracy `proposed` adalah **49,54%**.
  `no_fusion` lebih tinggi sekitar **14,35 poin persentase**, sedangkan
  `no_amfm` lebih tinggi sekitar **10,65 poin persentase**. Jika protokol aktual
  setara, hasil satu run ini belum mendukung klaim bahwa fusion atau AMFM
  meningkatkan klasifikasi.
- Dibandingkan `lds`, `proposed` memiliki accuracy sekitar **0,46 poin
  persentase lebih rendah**, tetapi Dice sekitar **0,0065** dan IoU sekitar
  **0,0058** lebih tinggi. Manfaat MA-LDS belum konsisten lintas metrik;
  selisih kecil ini belum membuktikan signifikansi statistik.
- `no_amfm` memiliki train accuracy tertinggi, tetapi validation accuracy sama
  dengan baseline. Train accuracy tinggi bukan bukti generalisasi lebih baik.
- Semua train accuracy lebih tinggi daripada validation accuracy; gap ini
  mengindikasikan keterbatasan generalisasi, bukan diagnosis sebab tunggal.

## Batas kesimpulan

1. **Model lengkap `proposed` sudah disertakan dari hasil validation Step 21.**
   Interpretasi kontribusi komponen bersyarat pada kesamaan split, data,
   preprocessing, dan konfigurasi aktual antar-run. Jangan membandingkan
   angka validation ini dengan test accuracy 24/25 Oktober atau rata-rata CV
   yang memakai pembagian data berbeda.
2. Nilai loss antar-varian tidak otomatis setara: baseline, LDS, dan MA-LDS
   memakai distribusi target klasifikasi berbeda. Loss boleh dipakai memilih
   checkpoint dalam setiap run, tetapi jangan menjadikannya satu-satunya dasar
   menentukan varian terbaik antar-objektif.
3. Pastikan implementasi serta konfigurasi aktual sesuai definisi varian.
   Nama `baseline` saja tidak membuktikan bahwa model tersebut ResNet50
   single-task murni. Varian baseline juga dapat mengubah beberapa komponen,
   sehingga bukan pengujian satu komponen yang terisolasi.
4. Output ini belum menyediakan Macro-F1, recall per kelas, atau confusion
   matrix. Metrik tersebut memerlukan prediksi validation, bukan dihitung dari
   accuracy saja.
5. Satu run per varian belum membuktikan signifikansi statistik atau kestabilan
   hasil. Test tidak digunakan untuk tuning.

## Narasi singkat laporan

> Pada enam varian ablation dan satu model lengkap yang dilaporkan, model tanpa fusi fitur memperoleh
> accuracy validasi tertinggi sebesar 63,89% dan Cohen's Kappa sebesar 0,4831.
> Sementara itu, model tanpa dilated convolution memperoleh Dice sebesar 0,5204
> dan IoU sebesar 0,3664, tertinggi di antara varian yang dilaporkan. Hasil ini
> menunjukkan bahwa varian dengan performa klasifikasi terbaik tidak selalu
> memiliki performa segmentasi terbaik. Model lengkap proposed memperoleh
> accuracy validasi 49,54%, Kappa 0,2978, Dice 0,5090, dan IoU 0,3571 pada
> checkpoint dengan total validation loss terendah (epoch 115). Dengan syarat
> protokol aktual antar-run setara, hasil ini belum mendukung peningkatan
> klasifikasi melalui AMFM dan fusion. Verifikasi kesetaraan eksperimen dan
> pengulangan dengan beberapa seed diperlukan untuk memperkuat kesimpulan.
