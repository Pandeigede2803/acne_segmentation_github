# Hasil Final Training Flowchart — 25 Oktober 2026

## Ringkasan

Final training varian **`proposed` (attention aktif)** tercatat lengkap sampai epoch 120 dan menghasilkan akurasi klasifikasi **56,25% pada 224 foto test**. Metrik test segmentasi juga tersedia: Dice **0,5025** dan IoU **0,3536**. Model mengenali sebagian besar foto `levle0` dan `levle2`, tetapi sering melewatkan `levle1`. Hasil antar-fold cross-validation berada di sekitar 48% akurasi validasi. Secara umum performa klasifikasi antar tingkat jerawat masih belum merata.

Berkas `metrics.json`, checkpoint, dan visualisasi di folder lokal memiliki waktu modifikasi 6 November 2026 pukul 12:06. Laporan ini merujuk eksperimen sebagai final flowchart 25 Oktober sesuai nama run; waktu modifikasi lokal bukan bukti bahwa proses masih berjalan.

Hasil ini belum mendukung klaim akurasi 90%. Akurasi rata-rata saja juga belum cukup untuk menilai apakah model berguna, karena hasil setiap kelas berbeda cukup jauh.

## Berkas hasil

- Kurva loss, akurasi, kappa, dan Dice training/validation: `/content/drive/MyDrive/acne_project/output/training_runs/2026-10-24_flowchart_proposed_filtered/training_curves_final.png`
- Confusion matrix test: `docs/25okt/confusion_matrix_test.png`
- Confusion matrix final pada test set: `/content/drive/MyDrive/acne_project/output/training_runs/2026-10-24_flowchart_proposed_filtered/confusion_matrix_test.png`
- Prediksi per foto test: `/content/drive/MyDrive/acne_project/output/training_runs/2026-10-24_flowchart_proposed_filtered/test_class_predictions.csv`
- Hasil cross-validation per fold: `/content/drive/MyDrive/acne_project/output/flowchart_experiments_2026-10-24/cross_validation/cross_validation_results.csv`
- Grafik metrik cross-validation: `/content/drive/MyDrive/acne_project/output/flowchart_experiments_2026-10-24/cross_validation/cross_validation_metrics.png`

## Cross-validation

Model diuji pada lima pembagian data train yang berbeda. Setiap foto dalam kumpulan train mendapat giliran menjadi data validasi. Hasil yang dilaporkan adalah rata-rata dan simpangan baku antar-fold.

| Metrik validasi | Rata-rata | Simpangan baku |
|---|---:|---:|
| Akurasi | 0,4798 (47,98%) | 0,0267 |
| Cohen's kappa | 0,2958 | 0,0341 |
| Dice segmentasi | 0,4924 | 0,0155 |
| IoU segmentasi | 0,3457 | 0,0135 |

Akurasi per-fold berkisar dari **45,10% sampai 51,96%**. Perbedaan antarfold relatif kecil, jadi performa validasi model terlihat cukup konsisten pada lima pembagian ini, meski tingkat akurasinya masih terbatas. Dice dan IoU di atas adalah metrik segmentasi pada validation folds. Untuk model final, Dice test **0,5025** dan IoU test **0,3536**.

## Visualisasi training dan evaluasi

Kurva memperlihatkan metrik training dan validation selama 120 epoch. Di akhir training, accuracy tercatat 80,33% pada train dan 54,17% pada validation, sehingga ada jarak performa yang perlu diperhatikan sebagai indikasi kemungkinan overfitting.

![Kurva training dan validation final](training_curves_final.png)

Confusion matrix berikut merangkum prediksi pada 224 foto test. Recall `levle1` rendah (24,7%; 24 dari 97 foto dikenali benar).

![Confusion matrix test final](confusion_matrix_test.png)

## Evaluasi klasifikasi final pada test set

Test set terdiri dari 224 foto dan hanya dipakai untuk evaluasi final. `support` adalah jumlah foto sebenarnya pada tiap kelas.

| Kelas | Precision | Recall | F1-score | Foto |
|---|---:|---:|---:|---:|
| `levle0` | 0,533 | 0,842 | 0,653 | 76 |
| `levle1` | 0,774 | 0,247 | 0,375 | 97 |
| `levle2` | 0,433 | 0,897 | 0,584 | 29 |
| `levle3` | 0,923 | 0,545 | 0,686 | 22 |
| **Macro average** | **0,666** | **0,633** | **0,575** | **224** |
| **Weighted average** | **0,663** | **0,563** | **0,527** | **224** |

- **Akurasi keseluruhan: 56,25%**, atau 126 prediksi benar dari 224 foto.
- **Macro F1: 0,575** memberi bobot yang sama pada empat kelas. Nilai ini menunjukkan performa rata-rata antar kelas masih sedang dan tidak merata.
- **Weighted F1: 0,527** memperhitungkan jumlah foto tiap kelas. Nilainya dipengaruhi kelas `levle1` yang paling banyak, tetapi recall kelas itu rendah.

### Pola kesalahan yang terlihat

- `levle1` hanya memiliki **recall 24,7%**: model menemukan sekitar 24 dari 97 foto kelas ini. Artinya, sekitar tiga perempat foto `levle1` salah diklasifikasikan sebagai kelas lain.
- `levle2` memiliki **recall 89,7%**, tetapi **precision 43,3%**. Model menemukan hampir semua contoh `levle2`, namun banyak foto dari kelas lain juga diprediksi sebagai `levle2`.
- `levle0` memiliki recall tinggi (**84,2%**) tetapi precision lebih rendah (**53,3%**), yang menandakan cukup banyak foto kelas lain ikut diprediksi sebagai `levle0`.
- `levle3` memiliki precision tinggi (**92,3%**) dan recall **54,5%**. Prediksi `levle3` biasanya benar, tetapi model melewatkan sekitar 10 dari 22 foto kelas tersebut.

Precision/recall menunjukkan jumlah serta arah kesalahan per kelas; confusion matrix yang tersimpan memperlihatkan kelas tujuan untuk masing-masing kesalahan. Label ditulis sebagai `levle0`–`levle3` mengikuti nama kelas di manifest proyek.

Sebagai pembanding sederhana, jika semua foto selalu ditebak sebagai kelas terbanyak (`levle1`), akurasinya 97/224 atau **43,3%**. Model mencapai 56,25%, lebih tinggi 12,95 poin persentase daripada tebakan mayoritas tersebut. Ini pembanding dasar saja; performa kelas `levle1` model tetap perlu diperbaiki.

## Catatan interpretasi

Cross-validation menghasilkan akurasi validasi rata-rata 47,98%, sedangkan test final menghasilkan 56,25%. Keduanya berasal dari subset yang berbeda, jadi selisih ini tidak membuktikan bahwa model membaik pada test. Test set hanya satu evaluasi berisi 224 foto.

File kurva training sudah dibuat, tetapi bentuk kurvanya belum dinilai dalam catatan ini karena gambar kurva tidak disertakan. Metrik `train_acc` dan `val_acc` yang muncul saat training menunjukkan jarak yang cukup besar; ini selaras dengan kemungkinan overfitting dan perlu dikonfirmasi dengan melihat kurva tersimpan.

Metrik segmentasi test pada `metrics.json` adalah Dice **0,5025** dan IoU **0,3536**. Nilai tersebut berasal dari evaluasi model final pada test set; nilai CV tetap dilaporkan terpisah sebagai rata-rata validation folds.

## Kesimpulan dan tindak lanjut

Model final memiliki akurasi klasifikasi **56,25%** dan performa tidak seimbang antar kelas. Fokus pemeriksaan berikutnya adalah kesalahan `levle1` serta prediksi berlebih pada `levle0` dan `levle2`. Tinjau confusion matrix dan CSV prediksi per foto untuk menemukan pola visual atau label yang keliru. Periksa pula kurva training dan laporkan metrik segmentasi test secara terpisah sebelum membuat klaim tentang kualitas mask.
