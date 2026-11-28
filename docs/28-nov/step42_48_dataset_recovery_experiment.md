# ACNE04-v2 — Dataset Recovery & Controlled Experiment Protocol

## 1. Objective dan scope

Investigasi penurunan recall Level 3 setelah offline augmentation 1.000/class, perbaiki validitas dataset, dan siapkan eksperimen yang dapat direproduksi. Jangan mengejar accuracy dengan mengorbankan minority-class recall atau integritas evaluasi.

Sumber: `audit_answers.md` (28 November 2026), `audit_cpu_bundle.zip`, `epoch_metrics.csv`. Run `step34-41_66f7794_20261127_recovery1`, commit `66f77940ae1d47dc4a80c08bfd76269de7eb7d80`. Temuan berikut berasal dari audit tersebut; usulan eksperimen di bawah bukan hasil yang sudah terbukti.

### Temuan terkonfirmasi

| Kelas | Original train setelah holdout | Synthetic | Total train | Val | Test |
|---|---:|---:|---:|---:|---:|
| levle0 | 240 | 760 | 1000 | 42 | 188 |
| levle1 | 295 | 705 | 1000 | 52 | 232 |
| levle2 | 67 | 933 | 1000 | 12 | 54 |
| levle3 | 11 | 989 | 1000 | 2 | 9 |
| Total | 613 | 3387 | 4000 | 108 | 483 |

- 17 grup exact duplicate lintas split: 13 train–test, 3 val–test, 1 train–val. 16/483 test memiliki citra identik di train/val. Lima grup memiliki konflik severity; semua 17 grup memiliki mask V2 berbeda.
- Tidak tersedia patient ID; independensi antar pasien belum terbukti. Near-duplicate belum diaudit.
- Level 3: rata-rata 89,91 synthetic/original, maksimum 103. Level 2: rata-rata 13,93, maksimum 28.
- Offline augmentation: rotasi ±15°, flip horizontal p=0,5, brightness/contrast 0,85–1,15. Tidak ada crop/scaling; transformasi mask dan image sinkron. Online augmentation tambahan aktif saat training.
- P95 perubahan absolut area mask Level 2/3 masing-masing 1,28%/0,86%; tidak ada mask kosong atau susut >10% pada kedua kelas. Ini tidak membuktikan seluruh lesi tetap terlihat.
- MA-LDS mean epsilon Level 2: 0,093516 → 0,093514; Level 3: 0,144611 → 0,145528. Epsilon online aktual belum direkam.
- Run pembanding tanpa offline augmentation masih menggunakan online augmentation. Train 721 tanpa val vs 613 + 108 val; class weights [0,6392; 0,5195; 2,2816; 13,8654] vs [1,1,1,1]; final epoch 50 vs min val loss epoch 21. Jadi efek augmentasi belum terisolasi.
- Epoch 21 val loss 1,087702; epoch 26 val Macro F1 0,711538 dan val loss 1,088146; epoch 50 val Macro F1 0,447525. Checkpoint epoch 26 tidak tersedia. Penurunan val F1 mengindikasikan masalah generalisasi/stabilitas, bukan bukti tunggal overfitting.
- Recall test Level 3 sebelumnya 8/9 vs terbaru 2/9. Sampel kecil; jangan menyimpulkan efek kausal atau memilih konfigurasi berdasarkan test.

## 2. Guardrails

- Step 42–46 CPU only. STOP setelah laporan dan proposal konfigurasi; training hanya setelah persetujuan eksplisit.
- Jangan overwrite dataset, manifest, notebook cell lama, checkpoint atau hasil eksperimen. Gunakan direktori versioned baru, simpan commit, seed, hash dan command.
- Pertahankan RGB 512×512, mask V2, ResNet50, Dilated UNet + Spatial Attention, AMFM, MA-LDS dan loss weights 1/1/0.2. Jangan membuat V3 atau menambah fitur morphology pada tahap ini.
- Audit implementasi aktual; jangan mengarang path, fungsi, hasil, patient ID atau label klinis. Jika artefak hilang, tandai BLOCKED.
- Jangan melakukan model inference pada test untuk memilih konfigurasi. Test lama telah digunakan berulang kali untuk mengarahkan penelitian; split ulang dari dataset sama tidak otomatis menjadi test independen baru.

## Step 42 — Exact duplicates dan konflik anotasi

1. Baca manifest original sebelum augmentasi, audit CSV duplicate dan file image/mask. Verifikasi identitas dan hash artefak.
2. Hitung SHA-256 file dan hash decoded RGB dengan dimensi tercatat; bedakan duplikat byte dan pixel-identical. Audit seluruh original, termasuk duplikat dalam split yang sama.
3. Bentuk duplicate group ID deterministik. Catat source ID, label, split lama, image hash, mask hash, annotation count jika tersedia.
4. Buat contact sheet citra identik dengan overlay setiap mask dan label. Jangan memilih label berdasarkan prediksi model atau mayoritas label secara otomatis.
5. Untuk konflik label/mask, minta adjudikasi berdasarkan anotasi sumber atau reviewer kompeten. Jika belum terselesaikan, karantina grup dari eksperimen utama. Untuk grup valid, pilih satu representasi canonical berdasarkan kualitas/provenance atau pertahankan sebagai satu group dengan alasan terdokumentasi.
6. Jangan menghapus file sumber. Buat manifest clean baru dan exclusion log.

Output: `exact_duplicate_groups.csv`, `label_mask_conflicts.csv`, `duplicate_review_grid.png`, `quarantine_manifest.csv`, `clean_original_manifest.csv`, `step42_report.md`.

Acceptance: seluruh konflik memiliki status resolved/quarantined; tidak ada penghapusan tanpa jejak; jumlah sebelum/sesudah per kelas dapat direkonsiliasi.

## Step 43 — Near-duplicate dan split

1. Cari near-duplicate pada original dengan perceptual hash (misalnya pHash/dHash), termasuk versi flip; gunakan similarity hanya sebagai kandidat. Review visual pasangan untuk menghindari false match. Catat threshold dan alasan pemilihan; jangan menyamakan kemiripan wajah dengan patient identity.
2. Jika metadata pasien tersedia dari sumber resmi, gunakan patient-level grouping. Jika tidak, gunakan duplicate/near-duplicate group dan laporkan keterbatasan.
3. Split original sebelum augmentasi menggunakan stratified group split jika memungkinkan. Pertahankan 60/40 sebagai referensi historis, bukan kewajiban jika membuat validation severe tidak memadai. Usulkan distribusi train/val/test dan jumlah grup independen per kelas sebelum mengunci split. Jangan memilih split berdasarkan skor model.
4. Karena validation Level 3 sebelumnya hanya dua citra, pertimbangkan grouped stratified cross-validation pada development set dengan jumlah fold yang layak berdasarkan grup Level 3; laporkan biaya komputasi dan ketidakpastian. Jangan menjanjikan recall stabil dari sampel kecil.
5. Jika tersedia, prioritaskan test eksternal yang belum pernah dipakai. Jika tidak, nyatakan evaluasi internal eksploratif dan keterbatasan test reuse. Bekukan manifest dan hash sebelum training.
6. Assert group disjoint, exact hash disjoint, tidak ada synthetic pada val/test, semua image/mask valid, dan setiap kelas tersedia pada split evaluasi yang direncanakan.

Output: `near_duplicate_candidates.csv`, `near_duplicate_review.csv`, `split_proposal.md`, `split_manifest.csv`, `split_counts.csv`, `split_integrity.json`.

Acceptance: zero known duplicate group overlap; unresolved near-duplicate dikarantina atau digabung secara konservatif; keterbatasan patient ID dan test reuse eksplisit.

## Step 44 — Visual audit augmentation Level 2/3

Gunakan dataset augmented lama yang sudah ada; tidak perlu generate baru untuk audit.

- Level 3: seluruh 11 original lama, 10 synthetic per original (seed tetap); sertakan sampel dengan perubahan area terbesar. Level 2: 30 original (atau semua bila feasible), 10 synthetic per original. Level 0/1: 10 original per kelas sebagai kontrol. Catat sampling aktual.
- Grid: original RGB, original overlay mask, augmented RGB, augmented overlay mask, source ID, synthetic ID, label, area ratio, transform metadata jika tersedia. Jangan mengarang parameter transform individual jika tidak tercatat.
- Review alignment, lesion visibility, clipping, brightness/contrast, artefak tepi hitam, perubahan distribusi spasial. Gunakan inverse transform jika parameter tersedia untuk membandingkan annotation retention; area ratio bukan pengganti lesion count.
- Audit seluruh synthetic secara statistik: empty mask, area change, morphology, source reuse, duplicate output, dan perubahan intensitas di dalam mask. Bedakan temuan objektif dan penilaian visual.
- Tandai pass / suspect / fail; review label klinis hanya jika ada referensi atau reviewer kompeten.

Output: `visual_audit_level2/`, `visual_audit_level3/`, `visual_audit_controls/`, `visual_audit_scores.csv`, `augmentation_quality_report.md`.

Acceptance: seluruh original Level 3 tercakup, kasus suspect/fail memiliki bukti dan rekomendasi; tidak menyimpulkan kualitas klinis hanya dari luas mask.

## Step 45 — Controlled offline augmentation

Siapkan kandidat setelah Step 42–44 lolos dan split disetujui. Target awal 300/class dengan cap total 7x per original (termasuk original); bukan angka optimal yang sudah terbukti.

Formula kelas: `target_c = max(n_c, min(300, 7*n_c))`. Alokasi synthetic merata per source, selisih maksimal satu jika memungkinkan. Tidak undersampling original. Simpan lineage dan parameter transform. Assert setiap source maksimal 6 synthetic dan setiap synthetic berasal dari train.

Simulasi berdasarkan jumlah lama (hitung ulang setelah clean split):

| Kelas | Original | Target total | Synthetic |
|---|---:|---:|---:|
| levle0 | 240 | 300 | 60 |
| levle1 | 295 | 300 | 5 |
| levle2 | 67 | 300 | 233 |
| levle3 | 11 | 77 | 66 |
| Total | 613 | 977 | 364 |

Pengurangan ukuran dataset vs 4000 = 75,575%. Bukan jaminan pengurangan waktu training jika update budget disamakan.

Pertahankan transform lama sebagai kontrol jika lolos audit. Jika transform gagal, buat eksperimen transform terpisah; jangan mengubah transform dan oversampling sekaligus tanpa mencatat confounder. Mask nearest-neighbor, image bilinear, geometri sinkron. Online augmentation tetap identik antar eksperimen.

Output: `controlled_aug_manifest.csv`, `augmentation_lineage.csv`, `augmentation_config.json`, `controlled_aug_integrity.json`, `controlled_aug_report.md`.

## Step 46 — Class weighting dan experiment design

Audit loss aktual: bagaimana balanced weight dihitung, normalisasi, reduction, interaksi dengan soft target MA-LDS, dan apakah weight diterapkan per target class atau sample. Laporkan effective weight dan kontribusi loss per kelas. Area dan density saat ini identik pada binary mask; catat sebagai keterbatasan, jangan ubah dulu.

Desain utama (semua menggunakan clean split, V2, seed dan transform sama):

| Run | Offline augmentation | Class weight |
|---|---|---|
| E1 | None | Uniform [1,1,1,1] |
| E2 | Controlled 300/cap7 | Uniform [1,1,1,1] |
| E3 | 1000/class | Uniform [1,1,1,1] |
| E4 (opsional) | Controlled, sama dengan E2 | Balanced dari original train |

E1–E3 menguji offline augmentation dengan weight tetap. E2–E4 menguji class weighting dengan dataset tetap. Jangan menghitung ulang balanced weight dari augmented distribution pada eksperimen yang membutuhkan weight tetap. Hindari kombinasi oversampling ekstrem dan inverse-frequency weight tanpa audit karena dapat overcompensate. E3/E4 opsional sesuai anggaran; jika hanya E1/E2, jangan mengklaim perbandingan kausal dengan run lama.

Output: `class_weight_audit.md`, `experiment_matrix.yaml`, `training_budget.md`, `pretraining_go_no_go.md`.

### STOP — Approval gate

Sebelum GPU: kirim ringkasan jumlah data bersih, status konflik, split, visual audit, class weight, estimasi optimizer steps dan biaya. Jangan training sampai user menyetujui.

## Step 47 — Training (setelah approval)

Konfigurasi referensi: batch 4, LR freeze 0.001, unfreeze 0.0001, pretrained ImageNet, maksimum 50 epoch (20 freeze + 30 unfreeze pada referensi), seed 42, loss weights seg/cls/morph 1/1/0.2. Catat optimizer/scheduler/gradient accumulation aktual.

Penting: 50 epoch tidak setara antar dataset. Dengan batch 4 tanpa drop_last/accumulation, dataset lama 613/977/4000 menghasilkan sekitar 154/245/1000 updates per epoch. Tetapkan budget optimizer updates, batas fase freeze/unfreeze dan interval validasi yang sama untuk perbandingan utama; gunakan batas 50 epoch ekuivalen pada dataset referensi yang dipilih. Jika mempertahankan 50 full passes tiap dataset, laporkan sebagai eksperimen fixed-epoch terpisah, bukan fixed-update. Dokumentasikan kompromi dan minta approval.

- Pilih checkpoint berdasarkan validation Macro F1, tie-break val loss (aturan ditetapkan sebelum training). Simpan juga best val loss, best Dice, last checkpoint dan checkpoint berkala (misalnya tiap 5 validation intervals), beserta optimizer/scheduler/RNG state.
- Early stopping patience 5 validation intervals setelah unfreeze, bukan otomatis 5 epoch jika interval berbasis update. Laporkan actual steps; early stopping dapat menghasilkan budget aktual berbeda. Untuk ablation fixed-budget, simpan best checkpoint tetapi jalankan sampai budget sama jika disetujui.
- Log train/val loss per komponen, Macro F1, balanced accuracy, accuracy, per-class precision/recall/F1/support, Dice/IoU, Kappa, confusion matrix, epsilon aktual online dan LR/update count.
- Gunakan seed tambahan atau grouped CV jika anggaran memungkinkan. Jangan memilih seed terbaik berdasarkan test.
- Jangan ubah MA-LDS/AMFM/morphology selama eksperimen utama. Ablation komponen merupakan tahap terpisah bila bukti mendukung.

Output: `run_config.yaml`, `epoch_metrics.csv`, `step_metrics.csv`, `validation_predictions.csv`, `epsilon_online_summary.csv`, `checkpoints/`, `training_report.md`.

## Step 48 — Evaluasi final

Bekukan konfigurasi berdasarkan validation saja. Evaluasi test yang telah dikunci setelah model selection selesai; bila tidak ada test baru yang independen, jelaskan bahwa hasil merupakan evaluasi internal dengan riwayat test reuse.

- Confusion matrix raw dan row-normalized, label order levle0–levle3; assert jumlah diagonal dan support sesuai metrics.
- Macro F1 sebagai primary classification metric; balanced accuracy, accuracy, weighted F1, per-class recall/precision, Cohen kappa (nyatakan jenis weighting), ordinal MAE dan under/overestimation sebagai secondary.
- Dice/IoU pada target mask V2 yang identik; jangan membandingkan langsung dengan target mask berbeda sebagai bukti peningkatan segmentasi.
- Laporkan uncertainty, terutama Level 3; gunakan confidence interval per-class recall dan group-level bootstrap bila grup tersedia. Jangan mengklaim clinical validity dari sampel kecil.
- Bandingkan hasil lama hanya sebagai historical reference, karena split, class weight, checkpoint dan budget berbeda.

Target aspiratif (bukan syarat validitas): Macro F1 ≥0.70, balanced accuracy ≥0.70, accuracy ≥0.75, recall Level 2 ≥0.70 dan Level 3 ≥0.75 dengan precision yang memadai. Laporkan hasil aktual meskipun target gagal. Jangan tuning berdasarkan test untuk mengejar target.

Output: `test_metrics.json`, `classification_report.csv`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `test_predictions.csv`, `experiment_comparison.csv`, `final_research_report.md`.

## Final report format

Untuk setiap step: status PASS/FAIL/BLOCKED, input dan hash, command, output path, temuan faktual, keterbatasan, rekomendasi dan keputusan yang masih membutuhkan persetujuan. Bedakan hipotesis dari bukti. Jangan mengklaim bahwa V2, oversampling, MA-LDS atau class weighting adalah akar penyebab sebelum eksperimen terkontrol.
