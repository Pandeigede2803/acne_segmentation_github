# Step 33 — Filter Wajah + Kulit (terpisah, tanpa training)

Step ini membuat **salinan** data 512 di mana hanya area wajah + kulit yang
dipertahankan. Step 31 (data mentah) dan Step 31b (training) tidak diubah.

## Posisi

```text
Step 31   → data 512 mentah + split/augmentasi
Step 32   → audit cakupan wajah/anotasi
Step 33   → data 512 terfilter wajah+kulit (step ini)
Step 31b  → training; set TRAINING_VARIANT='skinfilter'
```

## Metode filter

1. Deteksi wajah dengan Haar cascade (frontal + profile).
2. Peta kulit dengan threshold YCrCb, dibatasi di sekitar wajah.
3. Region = kotak wajah ∪ peta kulit.
4. Piksel di luar region diisi:
   - `black` (default)
   - `gray`
   - `mean` (rata-rata warna region)
5. Mask **disalin tanpa diubah** secara default. `--clip-masks` hanya dipakai
   jika sengaja ingin mengubah target.

## Output

```text
output/experiments_512_skin/skinfilter_aug1000_img512_v1/
├── preprocess_skin/
│   ├── refined_manifest_skin.csv
│   ├── images/                      # citra 512 terfilter
│   ├── masks/                       # mask salinan
│   ├── skin_filter_provenance.json
│   └── visual_examples/
├── split_60_40/
│   ├── manifest_split.csv
│   ├── manifest_aug.csv
│   ├── aug_pairs/
│   └── split_aug_provenance.json
├── split_70_30/...
└── split_aug_preparation.csv
```

Split/augmentasi tetap memakai `split_train_test_manifest.py` dan
`augment_train_pairs_offline.py` (1000/kelas, tanpa `val`).

## Menjalankan

1. Jalankan Step 31 sampai selesai.
2. Set `RUN_SKIN_FILTER = True`, jalankan Step 33.
3. Untuk training, buka Step 31b:
   ```python
   TRAINING_VARIANT = 'skinfilter'
   ```
   lalu jalankan. Hasil test masuk ke
   `experiments_512_skin/.../split_aug_results.csv`.

## Catatan penting

- Filter diterapkan pada **input**; mask target tetap. Jika anotasi berada di
  luar wajah/kulit, model tidak bisa memprediksinya dan Dice bisa turun.
- `black` menimbulkan domain shift; bandingkan dengan Step 31 mentah, jangan
  mencampur metriknya.
- Deteksi Haar bisa gagal pada Very Severe; lihat `face_detected` di
  provenance.
- Semua output baru; tidak ada folder lama yang ditimpa.
