# Audit Wajah vs Anotasi — Step 32

Dokumen ini menjelaskan urutan kerja untuk memastikan masalah segmentasi
berasal dari data (wajah/kulit/anotasi), bukan langsung dari arsitektur.

## Urutan kerja

| # | Isi | Biaya |
|---|---|---|
| 1 | Deteksi wajah pada 1.204 citra; catat kotak wajah dan persentase luas wajah | offline, menit |
| 2 | Hitung persentase area anotasi yang masuk ke kotak wajah atau peta kulit | offline, menit |
| 3 | Lihat 4–6 contoh visual sebelum vs sesudah filter wajah+kulit, termasuk Very Severe | sekilas |

## Metode

- **Wajah**: OpenCV Haar cascade (frontal default, frontal alt2, profile kiri/kanan).
  Wajah terbesar dipilih bila ada beberapa deteksi.
- **Peta kulit**: threshold YCrCb (`Cr 133–180`, `Cb 77–133`) + `V ≥ 30`,
  dibatasi di sekitar kotak wajah (skala 1,1) bila wajah terdeteksi.
- **Anotasi**: lingkaran ACNE04-v2 dirasterisasi pada resolusi asli.
- **Cakupan**: `area(anotasi ∩ region) / area(anotasi)`, dengan
  `region = kotak wajah ∪ peta kulit`. Dilaporkan terpisah untuk wajah,
  kulit, dan gabungan.

## Output

```text
output/face_audit_v1/
├── face_annotation_audit.csv
├── face_annotation_summary.json
└── visual_examples/*.png
```

Kolom penting pada CSV:

- `face_detected`, `face_x/y/w/h`, `face_area_pct`
- `n_circles`, `annotation_area_pct`
- `annotation_inside_face_pct`
- `annotation_inside_skin_pct`
- `annotation_inside_union_pct`
- `circles_center_inside_union`

Ringkasan JSON memuat rata-rata keseluruhan, per kelas, `images_fully_inside_union`,
`images_below_90pct`, `images_below_50pct`, `images_zero_coverage`, dan 10 kasus
terburuk.

## Cara membaca

- ✅ `mean_inside_union_pct ≥ 90%` dan `images_zero_coverage` kecil:
  filter/crop wajah relatif aman untuk diuji sebagai eksperimen baru.
- ⚠️ Nilai rendah / banyak nol: wajah atau peta kulit membuang lesi.
  Jangan crop dulu; periksa contoh visual dan kasus terburuk.
- 🔎 Contoh visual: hijau = anotasi masuk region, merah = keluar.
  Panel “sesudah” menggelapkan area di luar wajah+kulit dan hanya
  mempertahankan anotasi yang masuk.

## Menjalankan di Colab

1. Jalankan Step 32 dengan `RUN_FACE_AUDIT = True`.
2. Cell memulihkan manifest asli dari provenance Step 11, lalu memakai
   `--annotations-json` dan `--image-dir` yang tercatat di sana.
3. Jika foto asli/JSON sudah hilang dari `/content`, jalankan ulang Step 5/6
   atau isi `FACE_AUDIT_ANNOTATIONS` dan `FACE_AUDIT_IMAGE_DIR`. Manifest asli
   selalu diambil dari provenance Step 11; override hanya lewat
   `FACE_AUDIT_MANIFEST`.

Audit ini tidak mengubah manifest, checkpoint, atau hasil training, dan tidak
membuka test untuk pemilihan model.


## Mode tanpa OAuth (folder publik)

Folder publik `1Wx3jxA0RRsLnPkbhmt_Na8jRCrjYHDaA` berisi output/scripts, bukan
dataset mentah. Folder ini bisa **diunduh read-only** dengan `gdown`; tidak bisa
di-mount read-write tanpa OAuth.

Step 32 mendukung mode ini:

```python
DRIVE_ACCESS_MODE = 'service_account'  # Step 2; perlu JSON key di Colab Secrets
USE_PUBLIC_MANIFEST = True   # Step 32
FACE_AUDIT_IMAGE_DIR = DETECTED_IMAGE_DIR
FACE_AUDIT_ANNOTATIONS = DETECTED_ANNOT_JSON
```

Cell mengunduh `refined_manifest.csv` dari folder publik memakai file ID
`1hwaX9eK2ueH55X9YyWtxOZnTqbxTAjuK` dan menulis hasil ke
`/content/public_output/face_audit_v1` pada mode lokal, lalu mengunggah hasil ke
folder Drive `face_audit_v1` bila service-account mode aktif. Foto asli + JSON
anotasi tetap berasal dari Step 5/6 (juga tanpa OAuth).

Untuk training/checkpoint, output tetap perlu penyimpanan persisten: mount
Drive sekali per sesi, atau simpan ke `/content` lalu unduh hasilnya.
