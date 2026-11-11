# Verifikasi Dataset ACNE04-v2 sebelum Training

## Tiga jumlah yang berbeda

1. **File gambar sumber:** bisa 1.457 karena gambar berasal dari ACNE04 original.
2. **Sampel asli v2:** tepat 1.204 nama unik dalam JSON resmi, dengan 32.443 anotasi.
3. **Baris setelah augmentasi:** train bertambah secara sintetis, validation/test
   tetap. Ini bukan jumlah citra asli independen atau jumlah pasien baru.

Distribusi awalan nama file pada JSON resmi: `levle0` 470, `levle1` 579,
`levle2` 133, `levle3` 22. Label ini diwarisi dari nama file original, bukan
hasil penentuan ulang severity menggunakan hitungan lesi v2.

## Menjalankan perbaikan di Colab

1. Gunakan versi terbaru `acne_colab.ipynb`.
2. Sediakan **`scripts/audit_acne04v2.py` di `SCRIPTS_DIR`**: unggah manual,
   atau publish script ke GitHub sebelum notebook mencoba mengunduhnya.
   Perubahan lokal tidak otomatis muncul pada notebook/ZIP di Drive atau GitHub.
3. Jalankan setup dan Step 6. JSON resmi diunduh dari commit
   `5930801fc75fb1a7084c26c1db3095d3e42d1f55`, diverifikasi SHA-256
   `ae03fc23fed0925c254dc4c4c9e55b539bd199b236151cb649857378a43539e9`.
4. Jalankan Step 8. Tampilan memisahkan jumlah file sumber dan 1.204 gambar v2
   yang cocok. Gambar hilang atau nama ganda dalam folder sumber menghentikan proses.
5. Jalankan Step 10 pada folder baru `acne04v2_official1204_letterbox_v1`.
   Manifest wajib cocok dengan semua nama dan jumlah lesi per gambar pada JSON,
   label, preprocessing, split, serta keberadaan file. Provenance disimpan.
6. Jalankan Step 10a. Hanya train diaugmentasi ke target 100%; cache diverifikasi.
7. Jalankan Step 11 dengan Run ID baru yang mencantumkan `official1204`.
8. Setelah itu, jalankan Step 20 augmented bila siap melatih tujuh varian.

**Jangan menggunakan checkpoint dari manifest yang berbeda untuk resume.**
Hasil sebelumnya tetap disimpan; tidak ada penghapusan atau konversi hasil lama
menjadi hasil v2 resmi. Jika audit menolak manifest tanpa provenance, pilih
folder preprocessing baru, bukan menghilangkan pemeriksaan atau membuat
provenance manual untuk artefak yang asalnya tidak diketahui.

## Bukti untuk laporan

- Identitas JSON resmi dan hash-nya.
- Jumlah sampel asli serta jumlah train/validation/test dari audit manifest.
- Jumlah train tiap kelas sebelum/sesudah augmentasi dan jumlah pasangan sintetis.
- Hash manifest dan script; konfigurasi dan folder setiap run.
- Hasil dari run baru yang sesuai sumber ini saja. Tabel hasil tetap kosong sampai tersedia.

Audit tidak memvalidasi diagnosis klinis, identitas pasien, kebocoran subjek,
atau isi piksel setiap foto/mask melalui hash. Pemeriksaan tersebut tetap
memerlukan audit tambahan. Nama file cocok bukan jaminan versi citra/piksel
atau ketepatan klinis mask. Training GPU belum dijalankan oleh perubahan ini.
