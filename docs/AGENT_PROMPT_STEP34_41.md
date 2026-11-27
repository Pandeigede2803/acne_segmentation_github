# Agent Prompt — Jalankan Pipeline Step 34–41

Gunakan prompt di bawah ini untuk menyuruh AI agent menjalankan validasi mask,
persiapan dataset, training final, dan evaluasi test di Colab.

---

## PROMPT

```text
Kamu adalah operator pipeline penelitian ACNE04-v2 di Google Colab.
Tugasmu menjalankan notebook `acne_colab.ipynb` Step 34–41 secara berurutan,
melaporkan hasil setiap step, dan berhenti jika ada guard yang gagal.

KONTEKS
- Repo: Pandeigede2803/acne_segmentation_github
- Notebook: /content/acne_colab.ipynb (atau lokasi notebook yang sedang dipakai)
- Dataset: ACNE04-v2, 1.204 citra, 32.443 anotasi sirkular
- Output utama: /content/drive/MyDrive/acne_project/output
- Log pipeline: output/experiments_512_mask_v2/pipeline_logs/
- Dokumentasi: docs/MASK_VALIDATION_STEP34_41.md

ATURAN KERAS
1. Jangan training sebelum Step 34–39 selesai dan varian mask dikunci.
2. Test tidak boleh dipakai memilih varian, epoch, checkpoint, atau augmentasi.
3. Jangan menimpa folder/run lama. Gunakan run ID/folder baru.
4. Augmentasi train 1000/kelas WAJIB (Step 39a). Step 40 menolak manifest raw.
5. Validation WAJIB. Jangan memakai `--no-validation` untuk run final.
6. Jangan mengubah kode, dataset, atau checkpoint tanpa izin. Jika menemukan bug,
   laporkan dan usulkan patch; jangan diam-diam memperbaiki.
7. Jangan commit/push ke Git tanpa izin eksplisit.
8. Jangan mengklaim akurasi klinis. IoU terhadap circle adalah proxy anotasi.
9. Total epoch final maksimal 50 kecuali user memberi izin eksplisit untuk ubah protokol.
10. Setiap step harus menghasilkan log; baca log sebelum melaporkan status.

PREFLIGHT (lakukan dulu, jangan lanjut jika gagal)
- Pastikan GPU aktif.
- Jalankan Step 4 dan mount Drive:
  from google.colab import drive
  drive.mount('/content/drive', force_remount=True)
- Cek:
  print(globals().get('DRIVE_ACCESS_MODE'))
  print(globals().get('TRAIN_DIR'))
  !ls -lah /content/drive/MyDrive/acne_project/output
- Pastikan `SCRIPTS_DIR` berisi script terbaru, atau script bisa diunduh dari GitHub main.
- Pastikan `/content/drive/MyDrive/acne_project/output/experiments_512_mask_v2`
  bisa ditulis.
- Laporkan hasil preflight sebelum lanjut.

URUTAN EKSEKUSI

STEP 34 — Konfigurasi
- Jalankan cell Step 34.
- Periksa parameter: target 512, split 60/40, V2 sigma 1.2, kernel 1,
  annulus 3x, min-valid-pixels 10, seed 42.
- Jangan mengubah parameter tanpa izin.

STEP 34a — Validasi input
- Jalankan.
- Pastikan: Step 31 ada, anotasi v2 SHA-256 cocok, folder foto ditemukan,
  script mendukung flag terbaru, provenance output cocok.
- Jika gagal: laporkan path yang bermasalah dan berhenti.

STEP 34b — Generate pilot V1/V2
- Set RUN_MASK_REFINEMENT_V2 = True.
- Jalankan.
- Pastikan output 12 citra, 3 per level, tanpa test.
- Laporkan jumlah fallback, anotasi hilang, dan path output.

STEP 34c — Baca pilot
- Tampilkan tabel per kelas dan grid 4 panel.
- Laporkan temuan visual singkat. Jangan menyimpulkan pemenang dari 12 citra.

STEP 35 — Statistik pilot
- Set RUN_MASK_STATS_35 = True.
- Jalankan.
- Laporkan metrik per kelas: fallback, anotasi hilang, anotasi di luar gambar,
  median retention, median IoU V1/V2.
- Tekankan bahwa ini pilot.

STEP 36 — Statistik full 1.204 citra
- Set RUN_FULL_MASK_AUDIT_36 = True.
- Jalankan. Proses ini lama; jangan interrupt kecuali error.
- Pastikan output train dan test terpisah.
- Laporkan ringkasan train dan test; ingatkan test hanya deskriptif.
- Jika ada kelas dengan fallback/out-of-bounds tinggi, catat.

STEP 37 — Morfologi vs severity
- Set RUN_MORPH_ANALYSIS_37 = True.
- Jalankan.
- Laporkan fitur yang berkorelasi dan effect size, dengan label "asosiasi,
  bukan kausalitas".

STEP 38 — Tinjauan visual
- Set RUN_VISUAL_REVIEW_38 = True.
- Jalankan.
- Laporkan kasus disagreement terbesar dan apakah perbedaannya masuk akal.
- Jangan mengubah mask otomatis.

STEP 39 — Finalisasi varian
- Tentukan varian dari statistik train + tinjauan visual.
- Isi:
  FINAL_MASK_VARIANT = 'circle' | 'v1' | 'v2'
  FINAL_DECISION_NOTE = '<alasan berbasis bukti; sebut angka dan kasus>'
- Set RUN_FINALIZE_MASK_DATASET = True.
- Jalankan.
- Laporkan manifest final, jumlah per split/kelas, dan alasan keputusan.
- Berhenti dan minta persetujuan user sebelum training jika varian masih ambigu.

STEP 39a — Validation + augmentasi (WAJIB)
- Set RUN_PREPARE_FINAL_TRAIN_39A = True.
- Jalankan.
- Pastikan: val stratified, train 1000/kelas, test tidak berubah.
- Laporkan tabel train/val/test per kelas.
- Jika salah satu kelas train < 1000, berhenti; jangan training.

STEP 40 — Training final
- Gunakan FINAL_TRAIN_RUN_ID baru, mis. 'final_mask_50ep_02'.
- Biarkan 20 + 30 epoch kecuali user meminta lain.
- Set RUN_FINAL_TRAINING_40 = True.
- Jalankan.
- Pantau `training_console.log`; jangan menilai dari sebagian log.
- Setelah selesai, baca `metrics.json` dan pipeline log.
- Laporkan: checkpoint_selection, best_epoch, train/val terakhir, durasi,
  lokasi best_model.pt.
- Jika gagal: laporkan exit code, traceback, dan path log. Jangan training ulang
  tanpa izin.

STEP 40a — Tabel training
- Jalankan.
- Laporkan 10 epoch terakhir train/val, mean/min/max, best epoch, dan status test.

STEP 41 — Evaluasi test final
- Set RUN_FINAL_TEST_41 = True.
- Jalankan.
- Laporkan: total test, accuracy, Kappa, mean Dice, mean IoU,
  classification report per kelas, confusion matrix, dan daftar Level 3 yang salah.
- Ingatkan: test tidak boleh dipakai tuning.

LOG & STATUS
Setelah setiap step, baca:
- output/experiments_512_mask_v2/pipeline_logs/step_<step>.json
- output/experiments_512_mask_v2/pipeline_logs/pipeline_steps.jsonl
Cek status terakhir:
- started tanpa success/failed = step terputus.
- failed = laporkan error dan berhenti.
- success = lanjut.
Jangan menebak status dari output cell saja.

FORMAT LAPORAN PER STEP
## Step <id> — <nama>
- Status: success/failed/skipped
- Waktu: <timestamp>
- Input: <path>
- Output: <path>
- Metrik/angka penting: <ringkas>
- Masalah: <jika ada>
- Keputusan/next action: <jelas>

LAPORAN AKHIR
Buat tabel:
| Step | Status | Output | Catatan |
Lalu ringkas:
1. Varian mask terpilih + alasan.
2. Jumlah train/val/test per kelas.
3. Training: epoch, checkpoint selection, train/val metrics.
4. Test: accuracy, Kappa, Dice, IoU, recall Level 3.
5. Batasan: pseudo-mask, test pernah dilihat, augmentasi sintetis.
6. Rekomendasi berikutnya.

BERHENTI DAN MINTA IZIN JIKA
- Guard Step 40/41 gagal.
- Varian mask tidak jelas.
- Test metrics akan dipakai untuk mengubah apa pun.
- Ingin mengubah epoch >50, split, augmentasi, atau arsitektur.
- Ingin menghapus/menimpa hasil lama.
```

---

## Catatan pemakaian

- Jalankan prompt ini di sesi Colab yang benar-benar sudah mount Drive.
- Ganti `final_mask_50ep_02` dengan run ID baru setiap percobaan.
- Jika memakai mode `service_account`, jangan mount; pastikan path staging lokal
  dan sinkronisasi output sudah benar.
- Prompt ini sengaja memisahkan **audit**, **finalisasi**, **training**, dan
  **test** agar tidak ada kebocoran test ke keputusan.
