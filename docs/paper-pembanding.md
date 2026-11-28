Ringkasan Paper Pembanding ACNE04
Saya sudah mencocokkan ringkasan ini dengan Tabel 2.1 proposal dan sumber paper. Dua studi melaporkan klasifikasi keparahan pada ACNE04, satu studi memakai ACNE04 untuk evaluasi tetapi melaporkan ICC dan AUC, sedangkan dua studi lain memakai dataset berbeda.

Gazeau dkk. (2024), AcneAI [1]
Preprocessing dan model: Model segmentasi dilatih pada 901 foto klinis. U-Net dengan encoder EfficientNet-B4 memproses citra 512 × 512. Lesi kemudian dipotong menjadi 12.682 citra kecil. Xception menerima crop 128 × 128 beserta peta ROI untuk memprediksi skor lesi, nanah, dan bekas. Skor lesi digabungkan menjadi skor keparahan 0–100, lalu dipetakan ke empat tingkat.

Hasil dan konteks: Paper melaporkan ICC 0,81 untuk kesesuaian skor keparahan pada ACNE04. AUC deteksi lesi pada ACNE04-v2 adalah 0,79, sedangkan AUC 0,88 berasal dari dataset klinis. Ini bukan akurasi klasifikasi empat kelas.

Prokhorov dan Kalinin (2024), IEEE ISBI [2]
Preprocessing dan model: Menggunakan ACNE04, 1.457 foto dan 18.983 bounding box. Jumlah lesi diubah menjadi distribusi label Gaussian dengan sigma 3. Label dilunakkan secara adaptif di sekitar batas tingkat keparahan. ResNet-50 menggabungkan tugas penghitungan lesi dan grading. Paper menyebut pembagian 80:20, yaitu 1.165 foto latih dan 292 foto uji, serta protokol lima fold pada eksperimen pembanding.

Hasil: Akurasi gabungan dilaporkan 84,11 ± 1,94%, dibanding baseline mereka 83,70 ± 1,53%. Selisihnya 0,41 poin persentase. Detail augmentasi dan preprocessing piksel tidak dirinci secara memadai di paper.

Alzahrani dkk. (2022), Computers [3]
Preprocessing dan model: Menggunakan ACNE04, 1.457 foto dan 18.983 bounding box. Pusat bounding box diubah menjadi titik, lalu menjadi peta kepadatan Gaussian. Citra diseragamkan ke 512 × 512. Faster R-CNN memberi petunjuk lokasi lesi kepada Dilated U-Net yang memprediksi peta kepadatan; hasilnya dijumlahkan untuk memperoleh hitungan dan tingkat keparahan. Augmentasi disebut digunakan, tetapi jenisnya tidak dijelaskan.

Hasil dan catatan: Narasi paper melaporkan akurasi 91,5%, MAE 1,76, dan MSE 3,57. Namun, confusion matrix untuk 292 foto uji menunjukkan 241 prediksi benar, setara 82,53%. Jadi angka akurasi yang dilaporkan dan confusion matrix paper tidak konsisten.

Fiaz dkk. (2025), HAM10000 [4]
Preprocessing dan model: Datasetnya HAM10000, bukan ACNE04, dengan 10.015 citra dermoskopik dan tujuh kelas. Citra dan mask diubah dari 600 × 450 menjadi 224 × 224, dinormalisasi, lalu diaugmentasi dengan rotasi, flip, dan perubahan kecerahan. Pembagian data stratified adalah 60:20:20. Model memakai U-Net ber-encoder ResNet34 untuk segmentasi dan EfficientNet-B0 untuk klasifikasi.

Hasil: Paper melaporkan akurasi klasifikasi 85,57% dan Dice segmentasi 0,8622 pada HAM10000. Angka ini tidak dapat disebut hasil pada ACNE04.

Suriani dkk. (2024), IJACSA [5]
Preprocessing dan model: Memakai 1.139 gambar acne dari sumber DermNet/Kaggle, bukan ACNE04, dengan tiga kelas. Augmentasinya mencakup rotasi, flip, zoom, perubahan brightness/contrast, dan Gaussian blur. Paper menyebut sembilan citra tambahan per citra asli pada data latih. MobileNetV2 dipilih untuk implementasi ringan di perangkat mobile.

Hasil dan catatan: Tabel eksperimen melaporkan test accuracy 92%, F1 0,88, dan waktu 18,2 ms per citra. Namun, bagian kesimpulan melaporkan akurasi 73% dan F1 per kelas yang berbeda. Paper ini juga bukan pengujian pada ACNE04.

Catatan untuk Tabel 2.1 proposal
Ganti judul kolom “Akurasi” menjadi “Metrik utama yang dilaporkan”, sebab Gazeau melaporkan ICC/AUC dan ada metrik segmentasi.
Untuk Gazeau, pisahkan konteks dataset: ICC 0,81 pada evaluasi keparahan ACNE04, AUC 0,79 pada deteksi ACNE04-v2, dan AUC 0,88 pada data klinis.
Prokhorov dan Alzahrani memakai ACNE04. Fiaz dan Suriani memakai dataset lain, jadi jangan diranking langsung dengan akurasi ACNE04.
Perbaiki metadata referensi Prokhorov, Gazeau, dan Alzahrani. Referensi Fiaz di daftar pustaka belum cocok dengan paper yang metrik dan metodenya diringkas pada tabel. Tambahkan referensi Suriani yang saat ini belum tercantum.
Untuk Alzahrani dan Suriani, beri catatan bahwa hasil di bagian paper tidak konsisten dengan tabel/confusion matrix atau kesimpulannya.
Rujukan
Gazeau dkk. “AcneAI: A new acne severity assessment method using digital images and deep learning.” MICCAI 2024. DOI: 10.1007/978-3-031-72086-4_7.
Prokhorov dan Kalinin. “Improving Acne Image Grading with Label Distribution Smoothing.” IEEE ISBI 2024. DOI: 10.1109/ISBI56570.2024.10635668.
Alzahrani dkk. “Attention Mechanism Guided Deep Regression Model for Acne Severity Grading.” Computers 11(3), artikel 31, 2022. DOI: 10.3390/computers11030031.
Fiaz dkk. “An explainable hybrid deep learning framework for precise skin lesion segmentation and multi-class classification.” Frontiers in Medicine, 2025. DOI: 10.3389/fmed.2025.1681542.
Suriani dkk. “Acne Severity Classification on Mobile Devices using Lighweight Deep Learning Approach.” IJACSA 15(6), 2024. DOI: 10.14569/IJACSA.2024.0150668.