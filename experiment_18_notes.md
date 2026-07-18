# Experiment 18 Notes

## Tujuan Experiment

- Mencoba menaikkan Macro F1 fusion ke kisaran 0.80 tanpa training ulang.
- Analisis dilakukan hanya pada level output probabilitas decision-level fusion.
- Fokus utamanya adalah menguji apakah threshold phone_use selain 0.5 dapat mengurangi false alarm `safe->phone` tanpa menaikkan `phone->safe` terlalu banyak.

## Alasan Threshold Tuning Dilakukan

- Baseline fusion saat ini memiliki Macro F1 0.78059 dengan threshold 0.50.
- Masalah utamanya adalah jumlah false alarm `safe->phone` yang masih tinggi dibanding `phone->safe`.
- Karena skor fusion sudah tersedia sebagai probabilitas, threshold tuning dapat dilakukan tanpa retraining dan tanpa mengubah model.

## Threshold Terbaik Berdasarkan Validation Set

### Average Fusion

- Threshold terbaik dari validation set: 0.49 dengan Macro F1 validation 0.75617.
- Hasil test baseline threshold 0.50: Macro F1 0.78059, accuracy 0.89091, safe->phone 20, phone->safe 4.
- Hasil test dengan threshold terpilih: Macro F1 0.72982, accuracy 0.87273, safe->phone 24, phone->safe 4.
- Apakah Macro F1 test tembus 0.80: Tidak.
- Apakah ada peningkatan Macro F1 dibanding baseline 0.50: Tidak.

### Adaptive Fusion Lama

- Threshold terbaik dari validation set: 0.49 dengan Macro F1 validation 0.75617.
- Hasil test baseline threshold 0.50: Macro F1 0.78059, accuracy 0.89091, safe->phone 20, phone->safe 4.
- Hasil test dengan threshold terpilih: Macro F1 0.72982, accuracy 0.87273, safe->phone 24, phone->safe 4.
- Apakah Macro F1 test tembus 0.80: Tidak.
- Apakah ada peningkatan Macro F1 dibanding baseline 0.50: Tidak.

## Perbandingan Dengan Baseline Threshold 0.5

- Perbandingan utama dilihat dari perubahan Macro F1 test, serta trade-off antara `safe->phone` dan `phone->safe`.
- Jika threshold terpilih menaikkan Macro F1 tetapi menambah `phone->safe` secara tajam, maka peningkatan tersebut perlu dibaca secara hati-hati.
- Jika threshold terpilih menurunkan `safe->phone` tanpa merusak `phone->safe` terlalu banyak, maka tuning threshold dapat dianggap berguna pada level fusion-output.

## Kesimpulan

- Threshold terbaik berdasarkan validation set adalah `0.49` untuk average fusion dan adaptive fusion lama.
- Namun, threshold tersebut tidak meningkatkan Macro F1 pada test set.
- Macro F1 test tetap tidak tembus `0.80`.
- Pada test set, threshold `0.49` justru menurunkan Macro F1 dari `0.78059` menjadi `0.72982`.
- False alarm `safe->phone` meningkat dari `20` menjadi `24`, sedangkan `phone->safe` tetap `4`.
- Average fusion dan adaptive fusion tetap identik, baik pada threshold baseline `0.50` maupun threshold terpilih `0.49`.
- Dengan demikian, hasil utama tetap menggunakan threshold `0.50` dengan Macro F1 `0.78059`.

## Catatan Penting

- Experiment 18 tidak menggunakan training ulang.
- Checkpoint front dan side tetap sama seperti eksperimen sebelumnya.
- Average fusion dan adaptive fusion lama tidak diubah rumusnya; yang dianalisis hanya threshold keputusan pada probabilitas output.
- Hasil experiment ini bersifat analisis lanjutan dan belum otomatis mengganti hasil utama sebelum direview.
