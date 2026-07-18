# Experiment 20 Adaptive Formula Notes

## Posisi Experiment 20

- Experiment 20 diposisikan sebagai exploratory/diagnostic analysis, bukan pengganti otomatis metode utama.
- Fokus eksperimen ini adalah mengecek apakah variasi adaptive confidence-weighting pada level probabilitas dapat menghasilkan prediksi akhir yang berbeda dari average fusion.
- Hasil utama penelitian sebelum Experiment 20 tetap Front14A + Side13 dengan Macro F1 0.78059.

## Tujuan Experiment 20

- Mengeksplorasi formula adaptive fusion baru yang lebih diskriminatif dibanding adaptive fusion lama tanpa retraining model.
- Menguji apakah perubahan definisi confidence dan mekanisme pembentukan bobot dapat mengubah keputusan akhir dibanding average fusion.
- Menjaga seluruh checkpoint, dataset, split, backbone, dan arsitektur tetap sama seperti eksperimen sebelumnya.
- Memilih formula hanya dari validation set, lalu mengevaluasi formula terpilih satu kali pada test set.

## Batasan Experiment 20

- Tidak ada retraining front maupun side.
- Tidak ada perubahan dataset.
- Tidak ada perubahan split train/validation/test.
- Tidak ada perubahan backbone EfficientNetV2-S.
- Tidak ada perubahan arsitektur model.
- Tidak ada perubahan checkpoint Front14A dan Side13 yang dievaluasi.
- Average fusion lama tidak diubah.
- Adaptive fusion lama tidak dihapus dan tetap dipertahankan sebagai pembanding resmi.

## Mengapa Adaptive Lama Dianalisis Ulang

- Experiment 17 menunjukkan `legacy_adaptive_softmax` identik dengan average fusion pada test set.
- Rata-rata bobot front adaptive lama berada di sekitar 0.52368 dengan 167 dari 220 sampel berada pada rentang 0.45-0.55.
- Adaptive sharpening alpha 1, 2, 3, 5, 10 juga tidak mengubah hasil utama; Macro F1 tetap 0.78059.
- Ini mengarah pada dugaan bahwa masalah utama bukan hanya definisi confidence, tetapi juga mekanisme pembentukan bobot yang terlalu meredam perbedaan confidence.

## Formula Yang Diuji

1. `average_fusion`
2. `legacy_adaptive_softmax`
3. `legacy_adaptive_ratio`
4. `margin_adaptive_ratio`
5. `entropy_adaptive_ratio`
6. `maxprob_adaptive_ratio`
7. `reliability_weighted_static_fusion`

## Catatan Matematis Singkat

- Pada klasifikasi biner, `abs(prob_phone - prob_safe)` proporsional dengan `abs(prob_phone - 0.5)`.
- Karena itu, `margin_adaptive_ratio` secara teoritis berpotensi identik dengan formula legacy berbasis rasio dan dijalankan sebagai verifikasi empiris.
- `legacy_adaptive_ratio` ditambahkan untuk mengisolasi apakah penyebab utama ketidakberbedaan adaptive lama berasal dari softmax weighting, bukan dari definisi confidence-nya.
- Fusion berbentuk `w_front * P_front + w_side * P_side` dengan bobot positif dan jumlah bobot 1 adalah convex combination, sehingga probabilitas fusion selalu berada di antara `P_front` dan `P_side`.
- Jika perubahan bobot tidak mendorong skor gabungan melintasi decision boundary 0.5, maka prediksi akhir tetap sama walaupun distribusi bobot berubah cukup jauh.

## Hasil Validation Semua Formula

| Formula | Macro F1 | Accuracy | safe->phone | phone->safe | diff_vs_avg | diff_vs_legacy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `average_fusion` | 0.74398 | 0.88000 | 17 | 10 | 0 | 0 |
| `legacy_adaptive_softmax` | 0.74398 | 0.88000 | 17 | 10 | 0 | 0 |
| `legacy_adaptive_ratio` | 0.74398 | 0.88000 | 17 | 10 | 0 | 0 |
| `margin_adaptive_ratio` | 0.74398 | 0.88000 | 17 | 10 | 0 | 0 |
| `entropy_adaptive_ratio` | 0.74398 | 0.88000 | 17 | 10 | 0 | 0 |
| `maxprob_adaptive_ratio` | 0.74398 | 0.88000 | 17 | 10 | 0 | 0 |
| `reliability_weighted_static_fusion` | 0.74398 | 0.88000 | 17 | 10 | 0 | 0 |

## Formula Terpilih Dari Validation

- Formula terpilih: `average_fusion`.
- Validation Macro F1: 0.74398.
- Tie-break yang dipakai: phone->safe lebih rendah, lalu safe->phone lebih rendah, lalu urutan kesederhanaan kandidat.
- Karena seluruh kandidat seri pada validation dan seluruh `diff_vs_avg = 0`, formula paling sederhana yaitu `average_fusion` dipilih sebagai hasil final validation.

## Hasil Test Formula Terpilih

- Test Macro F1: 0.78059.
- Confusion matrix: [[20, 20], [4, 176]].
- safe->phone: 20; phone->safe: 4.
- Prediksi berbeda dari average fusion: 0.
- Prediksi berbeda dari legacy adaptive: 0.

## Perbandingan Dengan Baseline Utama

- Baseline average fusion utama: Macro F1 0.78059.
- Formula terpilih pada test tidak mengungguli baseline; delta Macro F1 = +0.00000.
- Apakah prediksi baru benar-benar berubah dari average: tidak.
- Trade-off safe->phone dan phone->safe terhadap baseline: +0 dan +0.

## Interpretasi Final

- Seluruh weighted probability adaptive variants tetap tidak mengubah keputusan akhir terhadap average fusion, baik pada validation maupun pada test.
- `legacy_adaptive_ratio` dan `entropy_adaptive_ratio` memang menghasilkan bobot yang lebih ekstrem, tetapi perubahan bobot tersebut tetap tidak cukup untuk mendorong probabilitas fusion melintasi threshold keputusan 0.5 pada sampel mana pun.
- Pada konfigurasi checkpoint Front14A + Side13, average fusion atas probabilitas mentah sudah secara implisit membawa unsur confidence, sehingga confidence-weighting eksplisit menjadi redundant.
- Experiment 20 dapat dibaca sebagai early-stop metodologis untuk jalur adaptive fusion berbasis weighted probability average pada konfigurasi ini.

## Status Reliability-Weighted Static Fusion

- `reliability_weighted_static_fusion` hanya pembanding statis berbasis Macro F1 validation single-view.
- Formula ini bukan adaptive fusion per-sample dan tidak boleh dilabeli sebagai adaptive.

## Status Kesimpulan

- Adaptive confidence-weighting belum terbukti mengungguli average fusion.
- Experiment 20 tidak mengubah rumus adaptive fusion utama.
- Average fusion tetap strategi yang paling sederhana dan stabil pada konfigurasi ini.
- Hasil Experiment 20 bersifat eksploratif dan belum otomatis menggantikan hasil utama penelitian.
- Jika hasil tampak menjanjikan, formula baru tetap perlu dikonsultasikan ke pembimbing sebelum dijadikan metode utama.
- Validation set yang relatif kecil membuat beberapa kandidat perlu ditafsirkan hati-hati.
