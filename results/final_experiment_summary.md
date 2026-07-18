# Ringkasan Akhir Eksperimen

## Konteks

Tahap akhir penelitian ini tidak melakukan training ulang, tuning tambahan,
perubahan arsitektur model, perubahan dataset, maupun perubahan metode fusion.
Ringkasan ini sepenuhnya disusun dari hasil yang sudah tersedia pada
Experiment 14A revisi, Experiment 15, Experiment 16, Experiment 17,
Experiment 18, dan Experiment 19.

Fokus evaluasi utama adalah **Macro F1-score**, karena penelitian ini perlu
menilai keseimbangan performa pada dua kelas, yaitu `safe_driving` dan
`phone_use`, bukan hanya akurasi total.

## Ringkasan Hasil Utama

### Front single-view (Experiment 14A revisi)

- Macro F1: **0.76626**
- Accuracy: **0.86364**
- Precision macro: **0.77183**
- Recall macro: **0.76111**
- ECE: **0.12513**
- Brier Score: **0.11311**

Confusion matrix front:
- True `safe_driving`: 24 benar, 16 salah menjadi `phone_use`
- True `phone_use`: 166 benar, 14 salah menjadi `safe_driving`

Hasil ini menunjukkan bahwa front single-view tetap menjadi baseline terkuat
di antara model single-view. Namun, kesalahan front cukup terlihat pada kelas
`safe_driving`, khususnya ketika sampel aman diprediksi sebagai `phone_use`.

### Side single-view (Experiment 13)

- Macro F1: **0.62023**
- Accuracy: **0.81364**
- Precision macro: **0.66553**
- Recall macro: **0.60417**

Secara kuantitatif, side single-view berada di bawah front single-view pada
seluruh metrik utama. Dengan demikian, side view tidak lebih kuat sebagai
model tunggal, tetapi tetap relevan sebagai sumber informasi tambahan.

### Fusion

#### Average fusion

- Macro F1: **0.78059**
- Accuracy: **0.89091**
- Precision macro: **0.86565**
- Recall macro: **0.73889**

#### Adaptive fusion

- Macro F1: **0.78059**
- Accuracy: **0.89091**
- Precision macro: **0.86565**
- Recall macro: **0.73889**

Confusion matrix fusion:
- True `safe_driving`: 20 benar, 20 salah menjadi `phone_use`
- True `phone_use`: 176 benar, 4 salah menjadi `safe_driving`

Hasil ini menunjukkan bahwa fusion meningkatkan Macro F1 dari **0.76626**
pada front single-view menjadi **0.78059**. Dengan kata lain, integrasi front
dan side memberi manfaat terhadap performa keseluruhan. Namun, peningkatan ini
tidak merata pada semua jenis error, karena fusion juga meningkatkan false
alarm pada kelas `safe_driving`.

## Ringkasan Analisis Error Numerik (Experiment 15)

Experiment 15 dilakukan untuk membaca pola error per-sample tanpa melakukan
training baru. Dari total 220 sampel uji:

- Front correct: 190
- Side correct: 179
- Average fusion correct: 196
- Adaptive fusion correct: 196
- Both correct: 163
- Front only correct: 27
- Side only correct: 16
- Both wrong: 14
- Fusion fixed front error: 11
- Fusion broke front correct: 5

Ringkasan misclassification:
- Front `safe -> phone`: 16
- Front `phone -> safe`: 14
- Side `safe -> phone`: 29
- Side `phone -> safe`: 12
- Fusion `safe -> phone`: 20
- Fusion `phone -> safe`: 4

Interpretasi awal dari angka tersebut adalah:
- Fusion berhasil mengoreksi sebagian error front, terbukti dari 11 kasus
  `fusion_fixed_front_error`.
- Risiko negatif fusion tetap ada, tetapi jumlahnya lebih kecil, yaitu 5 kasus
  `fusion_broke_front_correct`.
- Fusion sangat menurunkan error `phone_use -> safe`, dari 14 pada front
  menjadi 4 pada fusion.
- Sebaliknya, fusion meningkatkan error `safe -> phone`, dari 16 pada front
  menjadi 20 pada fusion.

## Ringkasan Analisis Visual (Experiment 16)

Experiment 16 memilih sampel representatif dari empat kategori:
- `fusion_fixed_front_error`
- `fusion_broke_front_correct`
- `side_only_correct`
- `both_wrong`

Analisis visual menunjukkan bahwa side view berperan sebagai informasi
komplementer pada beberapa kasus `phone_use` ketika front view kurang jelas.
Namun, fusion juga meningkatkan kecenderungan false alarm pada kelas
`safe_driving`, terutama ketika pose tangan pada side view menyerupai pola
penggunaan ponsel. Hal ini menjelaskan mengapa fusion meningkatkan Macro F1
secara keseluruhan, tetapi safe recall menurun dibanding front single-view.

## Ringkasan Analisis Lanjutan Fusion (Experiment 17 dan 18)

Experiment 17 menunjukkan bahwa adaptive fusion lama tetap identik dengan
average fusion karena distribusi bobotnya masih sangat dekat ke 0.5. Bahkan
setelah diuji dengan adaptive sharpening pada alpha 1, 2, 3, 5, dan 10, hasil
akhir tetap tidak berubah dan Macro F1 tetap berada di **0.78059**.

Experiment 18 kemudian menguji threshold tuning berbasis validation set tanpa
training ulang. Threshold terbaik validation untuk average fusion dan adaptive
fusion sama-sama **0.49**, tetapi ketika diuji pada test set, hasilnya justru
menurun dari Macro F1 **0.78059** menjadi **0.72982**. False alarm
`safe->phone` juga naik dari **20** menjadi **24**, sementara `phone->safe`
tetap **4**.

Dengan demikian, hasil utama tetap menggunakan threshold **0.50** dengan Macro
F1 **0.78059**, dan peningkatan lebih lanjut belum berhasil dicapai melalui
adaptive sharpening maupun threshold tuning pada konfigurasi saat ini.

## Ringkasan Side Class-Weighting (Experiment 19)

Experiment 19 memindahkan arah perbaikan dari level fusion ke modalitas side
single-view melalui class-weighting sweep. Motivasi utamanya adalah karena side
baseline masih relatif lemah, terutama dalam membedakan sampel aman dari
`phone_use`, sehingga kontribusi side pada fusion belum optimal.

Baseline fusion lama antara Front14A dan Side13 memiliki Macro F1
**0.78059** dengan confusion matrix `[[20, 20], [4, 176]]`. Pada sweep
eksploratif berbasis test, `gamma = 0.95` sempat terlihat sebagai test-best
dengan Macro F1 **0.79323** dan confusion matrix `[[25, 15], [11, 169]]`.
Namun, hasil tersebut tidak dijadikan final karena gamma dipilih berdasarkan
test set, sehingga berisiko menimbulkan test set leakage.

Setelah protokol diperbaiki, gamma dipilih ulang secara benar menggunakan
validation set. Hasil validation-based selection menunjukkan bahwa gamma
terpilih adalah **0.50** dengan validation Macro F1 **0.76929**. Ketika
gamma terpilih ini dievaluasi satu kali pada test set, hasilnya adalah Macro
F1 **0.77992** dengan confusion matrix `[[21, 19], [6, 174]]`.

Dengan demikian, class weighting pada side **belum terbukti mengungguli
baseline** Front14A + Side13. Trade-off yang muncul adalah `safe->phone`
membaik tipis dari **20** menjadi **19**, tetapi `phone->safe` memburuk dari
**4** menjadi **6**. Tidak ada konfigurasi validation-based final yang
menembus Macro F1 **0.80**.

## Ringkasan Eksplorasi Formula Adaptive Fusion (Experiment 20)

Experiment 20 dilakukan sebagai eksplorasi lanjutan pada level decision-level
fusion tanpa retraining. Tujuannya adalah menguji beberapa variasi adaptive
atau re-weighted fusion berbasis probabilitas untuk melihat apakah perubahan
definisi confidence dan mekanisme pembobotan dapat menghasilkan prediksi akhir
yang berbeda dari average fusion.

Seluruh kandidat yang diuji, yaitu `average_fusion`,
`legacy_adaptive_softmax`, `legacy_adaptive_ratio`,
`margin_adaptive_ratio`, `entropy_adaptive_ratio`,
`maxprob_adaptive_ratio`, dan `reliability_weighted_static_fusion`,
memperoleh hasil validation yang identik. Semua formula sama-sama menghasilkan
Macro F1 **0.74398** pada validation, dan seluruhnya memiliki
`diff_vs_avg = 0`, sehingga tidak ada satu pun formula yang menghasilkan
prediksi berbeda dari average fusion.

Karena seluruh kandidat seri pada validation, tie-break memilih formula paling
sederhana, yaitu `average_fusion`. Ketika formula terpilih ini dievaluasi satu
kali pada test set, hasilnya tetap sama dengan baseline utama, yaitu Macro F1
**0.78059** dengan confusion matrix `[[20, 20], [4, 176]]`,
`safe->phone = 20`, dan `phone->safe = 4`.

Interpretasi utama Experiment 20 adalah bahwa variasi weighted probability
adaptive fusion belum cukup untuk mengubah keputusan akhir pada konfigurasi
Front14A + Side13. Dengan kata lain, average fusion pada probabilitas mentah
sudah secara implisit membawa informasi confidence, sehingga confidence-based
re-weighting menjadi redundant pada eksperimen ini. Hasil utama penelitian
tetap tidak berubah, dan adaptive fusion belum terbukti lebih efektif daripada
average fusion.

## Poin Utama Untuk Bab Hasil dan Pembahasan

- Macro F1 tetap menjadi metrik utama untuk menarik kesimpulan.
- Front single-view adalah baseline terbaik pada model tunggal.
- Side single-view lebih lemah sebagai classifier tunggal, tetapi berguna
  sebagai informasi komplementer.
- Average fusion dan adaptive fusion sama-sama meningkatkan Macro F1 menjadi
  0.78059.
- Adaptive fusion belum mengungguli average fusion karena hasil akhirnya
  identik pada seluruh metrik evaluasi utama.
- Threshold tuning pada Experiment 18 juga belum memperbaiki hasil, sehingga
  threshold 0.50 tetap dipakai sebagai hasil utama.
- Experiment 19 menunjukkan bahwa perbaikan lebih lanjut lebih mungkin datang
  dari penguatan kualitas side view daripada dari modifikasi bobot fusion.
- Hasil eksploratif terbaik Experiment 19 muncul pada `gamma = 0.95`, tetapi
  itu tidak dipakai sebagai hasil final karena dipilih berdasarkan test set.
- Gamma final yang valid dari validation-based selection adalah `0.50` dengan
  test Macro F1 `0.77992`, sehingga class weighting belum terbukti
  mengungguli baseline `0.78059`.
- Experiment 20 mengevaluasi beberapa variasi adaptive/re-weighted fusion
  tanpa retraining, tetapi semua kandidat seri pada validation dan identik
  dengan average fusion.
- Formula terpilih Experiment 20 adalah `average_fusion` karena paling
  sederhana, dan hasil test tetap berada pada Macro F1 `0.78059`.
- Adaptive fusion berbasis confidence-weighting belum terbukti lebih efektif
  daripada average fusion pada konfigurasi checkpoint yang dipakai.
- Calibration belum menjadi fokus utama pada tahap ini, tetapi nilai ECE dan
  Brier Score front menunjukkan confidence model masih layak dianalisis lebih
  lanjut pada tahap lanjutan.

## Kesimpulan Sementara

Berdasarkan hasil yang tersedia, strategi fusion tetap layak dipertahankan
sebagai hasil terbaik penelitian ini karena memberikan peningkatan Macro F1
terhadap baseline front single-view. Hasil baseline fusion berada pada
**0.78059**, dan nilai ini tetap menjadi hasil utama yang valid setelah
Experiment 19 dievaluasi ulang dengan protokol pemilihan gamma berbasis
validation.

Experiment 19 tetap berguna sebagai ablation class weighting karena menunjukkan
bahwa perubahan bobot kelas memang menggeser trade-off antara `safe->phone` dan
`phone->safe`. Namun, pada protokol yang benar, gamma final terpilih `0.50`
hanya mencapai Macro F1 **0.77992**, sehingga class weighting belum terbukti
mengungguli baseline. Karena itu, hasil utama penelitian tidak berubah:
Front14A + Side13 dengan fusion Macro F1 **0.78059**, dan average fusion serta
adaptive fusion tetap identik. Experiment 20 kemudian menegaskan bahwa variasi
formula adaptive fusion berbasis weighted probability average juga belum
mengubah keputusan akhir maupun meningkatkan hasil utama tersebut.
