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
**0.78059** dengan confusion matrix `[[20, 20], [4, 176]]`. Setelah dilakukan
sweep class weighting pada side, kandidat terbaik diperoleh pada
**gamma = 0.95** dengan hasil fusion Macro F1 **0.79323** dan confusion matrix
`[[25, 15], [11, 169]]`. Ini berarti Experiment 19 berhasil meningkatkan Macro
F1 fusion dibanding baseline fusion sebelumnya.

Namun, perbaikannya tetap disertai trade-off. Error `safe->phone` membaik dari
**20** menjadi **15**, tetapi error `phone->safe` memburuk dari **4** menjadi
**11**. Dengan demikian, hasil terbaik Experiment 19 tidak dapat dibaca
sebagai perbaikan menyeluruh pada semua aspek, melainkan sebagai peningkatan
Macro F1 dengan perubahan karakter kesalahan.

Sebagai pembanding, konfigurasi **gamma = 1.00** menghasilkan Macro F1 fusion
**0.79166** dengan confusion matrix `[[26, 14], [13, 167]]`. Narrow sweep pada
gamma `0.93`, `0.94`, `0.96`, `0.97`, dan `0.98` juga tidak berhasil
melampaui `gamma = 0.95`. Tidak ada konfigurasi Experiment 19 yang menembus
Macro F1 **0.80**.

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
- Kandidat terbaik Experiment 19 adalah `gamma = 0.95` dengan Macro F1 fusion
  `0.79323`, tetapi hasil ini tetap memiliki trade-off pada error
  `phone->safe`.
- Calibration belum menjadi fokus utama pada tahap ini, tetapi nilai ECE dan
  Brier Score front menunjukkan confidence model masih layak dianalisis lebih
  lanjut pada tahap lanjutan.

## Kesimpulan Sementara

Berdasarkan hasil yang tersedia, strategi fusion tetap layak dipertahankan
sebagai hasil terbaik penelitian ini karena memberikan peningkatan Macro F1
terhadap baseline front single-view. Hasil baseline fusion berada pada
**0.78059**, sedangkan Experiment 19 mendorongnya naik menjadi **0.79323**
melalui side class-weighting pada `gamma = 0.95`.

Meski demikian, peningkatan tersebut tetap disertai trade-off. Pada baseline
fusion, masalah utama berada pada false alarm kelas `safe_driving`, sedangkan
pada kandidat terbaik Experiment 19, false alarm tersebut membaik tetapi
error `phone->safe` justru meningkat. Karena itu, interpretasi hasil perlu
menekankan bahwa peningkatan Macro F1 tidak berarti seluruh aspek performa
membaik secara bersamaan.
