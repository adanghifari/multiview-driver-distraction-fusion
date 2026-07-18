# Experiment 19 Notes

## Tujuan Experiment 19 Revised v2

Experiment 19 revised v2 bertujuan memperbaiki kelemahan utama side view pada
kelas `safe_driving`. Fokus revisi ini adalah mengurangi bias side terhadap
kelas `phone_use`, menaikkan safe recall, dan melihat apakah side yang lebih
seimbang dapat memberi sinyal komplementer yang lebih baik ke fusion.

## Alasan Fokus Ke Class Imbalance Pada Side

Pada baseline side lama, confusion matrix adalah `[[11, 29], [12, 168]]`.
Artinya:
- safe recall = `11 / 40 = 0.275`
- phone recall = `168 / 180 = 0.933`

Pola ini menunjukkan bahwa side sangat bias ke `phone_use`. Karena itu, revisi
v2 mencoba class-balance rescue dengan balanced class weights yang dihitung
langsung dari distribusi label training, bukan hardcoded.

## Konfigurasi Hyperparameter Revised v2

- View: `side`
- Experiment preset: `experiment_19_side`
- Backbone: `EfficientNetV2-S`
- Optimizer: `AdamW`
- Learning rate: `3e-5`
- Weight decay: `1e-4`
- Dropout: `0.3`
- Label smoothing: `0.0`
- Frozen stages: `4`
- Scheduler: `ReduceLROnPlateau`
- Scheduler factor: `0.5`
- Scheduler patience: `1`
- Early stopping patience: `5`
- Batch size: `32`
- Max epochs: `30`
- Best checkpoint dipilih berdasarkan validation Macro F1

## Class Weight Yang Digunakan

Balanced class weights dihitung dari train split:
- Train safe_driving: `176`
- Train phone_use: `908`

Bobot loss yang dipakai:
- safe_driving: `3.0795`
- phone_use: `0.5969`

## Riwayat Singkat Experiment 19

### 1. Experiment 19 awal gagal

- Side Macro F1 = `0.58625`
- Fusion Macro F1 = `0.76827`
- Indikasi: underfit / terlalu konservatif, dan konfigurasi yang terlalu mirip
  Front14A tidak cocok untuk side view.

### 2. Experiment 19 revisi pertama

- Side Macro F1 = `0.62023`
- Fusion Macro F1 = `0.78059`
- Hasil ini berhasil memulihkan performa ke baseline Side13, tetapi belum
  meningkat di atas baseline.

### 3. Experiment 19 revised v2

- Menggunakan balanced class weight.
- Target utamanya adalah memperbaiki safe recall side dan menekan error
  `safe->phone`.

## Perbandingan Side13 vs Side19 Awal vs Side19 Revisi Pertama vs Side19 Revised v2

### Side13 lama

- Macro F1: `0.62023`
- Confusion matrix: `[[11, 29], [12, 168]]`
- safe recall: `0.275`
- phone recall: `0.933`
- safe->phone: `29`
- phone->safe: `12`

### Side19 awal

- Macro F1: `0.58625`
- Confusion matrix: `[[10, 30], [17, 163]]`
- safe recall: `0.250`
- phone recall: `0.906`
- safe->phone: `30`
- phone->safe: `17`

### Side19 revisi pertama

- Macro F1: `0.62023`
- Confusion matrix: `[[11, 29], [12, 168]]`
- safe recall: `0.275`
- phone recall: `0.933`
- safe->phone: `29`
- phone->safe: `12`

### Side19 revised v2

- Best epoch: `8`
- Train loss best: `0.58090`
- Val loss best: `0.61449`
- Loss gap: `0.03358`
- Validation Macro F1 terbaik: `0.57951`
- Side test Macro F1: `0.55772`
- Confusion matrix side revised v2: `[[22, 18], [60, 120]]`
- safe recall side revised v2: `0.550`
- phone recall side revised v2: `0.667`
- side safe->phone: `18`
- side phone->safe: `60`

## Analisis Trade-off Side Revised v2

Balanced class weight memang berhasil menaikkan safe recall side secara besar,
dari `0.275` menjadi `0.550`, dan menurunkan error `safe->phone` dari `29`
menjadi `18`. Namun, trade-off-nya sangat mahal:
- phone recall turun dari `0.933` menjadi `0.667`
- phone->safe naik dari `12` menjadi `60`
- Macro F1 side justru turun ke `0.55772`

Jadi, pada level single-view, revised v2 berhasil membuat side lebih seimbang
ke arah `safe_driving`, tetapi mengorbankan terlalu banyak performa pada kelas
`phone_use`.

## Hasil Fusion Baseline

Baseline utama tetap:
- Front14A + Side13
- Fusion Macro F1 = `0.78059`
- Confusion matrix fusion = `[[20, 20], [4, 176]]`
- fusion safe->phone = `20`
- fusion phone->safe = `4`

## Hasil Fusion Front14A + Side19 Revised v2

- Average Fusion Macro F1 = `0.79166`
- Adaptive Fusion Macro F1 = `0.79166`
- Accuracy = `0.87727`
- Confusion matrix fusion = `[[26, 14], [13, 167]]`
- fusion safe->phone = `14`
- fusion phone->safe = `13`

## Analisis Fusion Revised v2

Hasil ini menunjukkan trade-off yang berbeda dari baseline:
- safe->phone turun dari `20` menjadi `14`
- phone->safe naik dari `4` menjadi `13`
- Macro F1 fusion naik dari `0.78059` menjadi `0.79166`

Artinya, meskipun side single-view revised v2 lebih buruk secara mandiri,
pergeseran bias side ke arah kelas aman justru memberi efek komplementer yang
berbeda saat digabung dengan front. Fusion menjadi lebih seimbang secara macro,
tetapi dengan konsekuensi missed detection `phone_use` yang lebih tinggi.

## Apakah Side19 Revised v2 Layak Menggantikan Side13

Jawabannya: **belum bisa diputuskan sebagai pengganti final secara otomatis**.

Alasannya:
- Jika fokus utama adalah side single-view, revised v2 jelas lebih buruk dari
  Side13.
- Jika fokus utama adalah fusion Macro F1, revised v2 justru lebih baik karena
  naik ke `0.79166`.
- Namun, kenaikan itu dibayar dengan lonjakan `phone->safe` dari `4` menjadi
  `13`, sehingga trade-off-nya cukup signifikan.

Dengan demikian, revised v2 layak dianggap sebagai kandidat analisis penting
karena berhasil menaikkan Macro F1 fusion, tetapi belum otomatis lebih aman
sebagai hasil final sebelum diputuskan apakah trade-off missed detection
`phone_use` masih dapat diterima.

## Apakah Fusion Tembus 0.80

Belum. Fusion revised v2 mencapai Macro F1 `0.79166`, sehingga masih berada di
bawah target `0.80`, meskipun lebih tinggi daripada baseline `0.78059`.
