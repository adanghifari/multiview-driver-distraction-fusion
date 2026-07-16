# Experiment 15 - Per-Sample Error Analysis

## Tujuan

Experiment 15 bertujuan menganalisis pola kesalahan per-sample pada test set
untuk membandingkan kontribusi:
- front view single-model,
- side view single-model,
- average fusion,
- adaptive fusion.

Eksperimen ini bukan training model baru dan bukan tuning hyperparameter.
Fokusnya adalah membaca checkpoint final yang sudah tersedia, menjalankan
inferensi pada paired test set, lalu mengekspor prediksi per-sample.

## Checkpoint

Checkpoint yang digunakan:
- Front: `checkpoints/front_best_exp14A.pt`
- Side: `checkpoints/side_best_exp13_backup.pt`

Side view tetap memakai checkpoint Experiment 13 sebagai kontrol stabil.

## Alasan Tidak Training Ulang

Experiment 15 mengevaluasi failure pattern dari kandidat stabil saat ini:
Front Experiment 14A revisi + Side Experiment 13. Training ulang akan mengubah
objek analisis, sehingga tidak dilakukan.

## Output

Script menghasilkan:
- `results/error_analysis_exp15.csv`
- `results/error_summary_exp15.json`

CSV berisi probabilitas, prediksi, status benar/salah, dan `case_type` untuk
setiap sample test.

## Penjelasan `case_type`

- `both_correct`: front dan side sama-sama benar.
- `front_only_correct`: front benar, side salah.
- `side_only_correct`: side benar, front salah.
- `both_wrong`: front dan side sama-sama salah, dan fusion tidak memperbaiki
  error front.
- `fusion_fixed_front_error`: front salah, tetapi average/adaptive fusion
  memperbaiki prediksi menjadi benar.
- `fusion_broke_front_correct`: front benar, tetapi average/adaptive fusion
  menjadi salah.

## Command

Jalankan dari root repository:

```powershell
python -m src.error_analysis --front-checkpoint checkpoints\front_best_exp14A.pt --side-checkpoint checkpoints\side_best_exp13_backup.pt --output-prefix exp15
```

Jika memakai virtualenv lokal:

```powershell
.\.venv\Scripts\python.exe -m src.error_analysis --front-checkpoint checkpoints\front_best_exp14A.pt --side-checkpoint checkpoints\side_best_exp13_backup.pt --output-prefix exp15
```

## Interpretasi Awal Yang Perlu Dilihat

Setelah hasil keluar, cek:
- apakah fusion lebih sering memperbaiki error front atau justru merusak
  prediksi front yang sudah benar;
- apakah error dominan berupa `safe_driving` yang diprediksi sebagai
  `phone_use`;
- apakah side view menyumbang koreksi pada sample yang gagal di front view;
- apakah average fusion dan adaptive fusion menghasilkan pola error yang sama.
