# Experiment 21E Notes

## Tujuan

Experiment 21E adalah satu percobaan kecil terakhir untuk mengecek apakah overfit side bisa dikurangi tanpa penurunan fusion seperti pada Experiment 21D.

Front model tetap tidak diubah karena front Exp21 stride30-best sudah berstatus OK.

## Protokol

- Dataset: 3MDAD.
- Task: binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama.
- Backbone: EfficientNetV2-S.
- Frame stride: 30.
- Threshold: 0.50.
- Front: locked dari `checkpoints/experiment_21/front_best_exp21.pt`.
- Side: retrain light-regularized.
- Fusion: average fusion dan adaptive confidence fusion.
- Output:
  - `results/experiment_21E/`
  - `checkpoints/experiment_21E/`

## Konfigurasi

| View | Status | LR | WD | Dropout | Label Smoothing | Freeze | Patience | Monitor | Class Weight |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Front | locked Exp21 stride30-best | 3e-5 | 5e-4 | 0.4 | 0.0 | 5 | 4 | validation Macro F1 | `[2.5, 1.0]` |
| Side | retrain | 2e-5 | 1e-3 | 0.3 | 0.0 | 4 | 6 | validation loss | `[2.5, 1.0]` |

## Rationale

Experiment 21D menurunkan side gap sedikit dan menaikkan side F1, tetapi fusion turun dari 0.82504 ke 0.81113. Karena itu, Experiment 21E hanya menaikkan weight decay dari 5e-4 ke 1e-3, sambil mengembalikan dropout 0.3 dan patience 6 seperti resep Exp21 stride30-best.

## Command

Jalankan dari root repo:

```powershell
.venv\Scripts\python.exe -m src.experiment_21E.run_experiment_21E
```

Jika training sudah selesai dan hanya ingin re-run evaluasi/summary:

```powershell
.venv\Scripts\python.exe -m src.experiment_21E.run_experiment_21E --skip-train
```

## Kriteria Baca Hasil

- Jika side gap turun dan fusion Macro F1 tetap mendekati atau melebihi 0.82504, Exp21E lebih menarik daripada Exp21D.
- Jika fusion tetap turun, maka percobaan overfit side dihentikan dan Exp21 stride30-best tetap menjadi hasil utama.
