# Experiment 21D Notes

## Tujuan

Experiment 21D adalah eksperimen diagnostik untuk menjawab pertanyaan: jika overfit pada side model Exp21 stride30-best dikurangi, apakah Macro F1 fusion tetap stabil, naik, atau turun?

Eksperimen ini tidak mengubah front model karena front Exp21 stride30-best sudah berstatus OK.

## Protokol

- Dataset: 3MDAD.
- Task: binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama.
- Backbone: EfficientNetV2-S.
- Frame stride: 30.
- Threshold: 0.50.
- Front: locked dari `checkpoints/experiment_21/front_best_exp21.pt`.
- Side: retrain regularized.
- Fusion: average fusion dan adaptive confidence fusion.
- Output:
  - `results/experiment_21D/`
  - `checkpoints/experiment_21D/`

## Konfigurasi

| View | Status | LR | WD | Dropout | Label Smoothing | Freeze | Patience | Monitor | Class Weight |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Front | locked Exp21 stride30-best | 3e-5 | 5e-4 | 0.4 | 0.0 | 5 | 4 | validation Macro F1 | `[2.5, 1.0]` |
| Side | retrain | 2e-5 | 1e-3 | 0.4 | 0.0 | 4 | 5 | validation loss | `[2.5, 1.0]` |

## Rationale

Exp21 stride30-best side memiliki train-validation loss gap 0.22406 pada best validation-loss checkpoint. Pola history menunjukkan side tetap membantu fusion, tetapi gap mulai melebar setelah epoch menengah. Karena itu, perubahan dibuat konservatif:

- Weight decay dinaikkan dari 5e-4 ke 1e-3.
- Dropout dinaikkan dari 0.3 ke 0.4.
- Early stopping patience diturunkan dari 6 ke 5.
- Freeze depth tetap 4 agar kapasitas side tidak dikurangi terlalu agresif.
- Monitor tetap validation loss karena target utama adalah menguji overfit/gap.

## Command

Jalankan dari root repo:

```powershell
.venv\Scripts\python.exe -m src.experiment_21D.run_experiment_21D
```

Jika training sudah selesai dan hanya ingin re-run evaluasi/summary:

```powershell
.venv\Scripts\python.exe -m src.experiment_21D.run_experiment_21D --skip-train
```

## Kriteria Baca Hasil

- Jika side gap turun dan fusion Macro F1 tetap >= 0.82504, maka regularisasi berhasil tanpa mengorbankan performa.
- Jika side gap turun tetapi fusion turun sedikit, overfit sebelumnya mungkin membantu decision boundary pada test set.
- Jika side gap turun tetapi fusion turun banyak, Exp21D tidak perlu mengganti hasil utama dan cukup menjadi analisis tambahan.
