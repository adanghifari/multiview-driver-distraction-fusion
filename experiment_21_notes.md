# Experiment 21 Notes

## Current Branch

- Branch: `experiment_21_stride30_best`
- Purpose: run Experiment 21 directly with the best recipe found from the stride-20 exploration, but under the final_v1 stride-30 protocol.
- Status: completed training/evaluation; statistical tests have been generated.

## Tujuan

Experiment 21 versi ini dibuat untuk membandingkan konfigurasi terbaik hasil eksplorasi dengan final_v1 secara apple-to-apple. Perbandingan ini memakai frame stride yang sama dengan final_v1, yaitu stride 30, sehingga support test sama-sama 220.

## Protokol

- Dataset: 3MDAD.
- Task: binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama dengan final pipeline.
- Backbone: EfficientNetV2-S.
- Input size: 224 x 224.
- Optimizer: AdamW.
- Scheduler: ReduceLROnPlateau.
- Decision threshold: 0.50.
- Frame stride: 30.
- Fusion: average fusion dan adaptive confidence fusion.
- Output:
  - `results/experiment_21/`
  - `checkpoints/experiment_21/`

## Konfigurasi

| View | Source Recipe | LR | WD | Dropout | Label Smoothing | Freeze | Patience | Monitor | Class Weight |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Front | Exp21B front stabilization | 3e-5 | 5e-4 | 0.4 | 0.0 | 5 | 4 | validation Macro F1 | `[2.5, 1.0]` |
| Side | Exp21C side stabilization | 2e-5 | 5e-4 | 0.3 | 0.0 | 4 | 6 | validation loss | `[2.5, 1.0]` |

## Command

Jalankan dari root repo:

```powershell
.venv\Scripts\python.exe -m src.experiment_21.run_experiment_21
```

Jika training sudah selesai dan hanya ingin re-run evaluasi/summary:

```powershell
.venv\Scripts\python.exe -m src.experiment_21.run_experiment_21 --skip-train
```

Setelah script Python selesai, buka dan run:

```text
notebooks/experiment_21.ipynb
```

## Hasil Aktual

Artefak hasil tersedia pada:

- `results/experiment_21/experiment_21_summary.json`
- `results/experiment_21/experiment_21_summary.md`
- `results/experiment_21/experiment_21_table.csv`
- `results/experiment_21/single_view_predictions_exp21.csv`
- `results/experiment_21/fusion_predictions_exp21.csv`

Ringkasan hasil:

| Metode | Accuracy | Precision Macro | Recall Macro | Macro F1 | safe->phone | phone->safe | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Front single-view | 0.86364 | 0.77183 | 0.76111 | 0.76626 | 16 | 14 | OK |
| Side single-view | 0.80909 | 0.69857 | 0.74722 | 0.71590 | 14 | 28 | OVERFIT |
| Average fusion | 0.90000 | 0.83967 | 0.81250 | 0.82504 | 13 | 9 | - |
| Adaptive fusion | 0.90000 | 0.83967 | 0.81250 | 0.82504 | 13 | 9 | - |

Compared with final_v1 average fusion:

- Final v1 Macro F1: 0.78059.
- Exp21 stride30-best Macro F1: 0.82504.
- Delta: +0.04445.
- Support: 220 vs 220.

## Statistical Tests

Artefak statistik tersedia pada:

- `results/experiment_21/statistical_tests_vs_final_v1/final_v1_vs_exp21_statistics.json`
- `results/experiment_21/statistical_tests_vs_final_v1/final_v1_vs_exp21_statistics_summary.csv`

Ringkasan:

| Test | Metric | Delta Exp21 - Final v1 | P-value | 95% CI | Significant 0.05 |
|---|---|---:|---:|---|---|
| McNemar exact | decision correctness @ 0.50 | +2 unique correct | 0.77441 | - | No |
| DeLong | ROC-AUC | +0.00292 | 0.85730 | - | No |
| Paired stratified bootstrap | Macro F1 | +0.04520 | 0.11760 | [-0.01040, 0.11005] | No |
| Paired stratified bootstrap | ROC-AUC | +0.00275 | 0.85520 | [-0.03084, 0.03444] | No |
| Paired stratified bootstrap | PR-AUC | +0.00163 | 0.71920 | [-0.00783, 0.01110] | No |

Interpretasi: Exp21 stride30-best lebih baik secara deskriptif, tetapi peningkatannya belum signifikan secara statistik pada alpha 0.05. Klaim yang aman adalah peningkatan empiris/deskriptif, bukan superioritas statistik yang kuat.

## Catatan Overfit

Front berstatus OK dengan train-validation loss gap 0.04383. Side berstatus OVERFIT dengan train-validation loss gap 0.22406, tetapi side tetap meningkatkan performa test dan membantu fusion naik dari 0.78059 menjadi 0.82504.

Eksplorasi berikutnya boleh fokus ke side overfit untuk pengetahuan: apakah regularisasi/early stopping yang lebih konservatif dapat menurunkan gap tanpa mengorbankan Macro F1 fusion. Tujuan awal eksplorasi ini bukan langsung mengganti hasil utama, tetapi menguji apakah memperbaiki overfit side membuat hasil turun, stabil, atau justru naik.
