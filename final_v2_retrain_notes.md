# Final v2 Retrain Notes

## Tujuan

`final_v2_retrain` merapikan hasil terbaik Experiment 21 stride30-best menjadi pipeline final yang terpisah dan mudah dibandingkan dengan `final_v1_retrain`.

## Protokol

- Dataset: 3MDAD.
- Task: binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama dengan final_v1.
- Backbone: EfficientNetV2-S.
- Frame stride: 30.
- Threshold: 0.50.
- Fusion: average fusion dan adaptive confidence fusion.
- Output:
  - `results/final_v2/`
  - `checkpoints/final_v2/`

## Konfigurasi

| View | Source Recipe | LR | WD | Dropout | Label Smoothing | Freeze | Patience | Monitor | Class Weight |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Front | Exp21B front stabilization | 3e-5 | 5e-4 | 0.4 | 0.0 | 5 | 4 | validation Macro F1 | `[2.5, 1.0]` |
| Side | Exp21C side stabilization | 2e-5 | 5e-4 | 0.3 | 0.0 | 4 | 6 | validation loss | `[2.5, 1.0]` |

## Command

Jalankan dari root repo:

```powershell
.venv\Scripts\python.exe -m src.final_v2.run_final_v2
```

Jika training sudah selesai dan hanya ingin re-run evaluasi/summary:

```powershell
.venv\Scripts\python.exe -m src.final_v2.run_final_v2 --skip-train
```

Setelah pipeline selesai, jalankan uji statistik:

```powershell
.venv\Scripts\python.exe -m src.final_v2.statistical_tests_final_v2_vs_final_v1
```

Setelah script Python selesai, buka dan run:

```text
notebooks/final_v2_retrain.ipynb
```

## Artifact Penting

- `results/final_v2/final_v2_summary.json`
- `results/final_v2/final_v2_summary.md`
- `results/final_v2/final_v2_table.csv`
- `results/final_v2/final_v2_single_view_predictions.csv`
- `results/final_v2/fusion_final_v2_predictions.csv`
- `results/final_v2/statistical_tests_vs_final_v1/final_v1_vs_final_v2_statistics_summary.csv`

## Kriteria Baca Hasil

- Final v2 dibandingkan langsung dengan final_v1 karena keduanya memakai stride 30 dan support test yang sama.
- Jika hasil retrain mereplikasi Exp21 stride30-best, fusion Macro F1 yang diharapkan sekitar 0.82504.
- Uji statistik tetap diperlukan sebelum klaim superioritas kuat.
