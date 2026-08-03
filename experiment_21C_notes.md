# Experiment 21C Notes

## Tujuan

Experiment 21C memperbaiki side view setelah Experiment 21B berhasil menstabilkan front view. Eksperimen ini tetap berada dalam batas proposal karena strategi fusion tetap average fusion dan adaptive fusion.

## Alasan

Experiment 21B menunjukkan front sudah kuat dan statusnya `OK`, tetapi fusion masih turun karena side cenderung menarik sampel `safe_driving` menjadi `phone_use`.

Diagnosis utama:

- Front 21B Macro F1: 0.78797, status `OK`.
- Side Exp21A Macro F1: 0.66044, status gap `OK`, tetapi confusion matrix menunjukkan bias ke `phone_use`.
- Fusion 21B menaikkan `safe->phone` dari 16 pada front menjadi 26.

Karena itu Experiment 21C fokus pada stabilisasi probabilitas side, bukan mengubah formula fusion.

## Hipotesis

Checkpoint side yang dipilih berdasarkan validation loss dapat menghasilkan probabilitas yang lebih stabil untuk fusion dibanding checkpoint yang dipilih berdasarkan validation Macro F1. Dengan freeze stage 4 dan patience 6, side diharapkan menjadi lebih tidak agresif terhadap kelas `phone_use`.

## Protokol

- Dataset: 3MDAD.
- Task: binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama.
- Backbone: EfficientNetV2-S.
- Frame stride: 20.
- Front checkpoint: dikunci dari Experiment 21B.
- Side: retrain.
- Checkpoint monitor: validation loss.
- Patience side: 6.
- Decision threshold: 0.50.
- Fusion: average fusion dan adaptive fusion.
- Output:
  - `results/experiment_21C/`
  - `checkpoints/experiment_21C/`

## Konfigurasi Side

| Parameter | Nilai |
|---|---:|
| LR | 2e-5 |
| WD | 5e-4 |
| Dropout | 0.3 |
| Label smoothing | 0.0 |
| Freeze stage | 4 |
| Early stopping patience | 6 |
| LR scheduler patience | 2 |
| Checkpoint monitor | validation loss |
| Class weight | `[2.5, 1.0]` |

## Command

Jalankan dari root repo:

```powershell
.venv\Scripts\python.exe -m src.experiment_21C.run_experiment_21C
```

Jika side 21C sudah dilatih dan hanya ingin re-run evaluasi/summary:

```powershell
.venv\Scripts\python.exe -m src.experiment_21C.run_experiment_21C --skip-train
```

## Kriteria Keputusan

Experiment 21C dianggap memperbaiki penyebab fusion turun jika:

- `safe->phone` side turun dari 39 pada side Exp21A.
- Fusion `safe->phone` turun dari 26 pada Exp 21B.
- Fusion Macro F1 naik dari Exp 21B baseline 0.75880.
- Idealnya fusion Macro F1 mendekati atau melampaui final_v1 0.78059.

## Hasil Aktual

Artefak hasil tersedia pada:

- `results/experiment_21C/experiment_21C_summary.json`
- `results/experiment_21C/experiment_21C_summary.md`
- `results/experiment_21C/experiment_21C_table.csv`
- `results/experiment_21C/single_view_predictions_exp21C.csv`
- `results/experiment_21C/fusion_predictions_exp21C.csv`

Ringkasan hasil:

| Metode | Accuracy | Precision Macro | Recall Macro | Macro F1 | safe->phone | phone->safe | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Front locked Exp21B | 0.86335 | 0.77094 | 0.81117 | 0.78797 | 16 | 28 | OK |
| Side single-view 21C | 0.85404 | 0.75608 | 0.76603 | 0.76085 | 22 | 25 | OVERFIT |
| Average fusion | 0.89441 | 0.83075 | 0.80389 | 0.81626 | 20 | 14 | - |
| Adaptive fusion | 0.89441 | 0.83075 | 0.80389 | 0.81626 | 20 | 14 | - |

Training diagnostics:

- Side checkpoint terpilih pada epoch 15 berdasarkan validation loss 0.38643.
- Validation Macro F1 pada checkpoint tersebut adalah 0.76723.
- Train-validation loss gap side adalah 0.15835, sehingga status generalisasi berbasis gap dikategorikan `OVERFIT`.
- Walaupun gap membesar, ECE side membaik menjadi 0.09635 dan confusion matrix side jauh lebih seimbang dibanding Exp21A.

Interpretasi:

- Target utama tercapai: `safe->phone` side turun dari 39 pada Exp21A menjadi 22 pada Exp21C.
- Fusion `safe->phone` turun dari 26 pada Exp21B menjadi 20 pada Exp21C.
- Fusion `phone->safe` juga membaik dari 18 pada Exp21B menjadi 14 pada Exp21C.
- Fusion Macro F1 naik dari 0.75880 pada Exp21B menjadi 0.81626 pada Exp21C.
- Exp21C fusion melampaui final_v1 reference 0.78059 pada protokol stride 20.
- Adaptive fusion masih identik dengan average fusion pada prediksi akhir.

Keputusan sementara:

- Exp21C adalah kandidat fusion terbaik pada rangkaian Exp21.
- Catatan kehati-hatian: side 21C memiliki gap loss kategori `OVERFIT`, sehingga hasil perlu dijelaskan sebagai trade-off antara generalization gap dan kualitas probabilitas/confusion matrix pada test set.
- Karena checkpoint dipilih berdasarkan validation loss, bukan test set, hasil ini masih lebih dapat dipertanggungjawabkan daripada pemilihan berbasis test-best.
