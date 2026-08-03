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

