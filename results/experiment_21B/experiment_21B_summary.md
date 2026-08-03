# Experiment 21B Summary

- Experiment 21B keeps the proposal-compliant fusion protocol from Experiment 21 but retrains only the front view. The side model is locked from Experiment 21A because its training status was OK and its Macro F1 improved over final_v1. The front model restores freeze depth to 5 to reduce the borderline train-validation gap while keeping stride 20, label smoothing 0.0, and average/adaptive fusion unchanged.
- Output disimpan di `results/experiment_21B/` dan `checkpoints/experiment_21B/`.
- Side model dikunci dari Experiment 21A; hanya front yang diretrain.
- Fusion tetap dibatasi pada average fusion dan adaptive fusion sesuai proposal.

## Protocol

- Dataset: 3MDAD binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama.
- Frame stride: 20.
- Primary metric: Macro F1.
- Threshold: 0.50.

## Configuration

- Front 21B: lr=3e-05, wd=0.0005, dropout=0.4, LS=0.0, freeze=5, patience=4.
- Side locked: checkpoint `D:\Skripsi\Experiment\checkpoints\experiment_21\side_best_exp21.pt` from experiment_21A.

## Results

- Front Macro F1: 0.78797; confusion: [[43, 16], [28, 235]]; status: OK.
- Side locked Macro F1: 0.66044; confusion: [[20, 39], [16, 247]]; status: OK.
- Average fusion Macro F1: 0.75880; confusion: [[33, 26], [18, 245]].
- Adaptive fusion Macro F1: 0.75880; confusion: [[33, 26], [18, 245]].
- Adaptive berbeda dari average pada 0 sampel.

## Reference Context

- Exp 21A fusion Macro F1: 0.76597; front status BORDERLINE.
- Final v1 fusion Macro F1: 0.78059.
- Exp 8 exploratory fusion Macro F1: 0.87494.
