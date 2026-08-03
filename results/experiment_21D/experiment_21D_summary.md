# Experiment 21D Summary

- Experiment 21D keeps the OK front model from Experiment 21 stride30-best locked and retrains only the side view with stronger regularization. The goal is diagnostic: test whether reducing side overfit preserves, improves, or lowers fusion performance.
- Output disimpan di `results/experiment_21D/` dan `checkpoints/experiment_21D/`.
- Front model dikunci dari Experiment 21 stride30-best; hanya side yang diretrain.
- Fusion tetap average fusion dan adaptive fusion sesuai proposal.

## Protocol

- Dataset: 3MDAD binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama.
- Frame stride: 30.
- Side checkpoint monitor: validation loss.
- Threshold: 0.50.

## Configuration

- Front locked: checkpoint `D:\Skripsi\Experiment\checkpoints\experiment_21\front_best_exp21.pt` from experiment_21_stride30_best.
- Side 21D: lr=2e-05, wd=0.001, dropout=0.4, LS=0.0, freeze=4, patience=5, monitor=val_loss.

## Results

- Front locked Macro F1: 0.76626; confusion: [[24, 16], [14, 166]]; status: OK.
- Side Macro F1: 0.73985; confusion: [[25, 15], [21, 159]]; status: OVERFIT.
- Average fusion Macro F1: 0.81113; confusion: [[25, 15], [8, 172]].
- Adaptive fusion Macro F1: 0.81113; confusion: [[25, 15], [8, 172]].
- Adaptive berbeda dari average pada 0 sampel.

## Diagnostic Comparison

- Exp21 stride30-best side gap: 0.22406.
- Exp21D side gap: 0.21484.
- Exp21 stride30-best fusion Macro F1: 0.82504.
- Final v1 fusion Macro F1: 0.78059.
- Exp 8 exploratory fusion Macro F1: 0.87494.
