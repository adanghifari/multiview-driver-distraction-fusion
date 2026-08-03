# Experiment 21E Summary

- Experiment 21E keeps the OK front model from Experiment 21 stride30-best locked and retrains only the side view with light regularization. It keeps dropout and patience from the best Exp21 side recipe while only increasing weight decay to test whether the overfit gap can improve without hurting fusion as much as Experiment 21D.
- Output disimpan di `results/experiment_21E/` dan `checkpoints/experiment_21E/`.
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
- Side 21E: lr=2e-05, wd=0.001, dropout=0.3, LS=0.0, freeze=4, patience=6, monitor=val_loss.

## Results

- Front locked Macro F1: 0.76626; confusion: [[24, 16], [14, 166]]; status: OK.
- Side Macro F1: 0.71590; confusion: [[26, 14], [28, 152]]; status: OVERFIT.
- Average fusion Macro F1: 0.82504; confusion: [[27, 13], [9, 171]].
- Adaptive fusion Macro F1: 0.82504; confusion: [[27, 13], [9, 171]].
- Adaptive berbeda dari average pada 0 sampel.

## Diagnostic Comparison

- Exp21 stride30-best side gap: 0.22406.
- Exp21E side gap: 0.22406.
- Exp21 stride30-best fusion Macro F1: 0.82504.
- Exp21D fusion Macro F1: 0.81113.
- Final v1 fusion Macro F1: 0.78059.
- Exp 8 exploratory fusion Macro F1: 0.87494.
