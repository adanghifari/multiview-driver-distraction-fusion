# Experiment 21C Summary

- Experiment 21C keeps the proposal-compliant average/adaptive fusion setup, locks the stabilized front model from Experiment 21B, and retrains only the side view. The side checkpoint is selected using validation loss with patience 6 to test whether a probability-stabilized side model reduces safe-to-phone errors in fusion.
- Output disimpan di `results/experiment_21C/` dan `checkpoints/experiment_21C/`.
- Front model dikunci dari Experiment 21B; hanya side yang diretrain.
- Fusion tetap average fusion dan adaptive fusion sesuai proposal.

## Protocol

- Dataset: 3MDAD binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama.
- Frame stride: 20.
- Side checkpoint monitor: validation loss.
- Threshold: 0.50.

## Configuration

- Front locked: checkpoint `D:\Skripsi\Experiment\checkpoints\experiment_21B\front_best_exp21B.pt` from experiment_21B.
- Side 21C: lr=2e-05, wd=0.0005, dropout=0.3, LS=0.0, freeze=4, patience=6, monitor=val_loss.

## Results

- Front locked Macro F1: 0.78797; confusion: [[43, 16], [28, 235]]; status: OK.
- Side Macro F1: 0.76085; confusion: [[37, 22], [25, 238]]; status: OVERFIT.
- Average fusion Macro F1: 0.81626; confusion: [[39, 20], [14, 249]].
- Adaptive fusion Macro F1: 0.81626; confusion: [[39, 20], [14, 249]].
- Adaptive berbeda dari average pada 0 sampel.

## Reference Context

- Exp 21B front Macro F1: 0.78797; fusion Macro F1: 0.75880.
- Exp 21A fusion Macro F1: 0.76597.
- Final v1 fusion Macro F1: 0.78059.
- Exp 8 exploratory fusion Macro F1: 0.87494.
