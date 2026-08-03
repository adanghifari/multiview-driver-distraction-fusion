# Experiment 21 Summary

- Experiment 21 adapts the strongest exploratory signal from Experiment 8 into a more disciplined final-style protocol. It uses stride 20 as a middle ground between Exp8 stride 15 and final_v1 stride 30, keeps label smoothing disabled following the final ablation, and relaxes freeze depth toward Exp8 to test whether more capacity helps under the stricter evaluation protocol.
- Output disimpan di `results/experiment_21/` dan `checkpoints/experiment_21/`.
- Notebook `notebooks/experiment_21.ipynb` membaca artefak setelah script Python selesai dijalankan.

## Protocol

- Dataset: 3MDAD binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama dengan final pipeline.
- Frame stride: 20.
- Primary metric: Macro F1.
- Threshold: 0.50.
- Fusion: decision-level average fusion dan adaptive confidence fusion.

## Configuration

- Front: lr=3e-05, wd=0.0005, dropout=0.4, LS=0.0, freeze=4, patience=4.
- Side: lr=2e-05, wd=0.0005, dropout=0.3, LS=0.0, freeze=3, patience=7.

## Results

- Front Macro F1: 0.77284; confusion: [[44, 15], [34, 229]]; status: BORDERLINE.
- Side Macro F1: 0.66044; confusion: [[20, 39], [16, 247]]; status: OK.
- Average fusion Macro F1: 0.76597; confusion: [[34, 25], [18, 245]].
- Adaptive fusion Macro F1: 0.76597; confusion: [[34, 25], [18, 245]].
- Adaptive berbeda dari average pada 0 sampel.

## Reference Context

- Final v1 reference fusion Macro F1: 0.78059 with stride 30 and support 220.
- Exp 8 exploratory fusion Macro F1: 0.87494 with stride 15 and support 419.
- Because stride/support differ, reference rows are context, not a direct statistical claim.
