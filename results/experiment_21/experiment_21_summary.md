# Experiment 21 Summary

- Experiment 21 applies the best stride-20 recipe under the final_v1 stride 30 sampling protocol: the front view uses the 21B stabilization recipe, the side view uses the 21C validation-loss stabilization recipe, and only proposal-compliant average/adaptive fusion is evaluated.
- Output disimpan di `results/experiment_21/` dan `checkpoints/experiment_21/`.
- Notebook `notebooks/experiment_21.ipynb` membaca artefak setelah script Python selesai dijalankan.

## Protocol

- Dataset: 3MDAD binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama dengan final pipeline.
- Frame stride: 30.
- Primary metric: Macro F1.
- Threshold: 0.50.
- Fusion: decision-level average fusion dan adaptive confidence fusion.

## Configuration

- Front: lr=3e-05, wd=0.0005, dropout=0.4, LS=0.0, freeze=5, patience=4.
- Side: lr=2e-05, wd=0.0005, dropout=0.3, LS=0.0, freeze=4, patience=6.

## Results

- Front Macro F1: 0.76626; confusion: [[24, 16], [14, 166]]; status: OK.
- Side Macro F1: 0.71590; confusion: [[26, 14], [28, 152]]; status: OVERFIT.
- Average fusion Macro F1: 0.82504; confusion: [[27, 13], [9, 171]].
- Adaptive fusion Macro F1: 0.82504; confusion: [[27, 13], [9, 171]].
- Adaptive berbeda dari average pada 0 sampel.

## Reference Context

- Final v1 reference fusion Macro F1: 0.78059 with stride 30 and support 220.
- Exp 8 exploratory fusion Macro F1: 0.87494 with stride 15 and support 419.
- Final v1 and Experiment 21 now share stride/support, so this comparison is apple-to-apple; Exp 8 remains exploratory context only.
