# Final v2 Summary

- final_v2_retrain adalah pipeline end-to-end yang melatih ulang front dan side menggunakan konfigurasi Exp21 stride30-best.
- Output final_v2 disimpan terpisah di `results/final_v2/` dan `checkpoints/final_v2/`.
- Tujuan final_v2 adalah pembanding rapi terhadap final_v1 pada protokol stride 30 yang sama.

## Konfigurasi Retrain

- Dataset: 3MDAD, task binary `safe_driving` vs `phone_use`.
- Frame stride: 30.
- Backbone: EfficientNetV2-S.
- Fusion: average fusion dan adaptive confidence fusion.
- Front mengikuti Experiment 21B front stabilization: lr=3e-05, wd=0.0005, dropout=0.4, freeze=5, monitor=validation Macro F1.
- Side mengikuti Experiment 21C side validation-loss stabilization: lr=2e-05, wd=0.0005, dropout=0.3, freeze=4, monitor=validation loss.

## Hasil Retrain Aktual

- Front best epoch: 9; val Macro F1: 0.72581; test Macro F1: 0.76626; confusion: [[24, 16], [14, 166]].
- Side best epoch: 25; val Macro F1: 0.73653; test Macro F1: 0.71590; confusion: [[26, 14], [28, 152]].
- Average fusion Macro F1: 0.82504; confusion: [[27, 13], [9, 171]].
- Adaptive fusion Macro F1: 0.82504; confusion: [[27, 13], [9, 171]].
- Jumlah prediksi adaptive berbeda dari average: 0.

## Perbandingan

- Final v1 average fusion Macro F1: 0.78059; final_v2 delta: +0.04445.
- Exp21 stride30-best average fusion Macro F1: 0.82504; final_v2 delta: +0.00000.
- Final v1 support: 220; final_v2 support: 220.

## Status Generalisasi dan Kalibrasi

- Front generalization status: OK; ECE: 0.12513; Brier Score: 0.11311.
- Side generalization status: OVERFIT; ECE: 0.14317; Brier Score: 0.13552.
- Uji statistik final_v2 vs final_v1 tersedia melalui `python -m src.final_v2.statistical_tests_final_v2_vs_final_v1` setelah pipeline selesai.
