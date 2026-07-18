# Final v1 Summary

- final_v1_retrain adalah pipeline end-to-end yang melatih ulang front dan side menggunakan konfigurasi final.
- Output final_v1_retrain disimpan terpisah di `results/final_v1/` dan `checkpoints/final_v1/`.
- Branch ini bukan eksperimen tuning baru, melainkan reproduksi final_v1 yang dikunci sementara dan bisa direproduksi.
- Hasil retrain dibandingkan dengan hasil referensi terkunci, bukan langsung menggantikan klaim final utama.
- Referensi Front14A: Macro F1 0.76626, confusion [[24, 16], [14, 166]].
- Referensi Side13: Macro F1 0.62023, confusion [[11, 29], [12, 168]].
- Referensi Fusion baseline: Macro F1 0.78059, confusion [[20, 20], [4, 176]].

## Konfigurasi Retrain

- Dataset tetap 3MDAD, task tetap binary `safe_driving` vs `phone_use`, view tetap front dan side.
- Backbone tetap EfficientNetV2-S dan fusion tetap decision-level fusion.
- Metrik utama tetap Macro F1; ECE dan Brier Score hanya diagnostics tambahan.
- Front mengikuti konfigurasi Front14A revised: lr=3e-05, wd=0.0003, dropout=0.4, freeze=5.
- Side mengikuti konfigurasi Side13 berdasarkan artefak repo: lr=2e-05, wd=0.002, dropout=0.3, freeze=4.
- Catatan Side13: Konfigurasi Side13 diturunkan dari history, ablation summary, dan config repo. Tidak ditemukan preset EXPERIMENT_CONFIGS khusus untuk Side13, tetapi LR=2e-5, WD=2e-3, dropout=0.3, label_smoothing=0.0, freeze=4, class_weights=[2.5,1.0], scheduler ReduceLROnPlateau, dan early stopping patience=7 konsisten dengan artefak yang tersedia.

## Hasil Retrain Aktual

- Front best epoch: 9; val Macro F1: 0.72581; test Macro F1: 0.76626; confusion: [[24, 16], [14, 166]].
- Side best epoch: 11; val Macro F1: 0.75000; test Macro F1: 0.62023; confusion: [[11, 29], [12, 168]].
- Average fusion Macro F1: 0.78059; confusion: [[20, 20], [4, 176]].
- Adaptive fusion Macro F1: 0.78059; confusion: [[20, 20], [4, 176]].
- Jumlah prediksi adaptive berbeda dari average: 0.
- Delta Macro F1 average fusion terhadap front single-view: +0.01433.
- Average fusion menurunkan phone->safe error dari 14 pada front menjadi 4.
- Average fusion menaikkan safe->phone error dari 16 pada front menjadi 20.

## Reproducibility Check

- Front reference F1 = 0.76626; actual = 0.76626; delta = +0.00000.
- Side reference F1 = 0.62023; actual = 0.62023; delta = +0.00000.
- Average Fusion reference F1 = 0.78059; actual = 0.78059; delta = +0.00000.
- Adaptive Fusion reference F1 = 0.78059; actual = 0.78059; delta = +0.00000.

## Status Generalisasi dan Kalibrasi

- Front generalization status: OK; ECE: 0.12513; Brier Score: 0.11311.
- Side generalization status: OK; ECE: 0.18263; Brier Score: 0.16537.
- Calibration diagnostics menunjukkan confidence model masih belum ideal, sehingga confidence-based adaptive fusion tidak otomatis lebih unggul.

## Interpretasi

- Front view adalah single-view terbaik pada final_v1_retrain.
- Side view lebih lemah sebagai standalone, terutama karena recall `safe_driving` rendah dan kecenderungan bias ke `phone_use`, tetapi tetap berguna sebagai informasi komplementer untuk fusion.
- Average fusion meningkatkan Macro F1 keseluruhan dan mengurangi missed detection `phone_use`, tetapi menambah false alarm pada `safe_driving`.
- Adaptive fusion berbasis confidence belum terbukti mengungguli average fusion pada konfigurasi ini.
- Average fusion menjadi strategi paling sederhana dan stabil untuk hasil utama final_v1_retrain.
- Pada run ini adaptive masih identik dengan average fusion.

## Hubungan Dengan Eksperimen 18-20

- Experiment 18 threshold tuning tidak mengganti hasil utama final_v1_retrain.
- Experiment 19 class weighting belum terbukti mengungguli baseline secara valid; gamma 0.95 hanya exploratory test-best, bukan hasil final.
- Validation-selected gamma 0.50 menghasilkan test Macro F1 0.77992, sangat dekat dengan baseline 0.78059 tetapi belum terbukti mengungguli.
- Experiment 20 adaptive formula exploration menunjukkan variasi weighted-probability fusion tetap identik dengan average fusion.
- Experiment 20 tidak mengubah rumus adaptive fusion utama; hasil utama tetap final_v1_retrain dengan fusion Macro F1 0.78059.
