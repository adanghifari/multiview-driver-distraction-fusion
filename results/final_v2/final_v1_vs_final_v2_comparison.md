# Final v1 vs Final v2 Comparison

## Ringkasan

Final v2 adalah versi retrain yang merapikan resep terbaik Experiment 21 stride30-best ke dalam pipeline final terpisah. Perbandingan dengan Final v1 bersifat apple-to-apple karena keduanya memakai split yang sama, frame stride 30, threshold 0.50, dan support test 220.

## Perbedaan Konfigurasi

| Komponen | Final v1 | Final v2 | Perubahan |
|---|---|---|---|
| Dataset | 3MDAD | 3MDAD | Sama |
| Task | binary safe_driving vs phone_use | binary safe_driving vs phone_use | Sama |
| Views | front + side | front + side | Sama |
| Backbone | EfficientNetV2-S | EfficientNetV2-S | Sama |
| Frame stride | 30 | 30 | Sama |
| Threshold | 0.50 | 0.50 | Sama |
| Image size | 224 x 224 | 224 x 224 | Sama |
| Batch size | 32 | 32 | Sama |
| Max epochs | 30 | 30 | Sama |
| Seed | SPLIT_SEED | SPLIT_SEED | Sama |
| Fusion | average + adaptive confidence | average + adaptive confidence | Sama |
| Front source recipe | Front14A revised | Exp21B front stabilization | Diganti |
| Front pretrained | True | True | Sama |
| Front optimizer | AdamW | AdamW | Sama |
| Front LR | 3e-5 | 3e-5 | Sama |
| Front weight decay | 3e-4 | 5e-4 | Naik |
| Front dropout | 0.4 | 0.4 | Sama |
| Front label smoothing | 0.0 | 0.0 | Sama |
| Front class weights | [2.5, 1.0] | [2.5, 1.0] | Sama |
| Front freeze depth | 5 | 5 | Sama |
| Front scheduler | ReduceLROnPlateau | ReduceLROnPlateau | Sama |
| Front scheduler factor | 0.5 | 0.5 | Sama |
| Front scheduler patience | 1 | 1 | Sama |
| Front patience | 4 | 4 | Sama |
| Front checkpoint monitor | validation Macro F1 | validation Macro F1 | Sama |
| Side source recipe | Side13 | Exp21C side validation-loss stabilization | Diganti |
| Side pretrained | True | True | Sama |
| Side optimizer | AdamW | AdamW | Sama |
| Side LR | 2e-5 | 2e-5 | Sama |
| Side weight decay | 2e-3 | 5e-4 | Turun |
| Side dropout | 0.3 | 0.3 | Sama |
| Side label smoothing | 0.0 | 0.0 | Sama |
| Side class weights | [2.5, 1.0] | [2.5, 1.0] | Sama |
| Side freeze depth | 4 | 4 | Sama |
| Side scheduler | ReduceLROnPlateau | ReduceLROnPlateau | Sama |
| Side scheduler factor | 0.5 | 0.5 | Sama |
| Side scheduler patience | 1 | 2 | Naik |
| Side patience | 7 | 6 | Turun |
| Side checkpoint monitor | validation Macro F1 | validation loss | Diganti |

## Perbedaan Hasil

| Method | Version | Accuracy | Precision Macro | Recall Macro | Macro F1 | safe->phone | phone->safe | ECE | Brier | Confusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Front single-view | Final v1 | 0.86364 | 0.77183 | 0.76111 | 0.76626 | 16 | 14 | 0.12513 | 0.11311 | [[24, 16], [14, 166]] |
| Front single-view | Final v2 | 0.86364 | 0.77183 | 0.76111 | 0.76626 | 16 | 14 | 0.12513 | 0.11311 | [[24, 16], [14, 166]] |
| Side single-view | Final v1 | 0.81364 | 0.66553 | 0.60417 | 0.62023 | 29 | 12 | 0.18263 | 0.16537 | [[11, 29], [12, 168]] |
| Side single-view | Final v2 | 0.80909 | 0.69857 | 0.74722 | 0.71590 | 14 | 28 | 0.14317 | 0.13552 | [[26, 14], [28, 152]] |
| Average fusion | Final v1 | 0.89091 | 0.86565 | 0.73889 | 0.78059 | 20 | 4 | - | - | [[20, 20], [4, 176]] |
| Average fusion | Final v2 | 0.90000 | 0.83967 | 0.81250 | 0.82504 | 13 | 9 | - | - | [[27, 13], [9, 171]] |
| Adaptive fusion | Final v1 | 0.89091 | 0.86565 | 0.73889 | 0.78059 | 20 | 4 | - | - | [[20, 20], [4, 176]] |
| Adaptive fusion | Final v2 | 0.90000 | 0.83967 | 0.81250 | 0.82504 | 13 | 9 | - | - | [[27, 13], [9, 171]] |

## Delta Hasil

| Method | Accuracy Delta | Precision Macro Delta | Recall Macro Delta | Macro F1 Delta | safe->phone Delta | phone->safe Delta | ECE Delta | Brier Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Front single-view | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0 | 0 | +0.00000 | +0.00000 |
| Side single-view | -0.00455 | +0.03304 | +0.14305 | +0.09567 | -15 | +16 | -0.03946 | -0.02985 |
| Average fusion | +0.00909 | -0.02598 | +0.07361 | +0.04445 | -7 | +5 | - | - |
| Adaptive fusion | +0.00909 | -0.02598 | +0.07361 | +0.04445 | -7 | +5 | - | - |

## Error Trade-off Fusion

| Error Type | Final v1 Average Fusion | Final v2 Average Fusion | Perubahan |
|---|---:|---:|---:|
| safe_driving -> phone_use | 20 | 13 | -7 |
| phone_use -> safe_driving | 4 | 9 | +5 |

Final v2 mengurangi false alarm safe-to-phone, tetapi menambah missed phone-to-safe. Secara Macro F1, trade-off ini tetap menguntungkan karena F1 naik dari 0.78059 menjadi 0.82504.

## Training Diagnostics

| View | Version | Best Epoch | Best Val Macro F1 | Train Loss | Val Loss | Gap | Status | Monitor |
|---|---|---:|---:|---:|---:|---:|---|---|
| Front | Final v1 | 9 | 0.72581 | 0.47353 | 0.51736 | 0.04383 | OK | validation Macro F1 |
| Front | Final v2 | 9 | 0.72581 | 0.47353 | 0.51736 | 0.04383 | OK | validation Macro F1 |
| Side | Final v1 | 11 | 0.75000 | 0.50153 | 0.56022 | 0.05869 | OK | validation Macro F1 |
| Side | Final v2 | 25 | 0.73653 | 0.20284 | 0.42690 | 0.22406 | OVERFIT | validation loss |

Catatan: peningkatan Final v2 terutama datang dari side model, tetapi side Final v2 memiliki train-validation loss gap lebih besar. Dengan kriteria diagnostics saat ini, side berubah dari OK pada Final v1 menjadi OVERFIT pada Final v2.

## Statistik Final v2 vs Final v1

| Test | Metric | Delta | P-value | 95% CI | Significant 0.05 |
|---|---|---:|---:|---|---|
| McNemar exact | decision correctness @ threshold 0.50 | +2 unique correct | 0.77441 | - | No |
| DeLong | ROC-AUC | +0.00292 | 0.85730 | - | No |
| Paired stratified bootstrap | Macro F1 | +0.04520 | 0.11760 | [-0.01040, 0.11005] | No |
| Paired stratified bootstrap | ROC-AUC | +0.00275 | 0.85520 | [-0.03084, 0.03444] | No |
| Paired stratified bootstrap | PR-AUC | +0.00163 | 0.71920 | [-0.00783, 0.01110] | No |

## Fusion Variant Probability Analysis

Analisis ini memakai prediksi Final v2 yang sama dan threshold 0.50. Pada keputusan akhir, Average Fusion, Adaptive Softmax, dan Adaptive Linear Normalization menghasilkan confusion matrix yang sama. Namun, skor probabilitasnya tidak identik, sehingga ROC-AUC, PR-AUC, ECE, dan Brier dapat berbeda.

| Method | Accuracy | Macro F1 | ROC-AUC | PR-AUC | ECE | Brier | safe->phone | phone->safe | ROC-AUC 95% CI | PR-AUC 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Average Fusion | 0.90000 | 0.82504 | 0.86847 | 0.96462 | 0.13322 | 0.10780 | 13 | 9 | [0.80152, 0.92625] | [0.94599, 0.98069] |
| Adaptive Softmax | 0.90000 | 0.82504 | 0.86486 | 0.96326 | 0.13446 | 0.10524 | 13 | 9 | [0.79569, 0.92444] | [0.94408, 0.98009] |
| Adaptive Linear Normalization | 0.90000 | 0.82504 | 0.84639 | 0.95122 | 0.10559 | 0.10071 | 13 | 9 | [0.76540, 0.91709] | [0.92259, 0.97689] |

| Pairwise | Hard Prediction Diff | Max Prob Abs Diff | Mean Prob Abs Diff | DeLong ROC-AUC Delta B-A | DeLong P-value | McNemar P-value |
|---|---:|---:|---:|---:|---:|---:|
| Average Fusion vs Adaptive Softmax | 0 | 0.05372269 | 0.01120039 | -0.00361 | 0.32083 | 1.00000 |
| Average Fusion vs Adaptive Linear Normalization | 0 | 0.21984770 | 0.04717839 | -0.02208 | 0.28848 | 1.00000 |
| Adaptive Softmax vs Adaptive Linear Normalization | 0 | 0.16739893 | 0.03597800 | -0.01847 | 0.30105 | 1.00000 |

Catatan: adaptive confidence utama di pipeline Final v2 adalah Adaptive Softmax sesuai rumus proposal. Adaptive Linear Normalization disimpan sebagai analisis tambahan untuk menunjukkan bahwa normalisasi bobot confidence biasa mengubah skor probabilitas, meskipun pada threshold 0.50 keputusan akhirnya tetap sama pada test set ini.

## Interpretasi

Final v2 unggul secara deskriptif terhadap Final v1 pada Macro F1 fusion (+0.04445), dan peningkatan utama berasal dari perbaikan side view. Namun, uji McNemar, DeLong, dan bootstrap belum menunjukkan signifikansi statistik pada alpha 0.05. Karena itu, klaim yang aman adalah Final v2 memberikan peningkatan empiris/deskriptif pada protokol evaluasi yang sama, bukan bukti superioritas statistik yang kuat.

## File Pendukung

- `results/final_v1/final_v1_summary.json`
- `results/final_v1/fusion_final_v1_predictions.csv`
- `results/final_v2/final_v2_summary.json`
- `results/final_v2/fusion_final_v2_predictions.csv`
- `results/final_v2/fusion_variant_analysis_final_v2.json`
- `results/final_v2/fusion_variant_analysis_final_v2.csv`
- `results/final_v2/statistical_tests_vs_final_v1/final_v1_vs_final_v2_statistics_summary.csv`
