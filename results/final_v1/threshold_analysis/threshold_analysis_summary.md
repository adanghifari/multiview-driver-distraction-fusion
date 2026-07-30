# Threshold Analysis Final V1

Eksperimen ini hanya menerapkan threshold berbeda pada probabilitas fused yang sudah tersedia. Tidak ada retraining, perubahan checkpoint, perubahan dataset, split, backbone, atau probabilitas dasar model.

- Threshold diuji: 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
- Total sampel: 220
- Urutan confusion matrix: [[safe_as_safe, safe_as_phone], [phone_as_safe, phone_as_phone]]
- ROC-AUC dan PR-AUC dihitung dari probabilitas, sehingga tetap sama untuk metode yang sama pada semua threshold.

## Threshold Dengan Perbedaan Prediksi
- Threshold 0.30: Softmax vs Average=0, Linear vs Average=1, Linear vs Softmax=1
- Threshold 0.35: Softmax vs Average=0, Linear vs Average=2, Linear vs Softmax=2
- Threshold 0.40: Softmax vs Average=0, Linear vs Average=3, Linear vs Softmax=3
- Threshold 0.45: Softmax vs Average=1, Linear vs Average=4, Linear vs Softmax=3
- Threshold 0.55: Softmax vs Average=2, Linear vs Average=4, Linear vs Softmax=2
- Threshold 0.60: Softmax vs Average=1, Linear vs Average=7, Linear vs Softmax=6

## Macro F1 Tertinggi
- Adaptive Linear Normalization threshold 0.50: Macro F1=0.78059, Balanced Accuracy=0.73889, MCC=0.59110, safe->phone=20, phone->safe=4
- Adaptive Softmax threshold 0.50: Macro F1=0.78059, Balanced Accuracy=0.73889, MCC=0.59110, safe->phone=20, phone->safe=4
- Average Fusion threshold 0.50: Macro F1=0.78059, Balanced Accuracy=0.73889, MCC=0.59110, safe->phone=20, phone->safe=4

## ROC-AUC dan PR-AUC
- Adaptive Linear Normalization: ROC-AUC=0.87694, PR-AUC=0.96921
- Adaptive Softmax: ROC-AUC=0.87056, PR-AUC=0.96530
- Average Fusion: ROC-AUC=0.86556, PR-AUC=0.96293

## Ranking Utama
| rank | method | threshold | macro_f1 | balanced_accuracy | mcc | roc_auc | pr_auc | safe_to_phone | phone_to_safe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | adaptive_linear_normalization | 0.50 | 0.78059 | 0.73889 | 0.5911 | 0.87694 | 0.96921 | 20 | 4 |
| 2 | adaptive_softmax | 0.50 | 0.78059 | 0.73889 | 0.5911 | 0.87056 | 0.9653 | 20 | 4 |
| 3 | average_fusion | 0.50 | 0.78059 | 0.73889 | 0.5911 | 0.86556 | 0.96293 | 20 | 4 |
| 4 | average_fusion | 0.55 | 0.77629 | 0.77361 | 0.55266 | 0.86556 | 0.96293 | 15 | 14 |
| 5 | adaptive_linear_normalization | 0.60 | 0.77517 | 0.78056 | 0.55066 | 0.87694 | 0.96921 | 14 | 16 |
| 6 | adaptive_softmax | 0.55 | 0.77174 | 0.76389 | 0.54424 | 0.87056 | 0.9653 | 16 | 13 |
| 7 | average_fusion | 0.60 | 0.7676 | 0.79167 | 0.54074 | 0.86556 | 0.96293 | 12 | 21 |
| 8 | adaptive_linear_normalization | 0.55 | 0.76694 | 0.75417 | 0.53604 | 0.87694 | 0.96921 | 17 | 12 |
| 9 | adaptive_softmax | 0.60 | 0.75852 | 0.77917 | 0.5214 | 0.87056 | 0.9653 | 13 | 21 |
| 10 | adaptive_linear_normalization | 0.45 | 0.68336 | 0.64444 | 0.45646 | 0.87694 | 0.96921 | 28 | 2 |
| 11 | adaptive_softmax | 0.45 | 0.67841 | 0.6375 | 0.48666 | 0.87056 | 0.9653 | 29 | 0 |
| 12 | average_fusion | 0.45 | 0.66154 | 0.625 | 0.46291 | 0.86556 | 0.96293 | 30 | 0 |
| 13 | adaptive_linear_normalization | 0.40 | 0.62585 | 0.6 | 0.41208 | 0.87694 | 0.96921 | 32 | 0 |
| 14 | adaptive_softmax | 0.40 | 0.56681 | 0.5625 | 0.3235 | 0.87056 | 0.9653 | 35 | 0 |
| 15 | average_fusion | 0.40 | 0.56681 | 0.5625 | 0.3235 | 0.86556 | 0.96293 | 35 | 0 |
| 16 | adaptive_linear_normalization | 0.35 | 0.52317 | 0.5375 | 0.24942 | 0.87694 | 0.96921 | 37 | 0 |
| 17 | adaptive_linear_normalization | 0.30 | 0.49988 | 0.525 | 0.20319 | 0.87694 | 0.96921 | 38 | 0 |
| 18 | adaptive_softmax | 0.30 | 0.47552 | 0.5125 | 0.14335 | 0.87056 | 0.9653 | 39 | 0 |
| 19 | adaptive_softmax | 0.35 | 0.47552 | 0.5125 | 0.14335 | 0.87056 | 0.9653 | 39 | 0 |
| 20 | average_fusion | 0.30 | 0.47552 | 0.5125 | 0.14335 | 0.86556 | 0.96293 | 39 | 0 |
| 21 | average_fusion | 0.35 | 0.47552 | 0.5125 | 0.14335 | 0.86556 | 0.96293 | 39 | 0 |

## Terbaik Per Metode
| rank | method | threshold | macro_f1 | balanced_accuracy | mcc | roc_auc | pr_auc | safe_to_phone | phone_to_safe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | adaptive_linear_normalization | 0.50 | 0.78059 | 0.73889 | 0.5911 | 0.87694 | 0.96921 | 20 | 4 |
| 2 | adaptive_softmax | 0.50 | 0.78059 | 0.73889 | 0.5911 | 0.87056 | 0.9653 | 20 | 4 |
| 3 | average_fusion | 0.50 | 0.78059 | 0.73889 | 0.5911 | 0.86556 | 0.96293 | 20 | 4 |

## Catatan Interpretasi
- Hasil ini untuk analisis threshold sesuai arahan dosen, bukan otomatis mengganti threshold final 0.50.
- Jangan memilih threshold final berdasarkan test set; gunakan hasil ini sebagai bukti analitis perilaku probabilitas dan label.
- Perbedaan ROC-AUC/PR-AUC menunjukkan perbedaan ranking probabilitas antar metode, bukan efek threshold.
