# Final v2 Threshold Exploration

## Tujuan

- Mengeksplorasi threshold `phone_use` pada output fusion Final v2 tanpa retraining.
- Threshold grid penuh 0.01 sampai 0.99 dicoba untuk validation dan test.
- Threshold protocol dipilih dari validation set, lalu dievaluasi satu kali pada test set.
- Best test threshold disertakan sebagai analisis deskriptif/oracle, bukan protocol selection resmi.
- Hasil utama Final v2 tetap threshold 0.50; eksplorasi ini adalah analisis tambahan.
- Adaptive Linear Normalization disertakan sebagai analisis tambahan, bukan pengganti adaptive fusion proposal.

## Ringkasan

### average_fusion

- Best validation threshold: 0.51 dengan Macro F1 validation 0.76291.
- Test baseline 0.50: Macro F1 0.82504, accuracy 0.90000, safe->phone 13, phone->safe 9.
- Test selected threshold: Macro F1 0.81897, accuracy 0.89545, safe->phone 13, phone->safe 10.
- Delta Macro F1 test: -0.00607.
- Best test threshold descriptive/oracle: 0.50 dengan Macro F1 test 0.82504.

### adaptive_fusion

- Best validation threshold: 0.51 dengan Macro F1 validation 0.76291.
- Test baseline 0.50: Macro F1 0.82504, accuracy 0.90000, safe->phone 13, phone->safe 9.
- Test selected threshold: Macro F1 0.82504, accuracy 0.90000, safe->phone 13, phone->safe 9.
- Delta Macro F1 test: +0.00000.
- Best test threshold descriptive/oracle: 0.50 dengan Macro F1 test 0.82504.

### adaptive_linear_normalization

- Best validation threshold: 0.51 dengan Macro F1 validation 0.76291.
- Test baseline 0.50: Macro F1 0.82504, accuracy 0.90000, safe->phone 13, phone->safe 9.
- Test selected threshold: Macro F1 0.82504, accuracy 0.90000, safe->phone 13, phone->safe 9.
- Delta Macro F1 test: +0.00000.
- Best test threshold descriptive/oracle: 0.50 dengan Macro F1 test 0.82504.

## Catatan Interpretasi

- Jika threshold terpilih menaikkan Macro F1 test, tetap baca sebagai hasil tuning berbasis validation, bukan retraining model.
- Perhatikan trade-off `safe->phone` dan `phone->safe`; threshold lebih tinggi biasanya mengurangi false alarm safe->phone tetapi dapat menaikkan missed phone->safe.
- Average fusion dan adaptive fusion tidak diubah rumusnya.
