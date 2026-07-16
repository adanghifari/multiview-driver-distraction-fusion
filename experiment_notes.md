# Experiment Notes

## experiment_14A - Front View Stabilization Revision

Experiment 14A dilakukan karena front view pada Experiment 13 masih menunjukkan
generalization gap antara train loss dan validation loss, sedangkan side view
sudah relatif stabil. Eksperimen ini hanya menyesuaikan konfigurasi front view;
side view tetap menjadi kontrol dari Experiment 13 dan tidak dilatih ulang.

### Catatan Experiment 14A Awal

Experiment 14A awal berhasil mengecilkan train-validation loss gap, tetapi gagal
mempertahankan kemampuan diskriminatif terhadap kelas `safe_driving`. Model
menjadi bias ke kelas mayoritas `phone_use`; confusion matrix menunjukkan hanya
1 dari 40 sampel `safe_driving` yang dikenali dengan benar, dan Test Macro F1
turun menjadi sekitar 0.4659.

Temuan ini menunjukkan bahwa pengurangan generalization gap tidak selalu berarti
model lebih sehat. Dalam kasus ini, regularisasi/freeze terlalu kuat membuat
front view kehilangan kapasitas belajar kelas minoritas.

Experiment 14A direvisi dengan mengembalikan freeze stage dari 6 ke 5 dan
mematikan label smoothing dari 0.05 ke 0.0. Hipotesis revisi: model akan kembali
mampu mengenali kelas `safe_driving`, sementara dropout 0.4 dan weight decay
3e-4 tetap memberi regularisasi ringan.

Konfigurasi yang diubah untuk front view:
- Backbone tetap EfficientNetV2-S.
- Optimizer tetap AdamW.
- Learning rate: 3e-5.
- Weight decay: 3e-4.
- Dropout: 0.4.
- Label smoothing: 0.0.
- Freeze stage: 5.
- Scheduler: ReduceLROnPlateau.
- Scheduler factor: 0.5.
- Scheduler patience: 1.
- Early stopping aktif, patience 4.
- Batch size: 32.
- Max epoch: 30.
- Best model dipilih berdasarkan validation Macro F1.

Indikator keberhasilan:
- Test Macro F1 front mendekati atau tidak jauh dari Experiment 13.
- Safe recall tidak collapse.
- Confusion matrix menunjukkan `safe_driving` masih dikenali dengan baik.
- Validation Macro F1 minimal sekitar 0.73 atau lebih.
- Train-validation loss gap tidak lebih buruk dari Experiment 13.
- Fusion dengan side Experiment 13 tidak turun jauh dari hasil Experiment 13.

Interpretasi jika hasil gagal:
- Jika train loss dan validation loss sama-sama tinggi, regularisasi terlalu
  kuat atau freeze terlalu banyak.
- Jika train loss rendah tetapi validation loss tetap tinggi, overfit masih
  terjadi.
- Jika Macro F1 naik tetapi validation loss memburuk, model belum sehat karena
  kemungkinan makin overconfident.
- Jika gap mengecil dan Macro F1 stabil atau turun sedikit, eksperimen masih
  dapat dianggap berhasil sebagai stabilisasi.

Command yang dijalankan manual:

```powershell
# Opsional, buat backup side Experiment 13 sebelum fusion
Copy-Item checkpoints\side_best.pt checkpoints\side_best_exp13_backup.pt

# Training front Experiment 14A saja
python -m src.train --view front --experiment experiment_14A

# Evaluasi front Experiment 14A
python -m src.evaluate --view front --checkpoint checkpoints\front_best_exp14A.pt

# Fusion memakai front 14A dan side Experiment 13
python -m src.fusion --front-checkpoint checkpoints\front_best_exp14A.pt --side-checkpoint checkpoints\side_best_exp13_backup.pt
```

Output utama:
- `checkpoints/front_best_exp14A.pt`
- `results/front_history_exp14A.json`
- `results/experiment_14A_front_metrics.json`
- `results/fusion_comparison_exp14A.json`

Catatan: jangan menjalankan training untuk side view pada Experiment 14A. Side
view tetap memakai checkpoint Experiment 13 sebagai kontrol stabil.
