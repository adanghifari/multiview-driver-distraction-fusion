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

Catatan augmentasi: front-view augmentation pada `src/config.py`
(`AUG_ROTATION_DEGREE_FRONT = 68`, `AUG_COLOR_JITTER_FACTOR_FRONT = 1.2`)
dipertahankan dari konfigurasi pipeline Experiment 14 yang sudah aktif
sebelum revisi 14A. Variabel utama Experiment 14A revisi adalah freeze stage,
label smoothing, dropout, weight decay, dan early stopping; augmentasi tidak
diubah khusus pada revisi ini.

Indikator keberhasilan:
- Test Macro F1 front mendekati atau tidak jauh dari Experiment 13.
- Safe recall tidak collapse.
- Confusion matrix menunjukkan `safe_driving` masih dikenali dengan baik.
- Validation Macro F1 minimal sekitar 0.73 atau lebih.
- Train-validation loss gap tidak lebih buruk dari Experiment 13.
- Fusion dengan side Experiment 13 tidak turun jauh dari hasil Experiment 13.

### Hasil Experiment 14A Revisi

Experiment 14A revisi berhasil sebagai eksperimen stabilisasi. Dibandingkan
Experiment 14A awal, model tidak lagi collapse ke kelas mayoritas `phone_use`.
Recall `safe_driving` pada test set kembali menjadi 0.60 (24 dari 40 sampel
safe dikenali benar).

Ringkasan front view:
- Best epoch: 9.
- Validation Macro F1: 0.72581.
- Train loss best epoch: 0.47353.
- Validation loss best epoch: 0.51736.
- Train-validation loss gap best epoch: 0.04383.
- Test accuracy: 0.86364.
- Test precision macro: 0.77183.
- Test recall macro: 0.76111.
- Test Macro F1: 0.76626.
- ECE: 0.12513.
- Brier Score: 0.11311.
- Confusion matrix front: `[[24, 16], [14, 166]]`.

Ringkasan fusion dengan side Experiment 13:
- Single front Macro F1: 0.76626.
- Single side Macro F1: 0.62023.
- Average fusion Macro F1: 0.78059.
- Adaptive fusion Macro F1: 0.78059.
- Confusion matrix fusion: `[[20, 20], [4, 176]]`.

Interpretasi: Experiment 14A revisi menghasilkan kompromi yang lebih sehat.
Macro F1 front sedikit di bawah Experiment 13, tetapi train-validation loss gap
jauh lebih kecil dan kelas `safe_driving` tetap dikenali. Fusion juga tetap
kompetitif terhadap Experiment 13.

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

## experiment_14B - Front View Stabilization Refinement

Experiment 14B adalah penyempurnaan kecil setelah Experiment 14A revisi berhasil
sebagai kandidat stabil front view. Fokusnya adalah mencoba meningkatkan
kemampuan model mengenali kelas minoritas `safe_driving` tanpa merusak stabilitas
generalisasi yang sudah membaik.

### Catatan Experiment 14B Awal

Experiment 14B awal menurunkan dropout dari 0.40 menjadi 0.35. Hasilnya lebih
buruk dari 14A revisi: Front Macro F1 turun menjadi 0.74552, safe recall turun
menjadi 22/40 (0.55), loss gap best naik menjadi 0.08135, dan fusion Macro F1
turun menjadi 0.71285. Karena itu, penurunan dropout tidak dipertahankan.

### Revisi Experiment 14B

Hipotesis revisi: Experiment 14A revisi sudah stabil dengan dropout 0.40,
freeze stage 5, dan label smoothing 0.0. Karena penurunan dropout ke 0.35
memperburuk hasil, dropout dikembalikan ke 0.40. Perubahan yang diuji pada 14B
revisi hanya menaikkan weight decay dari 3e-4 menjadi 5e-4. Regularisasi bobot
yang sedikit lebih kuat diharapkan dapat menekan pola spesifik train tanpa
mengurangi kapasitas model mengenali `safe_driving` secara berlebihan.

Konfigurasi front view:
- Backbone tetap EfficientNetV2-S.
- Optimizer tetap AdamW.
- Learning rate: 3e-5.
- Weight decay: 5e-4.
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

Perbedaan dari Experiment 14A revisi:
- Weight decay dinaikkan dari 3e-4 menjadi 5e-4.
- Parameter lain dipertahankan sama atau sedekat mungkin dengan 14A revisi.

Perbedaan dari Experiment 14B awal:
- Dropout dikembalikan dari 0.35 menjadi 0.40.
- Weight decay dinaikkan dari 3e-4 menjadi 5e-4.

Indikator keberhasilan:
- Front Test Macro F1 minimal menyamai atau melampaui 14A revisi (0.76626).
- Safe recall minimal tidak turun dari 0.60.
- Confusion matrix tidak collapse ke kelas `phone_use`.
- Loss gap tetap sehat dan tidak membesar ekstrem dari 14A revisi.
- Fusion Macro F1 minimal menyamai atau melampaui 14A revisi (0.78059).

Command yang dijalankan manual/terekam:

```powershell
# Training front Experiment 14B saja
python -m src.train --view front --experiment experiment_14B

# Evaluasi front Experiment 14B
python -m src.evaluate --view front --checkpoint checkpoints\front_best_exp14B.pt

# Fusion memakai front 14B dan side Experiment 13
python -m src.fusion --front-checkpoint checkpoints\front_best_exp14B.pt --side-checkpoint checkpoints\side_best_exp13_backup.pt
```

Output utama:
- `checkpoints/front_best_exp14B.pt`
- `results/front_history_exp14B.json`
- `results/experiment_14B_front_metrics.json`
- `results/fusion_comparison_exp14B.json`

Catatan: side view tetap memakai checkpoint Experiment 13 sebagai kontrol stabil
dan tidak dilatih ulang.

### Hasil Experiment 14B Revisi dan Keputusan Branch

Experiment 14B revisi dijalankan untuk menguji apakah peningkatan weight decay
ringan dapat memperbaiki generalisasi front view. Konfigurasi revisi
mengembalikan dropout ke 0.40 dan menaikkan weight decay dari 3e-4 menjadi
5e-4. Experiment 14A revisi tetap menjadi baseline stabil pembanding pada branch
ini.

Hasil 14B awal tidak dipertahankan sebagai kandidat karena:
- Front Macro F1 turun dari 0.76626 (14A revisi) menjadi 0.74552.
- Safe recall turun dari 0.60 menjadi 0.55.
- Loss gap best epoch naik dari 0.04383 menjadi 0.08135.
- Fusion Macro F1 turun dari 0.78059 menjadi 0.71285.

Validitas checkpoint 14B revisi dicek dengan SHA256 dan berbeda dari 14A:

```text
front_best_exp14A.pt
E2AF60A1F00C906E3F239484E361CF9BAB19215B67B6ABB7B1260FE5EC38979E

front_best_exp14B.pt
9BAF96C8AA7857585B2965DEAF27AB22BEA3796717C947FA923562797CC92FCF
```

Walaupun checkpoint 14B revisi berbeda dari checkpoint 14A, trajectory/history
dan metrik evaluasi yang tercatat identik dengan Experiment 14A revisi. Metrics
JSON hanya berbeda pada metadata checkpoint dan nilai `weight_decay`.

Ringkasan 14B revisi:
- Best epoch: 9.
- Validation Macro F1: 0.72581.
- Train-validation loss gap best epoch: 0.04383.
- Test Macro F1 front: 0.76626.
- Safe recall: 0.60 (24/40).
- Average fusion Macro F1: 0.78059.
- Adaptive fusion Macro F1: 0.78059.
- Confusion matrix front: `[[24, 16], [14, 166]]`.
- Confusion matrix fusion: `[[20, 20], [4, 176]]`.

Kesimpulan branch `experiment_14B`: 14B revisi valid sebagai ablasi, tetapi
tidak memberi peningkatan terukur dibandingkan 14A revisi. Kandidat stabil tetap
Experiment 14A revisi, bukan 14B.
