# Experiment 21 Notes

## Tujuan

Experiment 21 menguji apakah sinyal kuat dari Experiment 8 dapat dibawa ke protokol final yang lebih disiplin. Eksperimen ini bukan memakai ulang angka Exp 8, tetapi melakukan retrain dengan konfigurasi Exp8-informed dan evaluasi yang terpisah.

## Hipotesis

Stride 20 dapat menjadi kompromi antara:

- Exp 8 stride 15, yang menghasilkan performa tinggi tetapi masih lebih banyak redundansi temporal.
- final_v1 stride 30, yang lebih ketat tetapi mungkin terlalu agresif membuang informasi frame.

Jika stride 20 mempertahankan informasi visual yang penting tanpa terlalu banyak redundansi, Macro F1 fusion berpotensi naik dibanding final_v1.

## Protokol

- Dataset: 3MDAD.
- Task: binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama dengan pipeline final.
- Backbone: EfficientNetV2-S.
- Input size: 224 x 224.
- Optimizer: AdamW.
- Scheduler: ReduceLROnPlateau.
- Best checkpoint: validation Macro F1.
- Decision threshold: 0.50.
- Frame stride: 20.
- Output:
  - `results/experiment_21/`
  - `checkpoints/experiment_21/`

## Konfigurasi

| View | LR | WD | Dropout | Label Smoothing | Freeze | Patience | Class Weight |
|---|---:|---:|---:|---:|---:|---:|---|
| Front | 3e-5 | 5e-4 | 0.4 | 0.0 | 4 | 4 | `[2.5, 1.0]` |
| Side | 2e-5 | 5e-4 | 0.3 | 0.0 | 3 | 7 | `[2.5, 1.0]` |

## Command

Jalankan dari root repo:

```powershell
.venv\Scripts\python.exe -m src.experiment_21.run_experiment_21
```

Jika training sudah selesai dan hanya ingin re-run evaluasi/summary:

```powershell
.venv\Scripts\python.exe -m src.experiment_21.run_experiment_21 --skip-train
```

Train salah satu view saja:

```powershell
.venv\Scripts\python.exe -m src.experiment_21.run_experiment_21 --view front
.venv\Scripts\python.exe -m src.experiment_21.run_experiment_21 --view side
```

Setelah script Python selesai, buka dan run:

```text
notebooks/experiment_21.ipynb
```

## Kriteria Keputusan

Experiment 21 dapat dipertimbangkan sebagai kandidat pengganti final_v1 jika:

- Fusion Macro F1 lebih tinggi dari final_v1 baseline 0.78059.
- Train-validation gap tidak masuk kategori overfit berat.
- Peningkatan tidak hanya berasal dari bias ke kelas mayoritas.
- Confusion matrix tidak memperburuk `safe_driving -> phone_use` secara tidak proporsional.
- Analisis tambahan menunjukkan hasilnya stabil dan dapat dijelaskan.

## Catatan Interpretasi

Perbandingan dengan Exp 8 dan final_v1 harus dibaca sebagai konteks karena stride dan support test berbeda. Klaim utama Experiment 21 berada pada artefak `results/experiment_21/`, bukan pada angka historis Exp 8.
