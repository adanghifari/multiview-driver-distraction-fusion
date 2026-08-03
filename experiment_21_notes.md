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

## Hasil Aktual

Artefak hasil tersedia pada:

- `results/experiment_21/experiment_21_summary.json`
- `results/experiment_21/experiment_21_summary.md`
- `results/experiment_21/experiment_21_table.csv`
- `results/experiment_21/single_view_predictions_exp21.csv`
- `results/experiment_21/fusion_predictions_exp21.csv`

Ringkasan hasil:

| Metode | Accuracy | Precision Macro | Recall Macro | Macro F1 | safe->phone | phone->safe | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Front single-view | 0.84783 | 0.75131 | 0.80824 | 0.77284 | 15 | 34 | BORDERLINE |
| Side single-view | 0.82919 | 0.70960 | 0.63907 | 0.66044 | 39 | 16 | OK |
| Average fusion | 0.86646 | 0.78063 | 0.75392 | 0.76597 | 25 | 18 | - |
| Adaptive fusion | 0.86646 | 0.78063 | 0.75392 | 0.76597 | 25 | 18 | - |

Training diagnostics:

- Front best epoch 6, validation Macro F1 0.69548, train-validation loss gap 0.09876, sehingga dikategorikan `BORDERLINE`.
- Side best epoch 21, validation Macro F1 0.74767, train-validation loss gap 0.05569, sehingga dikategorikan `OK`.

Interpretasi sementara:

- Exp 21 meningkatkan front single-view sedikit dibanding final_v1 reference front Macro F1 0.76626 menjadi 0.77284.
- Fusion Exp 21 belum mengungguli final_v1 reference fusion Macro F1 0.78059; average/adaptive fusion hanya mencapai 0.76597.
- Fusion mengurangi `phone_use -> safe_driving` error dari 34 pada front menjadi 18, tetapi menaikkan `safe_driving -> phone_use` dari 15 menjadi 25. Trade-off ini membuat Macro F1 fusion turun dibanding front single-view.
- Adaptive fusion tetap identik dengan average fusion pada prediksi akhir.

Keputusan sementara:

- Exp 21A tidak mengganti final_v1.
- Konfigurasi ini tetap berguna karena menunjukkan stride 20 dan freeze lebih fleksibel tidak cukup untuk memperbaiki fusion baseline.
- Varian lanjutan yang masuk akal adalah menguji konfigurasi yang mempertahankan front improvement tetapi mengurangi dampak side terhadap false alarm `safe_driving`.
