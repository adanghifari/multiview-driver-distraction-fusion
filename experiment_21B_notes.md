# Experiment 21B Notes

## Tujuan

Experiment 21B adalah lanjutan dari Experiment 21A yang tetap berada dalam batas proposal. Fokusnya adalah stabilisasi model front view, bukan menambah strategi fusion baru.

## Alasan

Experiment 21A menunjukkan:

- Side view membaik dibanding final_v1 dan status generalisasinya `OK`.
- Front view sedikit membaik dibanding final_v1, tetapi statusnya masih `BORDERLINE`.
- Fusion average/adaptive belum mengungguli final_v1 karena trade-off error masih belum ideal.

Karena proposal membatasi strategi fusion pada average fusion dan adaptive fusion, Experiment 21B tidak memakai weighted fusion manual. Perbaikan difokuskan pada model single-view front.

## Hipotesis

Mengembalikan freeze depth front dari 4 ke 5 dapat mengurangi train-validation gap pada protokol stride 20, sambil mempertahankan peningkatan front view dari Experiment 21A.

## Protokol

- Dataset: 3MDAD.
- Task: binary `safe_driving` vs `phone_use`.
- Split: subject-based split yang sama.
- Backbone: EfficientNetV2-S.
- Frame stride: 20.
- Decision threshold: 0.50.
- Side checkpoint: dikunci dari Experiment 21A.
- Fusion: average fusion dan adaptive fusion saja.
- Output:
  - `results/experiment_21B/`
  - `checkpoints/experiment_21B/`

## Konfigurasi

| Komponen | Nilai |
|---|---:|
| Front LR | 3e-5 |
| Front WD | 5e-4 |
| Front dropout | 0.4 |
| Front label smoothing | 0.0 |
| Front freeze stage | 5 |
| Front patience | 4 |
| Side | Locked dari `checkpoints/experiment_21/side_best_exp21.pt` |

## Command

Jalankan dari root repo:

```powershell
.venv\Scripts\python.exe -m src.experiment_21B.run_experiment_21B
```

Jika front 21B sudah selesai dilatih dan hanya ingin re-run evaluasi/summary:

```powershell
.venv\Scripts\python.exe -m src.experiment_21B.run_experiment_21B --skip-train
```

Setelah script Python selesai, buka dan run:

```text
notebooks/experiment_21B.ipynb
```

## Kriteria Keputusan

Experiment 21B dianggap lebih baik dari 21A jika:

- Front status turun dari `BORDERLINE` menjadi `OK`, atau train-validation gap mengecil jelas.
- Front Macro F1 tidak turun besar dari 21A front 0.77284.
- Average/adaptive fusion naik dari 21A fusion 0.76597.
- Hasil fusion mendekati atau melampaui final_v1 fusion 0.78059.

Experiment 21B dapat dipertimbangkan mengganti final_v1 hanya jika fusion Macro F1 melampaui 0.78059 dengan interpretasi error yang masuk akal.

## Hasil Aktual

Artefak hasil tersedia pada:

- `results/experiment_21B/experiment_21B_summary.json`
- `results/experiment_21B/experiment_21B_summary.md`
- `results/experiment_21B/experiment_21B_table.csv`
- `results/experiment_21B/single_view_predictions_exp21B.csv`
- `results/experiment_21B/fusion_predictions_exp21B.csv`

Ringkasan hasil:

| Metode | Accuracy | Precision Macro | Recall Macro | Macro F1 | safe->phone | phone->safe | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Front single-view 21B | 0.86335 | 0.77094 | 0.81117 | 0.78797 | 16 | 28 | OK |
| Side locked Exp21A | 0.82919 | 0.70960 | 0.63907 | 0.66044 | 39 | 16 | OK |
| Average fusion | 0.86335 | 0.77556 | 0.74544 | 0.75880 | 26 | 18 | - |
| Adaptive fusion | 0.86335 | 0.77556 | 0.74544 | 0.75880 | 26 | 18 | - |

Training diagnostics:

- Front best epoch 6, validation Macro F1 0.71443, train-validation loss gap 0.06605, sehingga status front membaik dari `BORDERLINE` pada Exp 21A menjadi `OK`.
- Side tetap dikunci dari Exp 21A dengan status `OK`.

Interpretasi sementara:

- Stabilisasi front berhasil: Macro F1 front naik dari 0.77284 pada Exp 21A menjadi 0.78797 pada Exp 21B.
- Front 21B juga melampaui final_v1 front reference 0.76626.
- Namun average/adaptive fusion turun dari 0.76597 pada Exp 21A menjadi 0.75880 pada Exp 21B.
- Fusion tetap mengurangi `phone_use -> safe_driving` error dari 28 pada front menjadi 18, tetapi menaikkan `safe_driving -> phone_use` dari 16 menjadi 26. Trade-off ini membuat Macro F1 fusion lebih rendah dibanding front single-view.
- Adaptive fusion masih identik dengan average fusion pada prediksi akhir.

Keputusan sementara:

- Exp 21B berhasil memperbaiki front, tetapi belum memperbaiki fusion.
- Karena proposal menekankan perbandingan single-view terbaik vs average/adaptive fusion, hasil ini penting: pada protokol stride 20, single-view front 21B menjadi kandidat paling kuat, sedangkan multi-view fusion belum memberi peningkatan.
- Exp 21B belum mengganti final_v1 sebagai hasil fusion utama, tetapi dapat dijadikan bukti bahwa perbaikan front berhasil secara valid.
