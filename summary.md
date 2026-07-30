# Summary Eksperimen

Dokumen ini merangkum rangkaian eksperimen proyek deteksi penggunaan ponsel pengemudi berbasis dua sudut pandang, yaitu front view dan side view. Ringkasan ini disusun dari notebook eksperimen, catatan eksperimen, serta artefak hasil di folder `results/`.

## Konteks Penelitian

Penelitian menggunakan dataset 3MDAD untuk klasifikasi biner:

- `safe_driving`
- `phone_use`

Kelas `phone_use` berasal dari gabungan aktivitas AC5 sampai AC9, sedangkan `safe_driving` berasal dari AC1. Evaluasi utama memakai Macro F1 karena distribusi kelas tidak seimbang dan penelitian perlu menilai performa kedua kelas secara lebih adil.

Model single-view memakai EfficientNetV2-S. Dua model dilatih secara terpisah untuk front view dan side view, lalu hasil probabilitas keduanya digabung pada level keputusan menggunakan:

- Average fusion 50:50
- Adaptive fusion berbasis confidence

## Konfigurasi Dasar Pipeline

Konfigurasi umum pipeline, dengan beberapa variasi eksploratif per eksperimen:

| Komponen | Nilai |
|---|---|
| Dataset | 3MDAD |
| Task | Binary classification |
| View | Front dan side |
| Backbone | EfficientNetV2-S |
| Input size | 224 x 224 |
| Batch size | 32 |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |
| Early stopping | Berdasarkan validation Macro F1 |
| Decision threshold | 0.50 |
| Metrik utama | Macro F1 |
| Class weight dasar fase ablasi/final | Umumnya `[2.5, 1.0]` untuk `safe_driving` dan `phone_use`; beberapa eksperimen awal/gamma sweep memakai konfigurasi berbeda |

Catatan view: `RGB1` adalah side view dan `RGB2` adalah front view.

## Ringkasan Kronologis Eksperimen

Catatan protokol: Exp 1-3 masih memakai artefak awal dengan support sekitar 6003 sampel. Exp 4-7 memakai artefak dengan support sekitar 1215 sampel. Exp 8 memakai artefak eksploratif dengan support 419 sampel. Mulai Exp 9 sampai rangkaian final, evaluasi test utama memakai support 220 sampel. Karena itu, angka antar fase awal tidak selalu dapat dibandingkan langsung tanpa melihat perubahan protokol data.

| Eksperimen | Tujuan / Fokus | Konfigurasi Utama | Front F1 | Side F1 | Avg Fusion F1 | Adaptive F1 | Keputusan / Catatan |
|---|---|---|---:|---:|---:|---:|---|
| Exp 1 | Baseline awal pipeline | LR 1e-4; dropout 0.3; patience 5; belum ada freeze/WD khusus | 0.8225 | 0.8370 | 0.8410 | 0.8410 | Average/adaptive fusion sedikit unggul dari single-view. |
| Exp 2 | Perbaikan pipeline awal | LR 1e-4; dropout 0.3; WD 1e-4; freeze 4; scheduler aktif | 0.8093 | 0.7618 | 0.7920 | 0.7920 | Performa turun dibanding Exp 1 pada artefak branch GitHub. |
| Exp 3 | Perbaikan LR dan freeze awal | LR 5e-5; dropout 0.3; WD 1e-4; freeze 2; scheduler aktif | 0.8212 | 0.8344 | 0.8392 | 0.8392 | Mirip Exp 1, tidak memberi peningkatan besar. |
| Exp 4 | Regularisasi dan subsampling v4 | LR 5e-5; dropout 0.5; WD 5e-4; freeze 4; frame stride 5 | 0.8825 | 0.8032 | 0.8443 | 0.8443 | Front single-view menjadi sangat kuat; fusion tidak melampaui front. |
| Exp 5 | LR rendah dengan regularisasi v4 | LR 5e-6; dropout 0.5; WD 5e-4; freeze 4; frame stride 5 | 0.8375 | 0.7381 | 0.7990 | 0.7990 | LR rendah membuat training lebih stabil, tetapi fusion turun. |
| Exp 6 | Config v5 awal / regularisasi lebih kuat | LR 5e-6; dropout 0.6; WD 1e-3; freeze 5; frame stride 5 | 0.8293 | 0.7326 | 0.8287 | 0.8287 | Front dan fusion hampir sama; artefak branch lebih konsisten daripada output cell notebook lama. |
| Exp 7 | Optimasi side view | LR 5e-6; dropout 0.6; WD 1e-2; front freeze 5; side freeze 3; label smoothing 0.1; class weight `[5.15, 1.0]` | 0.8074 | 0.7301 | 0.8328 | 0.8328 | Fusion mulai lebih baik dari single-view pada konfigurasi ini. |
| Exp 8 | Golden Balance | Front LR 5e-5; side LR 2e-5; dropout 0.5; WD 5e-4; front freeze 4; side freeze 3; LS 0.1; class weight `[2.5, 1.0]`; stride 15 | 0.8631 | 0.7712 | 0.8749 | 0.8749 | Hasil eksploratif awal tertinggi, tetapi memakai support 419. |
| Exp 9 | Extreme temporal subsampling | Konfigurasi Exp 8 dengan frame stride dinaikkan menjadi 30 untuk memangkas redundansi temporal | 0.7850 | 0.7508 | 0.8044 | 0.8044 | Performa turun dibanding Exp 8, tetapi fusion tetap unggul. |
| Exp 10 | Penyeimbangan front view dan reduksi ECE | Front LR 3e-5; front dropout 0.6; side dropout 0.5; WD 5e-3; front freeze 5; side freeze 4; LS 0.12; stride 30 | 0.5999 | 0.6333 | 0.6433 | 0.6433 | Performa turun besar; konfigurasi terlalu konservatif untuk front. |
| Exp 11 | Optimasi regularization dan patience front | Front dropout 0.5; side dropout 0.5; WD 2e-3; patience front 4; LS 0.12; stride 30 | 0.7401 | 0.6333 | 0.7669 | 0.7669 | Front membaik dibanding Exp 10, fusion naik tetapi belum menjadi baseline final. |
| Exp 12 | Baseline ablation / side view rescue | Front dropout 0.5, WD 2e-3, patience 3; side dropout 0.3, patience 7; LS 0.12; stride 30 | 0.7399 | 0.6262 | 0.7614 | 0.7614 | Dipakai sebagai baseline ablasi. |
| Exp 13 Lama | Weight decay dan label smoothing campur | Dropout front 0.5; dropout side 0.3; label smoothing 0.0; weight decay 1e-2 | 0.7206 | 0.6262 | 0.7773 | - | Weight decay tinggi tidak memberi perbaikan jelas. |
| Exp 13 Baru | Isolasi label smoothing | Dropout front 0.5; dropout side 0.3; label smoothing 0.0; weight decay 2e-3 | 0.7740 | 0.6202 | 0.7841 | - | Menonaktifkan label smoothing menjadi faktor dominan peningkatan. |
| Exp 13 DO=0.60 | Uji dropout front lebih tinggi | Dropout front 0.60; side dikunci dari Exp 13 Baru | 0.7022 | 0.6202 | 0.6460 | - | Gap membaik, tetapi performa turun drastis. |
| Exp 13 DO=0.55 | Uji dropout front sedang | Dropout front 0.55; side dikunci dari Exp 13 Baru | 0.7075 | 0.6202 | 0.6729 | - | Regularisasi masih terlalu menekan performa. |
| Exp 14 / 14A branch | Stabilisasi front awal pada branch GitHub | Front hasil branch `experiment_14`; side tetap kontrol dari Exp 13 | 0.7556 | 0.6202 | 0.7447 | 0.7447 | Fusion turun dari Exp 13 Baru sehingga belum dipakai sebagai hasil final. |
| Exp 14A Awal | Stabilisasi front awal | Freeze stage 6; label smoothing 0.05; regularisasi lebih kuat | - | - | - | Gagal karena collapse ke kelas mayoritas `phone_use`. |
| Exp 14A Revisi | Stabilisasi front final | Front: lr 3e-5, WD 3e-4, dropout 0.4, label smoothing 0.0, freeze 5, patience 4. Side tetap kontrol dari Exp 13. | 0.7663 | 0.6202 | 0.7806 | 0.7806 | Konfigurasi stabil dan menjadi basis hasil final. |
| Exp 15 | Analisis error numerik | Tidak ada training ulang; memakai Front14A + Side13 | 0.7663 | 0.6202 | 0.7806 | 0.7806 | Menghitung pola error per-sample. |
| Exp 16 | Analisis visual failure case | Tidak ada training ulang; kurasi sampel dari Exp 15 | - | - | 0.7806 | 0.7806 | Menjelaskan kasus fusion memperbaiki atau merusak prediksi front. |
| Exp 17 | Adaptive sharpening | Uji alpha 1, 2, 3, 5, dan 10 tanpa training ulang | - | - | 0.7806 | 0.7806 | Semua alpha menghasilkan prediksi identik dengan average fusion. |
| Exp 18 | Threshold tuning | Threshold terbaik validation 0.49; diuji pada test | - | - | 0.7298 | 0.7298 | Threshold 0.49 menurunkan Macro F1; threshold final tetap 0.50. |
| Exp 19 Awal | Side class weighting awal | Revisi side dengan class weighting | - | 0.5863 | 0.7683 | - | Belum memperbaiki baseline. |
| Exp 19 Revisi 1 | Pemulihan baseline side | Side dikonfigurasi ulang agar mendekati Side13 | - | 0.6202 | 0.7806 | - | Kembali setara baseline final. |
| Exp 19 Full Balanced | Class weight full balanced | Side memakai class weight balanced penuh | - | 0.5577 | 0.7917 | - | Fusion tampak naik, tetapi side single-view turun. |
| Exp 19 Gamma Sweep | Mild class weighting | Gamma 0.25 sampai 0.98; test-best eksploratif gamma 0.95 | - | 0.5535 pada gamma 0.95 | 0.7932 pada gamma 0.95 | - | Tidak dijadikan final karena gamma dipilih berdasarkan test set. |
| Exp 19 Final Valid | Validation-based gamma selection | Gamma dipilih dari validation; gamma terpilih 0.50 | - | 0.6435 | 0.7799 | - | Belum mengungguli baseline final 0.7806. |
| Exp 20 | Eksplorasi formula adaptive fusion | Average fusion, legacy adaptive softmax, legacy ratio, margin ratio, entropy ratio, maxprob ratio, reliability static | - | - | 0.7806 | 0.7806 | Semua formula identik pada validation; average fusion dipilih dan hasil test tetap sama dengan baseline. |

## Ringkasan Ablasi Utama

| Konfigurasi | Front Config | Side Config | Front F1 | Side F1 | Fusion F1 | Status |
|---|---|---|---:|---:|---:|---|
| Exp 12 Baseline | DO 0.50, LS 0.12, WD 2e-3 | DO 0.30, LS 0.12, WD 2e-3 | 0.7399 | 0.6262 | 0.7614 | Baseline |
| Exp 13 Lama | DO 0.50, LS 0.00, WD 1e-2 | DO 0.30, LS 0.00, WD 1e-2 | 0.7206 | 0.6262 | 0.7773 | WD tinggi tidak dominan |
| Exp 13 Baru | DO 0.50, LS 0.00, WD 2e-3 | DO 0.30, LS 0.00, WD 2e-3 | 0.7740 | 0.6202 | 0.7841 | Ablasi lama terbaik |
| Exp 13 DO=0.60 | DO 0.60, LS 0.00, WD 2e-3 | Side dikunci | 0.7022 | 0.6202 | 0.6460 | Dropout terlalu kuat |
| Exp 13 DO=0.55 | DO 0.55, LS 0.00, WD 2e-3 | Side dikunci | 0.7075 | 0.6202 | 0.6729 | Dropout masih merugikan |
| Exp 14A Revisi | DO 0.40, LS 0.00, WD 3e-4, freeze 5, patience 4 | Side13 dikunci | 0.7663 | 0.6202 | 0.7806 | Final stabil |

Keterangan:

- DO = dropout
- LS = label smoothing
- WD = weight decay
- Fusion F1 adalah Macro F1 average fusion, kecuali jika disebut lain

## Hasil Final v1

Final v1 adalah reproduksi pipeline akhir yang dikunci. Pipeline ini bukan tuning baru, tetapi retrain/reproduksi hasil final agar artefak akhir berada di folder terpisah:

- `results/final_v1/`
- `checkpoints/final_v1/`

Konfigurasi final:

| View | Learning Rate | Weight Decay | Dropout | Label Smoothing | Freeze Stage | Patience | Class Weight |
|---|---:|---:|---:|---:|---:|---:|---|
| Front | 3e-5 | 3e-4 | 0.4 | 0.0 | 5 | 4 | `[2.5, 1.0]` |
| Side | 2e-5 | 2e-3 | 0.3 | 0.0 | 4 | 7 | `[2.5, 1.0]` |

Hasil final:

| Metode | Accuracy | Precision Macro | Recall Macro | Macro F1 | safe->phone | phone->safe | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Front single-view | 0.86364 | 0.77183 | 0.76111 | 0.76626 | 16 | 14 | 0.12513 | 0.11311 |
| Side single-view | 0.81364 | 0.66553 | 0.60417 | 0.62023 | 29 | 12 | 0.18263 | 0.16537 |
| Average fusion | 0.89091 | 0.86565 | 0.73889 | 0.78059 | 20 | 4 | - | - |
| Adaptive fusion | 0.89091 | 0.86565 | 0.73889 | 0.78059 | 20 | 4 | - | - |

Confusion matrix final:

| Metode | Confusion Matrix |
|---|---|
| Front single-view | `[[24, 16], [14, 166]]` |
| Side single-view | `[[11, 29], [12, 168]]` |
| Average fusion | `[[20, 20], [4, 176]]` |
| Adaptive fusion | `[[20, 20], [4, 176]]` |

Interpretasi final:

- Front view adalah single-view terbaik.
- Side view lebih lemah jika berdiri sendiri, tetapi tetap memberi informasi komplementer.
- Average fusion meningkatkan Macro F1 dari 0.76626 menjadi 0.78059.
- Fusion menurunkan error `phone_use -> safe_driving` dari 14 menjadi 4.
- Fusion menaikkan error `safe_driving -> phone_use` dari 16 menjadi 20.
- Adaptive fusion tidak berbeda dari average fusion pada konfigurasi final.

## Ranking Konfigurasi Berdasarkan Macro F1 Fusion

| Rank | Eksperimen | Skenario | Macro F1 Fusion | Status Pemakaian |
|---:|---|---|---:|---|
| 1 | Exp 8 | Golden Balance | 0.8749 | Eksploratif awal |
| 2 | Exp 4 | Regularisasi + stride 5 | 0.8443 | Eksploratif awal; protokol lama |
| 3 | Exp 1 | Baseline awal | 0.8410 | Eksploratif awal; test set besar |
| 4 | Exp 3 | Perbaikan LR dan freeze awal | 0.8392 | Eksploratif awal; test set besar |
| 5 | Exp 7 | Optimasi side view | 0.8328 | Eksploratif awal |
| 6 | Exp 6 | Config v5 awal | 0.8287 | Eksploratif awal |
| 7 | Exp 9 | Extreme temporal subsampling | 0.8044 | Eksploratif awal |
| 8 | Exp 5 | LR rendah + regularisasi v4 | 0.7990 | Eksploratif awal |
| 9 | Exp 2 | Perbaikan pipeline awal | 0.7920 | Eksploratif awal; test set besar |
| 10 | Exp 13 Baru | Isolasi label smoothing | 0.7841 | Ablasi lama terbaik |
| 11 | Exp 14A Revisi / Final v1 | Stabilisasi front | 0.7806 | Hasil final stabil |
| 12 | Exp 19 Final Valid | Gamma 0.50 validation-selected | 0.7799 | Ablasi class weighting, tidak mengganti final |
| 13 | Exp 13 Lama | WD dan LS campur | 0.7773 | Ablasi |
| 14 | Exp 11 | Optimasi front regularization | 0.7669 | Eksplorasi |
| 15 | Exp 12 | Baseline ablation | 0.7614 | Baseline ablasi |
| 16 | Exp 14 / 14A branch | Stabilisasi front awal pada branch GitHub | 0.7447 | Tidak dipakai |
| 17 | Exp 13 DO=0.55 | Dropout front 0.55 | 0.6729 | Tidak dipakai |
| 18 | Exp 13 DO=0.60 | Dropout front 0.60 | 0.6460 | Tidak dipakai |
| 19 | Exp 10 | Reduksi ECE | 0.6433 | Tidak dipakai |

Catatan penting: walaupun Exp 8 memiliki Macro F1 fusion tertinggi secara angka, hasil final penelitian tetap menggunakan konfigurasi final v1 / Exp 14A Revisi karena rangkaian ablasi akhir menekankan stabilitas, validitas protokol, dan reproduksibilitas artefak final. Ranking ini berguna untuk melihat riwayat eksperimen, tetapi fase Exp 1-8 memakai protokol evaluasi yang berbeda dari fase final.

## Kesimpulan Eksperimen

Rangkaian eksperimen menunjukkan bahwa perubahan regularisasi dan strategi fusion memiliki dampak besar terhadap Macro F1. Pada fase eksploratif awal, Exp 8 menghasilkan angka fusion tertinggi, tetapi eksperimen berikutnya menunjukkan perlunya konfigurasi yang lebih stabil dan dapat dipertanggungjawabkan.

Pada rangkaian ablasi akhir, menonaktifkan label smoothing menjadi perubahan yang paling berpengaruh. Sebaliknya, menaikkan dropout front untuk mengecilkan gap train-validation justru menurunkan performa test secara signifikan. Experiment 14A Revisi kemudian dipilih sebagai konfigurasi final karena memberi kompromi terbaik antara performa, stabilitas generalisasi, dan reproduksibilitas.

Hasil final menunjukkan bahwa average fusion adalah metode utama yang paling sederhana dan stabil. Adaptive fusion belum terbukti lebih baik karena menghasilkan prediksi dan metrik yang identik dengan average fusion. Eksperimen lanjutan pada threshold tuning, side class weighting, dan variasi formula adaptive fusion juga belum berhasil mengungguli baseline final secara valid.

Dengan demikian, hasil utama penelitian adalah konfigurasi Front14A Revisi + Side13 dengan average fusion, Macro F1 0.78059, accuracy 0.89091, precision macro 0.86565, dan recall macro 0.73889.

## Audit Kesesuaian Branch

Audit ini menjelaskan cabang Git yang dipakai sebagai rujukan angka utama pada ringkasan, sekaligus membatasi kasus yang tidak sepenuhnya berasal dari satu branch eksperimen.

| Rentang / Eksperimen | Branch / Artefak Rujukan | Status Kesesuaian |
|---|---|---|
| Exp 1 | `origin/experiment_1:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 2 | `origin/experiment_2:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 3 | `origin/experiment_3:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 4 | `origin/experiment_4:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 5 | `origin/experiment_5:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 6 | `origin/experiment_6:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 7 | `origin/experiment_7:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 8 | `origin/experiment_8:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 9 | `origin/experiment_9:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 10 | `origin/experiment_10:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 11 | `origin/experiment_11:results/fusion_comparison.json` | Sesuai dengan tabel kronologis. |
| Exp 12 | `origin/experiment_12:results/fusion_comparison.json` dan `results/ringkasan_akhir_ablasi.md` | Sesuai; Exp 12 menjadi baseline ablasi. |
| Exp 13 Lama / Baru / DO sweep | Artefak ablasi di `results/ringkasan_akhir_ablasi.md`, `results/fusion_comparison_ls_isolated.json`, notebook Exp 13, dan catatan run | Tidak memakai `origin/experiment_13:results/fusion_comparison.json` sebagai sumber utama karena branch tersebut berisi metrik lama yang sama dengan baseline awal, bukan paket ablasi akhir. |
| Exp 14 / 14A branch | `origin/experiment_14:results/fusion_comparison.json` dan `origin/experiment_14B:results/fusion_comparison.json` | Sesuai dengan baris Exp 14 / 14A branch. |
| Exp 14A Revisi | `origin/experiment_20_adaptive_formula:results/fusion_comparison_exp14A.json` dan `results/experiment_14A_front_metrics.json` | Sesuai; menjadi basis hasil final stabil. |
| Exp 15 | `results/error_summary_exp15.json` dan `experiment_15_notes.md` | Sesuai; analisis tanpa training ulang. |
| Exp 16 | `results/exp16_selected_failure_cases.csv`, folder `results/exp16_failure_case_images/`, dan `experiment_16_notes.md` | Sesuai; analisis visual tanpa training ulang. |
| Exp 17 | `results/adaptive/adaptive_sharpening_exp17B.json` dan `experiment_17_notes.md` | Sesuai; hasil adaptive tetap identik dengan average fusion. |
| Exp 18 | `results/threshold_test_exp18.json` dan `experiment_18_notes.md` | Sesuai; threshold 0.49 menurunkan hasil test. |
| Exp 19 | `origin/experiment_19` / `origin/experiment_20_adaptive_formula` pada artefak `results/fusion_comparison_exp19*.json`, `results/experiment_19_mild_class_weight_summary.json`, dan `results/gamma_selection_exp19_summary.json` | Sesuai; gamma test-best dicatat sebagai eksploratif, sedangkan gamma final valid dipilih dari validation. |
| Exp 20 | `origin/experiment_20_adaptive_formula:results/adaptive_formula_exp20_summary.json` dan `results/adaptive_formula_exp20_selected_test.json` | Sesuai; formula terpilih tetap average fusion. |
| Final v1 | `origin/final_v1_retrain:results/final_v1/final_v1_summary.json` | Sesuai; reproduksi artefak final di folder `results/final_v1/`. |

Catatan branch: file `summary.md` saat audit ini hanya ada pada branch `experiment_20_threshold_analysis`. Branch eksperimen lama tetap diperlakukan sebagai sumber artefak historis, bukan tempat penyimpanan salinan ringkasan final.

## Sumber Artefak Utama

- `main:results/fusion_comparison.json` atau `origin/experiment_1:results/fusion_comparison.json` untuk Exp 1
- `origin/experiment_2:results/fusion_comparison.json` untuk Exp 2
- `origin/experiment_3:results/fusion_comparison.json` untuk Exp 3
- `origin/experiment_4:results/fusion_comparison.json` untuk Exp 4
- `origin/experiment_5:results/fusion_comparison.json` untuk Exp 5
- `origin/experiment_6:results/fusion_comparison.json` untuk Exp 6
- `origin/experiment_7:results/fusion_comparison.json` untuk Exp 7
- `origin/experiment_8:results/fusion_comparison.json` untuk Exp 8
- `origin/experiment_9:results/fusion_comparison.json` untuk Exp 9
- `origin/experiment_10:results/fusion_comparison.json` untuk Exp 10
- `origin/experiment_11:results/fusion_comparison.json` untuk Exp 11
- `origin/experiment_12:results/fusion_comparison.json` untuk Exp 12
- `origin/experiment_14:results/fusion_comparison.json` dan `origin/experiment_14B:results/fusion_comparison.json` untuk Exp 14 / 14A branch
- `origin/experiment_20_adaptive_formula:results/fusion_comparison_exp14A.json` untuk Exp 14A Revisi
- `notebooks/experiment_6.ipynb` sampai `notebooks/experiment_13.ipynb`
- `experiment_notes.md`
- `experiment_15_notes.md`
- `experiment_16_notes.md`
- `experiment_17_notes.md`
- `experiment_18_notes.md`
- `experiment_19_notes.md`
- `experiment_20_adaptive_formula_notes.md`
- `results/ringkasan_akhir_ablasi.md`
- `results/final_experiment_summary.md`
- `results/final_v1/threshold_analysis/threshold_analysis_summary.md`
- `results/final_v1/final_v1_summary.md`
- `results/final_v1/final_v1_summary.json`
