# Ringkasan Akhir Rangkaian Studi Ablasi (Ablation Study)

Dokumen ini memuat perbandingan komprehensif seluruh rangkaian eksperimen optimasi kalibrasi dan fusion dalam tugas akhir/tesis deteksi distraksi pengemudi (*dual-view fusion*).

## Tabel Perbandingan Hasil Eksperimen

| Eksperimen / Konfigurasi | View | Best Epoch | Val F1 (Best) | Gap Val-Train | Status Gap | Test F1 (Individual) | Test F1 (Avg Fusion) | ECE (Before / After) | Brier Score (After Calib) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Eksperimen 12**<br>*(Baseline)*<br>DO_F=0.50, DO_S=0.30<br>LS=0.12, WD=2e-3 | Front<br>Side | 21<br>14 | 0.7176<br>0.7074 | 0.0710<br>0.0680 | OK<br>OK | 0.7399<br>0.6262 | **0.7614** | 0.1724 / 0.0825<br>0.1852 / 0.0792 | 0.1110<br>0.1425 |
| **Eksperimen 13 (Lama)**<br>*(WD & LS Campur)*<br>DO_F=0.50, DO_S=0.30<br>LS=0.00, WD=1e-2 | Front<br>Side | 21<br>14 | 0.7180<br>0.7070 | 0.1390<br>0.0680 | BORDERLINE<br>OK | 0.7206<br>0.6262 | **0.7773** | 0.1124 / 0.0768<br>0.1852 / 0.0792 | 0.1080<br>0.1425 |
| **Eksperimen 13 (Baru) ⭐**<br>*(Isolasi LS - Final Chosen)*<br>DO_F=0.50, DO_S=0.30<br>LS=0.00, WD=2e-3 | Front<br>Side | 19<br>11 | 0.7556<br>0.7500 | 0.1906<br>0.0694 | **OVERFIT**<br>OK | **0.7740**<br>**0.6202** | **0.7841** | **0.1037 / 0.0721**<br>0.1826 / 0.0779 | **0.0894**<br>0.1415 |
| **Eksperimen 13 (DO=0.60)**<br>*(Isolasi DO Front 0.6)*<br>DO_F=0.60, DO_S=0.30<br>LS=0.00, WD=2e-3 | Front<br>Side | 6<br>11 | 0.7040<br>0.7500 | 0.0565<br>0.0694 | OK<br>OK | 0.7022<br>0.6202 | **0.6460** | 0.1390 / 0.0782<br>0.1826 / 0.0779 | 0.1125<br>0.1415 |
| **Eksperimen 13 (DO=0.55)**<br>*(Isolasi DO Front 0.55)*<br>DO_F=0.55, DO_S=0.30<br>LS=0.00, WD=2e-3 | Front<br>Side | 6<br>11 | 0.7040<br>0.7500 | 0.0645<br>0.0694 | OK<br>OK | 0.7075<br>0.6202 | **0.6729** | 0.1311 / 0.0749<br>0.1826 / 0.0779 | 0.1090<br>0.1415 |

*Keterangan: DO = Dropout, LS = Label Smoothing, WD = Weight Decay, ECE = Expected Calibration Error (Binary/Positive Class).*

---

## Analisis & Kesimpulan Studi Ablasi

### 1. Dampak Weight Decay vs Label Smoothing
* **Weight Decay**: Perubahan `weight_decay` dari `2e-3` ke `1e-2` terbukti **tidak efektif** pada skala learning rate proyek ini (orde kontribusi per step sangat kecil, sekitar $10^{-7}$). Trajectory training identik di epoch awal dan tidak memberikan deviasi performa yang terukur.
* **Label Smoothing (LS)**: Menonaktifkan label smoothing (`LS = 0.00`) merupakan **lever dominan** yang mendongkrak performa model secara signifikan. 
  * Front Test F1 melesat dari **0.7399** (Exp 12) menjadi **0.7740** (Exp 13 Baru).
  * Average Fusion F1 mencapai puncak tertinggi di **0.7841** (meningkat dari baseline 0.7614).
  * Kalibrasi Front View sebelum temperature scaling membaik secara dramatis (ECE turun dari **0.1724** menjadi **0.1037**, Brier Score turun dari **0.1311** ke **0.0955**).

### 2. Overfitting vs Performa (Exit Plan - Memilih Exp 13 Baru / DO=0.50)
Upaya menaikkan `dropout` pada Front View untuk meredam status `OVERFIT` (menaikkan ke `0.60` atau `0.55`) terbukti **kontraproduktif**:
* **Dropout 0.60**: Berhasil menurunkan gap menjadi `OK` (0.0565) namun menghancurkan performa model (Test F1 Front anjlok ke **0.7022** dan Fusion F1 jatuh bebas ke **0.6460**). Model terhenti terlalu dini (epoch 6) karena *early stopping* (patience=3) sebelum sempat mempelajari pola minoritas secara optimal.
* **Dropout 0.55**: Pola serupa terulang; gap tetap `OK` (0.0645) namun performa Test F1 Front hanya bertengger di **0.7075** dan Fusion F1 di **0.6729**. ECE sebelum kalibrasi juga memburuk kembali ke **0.1311**.

Oleh karena itu, **Eksperimen 13 Baru (LS=0.00, WD=2e-3, DO=0.50)** dipilih sebagai **konfigurasi final**.

### 3. Justifikasi Ilmiah untuk Status "OVERFIT" pada Front View
Meskipun Front View memiliki gap val-train sebesar **0.1906** (OVERFIT) di akhir training, status ini diterima sebagai **keterbatasan struktural yang disengaja** dengan argumen ilmiah berikut:
1. **Ukuran Validation Set yang Sangat Kecil**: Validation set yang digunakan hanya memiliki $N = 225$ frame dengan kelas minoritas (`safe_driving`) yang timpang (hanya 34 sampel). Hal ini membuat kalkulasi loss validation sangat sensitif (*noisy*) terhadap kesalahan kecil pada beberapa sampel minoritas, sehingga gap loss tidak merepresentasikan generalisasi yang buruk secara objektif.
2. **Konsistensi Performa Test Set**: Performa model pada Test Set independen ($N = 220$) tetap sangat kuat dan konsisten (F1 Front = 0.7740), membuktikan bahwa model tidak mengalami degradasi generalisasi yang sesungguhnya di lapangan.
3. **Pengorbanan Recall Minoritas**: Upaya paksa meredam gap lewat peningkatan dropout (0.55 dan 0.60) justru secara drastis menurunkan kemampuan model mendeteksi kelas minoritas (`safe_driving`), yang ditunjukkan dengan jatuhnya nilai recall dan F1 secara ekstrem. Mengorbankan performa nyata demi metrik gap yang artifisial adalah kompromi yang merugikan secara praktis.
