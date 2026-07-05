# Konteks Proyek: Deteksi Penggunaan Ponsel Pengemudi (Multi-View, Decision-Level Fusion)

Ini adalah proyek Tugas Akhir S1 Informatika (Telkom University). Dokumen ini
merangkum konteks penelitian untuk AI coding assistant yang membantu di repo ini.
**Baca juga `[REVISI]Proposal Tugas Akhir.pdf` di root folder** untuk detail
lengkap tinjauan pustaka dan justifikasi metodologis — file ini hanya ringkasan
teknis + status progres implementasi.

## Tujuan Penelitian
Membandingkan performa deteksi *phone-use distracted driving* vs *safe driving*
menggunakan tiga skenario:
1. Model single-view (front saja, side saja) berbasis EfficientNetV2-S.
2. Multi-view dengan **average fusion** (bobot skor 50:50).
3. Multi-view dengan **adaptive fusion** (bobot dinamis berbasis confidence,
   softmax atas jarak skor ke threshold 0.5 — lihat Persamaan 3.2–3.4 proposal).

Evaluasi memakai akurasi, presisi, recall, dan **Macro F1-Score** (metrik utama,
karena distribusi kelas tidak seimbang: ~18% safe_driving vs ~82% phone_use).

## Dataset
- **3MDAD** (Jegham dkk., 2020), subset siang hari, modalitas RGB.
- Dua sudut pandang: `RGB1` = **side view**, `RGB2` = **front view** (dikonfirmasi
  manual, JANGAN dibalik).
- Klasifikasi biner: kelas asli **AC1** → `safe_driving` (label 0), **AC5–AC9**
  digabung → `phone_use` (label 1). Kelas AC2–4, AC10–16 tidak dipakai.
- **Known data issue (sudah diperbaiki di kode, bukan di file mentah):**
  Subjek S31 pada side view (RGB1) memiliki folder `AC9` dan `AC10` yang
  tertukar isinya (bug dataset yang juga dilaporkan komunitas untuk modalitas
  Depth di repo `kevinsu628/3MDAD`). Dikoreksi via `KNOWN_ACTIVITY_CORRECTIONS`
  di `src/config.py` — koreksi diterapkan di level metadata manifest, folder
  mentah di `data/raw/` TIDAK diubah.
- Setelah koreksi: 41.595 pasang frame tersinkronisasi (front-side match 100%,
  0 orphan). Proposal menyebut 41.574 (dari angka resmi paper) — selisih 21
  frame (~0.05%) didokumentasikan sebagai temuan minor, bukan bug pipeline.

## Struktur Folder
```
Experiment/
├── data/
│   ├── raw/{RGB1,RGB2}/S{subject}/AC{activity}/{view}_SUB{s}ACT{a}F{frame}.jpg  <- immutable, JANGAN diedit
│   ├── manifest_full.csv      # semua 16 kelas, kedua view, sudah dikoreksi metadata S31
│   ├── manifest_binary.csv    # hanya AC1 + AC5-9, kedua view
│   ├── manifest_paired.csv    # satu baris per pasangan frame (filepath_front, filepath_side) — untuk evaluasi fusion
│   └── manifest_split.csv     # manifest_binary + kolom 'split' (train/val/test) — untuk training single-view
├── src/
│   ├── config.py              # SUMBER KEBENARAN TUNGGAL: semua path, mapping view/label, hyperparameter, known corrections
│   ├── build_manifest.py       # scan data/raw -> manifest_full.csv & manifest_binary.csv (terapkan koreksi S31)
│   ├── check_view_pairing.py   # cek frame orphan antar front-side (debug tool)
│   ├── build_split.py          # subject-based split 35/7/8 -> manifest_paired.csv & manifest_split.csv
│   └── dataset.py              # PyTorch Dataset + DataLoader per-view (front/side), transform sesuai Subbab 3.2.3
├── notebooks/                  # untuk eksperimen interaktif (training, evaluasi, visualisasi)
├── checkpoints/                # tempat simpan bobot model terlatih
├── results/                    # tempat simpan metrik evaluasi, confusion matrix, dll.
└── [REVISI]Proposal Tugas Akhir.pdf
```

## Keputusan Desain Kunci (jangan diubah tanpa alasan kuat)
- Data mentah di `data/raw/` bersifat **immutable** — semua koreksi/filtering
  dilakukan di level manifest (CSV), bukan rename/edit file gambar.
- Subject-based split (bukan random per-frame) untuk mencegah data leakage
  antar partisi — split 35/7/8 subjek, seed=42, didefinisikan di `config.py`.
- Semua path, mapping label, dan hyperparameter didefinisikan **sekali** di
  `src/config.py`. Script lain mengimpor dari sana, tidak hardcode ulang.
- Backbone: **EfficientNetV2-S** (via `timm`), bukan varian M/L — sesuai
  batasan masalah proposal (tidak membandingkan arsitektur).
- Hyperparameter (Subbab 3.3, Bab 3): Adam lr=1e-4, dropout=0.3, batch=32,
  image size 224x224, normalisasi ImageNet mean/std, early stopping
  patience=5, cross-entropy loss, threshold keputusan 0.5.

## Status Progres (per hari ini)
✅ Data collection & manifest building (dengan koreksi bug S31)
✅ Subject-based split (train/val/test)
✅ PyTorch Dataset & DataLoader per-view
⬜ Model single-view EfficientNetV2-S (arsitektur + classifier head)
⬜ Training loop (fine-tuning, early stopping)
⬜ Evaluasi single-view (metrik + confusion matrix + analisis pola kegagalan)
⬜ Decision-level fusion (average & adaptive) menggunakan manifest_paired.csv
⬜ Evaluasi komparatif akhir (single-view terbaik vs average vs adaptive fusion)

## Environment
- Python venv (`.venv`), PyTorch 2.11+cu128 (GPU), torchvision, timm.
- Lihat `requirements.txt` di root untuk daftar lengkap.

## Preferensi Kerja
- Verifikasi klaim/angka terhadap sumber (proposal PDF, dataset asli) sebelum
  dikonfirmasi — jangan asumsikan tanpa cek.
- Bahasa hipotesis untuk hasil yang belum terbukti empiris ("berpotensi",
  bukan "akan" atau "terbukti").
- Perubahan kode sebaiknya inkremental dan bisa dijalankan ulang
  (reproducible), bukan one-off manual fix.
