"""
Membangun manifest dataset dari struktur folder mentah:

    data/raw/{RGB1,RGB2}/S{subject_id}/AC{activity_id}/{view}_SUB{s}ACT{a}F{f}.jpg

Menghasilkan dua file:
    - data/manifest_full.csv    -> seluruh 16 kelas aktivitas, kedua view
    - data/manifest_binary.csv  -> hanya A1 (safe_driving) & A5-A9 (phone_use)

Jalankan dari root proyek:
    python -m src.build_manifest
"""

import re
import sys
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    MANIFEST_FULL_PATH,
    MANIFEST_BINARY_PATH,
    VIEW_FOLDER_MAP,
    ACTIVITY_LABEL_NAMES,
    KNOWN_ACTIVITY_CORRECTIONS,
    activity_id_to_binary_label,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Contoh nama file: RGB1_SUB1ACT1F1.jpg
FILENAME_PATTERN = re.compile(
    r"^(?P<view_folder>RGB\d+)_SUB(?P<subject_from_name>\d+)"
    r"ACT(?P<activity_from_name>\d+)F(?P<frame>\d+)\.jpe?g$",
    re.IGNORECASE,
)


def scan_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Scan seluruh data/raw/{RGB1,RGB2}/S*/AC*/*.jpg dan parse metadatanya."""
    rows = []
    n_skipped = 0

    for view_folder, view_name in VIEW_FOLDER_MAP.items():
        view_dir = raw_dir / view_folder
        if not view_dir.exists():
            log.warning("Folder %s tidak ditemukan, dilewati.", view_dir)
            continue

        subject_dirs = sorted(
            [d for d in view_dir.iterdir() if d.is_dir() and d.name.startswith("S")]
        )
        log.info("Scanning %s (%s view): %d folder subjek", view_folder, view_name, len(subject_dirs))

        for subject_dir in subject_dirs:
            subject_match = re.match(r"S(\d+)", subject_dir.name)
            if not subject_match:
                log.warning("Nama folder subjek tidak dikenali: %s", subject_dir)
                continue
            subject_id_from_folder = int(subject_match.group(1))

            activity_dirs = sorted(
                [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("AC")]
            )
            for activity_dir in activity_dirs:
                activity_match = re.match(r"AC(\d+)", activity_dir.name)
                if not activity_match:
                    log.warning("Nama folder aktivitas tidak dikenali: %s", activity_dir)
                    continue
                activity_id_from_folder = int(activity_match.group(1))

                for img_path in activity_dir.glob("*.jp*g"):
                    m = FILENAME_PATTERN.match(img_path.name)
                    if not m:
                        log.warning("Nama file tidak sesuai pola, dilewati: %s", img_path)
                        n_skipped += 1
                        continue

                    subject_from_name = int(m.group("subject_from_name"))
                    activity_from_name = int(m.group("activity_from_name"))
                    frame = int(m.group("frame"))

                    # Validasi silang: nama file vs nama folder harus konsisten
                    if subject_from_name != subject_id_from_folder:
                        log.warning(
                            "Subject mismatch pada %s (folder=S%d, nama file=SUB%d)",
                            img_path, subject_id_from_folder, subject_from_name,
                        )
                    if activity_from_name != activity_id_from_folder:
                        log.warning(
                            "Activity mismatch pada %s (folder=AC%d, nama file=ACT%d)",
                            img_path, activity_id_from_folder, activity_from_name,
                        )

                    rows.append({
                        "filepath": str(img_path.resolve()),
                        "view": view_name,
                        "subject_id": subject_id_from_folder,
                        "activity_id": activity_id_from_folder,
                        "activity_name": ACTIVITY_LABEL_NAMES.get(activity_id_from_folder, "unknown"),
                        "frame": frame,
                    })

    if n_skipped:
        log.warning("Total %d file dilewati karena nama tidak sesuai pola.", n_skipped)

    df = pd.DataFrame(rows)
    return df


def apply_known_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Perbaiki activity_id yang diketahui salah akibat bug folder tertukar
    di sumber dataset (lihat KNOWN_ACTIVITY_CORRECTIONS di config.py).

    Koreksi dilakukan di level metadata (kolom activity_id), bukan dengan
    memindah file fisik, supaya tetap reproducible dan auditable.
    """
    df = df.copy()
    df["activity_id_raw"] = df["activity_id"]  # simpan nilai asli untuk audit

    n_corrected_total = 0
    for (view, subject_id), swap_map in KNOWN_ACTIVITY_CORRECTIONS.items():
        for wrong_id, correct_id in swap_map.items():
            mask = (
                (df["view"] == view)
                & (df["subject_id"] == subject_id)
                & (df["activity_id_raw"] == wrong_id)
            )
            n = mask.sum()
            if n > 0:
                df.loc[mask, "activity_id"] = correct_id
                df.loc[mask, "activity_name"] = ACTIVITY_LABEL_NAMES.get(correct_id, "unknown")
                log.info(
                    "Koreksi diterapkan: %d frame (view=%s, S%d, folder AC%d -> AC%d sebenarnya)",
                    n, view, subject_id, wrong_id, correct_id,
                )
                n_corrected_total += n

    if n_corrected_total == 0:
        log.info("Tidak ada koreksi metadata yang diterapkan (tidak ada baris yang cocok).")
    else:
        log.info("Total %d baris dikoreksi.", n_corrected_total)

    return df


def build_binary_manifest(df_full: pd.DataFrame) -> pd.DataFrame:
    """Filter manifest lengkap ke kelas biner sesuai Subbab 3.2.1 proposal."""
    df = df_full.copy()
    df["binary_label"] = df["activity_id"].apply(activity_id_to_binary_label)
    df_binary = df.dropna(subset=["binary_label"]).reset_index(drop=True)
    return df_binary


def print_summary(df_full: pd.DataFrame, df_binary: pd.DataFrame) -> None:
    print("\n=== Ringkasan manifest_full ===")
    print(f"Total citra (semua kelas, kedua view): {len(df_full)}")
    print(df_full.groupby("view")["filepath"].count().rename("jumlah_citra"))
    print(f"Jumlah subjek unik: {df_full['subject_id'].nunique()}")

    print("\n=== Ringkasan manifest_binary ===")
    print(f"Total citra (A1 + A5-A9, kedua view): {len(df_binary)}")
    print(df_binary.groupby(["view", "binary_label"])["filepath"].count().rename("jumlah_citra"))

    # Cross-check terhadap Tabel 3.1 proposal (dalam pasang frame, bukan citra tunggal)
    n_pairs_expected = 41574
    n_single_expected = 83148
    print(f"\nEkspektasi proposal (Tabel 3.1): {n_pairs_expected} pasang frame -> {n_single_expected} citra tunggal")
    print(f"Hasil scan aktual (citra tunggal, kedua view): {len(df_binary)}")
    if len(df_binary) != n_single_expected:
        log.warning(
            "Jumlah citra hasil scan berbeda dari estimasi proposal. "
            "Hal ini WAJAR jika stride ekstraksi frame di dataset mentah berbeda dari asumsi awal — "
            "verifikasi manual sebelum melanjutkan ke tahap split."
        )


def main():
    if not RAW_DATA_DIR.exists():
        log.error("Direktori data mentah tidak ditemukan: %s", RAW_DATA_DIR)
        sys.exit(1)

    log.info("Memulai scan dari: %s", RAW_DATA_DIR)
    df_full = scan_raw_data(RAW_DATA_DIR)

    if df_full.empty:
        log.error("Tidak ada data yang berhasil di-parse. Periksa struktur folder dan pola nama file.")
        sys.exit(1)

    df_full = apply_known_corrections(df_full)
    df_binary = build_binary_manifest(df_full)

    MANIFEST_FULL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(MANIFEST_FULL_PATH, index=False)
    df_binary.to_csv(MANIFEST_BINARY_PATH, index=False)

    log.info("Manifest lengkap disimpan: %s", MANIFEST_FULL_PATH)
    log.info("Manifest biner disimpan: %s", MANIFEST_BINARY_PATH)

    print_summary(df_full, df_binary)


if __name__ == "__main__":
    main()
