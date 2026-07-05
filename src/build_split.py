"""
Membangun manifest berpasangan (paired) dari manifest_binary.csv, lalu
melakukan subject-based split sesuai Tabel 3.2 proposal (35/7/8 subjek).

Manifest berpasangan hanya menyertakan frame_key (subject_id, activity_id, frame)
yang tersedia di KEDUA view -- 21 frame orphan di S31/AC9 (hanya ada di front)
otomatis ter-drop di sini.

Output:
    data/manifest_paired.csv       -> satu baris per pasangan frame,
                                       kolom filepath_front & filepath_side
    data/manifest_split.csv        -> manifest_binary.csv + kolom 'split'
                                       (train/val/test), per baris per view

Jalankan dari root proyek:
    python -m src.build_split
"""

import logging
import numpy as np
import pandas as pd

from src.config import (
    MANIFEST_BINARY_PATH,
    MANIFEST_PAIRED_PATH,
    MANIFEST_SPLIT_PATH,
    SPLIT_SEED,
    SPLIT_RATIOS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def build_paired_manifest(df_binary: pd.DataFrame) -> pd.DataFrame:
    """Pivot manifest_binary (long, per-view) menjadi manifest berpasangan (wide)."""
    front = df_binary[df_binary["view"] == "front"].set_index(
        ["subject_id", "activity_id", "frame"], drop=False
    )
    side = df_binary[df_binary["view"] == "side"].set_index(
        ["subject_id", "activity_id", "frame"], drop=False
    )

    common_index = front.index.intersection(side.index)
    n_dropped_front = len(front) - len(common_index)
    n_dropped_side = len(side) - len(common_index)
    if n_dropped_front or n_dropped_side:
        log.warning(
            "Drop frame tidak berpasangan: %d dari front, %d dari side.",
            n_dropped_front, n_dropped_side,
        )

    paired = pd.DataFrame({
        "filepath_front": front.loc[common_index, "filepath"],
        "filepath_side": side.loc[common_index, "filepath"],
        "binary_label": front.loc[common_index, "binary_label"],
    }).reset_index()

    return paired


def subject_based_split(subject_ids: np.ndarray, seed: int, ratios: dict) -> dict:
    """Bagi array subject_id menjadi train/val/test secara acak per-subjek."""
    rng = np.random.default_rng(seed)
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    # sisanya masuk test, supaya total pas tanpa kehilangan/duplikasi subjek
    n_test = n - n_train - n_val

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    log.info(
        "Split subjek -> train: %d, val: %d, test: %d (total %d)",
        len(train_ids), len(val_ids), len(test_ids), n,
    )
    return {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}


def assign_split(subject_id: int, split_map: dict) -> str:
    for split_name, id_set in split_map.items():
        if subject_id in id_set:
            return split_name
    raise ValueError(f"subject_id {subject_id} tidak ada di split manapun")


def print_summary(df_split: pd.DataFrame, df_paired: pd.DataFrame) -> None:
    print("\n=== Ringkasan split (per view, jumlah citra tunggal) ===")
    print(df_split.groupby(["split", "view", "binary_label"])["filepath"].count())

    print("\n=== Ringkasan split (jumlah subjek per partisi) ===")
    print(df_split.groupby("split")["subject_id"].nunique())

    print(f"\n=== Total pasangan frame tersinkronisasi: {len(df_paired)} ===")
    print("(bandingkan dengan Tabel 3.1 proposal: 41.574 pasang frame)")


def main():
    df_binary = pd.read_csv(MANIFEST_BINARY_PATH)

    # 1. Bangun manifest berpasangan (untuk evaluasi decision-level fusion)
    df_paired = build_paired_manifest(df_binary)
    df_paired.to_csv(MANIFEST_PAIRED_PATH, index=False)
    log.info("Manifest berpasangan disimpan: %s (%d pasangan)", MANIFEST_PAIRED_PATH, len(df_paired))

    # 2. Subject-based split (subjek yang sama harus di partisi yang sama,
    #    dan sama untuk front maupun side -- lihat Subbab 3.2.2 proposal)
    subject_ids = df_binary["subject_id"].unique()
    split_map = subject_based_split(subject_ids, seed=SPLIT_SEED, ratios=SPLIT_RATIOS)

    df_split = df_binary.copy()
    df_split["split"] = df_split["subject_id"].apply(lambda s: assign_split(s, split_map))
    df_split.to_csv(MANIFEST_SPLIT_PATH, index=False)
    log.info("Manifest dengan kolom split disimpan: %s", MANIFEST_SPLIT_PATH)

    print_summary(df_split, df_paired)


if __name__ == "__main__":
    main()
