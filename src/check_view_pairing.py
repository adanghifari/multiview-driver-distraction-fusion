"""
Mengecek kesesuaian pasangan frame antara front view dan side view.

Decision-level fusion butuh pasangan (subject_id, activity_id, frame) yang
identik di kedua view. Script ini mengidentifikasi frame yang orphan
(hanya ada di satu view) sehingga bisa diputuskan: didrop, atau dicek manual.

Jalankan dari root proyek:
    python -m src.check_view_pairing
"""

import logging
import pandas as pd

from src.config import MANIFEST_BINARY_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    df = pd.read_csv(MANIFEST_BINARY_PATH)

    df["frame_key"] = list(zip(df["subject_id"], df["activity_id"], df["frame"]))

    front_keys = set(df.loc[df["view"] == "front", "frame_key"])
    side_keys = set(df.loc[df["view"] == "side", "frame_key"])

    only_front = sorted(front_keys - side_keys)
    only_side = sorted(side_keys - front_keys)

    print(f"Frame hanya ada di front (tidak ada pasangan di side): {len(only_front)}")
    print(f"Frame hanya ada di side (tidak ada pasangan di front): {len(only_side)}")

    if only_front:
        print("\nContoh frame orphan di front (subject_id, activity_id, frame):")
        for key in only_front[:30]:
            subj, act, frame = key
            row = df[(df["view"] == "front") & (df["subject_id"] == subj)
                      & (df["activity_id"] == act) & (df["frame"] == frame)].iloc[0]
            print(f"  S{subj} AC{act} F{frame} -> {row['filepath']}")

    if only_side:
        print("\nContoh frame orphan di side (subject_id, activity_id, frame):")
        for key in only_side[:30]:
            subj, act, frame = key
            row = df[(df["view"] == "side") & (df["subject_id"] == subj)
                      & (df["activity_id"] == act) & (df["frame"] == frame)].iloc[0]
            print(f"  S{subj} AC{act} F{frame} -> {row['filepath']}")

    n_paired = len(front_keys & side_keys)
    print(f"\nJumlah pasangan frame valid (ada di kedua view): {n_paired}")


if __name__ == "__main__":
    main()
