import logging

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE, BINARY_LABEL_MAP, FRAME_STRIDE, MANIFEST_PAIRED_PATH, MANIFEST_SPLIT_PATH
from src.dataset import get_dataloader, get_transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class PairedFinalV1Dataset(Dataset):
    """Stable paired dataset for final_v1 validation/test fusion evaluation."""

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_front = Image.open(row["filepath_front"]).convert("RGB")
        img_side = Image.open(row["filepath_side"]).convert("RGB")
        img_front = self.transform(img_front)
        img_side = self.transform(img_side)
        label = BINARY_LABEL_MAP[row["binary_label"]]
        return img_front, img_side, label, idx


def get_single_view_loader(view: str, split: str, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Reuse the existing repo dataloader so split and preprocessing stay identical."""
    return get_dataloader(view, split, batch_size=batch_size, num_workers=0)


def load_paired_split_dataframe(split_name: str) -> pd.DataFrame:
    """Load paired rows from the existing subject split without creating a new split."""
    if split_name not in {"val", "test"}:
        raise ValueError(f"split_name harus 'val' atau 'test', dapat: {split_name}")

    df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
    subject_ids = set(df_split[df_split["split"] == split_name]["subject_id"].unique())
    if not subject_ids:
        raise ValueError(f"Tidak ada subject untuk split '{split_name}'.")

    df_paired = pd.read_csv(MANIFEST_PAIRED_PATH)
    df = df_paired[df_paired["subject_id"].isin(subject_ids)].copy()
    if df.empty:
        raise ValueError(f"Tidak ada data paired untuk split '{split_name}'.")

    if FRAME_STRIDE > 1:
        df = df[(df["frame"] - 1) % FRAME_STRIDE == 0].copy()
        log.info(
            "Frame subsampling diterapkan pada final_v1 split=%s (stride=%d): %d pasangan tersisa",
            split_name,
            FRAME_STRIDE,
            len(df),
        )

    sort_cols = [col for col in ["subject_id", "activity_id", "frame"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    df = df.reset_index(drop=True)
    df["sample_id"] = df.apply(
        lambda row: f"S{int(row['subject_id']):02d}_AC{int(row['activity_id']):02d}_F{int(row['frame']):05d}",
        axis=1,
    )
    return df


def get_paired_split_loader(split_name: str, batch_size: int = BATCH_SIZE) -> DataLoader:
    df = load_paired_split_dataframe(split_name)
    transform = get_transforms("front", "test")
    dataset = PairedFinalV1Dataset(df, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
