"""
Dataset dan DataLoader PyTorch untuk model klasifikasi single-view.

Setiap sudut pandang (front/side) dilatih sebagai model terpisah, sehingga
Dataset ini dibangun secara eksplisit per-view -- bukan menggabungkan front+side
dalam satu Dataset. Kombinasi front+side baru dipakai nanti khusus untuk
evaluasi decision-level fusion (dari manifest_paired.csv, ditangani terpisah).

Preprocessing mengikuti Subbab 3.2.3 proposal:
    - Resize ke 224x224
    - Normalisasi mean/std ImageNet
    - Augmentasi (flip, rotation, color jitter) HANYA pada data latih
    - Data validasi & uji tanpa augmentasi
"""

import logging

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import (
    MANIFEST_SPLIT_PATH,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BATCH_SIZE,
    BINARY_LABEL_MAP,
    FRAME_STRIDE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class DriverViewDataset(Dataset):
    """Dataset untuk satu sudut pandang (front ATAU side) pada satu partisi
    (train/val/test), diambil dari manifest_split.csv."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)
        label = BINARY_LABEL_MAP[row["binary_label"]]
        return image, label


def get_transforms(split: str) -> transforms.Compose:
    """Transform training menyertakan augmentasi; val/test tidak (Subbab 3.2.3)."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),     # [v4] dinaikkan dari 10 → 15 derajat
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # [v4] dari 0.2 → 0.3
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_split_dataframe(view: str, split: str) -> pd.DataFrame:
    """Ambil subset manifest_split.csv untuk satu view dan satu partisi.

    Menerapkan frame subsampling sesuai FRAME_STRIDE dari config.py:
    hanya frame dengan (frame - 1) % FRAME_STRIDE == 0 yang diambil
    (frame ke-1, 6, 11, 16, ...). Diterapkan pada semua split secara
    konsisten agar tidak ada ketidaksesuaian antara train/val/test.
    """
    df = pd.read_csv(MANIFEST_SPLIT_PATH)
    subset = df[(df["view"] == view) & (df["split"] == split)]
    if subset.empty:
        raise ValueError(f"Tidak ada data untuk view='{view}', split='{split}'. Cek manifest_split.csv.")

    # Frame subsampling: ambil 1 dari setiap FRAME_STRIDE frame [v4]
    if FRAME_STRIDE > 1:
        subset = subset[(subset["frame"] - 1) % FRAME_STRIDE == 0]
        log.info("Frame subsampling diterapkan (stride=%d): %d frame tersisa untuk view=%s split=%s",
                 FRAME_STRIDE, len(subset), view, split)

    return subset


def get_dataloader(view: str, split: str, batch_size: int = BATCH_SIZE,
                    num_workers: int = 4, shuffle: bool = None) -> DataLoader:
    """Bangun satu DataLoader untuk kombinasi view + split tertentu.

    view: 'front' atau 'side'
    split: 'train', 'val', atau 'test'
    """
    if view not in ("front", "side"):
        raise ValueError(f"view harus 'front' atau 'side', dapat: {view}")
    if split not in ("train", "val", "test"):
        raise ValueError(f"split harus 'train', 'val', atau 'test', dapat: {split}")

    df_subset = load_split_dataframe(view, split)
    transform = get_transforms(split)
    dataset = DriverViewDataset(df_subset, transform)

    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    log.info("DataLoader dibuat: view=%s split=%s n=%d batch_size=%d shuffle=%s",
              view, split, len(dataset), batch_size, shuffle)
    return loader


def get_all_dataloaders(view: str, batch_size: int = BATCH_SIZE, num_workers: int = 4) -> dict:
    """Shortcut: bangun train/val/test DataLoader sekaligus untuk satu view."""
    return {
        split: get_dataloader(view, split, batch_size=batch_size, num_workers=num_workers)
        for split in ("train", "val", "test")
    }


if __name__ == "__main__":
    # Sanity check cepat: pastikan loader bisa dibuat dan satu batch bisa ditarik
    for view in ("front", "side"):
        loaders = get_all_dataloaders(view, num_workers=0)
        images, labels = next(iter(loaders["train"]))
        print(f"[{view}] batch images: {images.shape}, labels: {labels[:8].tolist()}")
