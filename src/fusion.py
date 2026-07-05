"""
Decision-level fusion: average dan adaptive.

Menggunakan manifest_paired.csv untuk evaluasi pada test set.
Kedua model single-view (front & side) sudah harus dilatih terlebih dahulu.

Metode fusion (sesuai Persamaan 3.2–3.4 proposal):

  Average fusion:
    S_fused = 0.5 × S_front + 0.5 × S_side

  Adaptive fusion (bobot berbasis confidence):
    d_i     = |S_i − 0.5|                           ... (3.2)
    w_i     = exp(d_i) / (exp(d_front) + exp(d_side)) ... (3.3)
    S_fused = w_front × S_front + w_side × S_side     ... (3.4)

  Prediksi akhir:
    ŷ = 1 (phone_use)  jika S_fused ≥ 0.5
    ŷ = 0 (safe_driving) jika S_fused < 0.5

Penggunaan:
  python -m src.fusion
"""

import json
import logging

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.config import (
    MANIFEST_PAIRED_PATH,
    MANIFEST_SPLIT_PATH,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    DECISION_THRESHOLD,
    BINARY_LABEL_MAP,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BATCH_SIZE,
)
from src.dataset import get_transforms
from src.evaluate import load_trained_model, compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset untuk pasangan frame (front + side)
# ──────────────────────────────────────────────────────────────────────────────

class PairedTestDataset(Dataset):
    """Dataset yang memuat pasangan frame front & side secara tersinkronisasi.
    Digunakan khusus untuk evaluasi decision-level fusion pada test set."""

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
        return img_front, img_side, label


def get_paired_test_loader(batch_size: int = BATCH_SIZE,
                           num_workers: int = 4) -> DataLoader:
    """Bangun DataLoader untuk pasangan frame test set."""
    # Ambil subject_id yang masuk test split
    df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
    test_subjects = set(df_split[df_split["split"] == "test"]["subject_id"].unique())

    # Filter manifest_paired ke test subjects saja
    df_paired = pd.read_csv(MANIFEST_PAIRED_PATH)
    df_test = df_paired[df_paired["subject_id"].isin(test_subjects)]

    if df_test.empty:
        raise ValueError("Tidak ada data test di manifest_paired.csv")

    transform = get_transforms("test")  # tanpa augmentasi
    dataset = PairedTestDataset(df_test, transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    log.info("Paired test DataLoader: n=%d pairs, batch_size=%d", len(dataset), batch_size)
    return loader


# ──────────────────────────────────────────────────────────────────────────────
# Fusion methods
# ──────────────────────────────────────────────────────────────────────────────

def average_fusion(scores_front: np.ndarray, scores_side: np.ndarray) -> np.ndarray:
    """Average fusion: S_fused = 0.5 × S_front + 0.5 × S_side."""
    return 0.5 * scores_front + 0.5 * scores_side


def adaptive_fusion(scores_front: np.ndarray, scores_side: np.ndarray,
                    threshold: float = DECISION_THRESHOLD) -> np.ndarray:
    """Adaptive fusion berbasis confidence (Persamaan 3.2–3.4 proposal).

    d_i     = |S_i - threshold|
    w_i     = exp(d_i) / (exp(d_front) + exp(d_side))   (softmax)
    S_fused = w_front × S_front + w_side × S_side
    """
    d_front = np.abs(scores_front - threshold)
    d_side = np.abs(scores_side - threshold)

    # Softmax over [d_front, d_side] per sample
    exp_front = np.exp(d_front)
    exp_side = np.exp(d_side)
    sum_exp = exp_front + exp_side

    w_front = exp_front / sum_exp
    w_side = exp_side / sum_exp

    return w_front * scores_front + w_side * scores_side


# ──────────────────────────────────────────────────────────────────────────────
# Inferensi & evaluasi
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_paired_scores(model_front, model_side, paired_loader, device):
    """Jalankan kedua model pada paired test set.
    Return (scores_front, scores_side, labels) sebagai numpy arrays."""
    model_front.eval()
    model_side.eval()

    all_scores_front = []
    all_scores_side = []
    all_labels = []

    for img_front, img_side, labels in paired_loader:
        img_front = img_front.to(device)
        img_side = img_side.to(device)

        logits_front = model_front(img_front)
        logits_side = model_side(img_side)

        probs_front = torch.sigmoid(logits_front).cpu().numpy()
        probs_side = torch.sigmoid(logits_side).cpu().numpy()

        all_scores_front.extend(probs_front.tolist())
        all_scores_side.extend(probs_side.tolist())
        all_labels.extend(labels.tolist())

    return (
        np.array(all_scores_front),
        np.array(all_scores_side),
        np.array(all_labels),
    )


def evaluate_fusion(method_name: str, fused_scores: np.ndarray,
                    labels: np.ndarray, threshold: float = DECISION_THRESHOLD) -> dict:
    """Evaluasi satu metode fusion."""
    preds = (fused_scores >= threshold).astype(int)
    metrics = compute_metrics(labels.tolist(), preds.tolist())
    metrics["method"] = method_name
    print_metrics(method_name, metrics)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load kedua model ──
    model_front, ckpt_front = load_trained_model("front", device)
    model_side, ckpt_side = load_trained_model("side", device)

    # ── Load paired test data ──
    paired_loader = get_paired_test_loader()

    # ── Dapatkan skor dari kedua model ──
    log.info("Running inference on paired test set...")
    scores_front, scores_side, labels = get_paired_scores(
        model_front, model_side, paired_loader, device
    )
    log.info("Inference selesai: %d paired frames", len(labels))

    # ── Evaluasi single-view (dari skor paired, sebagai baseline) ──
    log.info("")
    eval_front = evaluate_fusion("Single-view FRONT", scores_front, labels)
    eval_side = evaluate_fusion("Single-view SIDE", scores_side, labels)

    # ── Average fusion ──
    scores_avg = average_fusion(scores_front, scores_side)
    eval_avg = evaluate_fusion("Average Fusion (50:50)", scores_avg, labels)

    # ── Adaptive fusion ──
    scores_adapt = adaptive_fusion(scores_front, scores_side)
    eval_adapt = evaluate_fusion("Adaptive Fusion (confidence)", scores_adapt, labels)

    # ── Tabel perbandingan ──
    log.info("")
    log.info("=" * 70)
    log.info("PERBANDINGAN AKHIR (Test Set — Paired Frames)")
    log.info("=" * 70)
    log.info("%-35s  Acc      Prec     Rec      F1", "Method")
    log.info("-" * 70)
    for e in [eval_front, eval_side, eval_avg, eval_adapt]:
        log.info("%-35s  %.4f   %.4f   %.4f   %.4f",
                 e["method"], e["accuracy"], e["precision_macro"],
                 e["recall_macro"], e["f1_macro"])
    log.info("=" * 70)

    # ── Simpan semua hasil ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {
        "single_front": eval_front,
        "single_side": eval_side,
        "average_fusion": eval_avg,
        "adaptive_fusion": eval_adapt,
    }
    results_path = RESULTS_DIR / "fusion_comparison.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Hasil disimpan di: %s", results_path)


if __name__ == "__main__":
    main()
