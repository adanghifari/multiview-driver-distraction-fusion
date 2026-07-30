# %% [markdown]
# # Decision-Level Fusion: Average & Adaptive
# Menggabungkan prediksi model front dan side.
#
# - **Average fusion**: S_fused = 0.5 × S_front + 0.5 × S_side
# - **Adaptive fusion**: bobot berbasis confidence tiap view (Persamaan 3.2–3.4)
#
# **Penggunaan CLI:** `python -m src.fusion`

# %%
# Imports & setup logging
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
from pathlib import Path

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
    FRAME_STRIDE,
)
from src.dataset import get_transforms
from src.evaluate import load_trained_model, compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# %%
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
                           num_workers: int = 0) -> DataLoader:
    """Bangun DataLoader untuk pasangan frame test set.

    Menerapkan frame subsampling (FRAME_STRIDE) yang konsisten dengan
    loader single-view di dataset.py agar evaluasi fusion sebanding.
    """
    # Ambil subject_id yang masuk test split
    df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
    test_subjects = set(df_split[df_split["split"] == "test"]["subject_id"].unique())

    # Filter manifest_paired ke test subjects saja
    df_paired = pd.read_csv(MANIFEST_PAIRED_PATH)
    df_test = df_paired[df_paired["subject_id"].isin(test_subjects)]

    if df_test.empty:
        raise ValueError("Tidak ada data test di manifest_paired.csv")

    # Frame subsampling: konsisten dengan dataset.py [v4]
    if FRAME_STRIDE > 1:
        df_test = df_test[(df_test["frame"] - 1) % FRAME_STRIDE == 0]
        log.info("Frame subsampling diterapkan pada paired test loader (stride=%d): %d pasangan tersisa",
                 FRAME_STRIDE, len(df_test))

    transform = get_transforms("front", "test")  # tanpa augmentasi
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


# %%
# Fusion methods (average & adaptive)
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


def adaptive_linear_normalization_fusion(
    scores_front: np.ndarray,
    scores_side: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> np.ndarray:
    """Adaptive fusion pembanding dengan normalisasi linear confidence.

    d_i     = |S_i - threshold|
    w_i     = d_i / (d_front + d_side)
    S_fused = w_front * S_front + w_side * S_side

    Jika d_front + d_side = 0, bobot dibuat 0.5 dan 0.5.
    """
    d_front = np.abs(scores_front - threshold)
    d_side = np.abs(scores_side - threshold)
    denom = d_front + d_side

    w_front = np.divide(d_front, denom, out=np.full_like(d_front, 0.5, dtype=float), where=denom != 0)
    w_side = np.divide(d_side, denom, out=np.full_like(d_side, 0.5, dtype=float), where=denom != 0)

    return w_front * scores_front + w_side * scores_side


# %%
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

        probs_front = torch.softmax(logits_front, dim=1)[:, 1].cpu().numpy()
        probs_side = torch.softmax(logits_side, dim=1)[:, 1].cpu().numpy()

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


# %%
# Main — jalankan sel ini untuk evaluasi fusion secara interaktif
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Decision-level fusion untuk front & side")
    parser.add_argument("--exp_id", type=str, default="",
                        help="ID Eksperimen (opsional, misal 'exp3')")
    parser.add_argument("--front-checkpoint", type=str, default=None,
                        help="Path checkpoint front eksplisit, misal checkpoints/front_best_exp14A.pt")
    parser.add_argument("--side-checkpoint", type=str, default=None,
                        help="Path checkpoint side eksplisit, misal checkpoints/side_best_exp13_backup.pt")
    parser.add_argument("--output", type=str, default=None,
                        help="Path output JSON eksplisit, misal results/fusion_comparison_exp19.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load kedua model ──
    model_front, ckpt_front = load_trained_model(
        "front",
        device,
        exp_id=args.exp_id,
        checkpoint_path=args.front_checkpoint,
    )
    model_side, ckpt_side = load_trained_model(
        "side",
        device,
        exp_id=args.exp_id,
        checkpoint_path=args.side_checkpoint,
    )

    # ── Load paired test data ──
    paired_loader = get_paired_test_loader()

    # ── Dapatkan skor dari kedua model ──
    log.info("Running inference on paired test set...")
    scores_front, scores_side, labels = get_paired_scores(
        model_front, model_side, paired_loader, device
    )
    log.info("Inference selesai: %d paired frames", len(labels))

    # ── Sanity check: pastikan jumlah sampel di front/side test metrics sinkron dengan paired test set [v5/Exp8] ──
    suffix = f"_{args.exp_id}" if args.exp_id else ""
    custom_checkpoint_used = bool(args.front_checkpoint or args.side_checkpoint)
    for view in ("front", "side"):
        if custom_checkpoint_used:
            log.info("Checkpoint eksplisit digunakan; sanity check file metrics default untuk %s dilewati.", view)
            continue
        metrics_path = RESULTS_DIR / f"{view}_test_metrics{suffix}.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                m = json.load(f)
            # Hitung total sampel dari sum confusion matrix
            total_samples = sum(sum(r) for r in m.get("confusion_matrix", []))
            if total_samples != len(labels):
                raise AssertionError(
                    f"Mismatch jumlah sampel pada {view} view: "
                    f"metrics di {metrics_path.name} memiliki {total_samples} sampel, "
                    f"sedangkan paired test set memiliki {len(labels)} sampel. "
                    f"Silakan re-run evaluasi single-view terlebih dahulu: "
                    f"'python -m src.evaluate --view {view}'"
                )
        else:
            log.warning(f"File metrics {metrics_path.name} belum ada, sanity check sampel dilewati.")

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
    if args.output:
        results_path = Path(args.output)
        if not results_path.is_absolute():
            results_path = Path.cwd() / results_path
    elif ckpt_front.get("experiment") == "experiment_14A":
        results_path = RESULTS_DIR / "fusion_comparison_exp14A.json"
    else:
        suffix = f"_{args.exp_id}" if args.exp_id else ""
        results_path = RESULTS_DIR / f"fusion_comparison{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Hasil disimpan di: %s", results_path)

    # ── Jalankan audit/diagnostik fusion secara otomatis agar fusion_debug.csv tidak stale ──
    if custom_checkpoint_used:
        log.info("Audit fusion_debug otomatis dilewati karena checkpoint eksplisit digunakan.")
    else:
        import subprocess
        import sys
        cmd = [sys.executable, "-m", "src.fusion_debug"]
        if args.exp_id:
            cmd.extend(["--exp_id", args.exp_id])
        log.info("Menjalankan audit/diagnostik fusion secara otomatis...")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            log.error("Gagal menjalankan fusion_debug secara otomatis: %s", e)


if __name__ == "__main__":
    main()
