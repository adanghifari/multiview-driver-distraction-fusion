"""
Calibration Diagnostic Script (Offline / Isolated)
For auditing and analyzing model calibration using Temperature Scaling.

Features:
- Loads the trained Front and Side view models from checkpoints.
- Extracts raw logits (before softmax) and targets from the validation set.
- Fits a Temperature scaling parameter (T) per view using LBFGS optimization on the validation set.
- Evaluates the Expected Calibration Error (ECE) and Brier Score on the test set before and after calibration.
- Computes a bootstrap confidence interval (95% CI, 1000 iterations) for ECE.

This script does NOT modify the main pipeline or the adaptive fusion calculations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHECKPOINT_DIR,
    NUM_STAGES_TO_FREEZE_FRONT,
    NUM_STAGES_TO_FREEZE_SIDE,
)
from src.dataset import get_dataloader
from src.model import build_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def compute_ece_binary(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Class-wise ECE for the positive class (consistent with fusion_debug.py)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_ece_standard(logits: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE based on the max probability class (predicted class confidence)."""
    # Softmax to get full probability distribution
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(logits)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score for the positive class (consistent with fusion_debug.py)."""
    return float(np.mean((probs - labels) ** 2))


def bootstrap_ece_ci(probs: np.ndarray, labels: np.ndarray, n_iterations: int = 1000, alpha: float = 0.05) -> tuple:
    """Compute bootstrap confidence interval for positive class ECE."""
    n_samples = len(probs)
    eces = []
    # Seed for reproducibility
    rng = np.random.default_rng(42)
    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        resampled_probs = probs[indices]
        resampled_labels = labels[indices]
        ece = compute_ece_binary(resampled_probs, resampled_labels)
        eces.append(ece)
        
    eces = np.sort(eces)
    lower_idx = int(n_iterations * (alpha / 2.0))
    upper_idx = int(n_iterations * (1.0 - alpha / 2.0))
    return eces[lower_idx], eces[upper_idx]


def load_model(view: str, device: torch.device, exp_id: str = ""):
    """Load model checkpoint for a specific view."""
    suffix = f"_{exp_id}" if exp_id else ""
    ckpt_path = CHECKPOINT_DIR / f"{view}{suffix}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan di {ckpt_path}")
        
    freeze_stages = NUM_STAGES_TO_FREEZE_FRONT if view == "front" else NUM_STAGES_TO_FREEZE_SIDE
    model = build_model(pretrained=False, num_stages_to_freeze=freeze_stages)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    log.info(f"Berhasil memuat model {view} dari epoch {checkpoint['epoch']} (Val F1: {checkpoint['val_macro_f1']:.4f})")
    return model


@torch.no_grad()
def collect_logits_and_labels(model, dataloader, device):
    """Run model on dataloader and collect logits and labels."""
    all_logits = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)  # (B, 2)
        all_logits.append(logits.cpu())
        all_labels.append(labels.clone())
        
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0).long()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor):
    """Fit temperature parameter T using LBFGS optimizer on the validation set.
    Uses log(T) reparameterization and clamps T >= 0.1 to avoid degenerate solutions.
    """
    log_temperature = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    optimizer = optim.LBFGS([log_temperature], lr=0.01, max_iter=100)
    criterion = nn.CrossEntropyLoss()
    
    # Calculate initial loss (T = 1.0)
    initial_loss = criterion(logits, labels).item()
    
    def eval_val():
        optimizer.zero_grad()
        T = torch.exp(log_temperature)
        T = torch.clamp(T, min=0.1, max=10.0)
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_val)
    
    T_val = torch.exp(log_temperature).item()
    T_val = max(0.1, min(10.0, T_val))
    
    # Calculate final loss
    final_loss = criterion(logits / T_val, labels).item()
    
    return T_val, initial_loss, final_loss


def analyze_view_calibration(view: str, device: torch.device, exp_id: str = "", n_bootstrap: int = 1000):
    """Run full calibration analysis for a view."""
    log.info("-" * 65)
    log.info(f"ANALISIS KALIBRASI UNTUK VIEW: {view.upper()}")
    log.info("-" * 65)
    
    # Load model and datasets
    model = load_model(view, device, exp_id)
    val_loader = get_dataloader(view, "val")
    test_loader = get_dataloader(view, "test")
    
    # Collect validation logits and fit temperature
    log.info("Mengambil logit dari validation set...")
    val_logits, val_labels = collect_logits_and_labels(model, val_loader, device)
    
    # Val set size sanity check
    val_labels_np = val_labels.numpy()
    n_total = len(val_labels_np)
    n_pos = np.sum(val_labels_np == 1)
    n_neg = np.sum(val_labels_np == 0)
    log.info(f"Distribusi Val Set: Total={n_total}, Positif (phone_use)={n_pos}, Negatif (safe_driving)={n_neg}")
    
    if n_total < 300 or min(n_pos, n_neg) < 50:
        log.warning(f"⚠️ UKURAN VAL SET KECIL/SANGAT KECIL (N={n_total}, Kelas Minoritas={min(n_pos, n_neg)} sampel).")
        log.warning("   Ada risiko tinggi temperature scaling mengalami OVERFITTING pada validation set.")
        log.warning("   Hasil kalibrasi pada test set harus diinterpretasikan dengan sangat hati-hati!")
        
    log.info("Optimasi parameter T (Temperature Scaling) dengan LBFGS...")
    T, init_loss, final_loss = fit_temperature(val_logits, val_labels)
    log.info(f"Optimal Temperature (T) untuk {view.upper()} view: {T:.4f} (Val Loss: {init_loss:.4f} -> {final_loss:.4f})")
    
    # Sanity check on T value
    if T <= 0.1001 or T >= 9.999:
        log.warning(f"⚠️ KEMUNGKINAN OPTIMASI TIDAK KONVERGEN (T mencapai batas clamp: {T:.4f})")
        log.warning(f"   Initial Val Loss: {init_loss:.4f} -> Final Val Loss: {final_loss:.4f}")
    
    # Collect test set logits
    log.info("Mengambil logit dari test set...")
    test_logits, test_labels = collect_logits_and_labels(model, test_loader, device)
    test_labels_np = test_labels.numpy()
    
    # Predictions before calibration
    probs_before = torch.softmax(test_logits, dim=1)[:, 1].numpy()
    ece_bin_before = compute_ece_binary(probs_before, test_labels_np)
    ece_std_before = compute_ece_standard(test_logits.numpy(), test_labels_np)
    bs_before = compute_brier_score(probs_before, test_labels_np)
    
    # Predictions after calibration
    probs_after = torch.softmax(test_logits / T, dim=1)[:, 1].numpy()
    ece_bin_after = compute_ece_binary(probs_after, test_labels_np)
    ece_std_after = compute_ece_standard((test_logits / T).numpy(), test_labels_np)
    bs_after = compute_brier_score(probs_after, test_labels_np)
    
    # Bootstrap CI for ECE
    ci_before_lower, ci_before_upper = bootstrap_ece_ci(probs_before, test_labels_np, n_iterations=n_bootstrap)
    ci_after_lower, ci_after_upper = bootstrap_ece_ci(probs_after, test_labels_np, n_iterations=n_bootstrap)
    
    # Print reports
    print(f"\n[HASIL DIAGNOSTIK KALIBRASI {view.upper()} VIEW]")
    print(f"  Optimal Temperature (T) : {T:.4f}")
    print(f"  Test Set (N = {len(test_labels_np)}):")
    print(f"    - ECE (Binary / Positive Class):")
    print(f"        Sebelum Kalibrasi: {ece_bin_before:.5f}  (95% CI: [{ci_before_lower:.5f}, {ci_before_upper:.5f}])")
    print(f"        Setelah Kalibrasi: {ece_bin_after:.5f}  (95% CI: [{ci_after_lower:.5f}, {ci_after_upper:.5f}])")
    print(f"    - ECE (Standard / Multiclass):")
    print(f"        Sebelum Kalibrasi: {ece_std_before:.5f}")
    print(f"        Setelah Kalibrasi: {ece_std_after:.5f}")
    print(f"    - Brier Score (Positive Class):")
    print(f"        Sebelum Kalibrasi: {bs_before:.5f}")
    print(f"        Setelah Kalibrasi: {bs_after:.5f}")
    print(f"    - Selisih ECE (Binary)  : {ece_bin_after - ece_bin_before:.5f} (Penurunan: {abs(ece_bin_after - ece_bin_before)/ece_bin_before*100:.1f}%)" if ece_bin_before > 0 else "")
    print(f"    - Selisih Brier Score   : {bs_after - bs_before:.5f}")
    print()
    
    return {
        "temperature": T,
        "ece_before": ece_bin_before,
        "ece_after": ece_bin_after,
        "ci_before": (ci_before_lower, ci_before_upper),
        "ci_after": (ci_after_lower, ci_after_upper),
        "brier_before": bs_before,
        "brier_after": bs_after,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibration Diagnostics with Temperature Scaling & Bootstrap CI")
    parser.add_argument("--exp_id", type=str, default="",
                        help="ID Eksperimen (opsional, misal 'exp10')")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Jumlah iterasi bootstrap untuk ECE CI (default: 1000)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Menggunakan device: {device}")
    
    results = {}
    for view in ("front", "side"):
        try:
            res = analyze_view_calibration(view, device, exp_id=args.exp_id, n_bootstrap=args.bootstrap)
            results[view] = res
        except Exception as e:
            log.error(f"Gagal melakukan analisis kalibrasi untuk {view}: {e}")
            
    if len(results) == 2:
        print("=" * 65)
        print("RINGKASAN PERBANDINGAN KALIBRASI")
        print("=" * 65)
        print(f"{'View':<10} | {'T':<6} | {'ECE Before (95% CI)':<30} | {'ECE After (95% CI)':<30} | {'Brier Bef':<9} | {'Brier Aft':<9}")
        print("-" * 115)
        for view in ("front", "side"):
            r = results[view]
            ci_b = f"{r['ece_before']:.4f} [{r['ci_before'][0]:.4f}, {r['ci_before'][1]:.4f}]"
            ci_a = f"{r['ece_after']:.4f} [{r['ci_after'][0]:.4f}, {r['ci_after'][1]:.4f}]"
            print(f"{view:<10} | {r['temperature']:.4f} | {ci_b:<30} | {ci_a:<30} | {r['brier_before']:.4f} | {r['brier_after']:.4f}")
        print("=" * 115)
        print("\nCatatan: ECE biner dihitung berdasarkan probabilitas kelas positif (phone_use).")
        print("Semua data testing diambil dengan frame subsampling yang konsisten.\n")


if __name__ == "__main__":
    main()
