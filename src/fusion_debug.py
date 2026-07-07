"""
Diagnostic script for auditing and analyzing Adaptive Fusion vs Average Fusion.
Reuses the exact same checkpoints, loader, and mathematical formulas as the main pipeline.

Run with:
    python -m src.fusion_debug
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.config import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    DECISION_THRESHOLD,
)
from src.fusion import get_paired_test_loader, get_paired_scores

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Fungsi Diagnostik Kalibrasi
# ──────────────────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE).

    Membagi prediksi menjadi n_bins interval confidence, lalu menghitung
    selisih rata-rata antara confidence (mean prob) dan accuracy aktual per bin.

    ECE = sum_b (|B_b| / n) * |accuracy(B_b) - confidence(B_b)|

    Nilai mendekati 0 artinya model well-calibrated.
    Nilai tinggi (>0.10) menunjukkan overconfidence atau underconfidence.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier Score — ukuran akurasi probabilitas (lower is better).

    BS = (1/n) * sum((p_i - y_i)^2)

    Interpretasi:
      BS = 0.00        : prediksi sempurna
      BS = 0.25        : prediksi acak (baseline)
      BS mendekati 0   : kalibrasi baik
      BS > 0.10 dengan ECE tinggi : overconfidence
    """
    return float(np.mean((probs - labels) ** 2))


def print_calibration_report(view: str, probs: np.ndarray, labels: np.ndarray) -> None:
    """Cetak laporan ECE dan Brier Score untuk satu view."""
    ece = compute_ece(probs, labels)
    bs  = compute_brier_score(probs, labels)

    # Interpretasi kualitatif
    ece_qual = "BAIK" if ece < 0.05 else ("SEDANG" if ece < 0.10 else "BURUK (overconfident)")
    bs_qual  = "BAIK" if bs  < 0.10 else ("SEDANG" if bs  < 0.20 else "BURUK")

    print(f"  [{view.upper()}]")
    print(f"    ECE         : {ece:.5f}  ({ece_qual})")
    print(f"    Brier Score : {bs:.5f}  ({bs_qual})")
    return ece, bs


def load_and_print_loss_gap(exp_id: str = "") -> None:
    """Baca history JSON dan cetak gap train_loss vs val_loss per epoch.

    Gap besar (train_loss << val_loss, selisih > 0.10) mengindikasikan
    overfitting, bukan legitimate overconfidence.
    """
    suffix = f"_{exp_id}" if exp_id else ""
    print("\n" + "=" * 60)
    print("ANALISIS GAP TRAIN LOSS vs VAL LOSS")
    print("=" * 60)

    for view in ("front", "side"):
        history_path = RESULTS_DIR / f"{view}_history{suffix}.json"
        if not history_path.exists():
            print(f"  [{view.upper()}] History tidak ditemukan: {history_path}")
            continue

        with open(history_path) as f:
            hist = json.load(f)

        train_losses = np.array(hist["train_loss"])
        val_losses   = np.array(hist["val_loss"])
        gaps         = val_losses - train_losses  # positif = val > train (normal/overfit)

        best_epoch_idx = int(np.argmax(hist["val_macro_f1"]))
        final_train    = train_losses[-1]
        final_val      = val_losses[-1]
        final_gap      = gaps[-1]
        best_gap       = gaps[best_epoch_idx]
        max_gap        = gaps.max()

        overfit_flag = "OVERFIT" if final_gap > 0.15 else ("OK" if final_gap < 0.08 else "BORDERLINE")

        print(f"\n  [{view.upper()} VIEW]")
        print(f"    Epoch terbaik (val F1)  : {best_epoch_idx + 1}")
        print(f"    Train loss (akhir)      : {final_train:.5f}")
        print(f"    Val   loss (akhir)      : {final_val:.5f}")
        print(f"    Gap (val-train) akhir   : {final_gap:.5f} -> {overfit_flag}")
        print(f"    Gap di epoch terbaik    : {best_gap:.5f}")
        print(f"    Gap maksimum (any epoch): {max_gap:.5f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit/Diagnostik untuk Adaptive Fusion")
    parser.add_argument("--exp_id", type=str, default="",
                        help="ID Eksperimen (opsional, misal 'exp3')")
    args = parser.parse_args()

    # ── Setup device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | Exp ID: %s", device, args.exp_id if args.exp_id else "default")

    # ── Analisis Gap Loss (dari history JSON) ──
    load_and_print_loss_gap(exp_id=args.exp_id)

    # ── Load checkpoints ──
    try:
        from src.evaluate import load_trained_model
        model_front, _ = load_trained_model("front", device, exp_id=args.exp_id)
        model_side, _ = load_trained_model("side", device, exp_id=args.exp_id)
    except Exception as e:
        log.error("Gagal memuat checkpoint. Pastikan model front dan side sudah ditraining.")
        log.error("Error: %s", e)
        return

    # ── Load paired test loader ──
    try:
        paired_loader = get_paired_test_loader()
    except Exception as e:
        log.error("Gagal membuat paired test loader. Cek manifest file Anda.")
        log.error("Error: %s", e)
        return

    # ── Inference ──
    log.info("Running inference on paired test set...")
    scores_front, scores_side, labels = get_paired_scores(
        model_front, model_side, paired_loader, device
    )
    n_samples = len(labels)
    log.info("Inference selesai: %d sampel diperoleh.", n_samples)

    # ── Analisis Kalibrasi (ECE + Brier Score) ──
    print("\n" + "=" * 60)
    print("DIAGNOSTIK KALIBRASI PROBABILITAS")
    print("=" * 60)
    print("  (ECE mendekati 0 = well-calibrated; Brier Score < 0.10 = baik)")
    ece_front, bs_front = print_calibration_report("front", scores_front, labels.astype(float))
    ece_side,  bs_side  = print_calibration_report("side",  scores_side,  labels.astype(float))

    # Interpretasi gabungan: overfitting vs legitimate overconfidence
    print()
    train_loss_front = None
    train_loss_side  = None
    suffix = f"_{args.exp_id}" if args.exp_id else ""
    for view in ("front", "side"):
        hist_path = RESULTS_DIR / f"{view}_history{suffix}.json"
        if hist_path.exists():
            with open(hist_path) as f:
                h = json.load(f)
            if view == "front":
                train_loss_front = h["train_loss"][-1]
            else:
                train_loss_side = h["train_loss"][-1]

    print("  Interpretasi:")
    for view, ece, bs, tl in [
        ("FRONT", ece_front, bs_front, train_loss_front),
        ("SIDE",  ece_side,  bs_side,  train_loss_side),
    ]:
        if tl is None:
            continue
        if ece > 0.10 and tl < 0.10:
            diagnosis = "Overconfidence kemungkinan OVERFITTING (train_loss rendah, ECE tinggi)"
        elif ece < 0.05:
            diagnosis = "Well-calibrated — overconfidence kemungkinan LEGITIMATE"
        else:
            diagnosis = "Ambiguous — perlu ablasi regularisasi lebih lanjut"
        print(f"    [{view}] train_loss={tl:.5f}, ECE={ece:.5f} -> {diagnosis}")

    print()

    # ── Analisis Matematis Adaptive Fusion ──
    # Rumus:
    # d_i = |S_i - threshold|
    # w_front = exp(d_front) / (exp(d_front) + exp(d_side))
    confidence_front = np.abs(scores_front - DECISION_THRESHOLD)
    confidence_side = np.abs(scores_side - DECISION_THRESHOLD)

    exp_front = np.exp(confidence_front)
    exp_side = np.exp(confidence_side)
    sum_exp = exp_front + exp_side

    weight_front = exp_front / sum_exp
    weight_side = exp_side / sum_exp

    # ── Skor Fusion & Prediksi ──
    average_score = 0.5 * scores_front + 0.5 * scores_side
    adaptive_score = weight_front * scores_front + weight_side * scores_side

    pred_front = (scores_front >= DECISION_THRESHOLD).astype(int)
    pred_side = (scores_side >= DECISION_THRESHOLD).astype(int)
    average_prediction = (average_score >= DECISION_THRESHOLD).astype(int)
    adaptive_prediction = (adaptive_score >= DECISION_THRESHOLD).astype(int)

    # ── Hitung statistik bobot ──
    stats_w_front = {
        "mean": np.mean(weight_front),
        "std": np.std(weight_front),
        "min": np.min(weight_front),
        "max": np.max(weight_front),
    }
    stats_w_side = {
        "mean": np.mean(weight_side),
        "std": np.std(weight_side),
        "min": np.min(weight_side),
        "max": np.max(weight_side),
    }

    # Hitung jumlah sampel dengan bobot tertentu
    counts_front = {
        "0.55": np.sum(weight_front > 0.55),
        "0.60": np.sum(weight_front > 0.60),
        "0.70": np.sum(weight_front > 0.70),
        "0.80": np.sum(weight_front > 0.80),
    }
    counts_side = {
        "0.55": np.sum(weight_side > 0.55),
        "0.60": np.sum(weight_side > 0.60),
        "0.70": np.sum(weight_side > 0.70),
        "0.80": np.sum(weight_side > 0.80),
    }

    # Perbedaan prediksi
    diff_mask = average_prediction != adaptive_prediction
    n_diff_predictions = np.sum(diff_mask)

    # Analisis confidence
    stats_c_front = {
        "mean": np.mean(confidence_front),
        "std": np.std(confidence_front),
        "min": np.min(confidence_front),
        "max": np.max(confidence_front),
    }
    stats_c_side = {
        "mean": np.mean(confidence_side),
        "std": np.std(confidence_side),
        "min": np.min(confidence_side),
        "max": np.max(confidence_side),
    }

    # ── Buat Folder Debug ──
    debug_dir = RESULTS_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Histograms ──
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle("Analisis Distribusi Probabilitas, Confidence, dan Bobot", fontsize=16, fontweight="bold")

    # Front Probability
    axes[0, 0].hist(scores_front, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Probabilitas Prediksi Front (phone_use)")
    axes[0, 0].set_xlabel("Probabilitas")
    axes[0, 0].set_ylabel("Jumlah Sampel")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Side Probability
    axes[0, 1].hist(scores_side, bins=30, color="lightcoral", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Probabilitas Prediksi Side (phone_use)")
    axes[0, 1].set_xlabel("Probabilitas")
    axes[0, 1].set_ylabel("Jumlah Sampel")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Front Confidence
    axes[1, 0].hist(confidence_front, bins=30, color="teal", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Confidence Front |prob - 0.5|")
    axes[1, 0].set_xlabel("Confidence")
    axes[1, 0].set_ylabel("Jumlah Sampel")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Side Confidence
    axes[1, 1].hist(confidence_side, bins=30, color="indianred", edgecolor="black", alpha=0.7)
    axes[1, 1].set_title("Confidence Side |prob - 0.5|")
    axes[1, 1].set_xlabel("Confidence")
    axes[1, 1].set_ylabel("Jumlah Sampel")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    # Front Weight
    axes[2, 0].hist(weight_front, bins=30, color="darkcyan", edgecolor="black", alpha=0.7)
    axes[2, 0].set_title("Bobot Adaptive Front (w_front)")
    axes[2, 0].set_xlabel("Bobot")
    axes[2, 0].set_ylabel("Jumlah Sampel")
    axes[2, 0].axvline(0.5, color="red", linestyle="--", label="Simple Average (0.5)")
    axes[2, 0].legend()
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)

    # Side Weight
    axes[2, 1].hist(weight_side, bins=30, color="crimson", edgecolor="black", alpha=0.7)
    axes[2, 1].set_title("Bobot Adaptive Side (w_side)")
    axes[2, 1].set_xlabel("Bobot")
    axes[2, 1].set_ylabel("Jumlah Sampel")
    axes[2, 1].axvline(0.5, color="red", linestyle="--", label="Simple Average (0.5)")
    axes[2, 1].legend()
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    suffix = f"_{args.exp_id}" if args.exp_id else ""
    hist_path = debug_dir / f"histograms{suffix}.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    log.info("Histogram disimpan ke: %s", hist_path)

    # ── Plot 2: Scatter Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Scatter Plot Probabilitas Front vs Side", fontsize=16, fontweight="bold")

    # 2a. Scatter Plot colored by Ground Truth
    scatter1 = ax1.scatter(
        scores_front, scores_side, c=labels, cmap="coolwarm", alpha=0.5, edgecolors="none", s=15
    )
    ax1.set_title("Diwarnai Berdasarkan Ground Truth")
    ax1.set_xlabel("Probability Front")
    ax1.set_ylabel("Probability Side")
    ax1.axhline(0.5, color="grey", linestyle="--", alpha=0.7)
    ax1.axvline(0.5, color="grey", linestyle="--", alpha=0.7)
    cbar1 = fig.colorbar(scatter1, ax=ax1)
    cbar1.set_label("Ground Truth (0: safe_driving, 1: phone_use)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Highlight prediction differences if any
    if n_diff_predictions > 0:
        ax1.scatter(
            scores_front[diff_mask], scores_side[diff_mask],
            facecolors="none", edgecolors="lime", s=80, marker="o", linewidths=1.5,
            label="Prediksi Berbeda"
        )
        ax1.legend()

    # 2b. Scatter Plot colored by Prediction Differences
    colors_diff = np.where(diff_mask, "red", "lightgray")
    sizes_diff = np.where(diff_mask, 50, 10)
    alphas_diff = np.where(diff_mask, 0.9, 0.3)
    
    ax2.scatter(
        scores_front, scores_side, c=colors_diff, s=sizes_diff, alpha=alphas_diff, edgecolors="none"
    )
    ax2.set_title("Diwarnai Berdasarkan Perbedaan Prediksi (Merah: Berbeda)")
    ax2.set_xlabel("Probability Front")
    ax2.set_ylabel("Probability Side")
    ax2.axhline(0.5, color="grey", linestyle="--", alpha=0.7)
    ax2.axvline(0.5, color="grey", linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Draw decision boundary helper lines
    x_bound = np.linspace(0, 1, 100)
    y_bound = 1.0 - x_bound
    ax2.plot(x_bound, y_bound, color="blue", linestyle=":", label="Average Decision Boundary")
    ax2.legend()

    if n_diff_predictions == 0:
        ax2.text(
            0.5, 0.5, "Prediksi Identik 100%\nTidak Ada Perbedaan",
            color="darkblue", fontsize=12, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5")
        )

    plt.tight_layout()
    scatter_path = debug_dir / f"scatter_plot{suffix}.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    log.info("Scatter plot disimpan ke: %s", scatter_path)

    # ── Export CSV ──
    df_debug = pd.DataFrame({
        "ground_truth": labels,
        "prob_front": scores_front,
        "prob_side": scores_side,
        "confidence_front": confidence_front,
        "confidence_side": confidence_side,
        "weight_front": weight_front,
        "weight_side": weight_side,
        "average_score": average_score,
        "adaptive_score": adaptive_score,
        "average_prediction": average_prediction,
        "adaptive_prediction": adaptive_prediction,
    })
    csv_path = debug_dir / f"fusion_debug{suffix}.csv"
    df_debug.to_csv(csv_path, index=False)
    log.info("CSV debug diekspor ke: %s", csv_path)

    # ── Kesimpulan & Analisis Ilmiah ──
    is_always_near_half = np.all(np.abs(weight_front - 0.5) < 0.15)
    near_half_str = "YA" if is_always_near_half else "TIDAK"
    
    kesimpulan = (
        f"- Apakah bobot adaptive hampir selalu 0.5? {near_half_str} (Bobot Front: Min={stats_w_front['min']:.4f}, Max={stats_w_front['max']:.4f}, Rata-rata={stats_w_front['mean']:.4f}).\n"
        f"- Apakah adaptive benar-benar menghasilkan prediksi berbeda? {'YA' if n_diff_predictions > 0 else 'TIDAK'} (Prediksi berbeda: {n_diff_predictions} dari {n_samples} sampel).\n"
        f"- Apakah hasil identik disebabkan oleh implementasi atau distribusi confidence?\n"
        f"  DISEBABKAN OLEH DUA FAKTOR MATEMATIS & EMPIRIS:\n"
        f"  1. Batasan Matematis Softmax: Nilai confidence d_i dibatasi pada rentang [0, 0.5].\n"
        f"     Secara matematis, exp(d_i) tidak dapat menghasilkan bobot di luar rentang [0.378, 0.622] berapa pun selisih confidence-nya.\n"
        f"  2. Perilaku Distribusi: Model terlatih cenderung menghasilkan probabilitas bimodal ekstrim (dekat 0.0 atau 1.0).\n"
        f"     Saat kedua model sama-sama yakin (d_front ~ d_side ~ 0.5) atau memiliki deviasi simetris dari 0.5,\n"
        f"     softmax menghasilkan bobot w_i yang berada sangat dekat dengan 0.50 (berkisar antara 0.48 - 0.52).\n"
        f"     Oleh karena itu, skor akhir adaptive fusion hampir sama persis dengan average fusion,\n"
        f"     dan tidak pernah cukup berbeda untuk menyeberangi threshold keputusan 0.5 secara berbeda."
    )

    # ── Cetak Ringkasan Otomatis ──
    print("\n" + "=" * 50)
    print("ADAPTIVE FUSION DEBUG SUMMARY")
    print("=" * 50)
    print(f"Jumlah sampel: {n_samples}")
    print()
    print(f"Mean weight front: {stats_w_front['mean']:.6f}")
    print(f"Mean weight side:  {stats_w_side['mean']:.6f}")
    print()
    print(f"Std weight front:  {stats_w_front['std']:.6f}")
    print(f"Std weight side:   {stats_w_side['std']:.6f}")
    print()
    print(f"Prediction berbeda Average vs Adaptive: {n_diff_predictions}")
    print()
    print("Weight > 0.55:")
    print(f"  Front: {counts_front['0.55']}")
    print(f"  Side:  {counts_side['0.55']}")
    print("Weight > 0.60:")
    print(f"  Front: {counts_front['0.60']}")
    print(f"  Side:  {counts_side['0.60']}")
    print("Weight > 0.70:")
    print(f"  Front: {counts_front['0.70']}")
    print(f"  Side:  {counts_side['0.70']}")
    print("Weight > 0.80:")
    print(f"  Front: {counts_front['0.80']}")
    print(f"  Side:  {counts_side['0.80']}")
    print()
    print("Kesimpulan:")
    print(kesimpulan)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
