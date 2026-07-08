# %% [markdown]
# # Evaluasi Model Single-View
# Hitung: accuracy, precision, recall, Macro F1-Score, confusion matrix.
# Simpan hasil ke `results/{view}_test_metrics.json`.
#
# **Penggunaan CLI:**
# ```
# python -m src.evaluate --view front
# python -m src.evaluate --view side
# python -m src.evaluate   # evaluasi kedua view sekaligus
# ```

# %%
# Imports & setup logging
"""
Evaluasi model single-view pada test set.

Hitung: accuracy, precision, recall, Macro F1-Score, confusion matrix.
Simpan hasil ke results/{view}_test_metrics.json.

Penggunaan:
  python -m src.evaluate --view front
  python -m src.evaluate --view side
  python -m src.evaluate                # evaluasi kedua view sekaligus
"""

import argparse
import json
import logging

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.config import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    DECISION_THRESHOLD,
    BINARY_LABEL_MAP,
)
from src.dataset import get_dataloader
from src.model import build_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Nama kelas untuk laporan (urut sesuai label 0, 1)
CLASS_NAMES = [k for k, v in sorted(BINARY_LABEL_MAP.items(), key=lambda x: x[1])]
print("Class names:", CLASS_NAMES)


# %%
# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def load_trained_model(view: str, device: torch.device, exp_id: str = "", dropout_rate: float = None):
    """Muat checkpoint terbaik untuk view tertentu."""
    suffix = f"_{exp_id}" if exp_id else ""
    ckpt_path = CHECKPOINT_DIR / f"{view}{suffix}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt_path}")

    if dropout_rate is None:
        from src.config import DROPOUT_FRONT, DROPOUT_SIDE
        dropout_rate = DROPOUT_FRONT if view == "front" else DROPOUT_SIDE

    from src.config import NUM_STAGES_TO_FREEZE_FRONT, NUM_STAGES_TO_FREEZE_SIDE
    freeze_stages = NUM_STAGES_TO_FREEZE_FRONT if view == "front" else NUM_STAGES_TO_FREEZE_SIDE
    model = build_model(pretrained=False, num_stages_to_freeze=freeze_stages, dropout_rate=dropout_rate)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    log.info("Loaded %s checkpoint (%s): epoch=%d, val_F1=%.4f",
             view, exp_id if exp_id else "default", checkpoint["epoch"], checkpoint["val_macro_f1"])
    return model, checkpoint


@torch.no_grad()
def predict_test(model, test_loader, device, threshold=DECISION_THRESHOLD):
    """Jalankan inferensi pada test set. Return (labels, preds, probs)."""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        
        # Softmax probability untuk kelas positif (phone_use, indeks 1)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long().cpu()

        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return all_labels, all_preds, all_probs



def compute_metrics(labels, preds):
    """Hitung semua metrik evaluasi."""
    cm = confusion_matrix(labels, preds).tolist()
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": round(accuracy_score(labels, preds), 5),
        "precision_macro": round(precision_score(labels, preds, average="macro", zero_division=0), 5),
        "recall_macro": round(recall_score(labels, preds, average="macro", zero_division=0), 5),
        "f1_macro": round(f1_score(labels, preds, average="macro", zero_division=0), 5),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def print_metrics(view_or_method: str, metrics: dict):
    """Cetak metrik evaluasi ke console."""
    cm = metrics["confusion_matrix"]
    log.info("=" * 55)
    log.info("TEST RESULTS: %s", view_or_method.upper())
    log.info("=" * 55)
    log.info("  Accuracy : %.4f", metrics["accuracy"])
    log.info("  Precision: %.4f (macro)", metrics["precision_macro"])
    log.info("  Recall   : %.4f (macro)", metrics["recall_macro"])
    log.info("  F1-Score : %.4f (macro)", metrics["f1_macro"])
    log.info("  Confusion Matrix:")
    log.info("              Pred:safe  Pred:phone")
    log.info("  True:safe   %6d     %6d", cm[0][0], cm[0][1])
    log.info("  True:phone  %6d     %6d", cm[1][0], cm[1][1])
    log.info("-" * 55)


# %%
# ──────────────────────────────────────────────────────────────────────────────
# Evaluasi satu view
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_view(view: str, exp_id: str = "") -> dict:
    """Evaluasi model single-view pada test set, simpan hasil."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & test data
    from src.config import DROPOUT_FRONT, DROPOUT_SIDE
    dropout_rate = DROPOUT_FRONT if view == "front" else DROPOUT_SIDE
    model, checkpoint = load_trained_model(view, device, exp_id=exp_id, dropout_rate=dropout_rate)
    test_loader = get_dataloader(view, "test")

    # Predict & compute metrics
    labels, preds, probs = predict_test(model, test_loader, device)
    metrics = compute_metrics(labels, preds)

    # Tambah info checkpoint
    metrics["view"] = view
    metrics["best_epoch"] = checkpoint["epoch"]
    metrics["val_macro_f1"] = round(checkpoint["val_macro_f1"], 5)

    # Print
    print_metrics(f"{view} view", metrics)

    # Simpan
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{exp_id}" if exp_id else ""
    metrics_path = RESULTS_DIR / f"{view}_test_metrics{suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved to: %s", metrics_path)

    return metrics


# %%
# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point  (jalankan sel ini untuk evaluasi interaktif)
# ──────────────────────────────────────────────────────────────────────────────
# Atau gunakan langsung tanpa CLI:
#   evaluate_view("front")
#   evaluate_view("side")

def main():
    parser = argparse.ArgumentParser(description="Evaluasi single-view pada test set")
    parser.add_argument("--view", choices=["front", "side"],
                        help="View yang dievaluasi. Kosongkan untuk evaluasi keduanya.")
    parser.add_argument("--exp_id", type=str, default="",
                        help="ID Eksperimen (opsional, misal 'exp3')")
    args = parser.parse_args()

    if args.view:
        evaluate_view(args.view, exp_id=args.exp_id)
    else:
        # Evaluasi kedua view
        for v in ("front", "side"):
            evaluate_view(v, exp_id=args.exp_id)


if __name__ == "__main__":
    main()
