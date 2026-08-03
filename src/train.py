# %% [markdown]
# # Training Loop — Single-View EfficientNetV2-S
# Model: EfficientNetV2-S (binary classification: safe driving vs phone use)
#
# **Penggunaan CLI:**
# ```
# python -m src.train --view front
# python -m src.train --view side
# python -m src.train --view front --epochs 50
# ```

# %%
# Imports & setup logging
"""
Training loop untuk model single-view EfficientNetV2-S (klasifikasi biner).

Fitur:
  - BCEWithLogitsLoss + Adam optimizer
  - Validasi per-epoch dengan Macro F1-Score sebagai metrik utama
  - Early stopping berdasarkan val Macro F1 (patience dari config)
  - Simpan checkpoint terbaik ke checkpoints/{view}_best.pt
  - Simpan history training ke results/{view}_history.json

Penggunaan:
  python -m src.train --view front
  python -m src.train --view side
  python -m src.train --view front --epochs 50
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import (
    LEARNING_RATE_FRONT,
    LEARNING_RATE_SIDE,
    EARLY_STOPPING_PATIENCE_FRONT,
    EARLY_STOPPING_PATIENCE_SIDE,
    MAX_EPOCHS,
    DECISION_THRESHOLD,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    WEIGHT_DECAY,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE_FRONT,
    LR_SCHEDULER_PATIENCE_SIDE,
    BINARY_LABEL_MAP,
    NUM_STAGES_TO_FREEZE_FRONT,
    NUM_STAGES_TO_FREEZE_SIDE,
    LABEL_SMOOTHING,
    CLASS_WEIGHTS,
    DROPOUT_FRONT,
    DROPOUT_SIDE,
    SPLIT_SEED,
    EXPERIMENT_CONFIGS,
    FRAME_STRIDE,
)
from src.dataset import get_all_dataloaders, load_split_dataframe
from src.model import build_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def resolve_class_weights(view: str, exp_config, class_weight_gamma: float = 1.0) -> list[float]:
    """Resolve class weights, optionally computing balanced weights from train data."""
    raw_class_weights = exp_config["class_weights"] if exp_config else CLASS_WEIGHTS

    if raw_class_weights != "balanced":
        return raw_class_weights

    df_train = load_split_dataframe(view, "train")
    counts = {
        label_id: int((df_train["binary_label"] == label_name).sum())
        for label_name, label_id in BINARY_LABEL_MAP.items()
    }
    total = sum(counts.values())
    n_classes = len(counts)
    if total == 0 or any(v == 0 for v in counts.values()):
        raise ValueError("Distribusi label train tidak valid untuk menghitung balanced class weights.")

    weights_balanced = [total / (n_classes * counts[i]) for i in range(n_classes)]
    weights = [float(w ** class_weight_gamma) for w in weights_balanced]
    log.info(
        "Balanced class weights dihitung dari train split: safe_driving=%d, phone_use=%d",
        counts[0], counts[1],
    )
    log.info(
        "Balanced class weights sebelum gamma: safe_driving=%.4f, phone_use=%.4f",
        weights_balanced[0], weights_balanced[1],
    )
    log.info(
        "Class weight gamma: %.2f -> weights final: safe_driving=%.4f, phone_use=%.4f",
        class_weight_gamma, weights[0], weights[1],
    )
    return weights


def seed_everything(seed=42):
    """Kunci semua random seed agar eksperimen dapat direproduksi sepenuhnya."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"Random seed dikunci pada: {seed}")


# %%
# Training & validation satu epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Jalankan satu epoch training. Return rata-rata loss dan accuracy."""
    model.train()
    running_loss = 0.0
    n_samples = 0
    n_correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.long().to(device)  # target int untuk CrossEntropy

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        n_correct += (preds == labels).sum().item()
        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return running_loss / n_samples, n_correct / n_samples


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=DECISION_THRESHOLD):
    """Jalankan validasi. Return loss, F1, accuracy, precision, dan recall."""
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels_dev = labels.long().to(device)  # target int untuk CrossEntropy

        logits = model(images)
        loss = criterion(logits, labels_dev)

        # Softmax probability untuk kelas positif (phone_use, indeks 1)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long().cpu()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    avg_loss = running_loss / n_samples
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    return avg_loss, macro_f1, accuracy, precision, recall



# %%
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Early stopping berdasarkan val Macro F1-Score (higher is better)."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = -1.0
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Return True jika skor ini adalah yang terbaik (harus simpan model)."""
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return True     # skor membaik → simpan checkpoint
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False        # skor tidak membaik


# %%
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def run_training(view: str, max_epochs: int = MAX_EPOCHS, exp_id: str = "",
                 lr: float = None, experiment: str = "", class_weight_gamma: float = 1.0,
                 checkpoint_path: str = None, history_path: str = None, summary_path: str = None,
                 frame_stride: int = FRAME_STRIDE):
    """Latih model single-view dan simpan checkpoint + history."""
    seed_everything(SPLIT_SEED)

    exp_config = None
    if experiment:
        if experiment not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Experiment tidak dikenal: {experiment}. Pilihan: {list(EXPERIMENT_CONFIGS)}")
        exp_config = EXPERIMENT_CONFIGS[experiment]
        if view != exp_config["view"]:
            raise ValueError(
                f"{experiment} hanya untuk view='{exp_config['view']}'. "
                "Side view tetap memakai checkpoint Experiment 13 dan tidak dilatih ulang."
            )

    default_lr = (
        exp_config["learning_rate"] if exp_config
        else (LEARNING_RATE_FRONT if view == "front" else LEARNING_RATE_SIDE)
    )
    is_override = (lr is not None)
    if lr is None:
        lr = default_lr

    freeze_stages = (
        exp_config["num_stages_to_freeze"] if exp_config
        else (NUM_STAGES_TO_FREEZE_FRONT if view == "front" else NUM_STAGES_TO_FREEZE_SIDE)
    )
    early_stopping_patience = (
        exp_config["early_stopping_patience"] if exp_config
        else (EARLY_STOPPING_PATIENCE_FRONT if view == "front" else EARLY_STOPPING_PATIENCE_SIDE)
    )
    lr_scheduler_patience = (
        exp_config["lr_scheduler_patience"] if exp_config
        else (LR_SCHEDULER_PATIENCE_FRONT if view == "front" else LR_SCHEDULER_PATIENCE_SIDE)
    )
    dropout_rate = (
        exp_config["dropout"] if exp_config
        else (DROPOUT_FRONT if view == "front" else DROPOUT_SIDE)
    )
    weight_decay = exp_config["weight_decay"] if exp_config else WEIGHT_DECAY
    label_smoothing = exp_config["label_smoothing"] if exp_config else LABEL_SMOOTHING
    class_weights_values = resolve_class_weights(view, exp_config, class_weight_gamma=class_weight_gamma)

    # ── Tampilkan Ringkasan Konfigurasi Eksperimen ──
    log.info("\n" + "=" * 45)
    log.info("Experiment Configuration")
    log.info("=" * 45)
    log.info(f"  View           : {view}")
    log.info(f"  Optimizer      : AdamW")
    log.info(f"  Learning Rate  : {lr:.1e}" + (" (override)" if is_override else ""))
    log.info(f"  Weight Decay   : {weight_decay:.1e}")
    log.info(f"  Scheduler      : ReduceLROnPlateau")
    log.info(f"  Dropout        : {dropout_rate}")
    log.info(f"  Label Smoothing: {label_smoothing}")
    log.info(f"  ClassWeight γ  : {class_weight_gamma:.2f}")
    log.info(f"  Frozen Stages  : {freeze_stages}")
    log.info(f"  EarlyStopping  : patience={early_stopping_patience}")
    log.info(f"  Frame Stride   : {frame_stride}")
    log.info(f"  Experiment     : {experiment if experiment else 'default'}")
    log.info(f"  Experiment ID  : {exp_id if exp_id else 'None'}")
    log.info("=" * 45 + "\n")

    # ── Setup device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── DataLoaders ──
    loaders = get_all_dataloaders(view, frame_stride=frame_stride)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # ── Model, loss, optimizer ──
    model = build_model(pretrained=True, num_stages_to_freeze=freeze_stages, dropout_rate=dropout_rate).to(device)

    # ── Load static class weights dari config ──
    class_weights = torch.tensor(class_weights_values, dtype=torch.float).to(device)
    log.info(
        "Class weights (CrossEntropyLoss): safe_driving=%.4f, phone_use=%.4f",
        class_weights_values[0], class_weights_values[1],
    )
    log.info("Label smoothing: %.2f", label_smoothing)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── Verifikasi Weight Decay ──
    for i, g in enumerate(optimizer.param_groups):
        log.info(f"INFO: [VERIFIKASI] param_group[{i}] weight_decay aktual = {g['weight_decay']}")

    # ── Learning Rate Scheduler ──
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_SCHEDULER_FACTOR, patience=lr_scheduler_patience
    )

    # ── Early stopping ──
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    # ── Paths ──
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        history_out = Path(history_path) if history_path else (RESULTS_DIR / f"{view}_history_custom.json")
        if not history_out.is_absolute():
            history_out = Path.cwd() / history_out
        summary_out = Path(summary_path) if summary_path else None
        if summary_out is not None and not summary_out.is_absolute():
            summary_out = Path.cwd() / summary_out
        history_path = history_out
        summary_path = summary_out
    elif exp_config:
        ckpt_path = CHECKPOINT_DIR / exp_config["checkpoint_name"]
        history_path = RESULTS_DIR / exp_config["history_name"]
        summary_path = RESULTS_DIR / exp_config["summary_name"]
    elif exp_id:
        ckpt_path = CHECKPOINT_DIR / f"{view}_{exp_id}_best.pt"
        history_path = RESULTS_DIR / f"{view}_history_{exp_id}.json"
        summary_path = None
    else:
        ckpt_path = CHECKPOINT_DIR / f"{view}_best.pt"
        history_path = RESULTS_DIR / f"{view}_history.json"
        summary_path = None

    # ── History ──
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_accuracy": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "train_val_loss_gap": [],
        "lr": [],
    }

    log.info("=" * 65)
    log.info("Training: view=%s | max_epochs=%d | patience=%d", view, max_epochs, early_stopping_patience)
    log.info("=" * 65)

    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        t_epoch = time.time()

        # ── Train ──
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # ── Validate ──
        val_loss, val_f1, val_acc, val_precision, val_recall = validate(model, val_loader, criterion, device)
        loss_gap = val_loss - train_loss

        # ── Step Scheduler ──
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Record ──
        history["train_loss"].append(round(train_loss, 5))
        history["train_accuracy"].append(round(train_acc, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_macro_f1"].append(round(val_f1, 5))
        history["val_accuracy"].append(round(val_acc, 5))
        history["val_precision_macro"].append(round(val_precision, 5))
        history["val_recall_macro"].append(round(val_recall, 5))
        history["train_val_loss_gap"].append(round(loss_gap, 5))
        history["lr"].append(current_lr)

        # ── Early stopping check ──
        is_best = early_stopping.step(val_f1)
        if is_best:
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "val_precision_macro": val_precision,
                "val_recall_macro": val_recall,
                "train_val_loss_gap": loss_gap,
                "view": view,
                "experiment": experiment,
                "class_weights": class_weights.cpu().tolist(),
                "num_stages_to_freeze": freeze_stages,
                "hyperparameters": {
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "dropout": dropout_rate,
                    "label_smoothing": label_smoothing,
                    "num_stages_to_freeze": freeze_stages,
                    "early_stopping_patience": early_stopping_patience,
                    "lr_scheduler_factor": LR_SCHEDULER_FACTOR,
                    "lr_scheduler_patience": lr_scheduler_patience,
                    "class_weights": class_weights.cpu().tolist(),
                    "class_weight_gamma": class_weight_gamma,
                    "frame_stride": frame_stride,
                },
            }, ckpt_path)

        elapsed = time.time() - t_epoch
        status = "* BEST" if is_best else f"  wait {early_stopping.counter}/{early_stopping.patience}"
        log.info(
            (
                "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | gap=%.4f | "
                "train_acc=%.4f | val_acc=%.4f | val_prec=%.4f | val_rec=%.4f | "
                "val_F1=%.4f | lr=%.1e | best_epoch=%d | %s | %.0fs"
            ),
            epoch, max_epochs, train_loss, val_loss, loss_gap,
            train_acc, val_acc, val_precision, val_recall,
            val_f1, current_lr, best_epoch, status, elapsed,
        )

        if early_stopping.should_stop:
            log.info("Early stopping triggered at epoch %d. Best epoch: %d (val_F1=%.4f)",
                     epoch, best_epoch, early_stopping.best_score)
            break

    total_time = time.time() - t_start
    log.info("=" * 65)
    log.info("Training selesai dalam %.1f menit. Best epoch: %d (val_F1=%.4f)",
             total_time / 60, best_epoch, early_stopping.best_score)
    log.info("Checkpoint tersimpan di: %s", ckpt_path)

    # ── Simpan history ──
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("History tersimpan di: %s", history_path)

    if summary_path is not None:
        best_idx = max(range(len(history["val_macro_f1"])), key=lambda i: history["val_macro_f1"][i])
        summary = {
            "experiment": experiment,
            "view": view,
            "checkpoint_path": str(ckpt_path),
            "history_path": str(history_path),
            "hyperparameters": {
                "backbone": "EfficientNetV2-S",
                "optimizer": "AdamW",
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "dropout": dropout_rate,
                "label_smoothing": label_smoothing,
                "class_weight_gamma": class_weight_gamma,
                "scheduler": "ReduceLROnPlateau",
                "early_stopping": True,
                "early_stopping_patience": early_stopping_patience,
                "num_stages_to_freeze": freeze_stages,
                "metric_for_best_model": "validation Macro F1",
                "frame_stride": frame_stride,
            },
            "best_epoch": best_idx + 1,
            "best_validation_macro_f1": history["val_macro_f1"][best_idx],
            "train_loss_at_best_epoch": history["train_loss"][best_idx],
            "validation_loss_at_best_epoch": history["val_loss"][best_idx],
            "train_validation_loss_gap_at_best_epoch": history["train_val_loss_gap"][best_idx],
            "test_metrics": None,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Ringkasan awal eksperimen tersimpan di: %s", summary_path)

    return history


# %%
# CLI entry point (jalankan sel ini untuk mulai training)
# ──────────────────────────────────────────────────────────────────────────────
# Atau jalankan langsung tanpa CLI:
#   run_training(view="front", max_epochs=30)

def main():
    parser = argparse.ArgumentParser(description="Training single-view EfficientNetV2-S")
    parser.add_argument("--view", required=True, choices=["front", "side"],
                        help="Sudut pandang yang akan dilatih: 'front' atau 'side'")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                        help=f"Jumlah maksimum epoch (default: {MAX_EPOCHS})")
    parser.add_argument("--exp_id", type=str, default="",
                        help="ID Eksperimen (opsional, misal 'lr1e5' untuk suffix file)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate override (default: Front=5e-5, Side=2e-5 dari config.py). "
                             "Contoh: --lr 1e-5")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_CONFIGS.keys()), default="",
                        help="Preset eksperimen khusus, misalnya experiment_14A untuk front-only.")
    parser.add_argument("--class-weight-gamma", type=float, default=1.0,
                        help="Gamma untuk balanced class weight. Hanya berpengaruh bila preset memakai class_weights='balanced'.")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path checkpoint output eksplisit, misal checkpoints/side_best_exp19_gamma025.pt")
    parser.add_argument("--history-path", type=str, default=None,
                        help="Path history output eksplisit, misal results/side_history_exp19_gamma025.json")
    parser.add_argument("--summary-path", type=str, default=None,
                        help="Path summary output eksplisit, misal results/experiment_19_side_metrics_gamma025.json")
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE,
                        help=f"Frame subsampling stride untuk train/val/test (default: {FRAME_STRIDE})")
    args = parser.parse_args()

    run_training(
        view=args.view,
        max_epochs=args.epochs,
        exp_id=args.exp_id,
        lr=args.lr,
        experiment=args.experiment,
        class_weight_gamma=args.class_weight_gamma,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
        summary_path=args.summary_path,
        frame_stride=args.frame_stride,
    )


if __name__ == "__main__":
    main()
