import json
import logging
import time

import torch
import torch.nn as nn

from src.config import BINARY_LABEL_MAP, LR_SCHEDULER_FACTOR, SPLIT_SEED
from src.dataset import get_all_dataloaders, load_split_dataframe
from src.experiment_21C.config_experiment_21C import CHECKPOINTS_DIR_EXP21C, FRAME_STRIDE_EXP21C, MAX_EPOCHS_EXP21C, RESULTS_DIR_EXP21C, SIDE_CONFIG
from src.model import build_model
from src.train import seed_everything, train_one_epoch, validate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class MinEarlyStopping:
    """Early stopping for metrics where lower is better."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score < self.best_score:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def _class_weights_from_config() -> list[float]:
    raw = SIDE_CONFIG["class_weights"]
    if raw != "balanced":
        return raw
    df_train = load_split_dataframe("side", "train", frame_stride=FRAME_STRIDE_EXP21C)
    counts = {
        label_id: int((df_train["binary_label"] == label_name).sum())
        for label_name, label_id in BINARY_LABEL_MAP.items()
    }
    total = sum(counts.values())
    return [total / (len(counts) * counts[i]) for i in range(len(counts))]


def train_side_exp21C():
    seed_everything(SPLIT_SEED)
    CHECKPOINTS_DIR_EXP21C.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_EXP21C.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    loaders = get_all_dataloaders("side", frame_stride=FRAME_STRIDE_EXP21C)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    model = build_model(
        pretrained=True,
        num_stages_to_freeze=SIDE_CONFIG["num_stages_to_freeze"],
        dropout_rate=SIDE_CONFIG["dropout"],
    ).to(device)
    class_weights = torch.tensor(_class_weights_from_config(), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=SIDE_CONFIG["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=SIDE_CONFIG["learning_rate"], weight_decay=SIDE_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=SIDE_CONFIG["lr_scheduler_patience"],
    )
    early_stopping = MinEarlyStopping(SIDE_CONFIG["early_stopping_patience"])

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
    best_epoch = 0
    start = time.time()

    log.info("=" * 65)
    log.info("Training Experiment 21C side | checkpoint monitor=val_loss | patience=%d", SIDE_CONFIG["early_stopping_patience"])
    log.info("=" * 65)

    for epoch in range(1, MAX_EPOCHS_EXP21C + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_acc, val_precision, val_recall = validate(model, val_loader, criterion, device)
        loss_gap = val_loss - train_loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_loss, 5))
        history["train_accuracy"].append(round(train_acc, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_macro_f1"].append(round(val_f1, 5))
        history["val_accuracy"].append(round(val_acc, 5))
        history["val_precision_macro"].append(round(val_precision, 5))
        history["val_recall_macro"].append(round(val_recall, 5))
        history["train_val_loss_gap"].append(round(loss_gap, 5))
        history["lr"].append(current_lr)

        is_best = early_stopping.step(val_loss)
        if is_best:
            best_epoch = epoch
            torch.save(
                {
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
                    "view": "side",
                    "experiment": SIDE_CONFIG["experiment"],
                    "class_weights": class_weights.cpu().tolist(),
                    "num_stages_to_freeze": SIDE_CONFIG["num_stages_to_freeze"],
                    "checkpoint_monitor": SIDE_CONFIG["checkpoint_monitor"],
                    "hyperparameters": {
                        "learning_rate": SIDE_CONFIG["learning_rate"],
                        "weight_decay": SIDE_CONFIG["weight_decay"],
                        "dropout": SIDE_CONFIG["dropout"],
                        "label_smoothing": SIDE_CONFIG["label_smoothing"],
                        "num_stages_to_freeze": SIDE_CONFIG["num_stages_to_freeze"],
                        "early_stopping_patience": SIDE_CONFIG["early_stopping_patience"],
                        "lr_scheduler_factor": LR_SCHEDULER_FACTOR,
                        "lr_scheduler_patience": SIDE_CONFIG["lr_scheduler_patience"],
                        "class_weights": class_weights.cpu().tolist(),
                        "checkpoint_monitor": SIDE_CONFIG["checkpoint_monitor"],
                    },
                },
                SIDE_CONFIG["checkpoint_path"],
            )

        status = "* BEST_LOSS" if is_best else f"  wait {early_stopping.counter}/{early_stopping.patience}"
        log.info(
            (
                "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | gap=%.4f | "
                "train_acc=%.4f | val_acc=%.4f | val_prec=%.4f | val_rec=%.4f | "
                "val_F1=%.4f | lr=%.1e | best_epoch=%d | %s | %.0fs"
            ),
            epoch,
            MAX_EPOCHS_EXP21C,
            train_loss,
            val_loss,
            loss_gap,
            train_acc,
            val_acc,
            val_precision,
            val_recall,
            val_f1,
            current_lr,
            best_epoch,
            status,
            time.time() - epoch_start,
        )
        if early_stopping.should_stop:
            log.info("Early stopping triggered at epoch %d. Best epoch: %d (val_loss=%.4f)", epoch, best_epoch, early_stopping.best_score)
            break

    SIDE_CONFIG["history_path"].write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_idx = min(range(len(history["val_loss"])), key=lambda idx: history["val_loss"][idx])
    summary = {
        "experiment": SIDE_CONFIG["experiment"],
        "view": "side",
        "checkpoint_path": str(SIDE_CONFIG["checkpoint_path"]),
        "history_path": str(SIDE_CONFIG["history_path"]),
        "hyperparameters": {
            "backbone": "EfficientNetV2-S",
            "optimizer": "AdamW",
            "learning_rate": SIDE_CONFIG["learning_rate"],
            "weight_decay": SIDE_CONFIG["weight_decay"],
            "dropout": SIDE_CONFIG["dropout"],
            "label_smoothing": SIDE_CONFIG["label_smoothing"],
            "scheduler": "ReduceLROnPlateau",
            "early_stopping": True,
            "early_stopping_patience": SIDE_CONFIG["early_stopping_patience"],
            "num_stages_to_freeze": SIDE_CONFIG["num_stages_to_freeze"],
            "metric_for_best_model": "validation loss",
            "checkpoint_monitor": SIDE_CONFIG["checkpoint_monitor"],
        },
        "best_epoch": best_idx + 1,
        "best_validation_loss": history["val_loss"][best_idx],
        "best_validation_macro_f1": history["val_macro_f1"][best_idx],
        "train_loss_at_best_epoch": history["train_loss"][best_idx],
        "validation_loss_at_best_epoch": history["val_loss"][best_idx],
        "train_validation_loss_gap_at_best_epoch": history["train_val_loss_gap"][best_idx],
        "test_metrics": None,
        "training_minutes": round((time.time() - start) / 60, 2),
    }
    SIDE_CONFIG["summary_path"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return history


def main():
    train_side_exp21C()
    print(json.dumps({"side_checkpoint": str(SIDE_CONFIG["checkpoint_path"]), "results_dir": str(RESULTS_DIR_EXP21C)}, indent=2))


if __name__ == "__main__":
    main()
