import json
import logging

from src.final_v1.config_final_v1 import CHECKPOINTS_DIR_FINAL, FRONT_CONFIG, RESULTS_DIR_FINAL, SIDE_CONFIG
from src.train import run_training

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def train_front_final_v1():
    """Train final front model using the locked Front14A revised configuration."""
    CHECKPOINTS_DIR_FINAL.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_FINAL.mkdir(parents=True, exist_ok=True)

    run_training(
        view="front",
        max_epochs=FRONT_CONFIG["max_epochs"],
        experiment="experiment_14A",
        checkpoint_path=str(FRONT_CONFIG["checkpoint_path"]),
        history_path=str(FRONT_CONFIG["history_path"]),
        summary_path=str(RESULTS_DIR_FINAL / "front_train_summary_final_v1.json"),
    )


def train_side_final_v1():
    """Train final side model using the best-documented Side13 configuration."""
    CHECKPOINTS_DIR_FINAL.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_FINAL.mkdir(parents=True, exist_ok=True)

    run_training(
        view="side",
        max_epochs=SIDE_CONFIG["max_epochs"],
        lr=SIDE_CONFIG["learning_rate"],
        checkpoint_path=str(SIDE_CONFIG["checkpoint_path"]),
        history_path=str(SIDE_CONFIG["history_path"]),
        summary_path=str(RESULTS_DIR_FINAL / "side_train_summary_final_v1.json"),
    )


def read_best_training_snapshot(history_path):
    history = json.loads(history_path.read_text(encoding="utf-8"))
    best_idx = max(range(len(history["val_macro_f1"])), key=lambda idx: history["val_macro_f1"][idx])
    return {
        "best_epoch": best_idx + 1,
        "best_val_macro_f1": history["val_macro_f1"][best_idx],
        "train_loss_at_best_epoch": history["train_loss"][best_idx],
        "val_loss_at_best_epoch": history["val_loss"][best_idx],
        "train_val_loss_gap_at_best_epoch": history["train_val_loss_gap"][best_idx],
    }


def summarize_training_outputs():
    outputs = {}
    for view, config in (("front", FRONT_CONFIG), ("side", SIDE_CONFIG)):
        history_path = config["history_path"]
        if history_path.exists():
            outputs[view] = read_best_training_snapshot(history_path)
        else:
            outputs[view] = None
    return outputs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train final_v1 single-view models.")
    parser.add_argument("--view", choices=["front", "side"], default=None, help="Train only one view if needed.")
    args = parser.parse_args()

    if args.view == "front":
        train_front_final_v1()
    elif args.view == "side":
        train_side_final_v1()
    else:
        train_front_final_v1()
        train_side_final_v1()

    print(json.dumps(summarize_training_outputs(), indent=2))


if __name__ == "__main__":
    main()
