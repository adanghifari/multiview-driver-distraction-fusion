import json

from src.experiment_21.config_experiment_21 import (
    CHECKPOINTS_DIR_EXP21,
    FRAME_STRIDE_EXP21,
    FRONT_CONFIG,
    MAX_EPOCHS_EXP21,
    RESULTS_DIR_EXP21,
    SIDE_CONFIG,
)


def train_front_exp21():
    from src.train import run_training

    CHECKPOINTS_DIR_EXP21.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_EXP21.mkdir(parents=True, exist_ok=True)
    return run_training(
        view="front",
        max_epochs=MAX_EPOCHS_EXP21,
        experiment=FRONT_CONFIG["experiment"],
        checkpoint_path=str(FRONT_CONFIG["checkpoint_path"]),
        history_path=str(FRONT_CONFIG["history_path"]),
        summary_path=str(FRONT_CONFIG["summary_path"]),
        frame_stride=FRAME_STRIDE_EXP21,
    )


def train_side_exp21():
    from src.train import run_training

    CHECKPOINTS_DIR_EXP21.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_EXP21.mkdir(parents=True, exist_ok=True)
    return run_training(
        view="side",
        max_epochs=MAX_EPOCHS_EXP21,
        experiment=SIDE_CONFIG["experiment"],
        checkpoint_path=str(SIDE_CONFIG["checkpoint_path"]),
        history_path=str(SIDE_CONFIG["history_path"]),
        summary_path=str(SIDE_CONFIG["summary_path"]),
        frame_stride=FRAME_STRIDE_EXP21,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Experiment 21 single-view models.")
    parser.add_argument("--view", choices=["front", "side"], default=None)
    args = parser.parse_args()

    if args.view == "front":
        train_front_exp21()
    elif args.view == "side":
        train_side_exp21()
    else:
        train_front_exp21()
        train_side_exp21()

    print(json.dumps({"results_dir": str(RESULTS_DIR_EXP21), "checkpoints_dir": str(CHECKPOINTS_DIR_EXP21)}, indent=2))


if __name__ == "__main__":
    main()
