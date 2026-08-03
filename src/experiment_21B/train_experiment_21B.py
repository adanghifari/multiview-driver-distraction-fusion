import json

from src.experiment_21B.config_experiment_21B import CHECKPOINTS_DIR_EXP21B, FRAME_STRIDE_EXP21B, FRONT_CONFIG, MAX_EPOCHS_EXP21B, RESULTS_DIR_EXP21B


def train_front_exp21B():
    from src.train import run_training

    CHECKPOINTS_DIR_EXP21B.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_EXP21B.mkdir(parents=True, exist_ok=True)
    return run_training(
        view="front",
        max_epochs=MAX_EPOCHS_EXP21B,
        experiment=FRONT_CONFIG["experiment"],
        checkpoint_path=str(FRONT_CONFIG["checkpoint_path"]),
        history_path=str(FRONT_CONFIG["history_path"]),
        summary_path=str(FRONT_CONFIG["summary_path"]),
        frame_stride=FRAME_STRIDE_EXP21B,
    )


def main():
    train_front_exp21B()
    print(json.dumps({"front_checkpoint": str(FRONT_CONFIG["checkpoint_path"]), "results_dir": str(RESULTS_DIR_EXP21B)}, indent=2))


if __name__ == "__main__":
    main()
