from src.config import BATCH_SIZE, MAX_EPOCHS, PROJECT_ROOT
from src.experiment_21.config_experiment_21 import (
    CHECKPOINTS_DIR_EXP21,
    FRAME_STRIDE_EXP21,
    REFERENCE_EXP8,
    REFERENCE_FINAL_V1,
    RESULTS_DIR_EXP21,
    SIDE_CONFIG as SIDE_CONFIG_EXP21A,
    THRESHOLD,
)


EXPERIMENT_ID = "experiment_21B"
FRAME_STRIDE_EXP21B = FRAME_STRIDE_EXP21

RESULTS_DIR_EXP21B = PROJECT_ROOT / "results" / EXPERIMENT_ID
CHECKPOINTS_DIR_EXP21B = PROJECT_ROOT / "checkpoints" / EXPERIMENT_ID

FRONT_CONFIG = {
    "experiment": "experiment_21B_front",
    "learning_rate": 3e-5,
    "weight_decay": 5e-4,
    "dropout": 0.4,
    "label_smoothing": 0.0,
    "num_stages_to_freeze": 5,
    "early_stopping_patience": 4,
    "checkpoint_path": CHECKPOINTS_DIR_EXP21B / "front_best_exp21B.pt",
    "history_path": RESULTS_DIR_EXP21B / "front_history_exp21B.json",
    "summary_path": RESULTS_DIR_EXP21B / "front_train_summary_exp21B.json",
}

SIDE_CONFIG_LOCKED = {
    **SIDE_CONFIG_EXP21A,
    "checkpoint_path": CHECKPOINTS_DIR_EXP21 / "side_best_exp21.pt",
    "history_path": RESULTS_DIR_EXP21 / "side_history_exp21.json",
    "summary_path": RESULTS_DIR_EXP21 / "side_train_summary_exp21.json",
    "locked_from": "experiment_21A",
}

PIPELINE_DESCRIPTION = (
    "Experiment 21B keeps the proposal-compliant fusion protocol from Experiment 21 "
    "but retrains only the front view. The side model is locked from Experiment 21A "
    "because its training status was OK and its Macro F1 improved over final_v1. "
    "The front model restores freeze depth to 5 to reduce the borderline "
    "train-validation gap while keeping stride 20, label smoothing 0.0, and "
    "average/adaptive fusion unchanged."
)

REFERENCE_EXP21A = {
    "front_f1_macro": 0.77284,
    "side_f1_macro": 0.66044,
    "average_fusion_f1_macro": 0.76597,
    "adaptive_fusion_f1_macro": 0.76597,
    "front_generalization_status": "BORDERLINE",
    "side_generalization_status": "OK",
    "frame_stride": 20,
    "test_support": 322,
}

PRIMARY_METRIC = "Macro F1"
BATCH_SIZE_EXP21B = BATCH_SIZE
MAX_EPOCHS_EXP21B = MAX_EPOCHS

