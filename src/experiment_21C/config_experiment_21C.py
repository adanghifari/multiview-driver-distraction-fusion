from src.config import BATCH_SIZE, MAX_EPOCHS, PROJECT_ROOT
from src.experiment_21.config_experiment_21 import FRAME_STRIDE_EXP21, REFERENCE_EXP8, REFERENCE_FINAL_V1, THRESHOLD
from src.experiment_21B.config_experiment_21B import CHECKPOINTS_DIR_EXP21B, FRONT_CONFIG as FRONT_CONFIG_EXP21B, REFERENCE_EXP21A, RESULTS_DIR_EXP21B


EXPERIMENT_ID = "experiment_21C"
FRAME_STRIDE_EXP21C = FRAME_STRIDE_EXP21

RESULTS_DIR_EXP21C = PROJECT_ROOT / "results" / EXPERIMENT_ID
CHECKPOINTS_DIR_EXP21C = PROJECT_ROOT / "checkpoints" / EXPERIMENT_ID

FRONT_CONFIG_LOCKED = {
    **FRONT_CONFIG_EXP21B,
    "checkpoint_path": CHECKPOINTS_DIR_EXP21B / "front_best_exp21B.pt",
    "history_path": RESULTS_DIR_EXP21B / "front_history_exp21B.json",
    "summary_path": RESULTS_DIR_EXP21B / "front_train_summary_exp21B.json",
    "locked_from": "experiment_21B",
}

SIDE_CONFIG = {
    "experiment": "experiment_21C_side",
    "learning_rate": 2e-5,
    "weight_decay": 5e-4,
    "dropout": 0.3,
    "label_smoothing": 0.0,
    "num_stages_to_freeze": 4,
    "early_stopping_patience": 6,
    "lr_scheduler_patience": 2,
    "class_weights": [2.5, 1.0],
    "checkpoint_monitor": "val_loss",
    "checkpoint_path": CHECKPOINTS_DIR_EXP21C / "side_best_exp21C.pt",
    "history_path": RESULTS_DIR_EXP21C / "side_history_exp21C.json",
    "summary_path": RESULTS_DIR_EXP21C / "side_train_summary_exp21C.json",
}

PIPELINE_DESCRIPTION = (
    "Experiment 21C keeps the proposal-compliant average/adaptive fusion setup, "
    "locks the stabilized front model from Experiment 21B, and retrains only the "
    "side view. The side checkpoint is selected using validation loss with "
    "patience 6 to test whether a probability-stabilized side model reduces "
    "safe-to-phone errors in fusion."
)

REFERENCE_EXP21B = {
    "front_f1_macro": 0.78797,
    "side_f1_macro": 0.66044,
    "average_fusion_f1_macro": 0.75880,
    "adaptive_fusion_f1_macro": 0.75880,
    "front_generalization_status": "OK",
    "side_generalization_status": "OK",
    "frame_stride": 20,
    "test_support": 322,
}

PRIMARY_METRIC = "Macro F1"
BATCH_SIZE_EXP21C = BATCH_SIZE
MAX_EPOCHS_EXP21C = MAX_EPOCHS

