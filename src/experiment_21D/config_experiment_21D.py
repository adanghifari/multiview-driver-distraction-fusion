from src.config import BATCH_SIZE, MAX_EPOCHS, PROJECT_ROOT
from src.experiment_21.config_experiment_21 import (
    CHECKPOINTS_DIR_EXP21,
    FRAME_STRIDE_EXP21,
    FRONT_CONFIG,
    REFERENCE_EXP8,
    REFERENCE_FINAL_V1,
    RESULTS_DIR_EXP21,
    THRESHOLD,
)


EXPERIMENT_ID = "experiment_21D"
FRAME_STRIDE_EXP21D = FRAME_STRIDE_EXP21

RESULTS_DIR_EXP21D = PROJECT_ROOT / "results" / EXPERIMENT_ID
CHECKPOINTS_DIR_EXP21D = PROJECT_ROOT / "checkpoints" / EXPERIMENT_ID

FRONT_CONFIG_LOCKED = {
    **FRONT_CONFIG,
    "checkpoint_path": CHECKPOINTS_DIR_EXP21 / "front_best_exp21.pt",
    "history_path": RESULTS_DIR_EXP21 / "front_history_exp21.json",
    "summary_path": RESULTS_DIR_EXP21 / "front_train_summary_exp21.json",
    "locked_from": "experiment_21_stride30_best",
}

SIDE_CONFIG = {
    "experiment": "experiment_21D_side_regularized",
    "learning_rate": 2e-5,
    "weight_decay": 1e-3,
    "dropout": 0.4,
    "label_smoothing": 0.0,
    "num_stages_to_freeze": 4,
    "early_stopping_patience": 5,
    "lr_scheduler_patience": 2,
    "class_weights": [2.5, 1.0],
    "checkpoint_monitor": "val_loss",
    "checkpoint_path": CHECKPOINTS_DIR_EXP21D / "side_best_exp21D.pt",
    "history_path": RESULTS_DIR_EXP21D / "side_history_exp21D.json",
    "summary_path": RESULTS_DIR_EXP21D / "side_train_summary_exp21D.json",
}

REFERENCE_EXP21_STRIDE30_BEST = {
    "front_f1_macro": 0.76626,
    "side_f1_macro": 0.71590,
    "average_fusion_f1_macro": 0.82504,
    "adaptive_fusion_f1_macro": 0.82504,
    "front_generalization_status": "OK",
    "side_generalization_status": "OVERFIT",
    "side_train_validation_loss_gap": 0.22406,
    "frame_stride": 30,
    "test_support": 220,
}

PIPELINE_DESCRIPTION = (
    "Experiment 21D keeps the OK front model from Experiment 21 stride30-best "
    "locked and retrains only the side view with stronger regularization. The "
    "goal is diagnostic: test whether reducing side overfit preserves, improves, "
    "or lowers fusion performance."
)

PRIMARY_METRIC = "Macro F1"
BATCH_SIZE_EXP21D = BATCH_SIZE
MAX_EPOCHS_EXP21D = MAX_EPOCHS
REFERENCE_EXP8 = REFERENCE_EXP8
REFERENCE_FINAL_V1 = REFERENCE_FINAL_V1
