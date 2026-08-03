from pathlib import Path

from src.config import BATCH_SIZE, MAX_EPOCHS, PROJECT_ROOT, SPLIT_SEED


EXPERIMENT_ID = "experiment_21"
FRAME_STRIDE_EXP21 = 30
THRESHOLD = 0.50
SEED = SPLIT_SEED

RESULTS_DIR_EXP21 = PROJECT_ROOT / "results" / EXPERIMENT_ID
CHECKPOINTS_DIR_EXP21 = PROJECT_ROOT / "checkpoints" / EXPERIMENT_ID

FRONT_CONFIG = {
    "experiment": "experiment_21_front",
    "learning_rate": 3e-5,
    "weight_decay": 5e-4,
    "dropout": 0.4,
    "label_smoothing": 0.0,
    "num_stages_to_freeze": 4,
    "early_stopping_patience": 4,
    "checkpoint_path": CHECKPOINTS_DIR_EXP21 / "front_best_exp21.pt",
    "history_path": RESULTS_DIR_EXP21 / "front_history_exp21.json",
    "summary_path": RESULTS_DIR_EXP21 / "front_train_summary_exp21.json",
}

SIDE_CONFIG = {
    "experiment": "experiment_21_side",
    "learning_rate": 2e-5,
    "weight_decay": 5e-4,
    "dropout": 0.3,
    "label_smoothing": 0.0,
    "num_stages_to_freeze": 3,
    "early_stopping_patience": 7,
    "checkpoint_path": CHECKPOINTS_DIR_EXP21 / "side_best_exp21.pt",
    "history_path": RESULTS_DIR_EXP21 / "side_history_exp21.json",
    "summary_path": RESULTS_DIR_EXP21 / "side_train_summary_exp21.json",
}

REFERENCE_FINAL_V1 = {
    "front_f1_macro": 0.76626,
    "side_f1_macro": 0.62023,
    "average_fusion_f1_macro": 0.78059,
    "adaptive_fusion_f1_macro": 0.78059,
    "frame_stride": 30,
    "test_support": 220,
}

REFERENCE_EXP8 = {
    "front_f1_macro": 0.86314,
    "side_f1_macro": 0.77124,
    "average_fusion_f1_macro": 0.87494,
    "adaptive_fusion_f1_macro": 0.87494,
    "frame_stride": 15,
    "test_support": 419,
}

PIPELINE_DESCRIPTION = (
    "Experiment 21 adapts the strongest exploratory signal from Experiment 8 "
    "into a more disciplined final-style protocol. It uses stride 30 to make "
    "the comparison apple-to-apple with final_v1, keeps label smoothing "
    "disabled following the final ablation, and relaxes freeze depth toward "
    "Exp8 to test whether more capacity helps under the same sampling "
    "protocol."
)

PRIMARY_METRIC = "Macro F1"
BATCH_SIZE_EXP21 = BATCH_SIZE
MAX_EPOCHS_EXP21 = MAX_EPOCHS
