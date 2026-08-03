from pathlib import Path

from src.config import (
    BATCH_SIZE,
    FRAME_STRIDE,
    IMAGE_SIZE,
    MANIFEST_PAIRED_PATH,
    MANIFEST_SPLIT_PATH,
    MAX_EPOCHS,
    SPLIT_SEED,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_NAME = "3MDAD"
MANIFEST_PATHS = {
    "paired": MANIFEST_PAIRED_PATH,
    "split": MANIFEST_SPLIT_PATH,
}

LABEL_MAPPING = {
    "safe_driving": 0,
    "phone_use": 1,
}

BACKBONE = "tf_efficientnetv2_s"
NUM_CLASSES = 2
IMAGE_SIZE_FINAL = IMAGE_SIZE
BATCH_SIZE_FINAL = BATCH_SIZE
MAX_EPOCHS_FINAL = MAX_EPOCHS
THRESHOLD = 0.50
FRAME_STRIDE_FINAL = FRAME_STRIDE
SEED = SPLIT_SEED

RESULTS_DIR_FINAL = PROJECT_ROOT / "results" / "final_v2"
CHECKPOINTS_DIR_FINAL = PROJECT_ROOT / "checkpoints" / "final_v2"

FRONT_CONFIG = {
    "reference_name": "Experiment 21B front stabilization",
    "backbone": BACKBONE,
    "pretrained": True,
    "num_classes": NUM_CLASSES,
    "learning_rate": 3e-5,
    "weight_decay": 5e-4,
    "dropout": 0.4,
    "label_smoothing": 0.0,
    "num_stages_to_freeze": 5,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 1,
    "early_stopping_patience": 4,
    "batch_size": BATCH_SIZE_FINAL,
    "max_epochs": MAX_EPOCHS_FINAL,
    "class_weights": [2.5, 1.0],
    "metric_for_best_model": "validation Macro F1",
    "checkpoint_path": CHECKPOINTS_DIR_FINAL / "front_best_final_v2.pt",
    "history_path": RESULTS_DIR_FINAL / "front_history_final_v2.json",
    "summary_path": RESULTS_DIR_FINAL / "front_train_summary_final_v2.json",
}

SIDE_CONFIG = {
    "reference_name": "Experiment 21C side validation-loss stabilization",
    "backbone": BACKBONE,
    "pretrained": True,
    "num_classes": NUM_CLASSES,
    "learning_rate": 2e-5,
    "weight_decay": 5e-4,
    "dropout": 0.3,
    "label_smoothing": 0.0,
    "num_stages_to_freeze": 4,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "early_stopping_patience": 6,
    "batch_size": BATCH_SIZE_FINAL,
    "max_epochs": MAX_EPOCHS_FINAL,
    "class_weights": [2.5, 1.0],
    "metric_for_best_model": "validation loss",
    "checkpoint_monitor": "val_loss",
    "checkpoint_path": CHECKPOINTS_DIR_FINAL / "side_best_final_v2.pt",
    "history_path": RESULTS_DIR_FINAL / "side_history_final_v2.json",
    "summary_path": RESULTS_DIR_FINAL / "side_train_summary_final_v2.json",
}

REFERENCE_FINAL_V1 = {
    "front_f1_macro": 0.76626,
    "side_f1_macro": 0.62023,
    "average_fusion_f1_macro": 0.78059,
    "adaptive_fusion_f1_macro": 0.78059,
    "frame_stride": 30,
    "test_support": 220,
}

REFERENCE_EXP21_STRIDE30_BEST = {
    "front_f1_macro": 0.76626,
    "side_f1_macro": 0.71590,
    "average_fusion_f1_macro": 0.82504,
    "adaptive_fusion_f1_macro": 0.82504,
    "frame_stride": 30,
    "test_support": 220,
}

PRIMARY_METRIC = "f1_macro"
SUPPORTING_METRICS = ["accuracy", "precision_macro", "recall_macro"]
