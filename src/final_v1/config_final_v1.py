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

RESULTS_DIR_FINAL = PROJECT_ROOT / "results" / "final_v1"
CHECKPOINTS_DIR_FINAL = PROJECT_ROOT / "checkpoints" / "final_v1"

FRONT_CONFIG = {
    "reference_name": "Front14A revised",
    "backbone": BACKBONE,
    "pretrained": True,
    "num_classes": NUM_CLASSES,
    "learning_rate": 3e-5,
    "weight_decay": 3e-4,
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
    "checkpoint_path": CHECKPOINTS_DIR_FINAL / "front_best_final_v1.pt",
    "history_path": RESULTS_DIR_FINAL / "front_history_final_v1.json",
}

SIDE_CONFIG = {
    "reference_name": "Side13",
    "config_source": [
        "src/config.py",
        "results/side_history.json",
        "results/ringkasan_akhir_ablasi.md",
        "results/side_test_metrics.json",
    ],
    "backbone": BACKBONE,
    "pretrained": True,
    "num_classes": NUM_CLASSES,
    "learning_rate": 2e-5,
    "weight_decay": 2e-3,
    "dropout": 0.3,
    "label_smoothing": 0.0,
    "num_stages_to_freeze": 4,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 1,
    "early_stopping_patience": 7,
    "batch_size": BATCH_SIZE_FINAL,
    "max_epochs": MAX_EPOCHS_FINAL,
    "class_weights": [2.5, 1.0],
    "metric_for_best_model": "validation Macro F1",
    "best_epoch_reference": 11,
    "best_val_macro_f1_reference": 0.75,
    "notes": (
        "Konfigurasi Side13 diturunkan dari history, ablation summary, dan config repo. "
        "Tidak ditemukan preset EXPERIMENT_CONFIGS khusus untuk Side13, tetapi LR=2e-5, "
        "WD=2e-3, dropout=0.3, label_smoothing=0.0, freeze=4, class_weights=[2.5,1.0], "
        "scheduler ReduceLROnPlateau, dan early stopping patience=7 konsisten dengan artefak yang tersedia."
    ),
    "checkpoint_path": CHECKPOINTS_DIR_FINAL / "side_best_final_v1.pt",
    "history_path": RESULTS_DIR_FINAL / "side_history_final_v1.json",
}

REFERENCE_RESULTS = {
    "front": {
        "macro_f1": 0.76626,
        "confusion_matrix": [[24, 16], [14, 166]],
    },
    "side": {
        "macro_f1": 0.62023,
        "confusion_matrix": [[11, 29], [12, 168]],
    },
    "average_fusion": {
        "macro_f1": 0.78059,
        "confusion_matrix": [[20, 20], [4, 176]],
    },
    "adaptive_fusion": {
        "macro_f1": 0.78059,
        "confusion_matrix": [[20, 20], [4, 176]],
        "adaptive_different_from_average": 0,
    },
}

PRIMARY_METRIC = "f1_macro"
SUPPORTING_METRICS = ["accuracy", "precision_macro", "recall_macro"]
