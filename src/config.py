"""
Konfigurasi terpusat untuk pipeline eksperimen.

Semua path, mapping view, dan mapping label didefinisikan di sini SEKALI SAJA.
Script lain (build_manifest, dataset, training) mengimpor dari sini agar tidak
ada risiko salah asumsi (mis. RGB1/RGB2 tertukar) tersebar di banyak file.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Path dasar proyek
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

MANIFEST_FULL_PATH = MANIFEST_DIR / "manifest_full.csv"       # semua 16 kelas
MANIFEST_BINARY_PATH = MANIFEST_DIR / "manifest_binary.csv"   # hanya A1 + A5-A9
MANIFEST_PAIRED_PATH = MANIFEST_DIR / "manifest_paired.csv"   # pasangan front-side tersinkronisasi
MANIFEST_SPLIT_PATH = MANIFEST_DIR / "manifest_split.csv"     # manifest_binary + kolom split

# --------------------------------------------------------------------------
# Mapping folder RGB -> sudut pandang (view)
# Dikonfirmasi manual oleh peneliti: RGB1 = side view, RGB2 = front view
# --------------------------------------------------------------------------
VIEW_FOLDER_MAP = {
    "RGB1": "side",
    "RGB2": "front",
}

# --------------------------------------------------------------------------
# Mapping kelas aktivitas 3MDAD (AC1..AC16) ke label biner
# Sesuai Subbab 3.2.1 proposal:
#   A1        -> safe_driving      (label 0)
#   A5 - A9   -> phone_use         (label 1)
#   lainnya   -> tidak digunakan (di-drop saat filtering ke manifest_binary)
# --------------------------------------------------------------------------
ACTIVITY_LABEL_NAMES = {
    1: "safe_driving",
    2: "doing_hair_makeup",
    3: "adjusting_radio",
    4: "gps_operating",
    5: "writing_message_right",
    6: "writing_message_left",
    7: "talking_phone_right",
    8: "talking_phone_left",
    9: "having_picture",
    10: "talking_passenger",
    11: "singing_dancing",
    12: "fatigue_somnolence",
    13: "drinking_right",
    14: "drinking_left",
    15: "reaching_behind",
    16: "smoking",
}

SAFE_DRIVING_CLASSES = {1}
PHONE_USE_CLASSES = {5, 6, 7, 8, 9}
BINARY_CLASSES = SAFE_DRIVING_CLASSES | PHONE_USE_CLASSES

# --------------------------------------------------------------------------
# Koreksi metadata yang diketahui (known data issues)
# --------------------------------------------------------------------------
# Bug folder AC9/AC10 tertukar untuk subjek S31 pernah dilaporkan komunitas
# untuk modalitas Depth (lihat: github.com/kevinsu628/3MDAD, catatan
# "side_view_day_data/Depth1/S31/AC10 and AC9 should be swapped"). Verifikasi
# visual manual pada RGB mengonfirmasi swap yang sama terjadi pada RGB1
# (side view) untuk subjek ini. Isi folder AC9 sebenarnya AC10, dan
# sebaliknya. Dikoreksi di level manifest (bukan rename file mentah) agar
# tetap reproducible jika dataset di-extract ulang dari sumber asli.
#
# Format: (view, subject_id) -> {activity_id_di_folder: activity_id_sebenarnya}
KNOWN_ACTIVITY_CORRECTIONS = {
    ("side", 31): {9: 10, 10: 9},
}

BINARY_LABEL_MAP = {
    "safe_driving": 0,
    "phone_use": 1,
}


def activity_id_to_binary_label(activity_id: int):
    """Kembalikan nama label biner ('safe_driving'/'phone_use') atau None jika
    activity_id tidak termasuk dalam kelas yang dipakai penelitian ini."""
    if activity_id in SAFE_DRIVING_CLASSES:
        return "safe_driving"
    if activity_id in PHONE_USE_CLASSES:
        return "phone_use"
    return None


# --------------------------------------------------------------------------
# Parameter split subjek (Tabel 3.2 proposal)
# --------------------------------------------------------------------------
N_SUBJECTS_TOTAL = 50
SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# --------------------------------------------------------------------------
# Parameter preprocessing & training (Bab 3)
# --------------------------------------------------------------------------
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_NAME = "tf_efficientnetv2_s"      # identifier timm untuk EfficientNetV2-S
BATCH_SIZE = 32
LEARNING_RATE_FRONT = 3e-5      # [v5/Exp10] learning rate front diturunkan ke 3e-5 (moderat)
LEARNING_RATE_SIDE = 2e-5       # [v5/Exp8] learning rate side view (preventive untuk freeze 3 & stride 15)
DROPOUT_FRONT = 0.5               # [v6/Exp11] dropout front view direlaksasi ke 0.5
DROPOUT_SIDE = 0.3                # [v7/Exp12] dropout side view diturunkan untuk rescue
EARLY_STOPPING_PATIENCE_FRONT = 3  # [v6/Exp11] early stopping patience disesuaikan ke 3
EARLY_STOPPING_PATIENCE_SIDE = 7  # [v7/Exp12] patience side view dinaikkan untuk rescue
MAX_EPOCHS = 30
DECISION_THRESHOLD = 0.5

# Konfigurasi Eksperimen 13 (Calibration & Fusion Optimization — Final: Isolasi Label Smoothing)
WEIGHT_DECAY = 2e-3             # [v8/Exp13] dikembalikan ke baseline 2e-3 untuk mengisolasi efek label smoothing
NUM_STAGES_TO_FREEZE_FRONT = 5  # [v5/Exp10] freeze front stages dinaikkan ke 5 (reduksi overfitting)
NUM_STAGES_TO_FREEZE_SIDE = 4   # [v5/Exp10] freeze side stages dikembalikan ke 4 untuk stabilitas Side view
LABEL_SMOOTHING = 0.0           # [v8/Exp13] dinonaktifkan (0.00) untuk mengisolasi efeknya terhadap kalibrasi probabilitas biner
CLASS_WEIGHTS = [2.5, 1.0]      # [v5/Exp8] bobot kelas moderat untuk safe_driving
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE_FRONT = 1
LR_SCHEDULER_PATIENCE_SIDE = 1

# Intensitas Augmentasi (Subbab 3.2.3 proposal)
# Front View (Eksperimen 14 - Naik 1.5x untuk meredam overfitting)
AUG_ROTATION_DEGREE_FRONT = 68      # 1.5x dari 45 (67.5 -> 68)
AUG_COLOR_JITTER_FACTOR_FRONT = 1.2  # 1.5x dari 0.8

# Side View (Baseline Eksperimen 12/13)
AUG_ROTATION_DEGREE_SIDE = 45
AUG_COLOR_JITTER_FACTOR_SIDE = 0.8

# --------------------------------------------------------------------------
# Frame subsampling (Subbab 3.2.3 — Eksperimen v4)
# Ambil 1 dari setiap FRAME_STRIDE frame per sekuens untuk mengurangi
# redundansi temporal pada video 30Hz. Diterapkan pada SEMUA split
# (train/val/test) agar konsisten. Formula: (frame - 1) % FRAME_STRIDE == 0
# menghasilkan frame ke-1, 6, 11, 16, ... (selalu dimulai dari frame pertama).
# Set ke 1 untuk menonaktifkan subsampling (pakai semua frame).
# --------------------------------------------------------------------------
FRAME_STRIDE = 30               # [v5/Exp9] stride 30 untuk memangkas data redundan secara ekstrem (1 FPS)

# --------------------------------------------------------------------------
# Preset eksperimen khusus
# --------------------------------------------------------------------------
# Experiment 14A hanya ditujukan untuk stabilisasi FRONT view. Side view tetap
# memakai checkpoint Experiment 13 sebagai kontrol stabil dan tidak dilatih ulang.
EXPERIMENT_CONFIGS = {
    "experiment_14A": {
        "view": "front",
        "checkpoint_name": "front_best_exp14A.pt",
        "history_name": "front_history_exp14A.json",
        "summary_name": "experiment_14A_front_metrics.json",
        "fusion_name": "fusion_comparison_exp14A.json",
        "learning_rate": 3e-5,
        "weight_decay": 3e-4,
        "dropout": 0.4,
        "label_smoothing": 0.0,
        "num_stages_to_freeze": NUM_STAGES_TO_FREEZE_FRONT,
        "early_stopping_patience": 4,
        "lr_scheduler_patience": LR_SCHEDULER_PATIENCE_FRONT,
        "class_weights": CLASS_WEIGHTS,
        "description": (
            "Front-view stabilization revision: restore learning capacity "
            "by using Experiment 13 freeze depth and no label smoothing, "
            "while keeping light dropout and weight decay regularization."
        ),
    },
    "experiment_14B": {
        "view": "front",
        "checkpoint_name": "front_best_exp14B.pt",
        "history_name": "front_history_exp14B.json",
        "summary_name": "experiment_14B_front_metrics.json",
        "fusion_name": "fusion_comparison_exp14B.json",
        "learning_rate": 3e-5,
        "weight_decay": 5e-4,
        "dropout": 0.4,
        "label_smoothing": 0.0,
        "num_stages_to_freeze": NUM_STAGES_TO_FREEZE_FRONT,
        "early_stopping_patience": 4,
        "lr_scheduler_patience": LR_SCHEDULER_PATIENCE_FRONT,
        "class_weights": CLASS_WEIGHTS,
        "description": (
            "Front-view stabilization refinement: restore dropout to 0.40 "
            "and slightly increase weight decay from 3e-4 to 5e-4 while "
            "preserving the Experiment 14A revision training setup."
        ),
    },
}
