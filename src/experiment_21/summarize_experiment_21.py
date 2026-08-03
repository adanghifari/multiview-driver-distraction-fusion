import json

import pandas as pd

from src.experiment_21.config_experiment_21 import (
    FRONT_CONFIG,
    PIPELINE_DESCRIPTION,
    REFERENCE_EXP8,
    REFERENCE_FINAL_V1,
    RESULTS_DIR_EXP21,
    SIDE_CONFIG,
)
from src.final_v1.summarize_final_v1 import classify_generalization_status


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _training_diag(path):
    summary = _read_json(path)
    return {
        "best_epoch": summary.get("best_epoch"),
        "best_validation_macro_f1": summary.get("best_validation_macro_f1"),
        "train_loss_at_best_epoch": summary.get("train_loss_at_best_epoch"),
        "validation_loss_at_best_epoch": summary.get("validation_loss_at_best_epoch"),
        "train_validation_loss_gap_at_best_epoch": summary.get("train_validation_loss_gap_at_best_epoch"),
        "generalization_status": classify_generalization_status(
            summary.get("train_loss_at_best_epoch"),
            summary.get("validation_loss_at_best_epoch"),
            summary.get("best_validation_macro_f1"),
        ),
    }


def build_markdown(summary: dict) -> str:
    front = summary["front_single_view"]
    side = summary["side_single_view"]
    fusion = summary["fusion"]
    avg = fusion["average_fusion"]
    adapt = fusion["adaptive_fusion"]
    diagnostics = summary["training_diagnostics"]
    return "\n".join(
        [
            "# Experiment 21 Summary",
            "",
            f"- {PIPELINE_DESCRIPTION}",
            "- Output disimpan di `results/experiment_21/` dan `checkpoints/experiment_21/`.",
            "- Notebook `notebooks/experiment_21.ipynb` membaca artefak setelah script Python selesai dijalankan.",
            "",
            "## Protocol",
            "",
            "- Dataset: 3MDAD binary `safe_driving` vs `phone_use`.",
            "- Split: subject-based split yang sama dengan final pipeline.",
            "- Frame stride: 20.",
            "- Primary metric: Macro F1.",
            "- Threshold: 0.50.",
            "- Fusion: decision-level average fusion dan adaptive confidence fusion.",
            "",
            "## Configuration",
            "",
            f"- Front: lr={FRONT_CONFIG['learning_rate']}, wd={FRONT_CONFIG['weight_decay']}, dropout={FRONT_CONFIG['dropout']}, LS={FRONT_CONFIG['label_smoothing']}, freeze={FRONT_CONFIG['num_stages_to_freeze']}, patience={FRONT_CONFIG['early_stopping_patience']}.",
            f"- Side: lr={SIDE_CONFIG['learning_rate']}, wd={SIDE_CONFIG['weight_decay']}, dropout={SIDE_CONFIG['dropout']}, LS={SIDE_CONFIG['label_smoothing']}, freeze={SIDE_CONFIG['num_stages_to_freeze']}, patience={SIDE_CONFIG['early_stopping_patience']}.",
            "",
            "## Results",
            "",
            f"- Front Macro F1: {front['f1_macro']:.5f}; confusion: {front['confusion_matrix']}; status: {diagnostics['front']['generalization_status']}.",
            f"- Side Macro F1: {side['f1_macro']:.5f}; confusion: {side['confusion_matrix']}; status: {diagnostics['side']['generalization_status']}.",
            f"- Average fusion Macro F1: {avg['f1_macro']:.5f}; confusion: {avg['confusion_matrix']}.",
            f"- Adaptive fusion Macro F1: {adapt['f1_macro']:.5f}; confusion: {adapt['confusion_matrix']}.",
            f"- Adaptive berbeda dari average pada {fusion['adaptive_different_from_average']} sampel.",
            "",
            "## Reference Context",
            "",
            f"- Final v1 reference fusion Macro F1: {REFERENCE_FINAL_V1['average_fusion_f1_macro']:.5f} with stride {REFERENCE_FINAL_V1['frame_stride']} and support {REFERENCE_FINAL_V1['test_support']}.",
            f"- Exp 8 exploratory fusion Macro F1: {REFERENCE_EXP8['average_fusion_f1_macro']:.5f} with stride {REFERENCE_EXP8['frame_stride']} and support {REFERENCE_EXP8['test_support']}.",
            "- Because stride/support differ, reference rows are context, not a direct statistical claim.",
        ]
    ) + "\n"


def run_summary_exp21():
    front = _read_json(RESULTS_DIR_EXP21 / "front_metrics_exp21.json")
    side = _read_json(RESULTS_DIR_EXP21 / "side_metrics_exp21.json")
    fusion = _read_json(RESULTS_DIR_EXP21 / "fusion_metrics_exp21.json")
    diagnostics = {
        "front": _training_diag(FRONT_CONFIG["summary_path"]),
        "side": _training_diag(SIDE_CONFIG["summary_path"]),
    }
    rows = [
        {"method": "front_single_view", **{k: front.get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe", "ece_binary", "brier_score"]}, "generalization_status": diagnostics["front"]["generalization_status"]},
        {"method": "side_single_view", **{k: side.get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe", "ece_binary", "brier_score"]}, "generalization_status": diagnostics["side"]["generalization_status"]},
        {"method": "average_fusion", **{k: fusion["average_fusion"].get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe"]}, "ece_binary": None, "brier_score": None, "generalization_status": None},
        {"method": "adaptive_fusion", **{k: fusion["adaptive_fusion"].get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe"]}, "ece_binary": None, "brier_score": None, "generalization_status": None},
    ]
    summary = {
        "experiment": "experiment_21",
        "description": PIPELINE_DESCRIPTION,
        "front_single_view": front,
        "side_single_view": side,
        "fusion": fusion,
        "training_diagnostics": diagnostics,
        "reference_final_v1": REFERENCE_FINAL_V1,
        "reference_exp8": REFERENCE_EXP8,
    }
    pd.DataFrame(rows).to_csv(RESULTS_DIR_EXP21 / "experiment_21_table.csv", index=False)
    (RESULTS_DIR_EXP21 / "experiment_21_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR_EXP21 / "experiment_21_summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def main():
    print(json.dumps(run_summary_exp21(), indent=2))


if __name__ == "__main__":
    main()

