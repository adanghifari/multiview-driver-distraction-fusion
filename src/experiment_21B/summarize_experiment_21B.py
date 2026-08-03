import json

import pandas as pd

from src.experiment_21B.config_experiment_21B import (
    FRONT_CONFIG,
    PIPELINE_DESCRIPTION,
    REFERENCE_EXP21A,
    REFERENCE_EXP8,
    REFERENCE_FINAL_V1,
    RESULTS_DIR_EXP21B,
    SIDE_CONFIG_LOCKED,
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
    side = summary["side_locked_exp21A"]
    fusion = summary["fusion"]
    avg = fusion["average_fusion"]
    adapt = fusion["adaptive_fusion"]
    diag = summary["training_diagnostics"]
    return "\n".join(
        [
            "# Experiment 21B Summary",
            "",
            f"- {PIPELINE_DESCRIPTION}",
            "- Output disimpan di `results/experiment_21B/` dan `checkpoints/experiment_21B/`.",
            "- Side model dikunci dari Experiment 21A; hanya front yang diretrain.",
            "- Fusion tetap dibatasi pada average fusion dan adaptive fusion sesuai proposal.",
            "",
            "## Protocol",
            "",
            "- Dataset: 3MDAD binary `safe_driving` vs `phone_use`.",
            "- Split: subject-based split yang sama.",
            "- Frame stride: 20.",
            "- Primary metric: Macro F1.",
            "- Threshold: 0.50.",
            "",
            "## Configuration",
            "",
            f"- Front 21B: lr={FRONT_CONFIG['learning_rate']}, wd={FRONT_CONFIG['weight_decay']}, dropout={FRONT_CONFIG['dropout']}, LS={FRONT_CONFIG['label_smoothing']}, freeze={FRONT_CONFIG['num_stages_to_freeze']}, patience={FRONT_CONFIG['early_stopping_patience']}.",
            f"- Side locked: checkpoint `{SIDE_CONFIG_LOCKED['checkpoint_path']}` from {SIDE_CONFIG_LOCKED['locked_from']}.",
            "",
            "## Results",
            "",
            f"- Front Macro F1: {front['f1_macro']:.5f}; confusion: {front['confusion_matrix']}; status: {diag['front']['generalization_status']}.",
            f"- Side locked Macro F1: {side['f1_macro']:.5f}; confusion: {side['confusion_matrix']}; status: {diag['side_locked_exp21A']['generalization_status']}.",
            f"- Average fusion Macro F1: {avg['f1_macro']:.5f}; confusion: {avg['confusion_matrix']}.",
            f"- Adaptive fusion Macro F1: {adapt['f1_macro']:.5f}; confusion: {adapt['confusion_matrix']}.",
            f"- Adaptive berbeda dari average pada {fusion['adaptive_different_from_average']} sampel.",
            "",
            "## Reference Context",
            "",
            f"- Exp 21A fusion Macro F1: {REFERENCE_EXP21A['average_fusion_f1_macro']:.5f}; front status {REFERENCE_EXP21A['front_generalization_status']}.",
            f"- Final v1 fusion Macro F1: {REFERENCE_FINAL_V1['average_fusion_f1_macro']:.5f}.",
            f"- Exp 8 exploratory fusion Macro F1: {REFERENCE_EXP8['average_fusion_f1_macro']:.5f}.",
        ]
    ) + "\n"


def run_summary_exp21B():
    front = _read_json(RESULTS_DIR_EXP21B / "front_metrics_exp21B.json")
    side = _read_json(RESULTS_DIR_EXP21B / "side_locked_exp21A_metrics_exp21B.json")
    fusion = _read_json(RESULTS_DIR_EXP21B / "fusion_metrics_exp21B.json")
    diagnostics = {
        "front": _training_diag(FRONT_CONFIG["summary_path"]),
        "side_locked_exp21A": _training_diag(SIDE_CONFIG_LOCKED["summary_path"]),
    }
    rows = [
        {"method": "front_single_view_21B", **{k: front.get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe", "ece_binary", "brier_score"]}, "generalization_status": diagnostics["front"]["generalization_status"]},
        {"method": "side_locked_exp21A", **{k: side.get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe", "ece_binary", "brier_score"]}, "generalization_status": diagnostics["side_locked_exp21A"]["generalization_status"]},
        {"method": "average_fusion", **{k: fusion["average_fusion"].get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe"]}, "ece_binary": None, "brier_score": None, "generalization_status": None},
        {"method": "adaptive_fusion", **{k: fusion["adaptive_fusion"].get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe"]}, "ece_binary": None, "brier_score": None, "generalization_status": None},
    ]
    summary = {
        "experiment": "experiment_21B",
        "description": PIPELINE_DESCRIPTION,
        "front_single_view": front,
        "side_locked_exp21A": side,
        "fusion": fusion,
        "training_diagnostics": diagnostics,
        "reference_exp21A": REFERENCE_EXP21A,
        "reference_final_v1": REFERENCE_FINAL_V1,
        "reference_exp8": REFERENCE_EXP8,
    }
    pd.DataFrame(rows).to_csv(RESULTS_DIR_EXP21B / "experiment_21B_table.csv", index=False)
    (RESULTS_DIR_EXP21B / "experiment_21B_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR_EXP21B / "experiment_21B_summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def main():
    print(json.dumps(run_summary_exp21B(), indent=2))


if __name__ == "__main__":
    main()

