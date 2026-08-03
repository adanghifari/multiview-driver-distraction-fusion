import json

import pandas as pd

from src.experiment_21C.config_experiment_21C import (
    FRONT_CONFIG_LOCKED,
    PIPELINE_DESCRIPTION,
    REFERENCE_EXP21A,
    REFERENCE_EXP21B,
    REFERENCE_EXP8,
    REFERENCE_FINAL_V1,
    RESULTS_DIR_EXP21C,
    SIDE_CONFIG,
)
from src.final_v1.summarize_final_v1 import classify_generalization_status


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _training_diag(path):
    summary = _read_json(path)
    return {
        "best_epoch": summary.get("best_epoch"),
        "best_validation_loss": summary.get("best_validation_loss"),
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
    front = summary["front_locked_exp21B"]
    side = summary["side_single_view"]
    fusion = summary["fusion"]
    avg = fusion["average_fusion"]
    adapt = fusion["adaptive_fusion"]
    diag = summary["training_diagnostics"]
    return "\n".join(
        [
            "# Experiment 21C Summary",
            "",
            f"- {PIPELINE_DESCRIPTION}",
            "- Output disimpan di `results/experiment_21C/` dan `checkpoints/experiment_21C/`.",
            "- Front model dikunci dari Experiment 21B; hanya side yang diretrain.",
            "- Fusion tetap average fusion dan adaptive fusion sesuai proposal.",
            "",
            "## Protocol",
            "",
            "- Dataset: 3MDAD binary `safe_driving` vs `phone_use`.",
            "- Split: subject-based split yang sama.",
            "- Frame stride: 20.",
            "- Side checkpoint monitor: validation loss.",
            "- Threshold: 0.50.",
            "",
            "## Configuration",
            "",
            f"- Front locked: checkpoint `{FRONT_CONFIG_LOCKED['checkpoint_path']}` from {FRONT_CONFIG_LOCKED['locked_from']}.",
            f"- Side 21C: lr={SIDE_CONFIG['learning_rate']}, wd={SIDE_CONFIG['weight_decay']}, dropout={SIDE_CONFIG['dropout']}, LS={SIDE_CONFIG['label_smoothing']}, freeze={SIDE_CONFIG['num_stages_to_freeze']}, patience={SIDE_CONFIG['early_stopping_patience']}, monitor={SIDE_CONFIG['checkpoint_monitor']}.",
            "",
            "## Results",
            "",
            f"- Front locked Macro F1: {front['f1_macro']:.5f}; confusion: {front['confusion_matrix']}; status: {diag['front_locked_exp21B']['generalization_status']}.",
            f"- Side Macro F1: {side['f1_macro']:.5f}; confusion: {side['confusion_matrix']}; status: {diag['side']['generalization_status']}.",
            f"- Average fusion Macro F1: {avg['f1_macro']:.5f}; confusion: {avg['confusion_matrix']}.",
            f"- Adaptive fusion Macro F1: {adapt['f1_macro']:.5f}; confusion: {adapt['confusion_matrix']}.",
            f"- Adaptive berbeda dari average pada {fusion['adaptive_different_from_average']} sampel.",
            "",
            "## Reference Context",
            "",
            f"- Exp 21B front Macro F1: {REFERENCE_EXP21B['front_f1_macro']:.5f}; fusion Macro F1: {REFERENCE_EXP21B['average_fusion_f1_macro']:.5f}.",
            f"- Exp 21A fusion Macro F1: {REFERENCE_EXP21A['average_fusion_f1_macro']:.5f}.",
            f"- Final v1 fusion Macro F1: {REFERENCE_FINAL_V1['average_fusion_f1_macro']:.5f}.",
            f"- Exp 8 exploratory fusion Macro F1: {REFERENCE_EXP8['average_fusion_f1_macro']:.5f}.",
        ]
    ) + "\n"


def run_summary_exp21C():
    front = _read_json(RESULTS_DIR_EXP21C / "front_locked_exp21B_metrics_exp21C.json")
    side = _read_json(RESULTS_DIR_EXP21C / "side_metrics_exp21C.json")
    fusion = _read_json(RESULTS_DIR_EXP21C / "fusion_metrics_exp21C.json")
    diagnostics = {
        "front_locked_exp21B": _training_diag(FRONT_CONFIG_LOCKED["summary_path"]),
        "side": _training_diag(SIDE_CONFIG["summary_path"]),
    }
    rows = [
        {"method": "front_locked_exp21B", **{k: front.get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe", "ece_binary", "brier_score"]}, "generalization_status": diagnostics["front_locked_exp21B"]["generalization_status"]},
        {"method": "side_single_view_21C", **{k: side.get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe", "ece_binary", "brier_score"]}, "generalization_status": diagnostics["side"]["generalization_status"]},
        {"method": "average_fusion", **{k: fusion["average_fusion"].get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe"]}, "ece_binary": None, "brier_score": None, "generalization_status": None},
        {"method": "adaptive_fusion", **{k: fusion["adaptive_fusion"].get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe"]}, "ece_binary": None, "brier_score": None, "generalization_status": None},
    ]
    summary = {
        "experiment": "experiment_21C",
        "description": PIPELINE_DESCRIPTION,
        "front_locked_exp21B": front,
        "side_single_view": side,
        "fusion": fusion,
        "training_diagnostics": diagnostics,
        "reference_exp21B": REFERENCE_EXP21B,
        "reference_exp21A": REFERENCE_EXP21A,
        "reference_final_v1": REFERENCE_FINAL_V1,
        "reference_exp8": REFERENCE_EXP8,
    }
    pd.DataFrame(rows).to_csv(RESULTS_DIR_EXP21C / "experiment_21C_table.csv", index=False)
    (RESULTS_DIR_EXP21C / "experiment_21C_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR_EXP21C / "experiment_21C_summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def main():
    print(json.dumps(run_summary_exp21C(), indent=2))


if __name__ == "__main__":
    main()

