import json

import pandas as pd

from src.final_v1.summarize_final_v1 import classify_generalization_status, reference_delta
from src.final_v2.config_final_v2 import (
    FRAME_STRIDE_FINAL,
    FRONT_CONFIG,
    REFERENCE_EXP21_STRIDE30_BEST,
    REFERENCE_FINAL_V1,
    RESULTS_DIR_FINAL,
    SIDE_CONFIG,
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _training_diag(summary):
    if summary is None:
        return None
    train_loss = summary.get("train_loss_at_best_epoch")
    val_loss = summary.get("validation_loss_at_best_epoch")
    val_macro_f1 = summary.get("best_validation_macro_f1")
    hp = summary.get("hyperparameters", {})
    return {
        "best_epoch": summary.get("best_epoch"),
        "best_validation_loss": summary.get("best_validation_loss"),
        "best_validation_macro_f1": val_macro_f1,
        "train_loss_at_best_epoch": train_loss,
        "validation_loss_at_best_epoch": val_loss,
        "train_validation_loss_gap_at_best_epoch": summary.get("train_validation_loss_gap_at_best_epoch"),
        "generalization_status": classify_generalization_status(train_loss, val_loss, val_macro_f1),
        "dropout": hp.get("dropout"),
        "weight_decay": hp.get("weight_decay"),
        "label_smoothing": hp.get("label_smoothing"),
        "num_stages_to_freeze": hp.get("num_stages_to_freeze"),
        "metric_for_best_model": hp.get("metric_for_best_model"),
        "checkpoint_monitor": hp.get("checkpoint_monitor"),
        "frame_stride": hp.get("frame_stride"),
    }


def build_summary_markdown(front_metrics: dict, side_metrics: dict, fusion_metrics: dict, train_summaries: dict) -> str:
    avg = fusion_metrics["average_fusion"]
    adapt = fusion_metrics["adaptive_fusion"]
    diff = fusion_metrics["adaptive_different_from_average"]
    diagnostics = {
        "front": _training_diag(train_summaries.get("front")),
        "side": _training_diag(train_summaries.get("side")),
    }
    return "\n".join(
        [
            "# Final v2 Summary",
            "",
            "- final_v2_retrain adalah pipeline end-to-end yang melatih ulang front dan side menggunakan konfigurasi Exp21 stride30-best.",
            "- Output final_v2 disimpan terpisah di `results/final_v2/` dan `checkpoints/final_v2/`.",
            "- Tujuan final_v2 adalah pembanding rapi terhadap final_v1 pada protokol stride 30 yang sama.",
            "",
            "## Konfigurasi Retrain",
            "",
            "- Dataset: 3MDAD, task binary `safe_driving` vs `phone_use`.",
            f"- Frame stride: {FRAME_STRIDE_FINAL}.",
            "- Backbone: EfficientNetV2-S.",
            "- Fusion: average fusion dan adaptive confidence fusion.",
            f"- Front mengikuti {FRONT_CONFIG['reference_name']}: lr={FRONT_CONFIG['learning_rate']}, wd={FRONT_CONFIG['weight_decay']}, dropout={FRONT_CONFIG['dropout']}, freeze={FRONT_CONFIG['num_stages_to_freeze']}, monitor={FRONT_CONFIG['metric_for_best_model']}.",
            f"- Side mengikuti {SIDE_CONFIG['reference_name']}: lr={SIDE_CONFIG['learning_rate']}, wd={SIDE_CONFIG['weight_decay']}, dropout={SIDE_CONFIG['dropout']}, freeze={SIDE_CONFIG['num_stages_to_freeze']}, monitor={SIDE_CONFIG['metric_for_best_model']}.",
            "",
            "## Hasil Retrain Aktual",
            "",
            f"- Front best epoch: {front_metrics.get('best_epoch')}; val Macro F1: {front_metrics.get('val_macro_f1'):.5f}; test Macro F1: {front_metrics['f1_macro']:.5f}; confusion: {front_metrics['confusion_matrix']}.",
            f"- Side best epoch: {side_metrics.get('best_epoch')}; val Macro F1: {side_metrics.get('val_macro_f1'):.5f}; test Macro F1: {side_metrics['f1_macro']:.5f}; confusion: {side_metrics['confusion_matrix']}.",
            f"- Average fusion Macro F1: {avg['f1_macro']:.5f}; confusion: {avg['confusion_matrix']}.",
            f"- Adaptive fusion Macro F1: {adapt['f1_macro']:.5f}; confusion: {adapt['confusion_matrix']}.",
            f"- Jumlah prediksi adaptive berbeda dari average: {diff}.",
            "",
            "## Perbandingan",
            "",
            f"- Final v1 average fusion Macro F1: {REFERENCE_FINAL_V1['average_fusion_f1_macro']:.5f}; final_v2 delta: {reference_delta(avg['f1_macro'], REFERENCE_FINAL_V1['average_fusion_f1_macro']):+.5f}.",
            f"- Exp21 stride30-best average fusion Macro F1: {REFERENCE_EXP21_STRIDE30_BEST['average_fusion_f1_macro']:.5f}; final_v2 delta: {reference_delta(avg['f1_macro'], REFERENCE_EXP21_STRIDE30_BEST['average_fusion_f1_macro']):+.5f}.",
            f"- Final v1 support: {REFERENCE_FINAL_V1['test_support']}; final_v2 support: {fusion_metrics.get('support')}.",
            "",
            "## Status Generalisasi dan Kalibrasi",
            "",
            f"- Front generalization status: {diagnostics.get('front', {}).get('generalization_status', 'N/A')}; ECE: {front_metrics.get('ece_binary', 'N/A')}; Brier Score: {front_metrics.get('brier_score', 'N/A')}.",
            f"- Side generalization status: {diagnostics.get('side', {}).get('generalization_status', 'N/A')}; ECE: {side_metrics.get('ece_binary', 'N/A')}; Brier Score: {side_metrics.get('brier_score', 'N/A')}.",
            "- Uji statistik final_v2 vs final_v1 tersedia melalui `python -m src.final_v2.statistical_tests_final_v2_vs_final_v1` setelah pipeline selesai.",
        ]
    ) + "\n"


def run_summary():
    front_path = RESULTS_DIR_FINAL / "front_final_v2_metrics.json"
    side_path = RESULTS_DIR_FINAL / "side_final_v2_metrics.json"
    fusion_path = RESULTS_DIR_FINAL / "fusion_final_v2_metrics.json"
    front_train_path = RESULTS_DIR_FINAL / "front_train_summary_final_v2.json"
    side_train_path = RESULTS_DIR_FINAL / "side_train_summary_final_v2.json"

    for path in (front_path, side_path, fusion_path):
        if not path.exists():
            raise FileNotFoundError(f"File belum ada: {path}")

    front_metrics = _read_json(front_path)
    side_metrics = _read_json(side_path)
    fusion_metrics = _read_json(fusion_path)
    train_summaries = {
        "front": _read_json(front_train_path) if front_train_path.exists() else None,
        "side": _read_json(side_train_path) if side_train_path.exists() else None,
    }
    training_diagnostics = {
        "front": _training_diag(train_summaries["front"]),
        "side": _training_diag(train_summaries["side"]),
    }
    reference_comparison = {
        "final_v1_average_fusion_delta": reference_delta(
            fusion_metrics["average_fusion"]["f1_macro"], REFERENCE_FINAL_V1["average_fusion_f1_macro"]
        ),
        "exp21_stride30_best_average_fusion_delta": reference_delta(
            fusion_metrics["average_fusion"]["f1_macro"], REFERENCE_EXP21_STRIDE30_BEST["average_fusion_f1_macro"]
        ),
    }
    rows = [
        {"method": "front_single_view", **{k: front_metrics.get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe", "ece_binary", "brier_score"]}, "generalization_status": training_diagnostics.get("front", {}).get("generalization_status") if training_diagnostics.get("front") else None},
        {"method": "side_single_view", **{k: side_metrics.get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe", "ece_binary", "brier_score"]}, "generalization_status": training_diagnostics.get("side", {}).get("generalization_status") if training_diagnostics.get("side") else None},
        {"method": "average_fusion", **{k: fusion_metrics["average_fusion"].get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe"]}, "ece_binary": None, "brier_score": None, "generalization_status": None},
        {"method": "adaptive_fusion", **{k: fusion_metrics["adaptive_fusion"].get(k) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "safe_to_phone", "phone_to_safe"]}, "ece_binary": None, "brier_score": None, "generalization_status": None},
    ]
    summary = {
        "final_v2_retrain": True,
        "reference_final_v1": REFERENCE_FINAL_V1,
        "reference_exp21_stride30_best": REFERENCE_EXP21_STRIDE30_BEST,
        "front_single_view": front_metrics,
        "side_single_view": side_metrics,
        "fusion": fusion_metrics,
        "training_summaries": train_summaries,
        "training_diagnostics": training_diagnostics,
        "reference_comparison": reference_comparison,
        "scope": {
            "dataset": "3MDAD",
            "task": "binary safe_driving vs phone_use",
            "views": ["front", "side"],
            "backbone": "EfficientNetV2-S",
            "fusion_level": "decision-level fusion",
            "primary_metric": "Macro F1",
            "frame_stride": FRAME_STRIDE_FINAL,
        },
    }

    table_path = RESULTS_DIR_FINAL / "final_v2_table.csv"
    summary_json_path = RESULTS_DIR_FINAL / "final_v2_summary.json"
    summary_md_path = RESULTS_DIR_FINAL / "final_v2_summary.md"

    pd.DataFrame(rows).to_csv(table_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(
        build_summary_markdown(front_metrics, side_metrics, fusion_metrics, train_summaries),
        encoding="utf-8",
    )

    return {
        "table_path": table_path,
        "summary_json_path": summary_json_path,
        "summary_md_path": summary_md_path,
        "summary": summary,
    }


def main():
    outputs = run_summary()
    print(json.dumps(outputs["summary"], indent=2))


if __name__ == "__main__":
    main()
