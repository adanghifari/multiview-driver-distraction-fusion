import json

import pandas as pd

from src.final_v1.config_final_v1 import FRONT_CONFIG, REFERENCE_RESULTS, RESULTS_DIR_FINAL, SIDE_CONFIG
from src.final_v1.metrics_final_v1 import compute_brier_score, compute_ece_binary


def classify_generalization_status(train_loss, val_loss, val_macro_f1):
    if train_loss is None or val_loss is None or val_macro_f1 is None:
        return "N/A"
    gap = val_loss - train_loss
    if val_macro_f1 < 0.60 and train_loss > 0.60:
        return "UNDERFIT"
    if gap > 0.15:
        return "OVERFIT"
    if gap > 0.07:
        return "BORDERLINE"
    return "OK"


def build_training_diagnostics(train_summaries: dict) -> dict:
    diagnostics = {}
    for view, summary in train_summaries.items():
        if summary is None:
            diagnostics[view] = None
            continue
        hp = summary.get("hyperparameters", {})
        train_loss = summary.get("train_loss_at_best_epoch")
        val_loss = summary.get("validation_loss_at_best_epoch")
        val_macro_f1 = summary.get("best_validation_macro_f1")
        diagnostics[view] = {
            "best_epoch": summary.get("best_epoch"),
            "best_validation_macro_f1": val_macro_f1,
            "train_loss_at_best_epoch": train_loss,
            "validation_loss_at_best_epoch": val_loss,
            "train_validation_loss_gap_at_best_epoch": summary.get("train_validation_loss_gap_at_best_epoch"),
            "generalization_status": classify_generalization_status(train_loss, val_loss, val_macro_f1),
            "dropout": hp.get("dropout"),
            "weight_decay": hp.get("weight_decay"),
            "label_smoothing": hp.get("label_smoothing"),
            "num_stages_to_freeze": hp.get("num_stages_to_freeze"),
        }
    return diagnostics


def add_calibration_from_predictions(metrics: dict, view: str, predictions: pd.DataFrame | None) -> dict:
    metrics = dict(metrics)
    if metrics.get("ece_binary") is not None and metrics.get("brier_score") is not None:
        return metrics
    if predictions is None:
        return metrics

    view_df = predictions[predictions["view"] == view]
    if view_df.empty:
        return metrics

    metrics["ece_binary"] = compute_ece_binary(view_df["true_label"], view_df["prob_phone"])
    metrics["brier_score"] = compute_brier_score(view_df["true_label"], view_df["prob_phone"])
    return metrics


def reference_delta(actual: float, reference: float) -> float:
    return round(float(actual - reference), 5)


def build_summary_markdown(front_metrics: dict, side_metrics: dict, fusion_metrics: dict, train_summaries: dict) -> str:
    avg = fusion_metrics["average_fusion"]
    adapt = fusion_metrics["adaptive_fusion"]
    diff = fusion_metrics["adaptive_different_from_average"]
    diagnostics = build_training_diagnostics(train_summaries)

    front_ref = REFERENCE_RESULTS["front"]
    side_ref = REFERENCE_RESULTS["side"]
    avg_ref = REFERENCE_RESULTS["average_fusion"]
    adapt_ref = REFERENCE_RESULTS["adaptive_fusion"]
    fusion_delta = reference_delta(avg["f1_macro"], front_metrics["f1_macro"])

    return "\n".join(
        [
            "# Final v1 Summary",
            "",
            "- final_v1_retrain adalah pipeline end-to-end yang melatih ulang front dan side menggunakan konfigurasi final.",
            "- Output final_v1_retrain disimpan terpisah di `results/final_v1/` dan `checkpoints/final_v1/`.",
            "- Branch ini bukan eksperimen tuning baru, melainkan reproduksi final_v1 yang dikunci sementara dan bisa direproduksi.",
            "- Hasil retrain dibandingkan dengan hasil referensi terkunci, bukan langsung menggantikan klaim final utama.",
            f"- Referensi Front14A: Macro F1 {front_ref['macro_f1']:.5f}, confusion {front_ref['confusion_matrix']}.",
            f"- Referensi Side13: Macro F1 {side_ref['macro_f1']:.5f}, confusion {side_ref['confusion_matrix']}.",
            f"- Referensi Fusion baseline: Macro F1 {avg_ref['macro_f1']:.5f}, confusion {avg_ref['confusion_matrix']}.",
            "",
            "## Konfigurasi Retrain",
            "",
            "- Dataset tetap 3MDAD, task tetap binary `safe_driving` vs `phone_use`, view tetap front dan side.",
            "- Backbone tetap EfficientNetV2-S dan fusion tetap decision-level fusion.",
            "- Metrik utama tetap Macro F1; ECE dan Brier Score hanya diagnostics tambahan.",
            f"- Front mengikuti konfigurasi {FRONT_CONFIG['reference_name']}: lr={FRONT_CONFIG['learning_rate']}, wd={FRONT_CONFIG['weight_decay']}, dropout={FRONT_CONFIG['dropout']}, freeze={FRONT_CONFIG['num_stages_to_freeze']}.",
            f"- Side mengikuti konfigurasi {SIDE_CONFIG['reference_name']} berdasarkan artefak repo: lr={SIDE_CONFIG['learning_rate']}, wd={SIDE_CONFIG['weight_decay']}, dropout={SIDE_CONFIG['dropout']}, freeze={SIDE_CONFIG['num_stages_to_freeze']}.",
            f"- Catatan Side13: {SIDE_CONFIG['notes']}",
            "",
            "## Hasil Retrain Aktual",
            "",
            f"- Front best epoch: {front_metrics.get('best_epoch')}; val Macro F1: {front_metrics.get('val_macro_f1'):.5f}; test Macro F1: {front_metrics['f1_macro']:.5f}; confusion: {front_metrics['confusion_matrix']}.",
            f"- Side best epoch: {side_metrics.get('best_epoch')}; val Macro F1: {side_metrics.get('val_macro_f1'):.5f}; test Macro F1: {side_metrics['f1_macro']:.5f}; confusion: {side_metrics['confusion_matrix']}.",
            f"- Average fusion Macro F1: {avg['f1_macro']:.5f}; confusion: {avg['confusion_matrix']}.",
            f"- Adaptive fusion Macro F1: {adapt['f1_macro']:.5f}; confusion: {adapt['confusion_matrix']}.",
            f"- Jumlah prediksi adaptive berbeda dari average: {diff}.",
            f"- Delta Macro F1 average fusion terhadap front single-view: {fusion_delta:+.5f}.",
            f"- Average fusion menurunkan phone->safe error dari {front_metrics['phone_to_safe']} pada front menjadi {avg['phone_to_safe']}.",
            f"- Average fusion menaikkan safe->phone error dari {front_metrics['safe_to_phone']} pada front menjadi {avg['safe_to_phone']}.",
            "",
            "## Reproducibility Check",
            "",
            f"- Front reference F1 = {front_ref['macro_f1']:.5f}; actual = {front_metrics['f1_macro']:.5f}; delta = {reference_delta(front_metrics['f1_macro'], front_ref['macro_f1']):+.5f}.",
            f"- Side reference F1 = {side_ref['macro_f1']:.5f}; actual = {side_metrics['f1_macro']:.5f}; delta = {reference_delta(side_metrics['f1_macro'], side_ref['macro_f1']):+.5f}.",
            f"- Average Fusion reference F1 = {avg_ref['macro_f1']:.5f}; actual = {avg['f1_macro']:.5f}; delta = {reference_delta(avg['f1_macro'], avg_ref['macro_f1']):+.5f}.",
            f"- Adaptive Fusion reference F1 = {adapt_ref['macro_f1']:.5f}; actual = {adapt['f1_macro']:.5f}; delta = {reference_delta(adapt['f1_macro'], adapt_ref['macro_f1']):+.5f}.",
            "",
            "## Status Generalisasi dan Kalibrasi",
            "",
            f"- Front generalization status: {diagnostics.get('front', {}).get('generalization_status', 'N/A')}; ECE: {front_metrics.get('ece_binary', 'N/A')}; Brier Score: {front_metrics.get('brier_score', 'N/A')}.",
            f"- Side generalization status: {diagnostics.get('side', {}).get('generalization_status', 'N/A')}; ECE: {side_metrics.get('ece_binary', 'N/A')}; Brier Score: {side_metrics.get('brier_score', 'N/A')}.",
            "- Calibration diagnostics menunjukkan confidence model masih belum ideal, sehingga confidence-based adaptive fusion tidak otomatis lebih unggul.",
            "",
            "## Interpretasi",
            "",
            "- Front view adalah single-view terbaik pada final_v1_retrain.",
            "- Side view lebih lemah sebagai standalone, terutama karena recall `safe_driving` rendah dan kecenderungan bias ke `phone_use`, tetapi tetap berguna sebagai informasi komplementer untuk fusion.",
            "- Average fusion meningkatkan Macro F1 keseluruhan dan mengurangi missed detection `phone_use`, tetapi menambah false alarm pada `safe_driving`.",
            "- Adaptive fusion berbasis confidence belum terbukti mengungguli average fusion pada konfigurasi ini.",
            "- Average fusion menjadi strategi paling sederhana dan stabil untuk hasil utama final_v1_retrain.",
            f"- Pada run ini adaptive {'masih identik' if diff == 0 else 'tidak identik'} dengan average fusion.",
            "",
            "## Hubungan Dengan Eksperimen 18-20",
            "",
            "- Experiment 18 threshold tuning tidak mengganti hasil utama final_v1_retrain.",
            "- Experiment 19 class weighting belum terbukti mengungguli baseline secara valid; gamma 0.95 hanya exploratory test-best, bukan hasil final.",
            "- Validation-selected gamma 0.50 menghasilkan test Macro F1 0.77992, sangat dekat dengan baseline 0.78059 tetapi belum terbukti mengungguli.",
            "- Experiment 20 adaptive formula exploration menunjukkan variasi weighted-probability fusion tetap identik dengan average fusion.",
            "- Experiment 20 tidak mengubah rumus adaptive fusion utama; hasil utama tetap final_v1_retrain dengan fusion Macro F1 0.78059.",
        ]
    ) + "\n"


def run_summary():
    front_path = RESULTS_DIR_FINAL / "front_final_v1_metrics.json"
    side_path = RESULTS_DIR_FINAL / "side_final_v1_metrics.json"
    fusion_path = RESULTS_DIR_FINAL / "fusion_final_v1_metrics.json"
    front_train_path = RESULTS_DIR_FINAL / "front_train_summary_final_v1.json"
    side_train_path = RESULTS_DIR_FINAL / "side_train_summary_final_v1.json"

    for path in (front_path, side_path, fusion_path):
        if not path.exists():
            raise FileNotFoundError(f"File belum ada: {path}")

    front_metrics = json.loads(front_path.read_text(encoding="utf-8"))
    side_metrics = json.loads(side_path.read_text(encoding="utf-8"))
    fusion_metrics = json.loads(fusion_path.read_text(encoding="utf-8"))
    predictions_path = RESULTS_DIR_FINAL / "final_v1_single_view_predictions.csv"
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else None
    front_metrics = add_calibration_from_predictions(front_metrics, "front", predictions)
    side_metrics = add_calibration_from_predictions(side_metrics, "side", predictions)

    front_path.write_text(json.dumps(front_metrics, indent=2), encoding="utf-8")
    side_path.write_text(json.dumps(side_metrics, indent=2), encoding="utf-8")

    train_summaries = {
        "front": json.loads(front_train_path.read_text(encoding="utf-8")) if front_train_path.exists() else None,
        "side": json.loads(side_train_path.read_text(encoding="utf-8")) if side_train_path.exists() else None,
    }
    training_diagnostics = build_training_diagnostics(train_summaries)
    reference_comparison = {
        "front": {
            "reference_f1_macro": REFERENCE_RESULTS["front"]["macro_f1"],
            "actual_f1_macro": front_metrics["f1_macro"],
            "delta_f1_macro": reference_delta(front_metrics["f1_macro"], REFERENCE_RESULTS["front"]["macro_f1"]),
        },
        "side": {
            "reference_f1_macro": REFERENCE_RESULTS["side"]["macro_f1"],
            "actual_f1_macro": side_metrics["f1_macro"],
            "delta_f1_macro": reference_delta(side_metrics["f1_macro"], REFERENCE_RESULTS["side"]["macro_f1"]),
        },
        "average_fusion": {
            "reference_f1_macro": REFERENCE_RESULTS["average_fusion"]["macro_f1"],
            "actual_f1_macro": fusion_metrics["average_fusion"]["f1_macro"],
            "delta_f1_macro": reference_delta(
                fusion_metrics["average_fusion"]["f1_macro"], REFERENCE_RESULTS["average_fusion"]["macro_f1"]
            ),
        },
        "adaptive_fusion": {
            "reference_f1_macro": REFERENCE_RESULTS["adaptive_fusion"]["macro_f1"],
            "actual_f1_macro": fusion_metrics["adaptive_fusion"]["f1_macro"],
            "delta_f1_macro": reference_delta(
                fusion_metrics["adaptive_fusion"]["f1_macro"], REFERENCE_RESULTS["adaptive_fusion"]["macro_f1"]
            ),
        },
    }

    rows = [
        {
            "method": "front_single_view",
            "accuracy": front_metrics["accuracy"],
            "precision_macro": front_metrics["precision_macro"],
            "recall_macro": front_metrics["recall_macro"],
            "f1_macro": front_metrics["f1_macro"],
            "safe_to_phone": front_metrics["safe_to_phone"],
            "phone_to_safe": front_metrics["phone_to_safe"],
            "ece_binary": front_metrics.get("ece_binary"),
            "brier_score": front_metrics.get("brier_score"),
            "generalization_status": training_diagnostics.get("front", {}).get("generalization_status") if training_diagnostics.get("front") else None,
        },
        {
            "method": "side_single_view",
            "accuracy": side_metrics["accuracy"],
            "precision_macro": side_metrics["precision_macro"],
            "recall_macro": side_metrics["recall_macro"],
            "f1_macro": side_metrics["f1_macro"],
            "safe_to_phone": side_metrics["safe_to_phone"],
            "phone_to_safe": side_metrics["phone_to_safe"],
            "ece_binary": side_metrics.get("ece_binary"),
            "brier_score": side_metrics.get("brier_score"),
            "generalization_status": training_diagnostics.get("side", {}).get("generalization_status") if training_diagnostics.get("side") else None,
        },
        {
            "method": "average_fusion",
            "accuracy": fusion_metrics["average_fusion"]["accuracy"],
            "precision_macro": fusion_metrics["average_fusion"]["precision_macro"],
            "recall_macro": fusion_metrics["average_fusion"]["recall_macro"],
            "f1_macro": fusion_metrics["average_fusion"]["f1_macro"],
            "safe_to_phone": fusion_metrics["average_fusion"]["safe_to_phone"],
            "phone_to_safe": fusion_metrics["average_fusion"]["phone_to_safe"],
            "ece_binary": None,
            "brier_score": None,
            "generalization_status": None,
        },
        {
            "method": "adaptive_fusion",
            "accuracy": fusion_metrics["adaptive_fusion"]["accuracy"],
            "precision_macro": fusion_metrics["adaptive_fusion"]["precision_macro"],
            "recall_macro": fusion_metrics["adaptive_fusion"]["recall_macro"],
            "f1_macro": fusion_metrics["adaptive_fusion"]["f1_macro"],
            "safe_to_phone": fusion_metrics["adaptive_fusion"]["safe_to_phone"],
            "phone_to_safe": fusion_metrics["adaptive_fusion"]["phone_to_safe"],
            "ece_binary": None,
            "brier_score": None,
            "generalization_status": None,
        },
    ]

    summary = {
        "final_v1_retrain": True,
        "reference_results": REFERENCE_RESULTS,
        "side13_configuration_note": SIDE_CONFIG["notes"],
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
        },
    }

    table_path = RESULTS_DIR_FINAL / "final_v1_table.csv"
    summary_json_path = RESULTS_DIR_FINAL / "final_v1_summary.json"
    summary_md_path = RESULTS_DIR_FINAL / "final_v1_summary.md"

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
