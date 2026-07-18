"""
Experiment 17: adaptive fusion weight analysis and sharpening sweep.

This script keeps the existing average fusion and legacy adaptive fusion
unchanged. It analyzes why legacy adaptive fusion matches average fusion, then
tests sharpened adaptive weighting at decision-level only.

Run:
    python -m src.adaptive_fusion_analysis --front-checkpoint checkpoints/front_best_exp14A.pt --side-checkpoint checkpoints/side_best_exp13_backup.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import DECISION_THRESHOLD, RESULTS_DIR
from src.error_analysis import classify_case, load_paired_test_dataframe
from src.evaluate import compute_metrics, load_trained_model
from src.fusion import adaptive_fusion, average_fusion, get_paired_scores, get_paired_test_loader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


ALPHAS = [1, 2, 3, 5, 10]
LABEL_NAMES = {0: "safe_driving", 1: "phone_use"}


def compute_confidence(scores: np.ndarray, threshold: float = DECISION_THRESHOLD) -> np.ndarray:
    return np.abs(scores - threshold)


def compute_adaptive_weights(
    scores_front: np.ndarray,
    scores_side: np.ndarray,
    alpha: float = 1.0,
    threshold: float = DECISION_THRESHOLD,
):
    confidence_front = compute_confidence(scores_front, threshold=threshold)
    confidence_side = compute_confidence(scores_side, threshold=threshold)

    scaled_front = alpha * confidence_front
    scaled_side = alpha * confidence_side

    exp_front = np.exp(scaled_front)
    exp_side = np.exp(scaled_side)
    denom = exp_front + exp_side

    weight_front = exp_front / denom
    weight_side = exp_side / denom
    return confidence_front, confidence_side, weight_front, weight_side


def adaptive_sharpened_fusion(
    scores_front: np.ndarray,
    scores_side: np.ndarray,
    alpha: float,
    threshold: float = DECISION_THRESHOLD,
):
    _, _, weight_front, weight_side = compute_adaptive_weights(
        scores_front,
        scores_side,
        alpha=alpha,
        threshold=threshold,
    )
    scores_fused = weight_front * scores_front + weight_side * scores_side
    return scores_fused, weight_front, weight_side


def count_directional_errors(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "safe_to_phone": int(((labels == 0) & (preds == 1)).sum()),
        "phone_to_safe": int(((labels == 1) & (preds == 0)).sum()),
    }


def build_sample_dataframe(df_test: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df_base = df_test.copy().reset_index(drop=True)
    df_base["true_label"] = labels.astype(int)
    df_base["true_label_name"] = df_base["true_label"].map(LABEL_NAMES)
    return df_base


def run_analysis(front_checkpoint: str, side_checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model_front, _ = load_trained_model("front", device, checkpoint_path=front_checkpoint)
    model_side, _ = load_trained_model("side", device, checkpoint_path=side_checkpoint)

    df_test = load_paired_test_dataframe()
    paired_loader = get_paired_test_loader()
    log.info("Running inference on %d paired test samples...", len(df_test))
    scores_front, scores_side, labels = get_paired_scores(model_front, model_side, paired_loader, device)

    if len(df_test) != len(labels):
        raise AssertionError("Jumlah sampel dataframe dan skor inferensi tidak sinkron.")

    average_scores = average_fusion(scores_front, scores_side)
    adaptive_scores_legacy = adaptive_fusion(scores_front, scores_side)

    front_preds = (scores_front >= DECISION_THRESHOLD).astype(int)
    side_preds = (scores_side >= DECISION_THRESHOLD).astype(int)
    average_preds = (average_scores >= DECISION_THRESHOLD).astype(int)
    adaptive_preds_legacy = (adaptive_scores_legacy >= DECISION_THRESHOLD).astype(int)

    front_correct = front_preds == labels
    side_correct = side_preds == labels
    average_correct = average_preds == labels
    adaptive_correct_legacy = adaptive_preds_legacy == labels

    df_base = build_sample_dataframe(df_test, labels)

    # Experiment 17A: analyze the legacy adaptive weights (alpha=1)
    confidence_front, confidence_side, weight_front, weight_side = compute_adaptive_weights(
        scores_front,
        scores_side,
        alpha=1.0,
    )
    weight_diff_abs = np.abs(weight_front - 0.5)

    case_types = [
        classify_case(
            bool(front_correct[i]),
            bool(side_correct[i]),
            bool(average_correct[i]),
            bool(adaptive_correct_legacy[i]),
        )
        for i in range(len(labels))
    ]

    df_exp17a = pd.DataFrame(
        {
            "sample_id": df_base["sample_id"],
            "true_label": labels.astype(int),
            "front_prob_safe": np.round(1.0 - scores_front, 8),
            "front_prob_phone": np.round(scores_front, 8),
            "side_prob_safe": np.round(1.0 - scores_side, 8),
            "side_prob_phone": np.round(scores_side, 8),
            "front_confidence": np.round(confidence_front, 8),
            "side_confidence": np.round(confidence_side, 8),
            "adaptive_weight_front": np.round(weight_front, 8),
            "adaptive_weight_side": np.round(weight_side, 8),
            "average_pred": average_preds.astype(int),
            "adaptive_pred": adaptive_preds_legacy.astype(int),
            "average_correct": average_correct.astype(bool),
            "adaptive_correct": adaptive_correct_legacy.astype(bool),
            "weight_difference_abs": np.round(weight_diff_abs, 8),
            "case_type": case_types,
        }
    )

    csv_exp17a = RESULTS_DIR / "adaptive_weight_analysis_exp17A.csv"
    summary_exp17a = RESULTS_DIR / "adaptive_weight_summary_exp17A.json"
    df_exp17a.to_csv(csv_exp17a, index=False)

    between_045_055 = int(((weight_front >= 0.45) & (weight_front <= 0.55)).sum())
    pred_diff_avg_vs_adapt = int((average_preds != adaptive_preds_legacy).sum())
    adaptive_better_than_average = int(((adaptive_correct_legacy) & (~average_correct)).sum())
    adaptive_worse_than_average = int(((~adaptive_correct_legacy) & (average_correct)).sum())

    summary_a = {
        "total_samples": int(len(labels)),
        "mean_weight_front": round(float(weight_front.mean()), 8),
        "mean_weight_side": round(float(weight_side.mean()), 8),
        "std_weight_front": round(float(weight_front.std()), 8),
        "std_weight_side": round(float(weight_side.std()), 8),
        "min_weight_front": round(float(weight_front.min()), 8),
        "max_weight_front": round(float(weight_front.max()), 8),
        "count_weight_front_between_0_45_and_0_55": between_045_055,
        "predictions_adaptive_different_from_average": pred_diff_avg_vs_adapt,
        "adaptive_correct_when_average_wrong": adaptive_better_than_average,
        "adaptive_wrong_when_average_correct": adaptive_worse_than_average,
        "mean_confidence_front": round(float(confidence_front.mean()), 8),
        "mean_confidence_side": round(float(confidence_side.mean()), 8),
        "max_weight_difference_abs": round(float(weight_diff_abs.max()), 8),
        "mean_weight_difference_abs": round(float(weight_diff_abs.mean()), 8),
    }
    summary_exp17a.write_text(json.dumps(summary_a, indent=2), encoding="utf-8")

    # Experiment 17B: sharpening sweep
    rows_exp17b = []
    summary_exp17b = {
        "baseline": {
            "average_fusion": compute_metrics(labels.tolist(), average_preds.tolist()),
            "adaptive_fusion_legacy": compute_metrics(labels.tolist(), adaptive_preds_legacy.tolist()),
        },
        "alphas": {},
    }

    best_alpha = None
    best_macro_f1 = -1.0

    for alpha in ALPHAS:
        fused_scores, w_front_alpha, w_side_alpha = adaptive_sharpened_fusion(
            scores_front,
            scores_side,
            alpha=alpha,
        )
        preds_alpha = (fused_scores >= DECISION_THRESHOLD).astype(int)
        metrics_alpha = compute_metrics(labels.tolist(), preds_alpha.tolist())
        directional = count_directional_errors(labels, preds_alpha)
        diff_vs_avg = int((preds_alpha != average_preds).sum())
        diff_vs_legacy = int((preds_alpha != adaptive_preds_legacy).sum())
        safe_recall = metrics_alpha["classification_report"]["safe_driving"]["recall"]
        phone_recall = metrics_alpha["classification_report"]["phone_use"]["recall"]

        summary_exp17b["alphas"][str(alpha)] = {
            "alpha": alpha,
            "accuracy": metrics_alpha["accuracy"],
            "precision_macro": metrics_alpha["precision_macro"],
            "recall_macro": metrics_alpha["recall_macro"],
            "macro_f1": metrics_alpha["f1_macro"],
            "confusion_matrix": metrics_alpha["confusion_matrix"],
            "safe_to_phone": directional["safe_to_phone"],
            "phone_to_safe": directional["phone_to_safe"],
            "predictions_different_from_average": diff_vs_avg,
            "predictions_different_from_legacy_adaptive": diff_vs_legacy,
            "safe_recall": round(float(safe_recall), 8),
            "phone_recall": round(float(phone_recall), 8),
            "mean_weight_front": round(float(w_front_alpha.mean()), 8),
            "mean_weight_side": round(float(w_side_alpha.mean()), 8),
            "std_weight_front": round(float(w_front_alpha.std()), 8),
            "std_weight_side": round(float(w_side_alpha.std()), 8),
            "min_weight_front": round(float(w_front_alpha.min()), 8),
            "max_weight_front": round(float(w_front_alpha.max()), 8),
        }

        rows_exp17b.append(
            {
                "alpha": alpha,
                "accuracy": metrics_alpha["accuracy"],
                "precision_macro": metrics_alpha["precision_macro"],
                "recall_macro": metrics_alpha["recall_macro"],
                "macro_f1": metrics_alpha["f1_macro"],
                "confusion_matrix": json.dumps(metrics_alpha["confusion_matrix"]),
                "safe_to_phone": directional["safe_to_phone"],
                "phone_to_safe": directional["phone_to_safe"],
                "predictions_different_from_average": diff_vs_avg,
                "predictions_different_from_legacy_adaptive": diff_vs_legacy,
                "safe_recall": round(float(safe_recall), 8),
                "phone_recall": round(float(phone_recall), 8),
            }
        )

        if metrics_alpha["f1_macro"] > best_macro_f1:
            best_macro_f1 = metrics_alpha["f1_macro"]
            best_alpha = alpha

    csv_exp17b = RESULTS_DIR / "adaptive_sharpening_exp17B.csv"
    json_exp17b = RESULTS_DIR / "adaptive_sharpening_exp17B.json"
    pd.DataFrame(rows_exp17b).to_csv(csv_exp17b, index=False)
    json_exp17b.write_text(json.dumps(summary_exp17b, indent=2), encoding="utf-8")

    notes_path = Path("experiment_17_notes.md")
    notes_path.write_text(
        build_notes(summary_a, summary_exp17b, best_alpha, best_macro_f1),
        encoding="utf-8",
    )

    return {
        "exp17a_csv": csv_exp17a,
        "exp17a_json": summary_exp17a,
        "exp17b_csv": csv_exp17b,
        "exp17b_json": json_exp17b,
        "notes": notes_path,
        "summary_a": summary_a,
        "summary_b": summary_exp17b,
        "best_alpha": best_alpha,
        "best_macro_f1": best_macro_f1,
    }


def build_notes(summary_a: dict, summary_b: dict, best_alpha: int, best_macro_f1: float) -> str:
    baseline_avg_f1 = summary_b["baseline"]["average_fusion"]["f1_macro"]
    baseline_adapt_f1 = summary_b["baseline"]["adaptive_fusion_legacy"]["f1_macro"]
    pred_diff = summary_a["predictions_adaptive_different_from_average"]
    mid_weight_count = summary_a["count_weight_front_between_0_45_and_0_55"]
    total_samples = summary_a["total_samples"]

    improved_alphas = [
        int(alpha_str)
        for alpha_str, block in summary_b["alphas"].items()
        if block["macro_f1"] > baseline_avg_f1
    ]

    alpha_lines = []
    for alpha in ALPHAS:
        block = summary_b["alphas"][str(alpha)]
        tradeoff = []
        if block["safe_recall"] < 0.5:
            tradeoff.append("safe recall rendah")
        if block["phone_recall"] < 0.95:
            tradeoff.append("phone recall turun")
        tradeoff_text = ", ".join(tradeoff) if tradeoff else "trade-off besar tidak menonjol"
        alpha_lines.append(
            f"- alpha={alpha}: Macro F1 {block['macro_f1']:.5f}, accuracy {block['accuracy']:.5f}, "
            f"safe->phone {block['safe_to_phone']}, phone->safe {block['phone_to_safe']}, "
            f"beda vs average {block['predictions_different_from_average']}, {tradeoff_text}."
        )

    if pred_diff == 0:
        identik_reason = (
            "Adaptive fusion lama identik dengan average fusion karena bobot yang dihasilkan "
            "tetap sangat dekat ke 0.5, sehingga skor gabungan tidak cukup bergeser untuk "
            "mengubah keputusan threshold."
        )
    else:
        identik_reason = (
            "Adaptive fusion lama hampir identik dengan average fusion karena variasi bobotnya "
            "kecil dan hanya sedikit sampel yang berpindah keputusan."
        )

    if improved_alphas:
        improvement_text = (
            f"Ada alpha yang memperbaiki Macro F1, yaitu {improved_alphas}, "
            f"dengan nilai terbaik pada alpha={best_alpha} sebesar {best_macro_f1:.5f}."
        )
    else:
        improvement_text = (
            f"Tidak ada alpha yang benar-benar memperbaiki Macro F1 di atas baseline "
            f"average/adaptive lama ({baseline_avg_f1:.5f}). Nilai terbaik tetap {best_macro_f1:.5f} "
            f"pada alpha={best_alpha}."
        )

    return "\n".join(
        [
            "# Experiment 17 Notes",
            "",
            "## Tujuan Experiment 17",
            "",
            "- Menganalisis kenapa adaptive fusion lama identik dengan average fusion.",
            "- Melihat distribusi bobot adaptive lama pada checkpoint front Exp14A revisi dan side Exp13.",
            "- Mencoba adaptive sharpening dengan faktor alpha di level decision-level fusion saja.",
            "- Menjaga hasil utama tetap sama sampai ada bukti bahwa variasi baru benar-benar lebih baik.",
            "",
            "## Alasan Adaptive Fusion Lama Identik Dengan Average Fusion",
            "",
            f"- Mean bobot front: {summary_a['mean_weight_front']:.5f}; mean bobot side: {summary_a['mean_weight_side']:.5f}.",
            f"- Simpangan baku bobot front: {summary_a['std_weight_front']:.5f}; rentang bobot front: {summary_a['min_weight_front']:.5f} sampai {summary_a['max_weight_front']:.5f}.",
            f"- Jumlah sampel dengan bobot front di antara 0.45-0.55: {mid_weight_count} dari {total_samples}.",
            f"- Jumlah prediksi adaptive lama yang berbeda dari average fusion: {pred_diff}.",
            f"- {identik_reason}",
            "",
            "## Hipotesis Sharpening Alpha",
            "",
            "- Dengan menaikkan alpha pada softmax(confidence), bobot diharapkan menjadi lebih tajam.",
            "- Jika perbedaan confidence front dan side memang informatif, alpha yang lebih besar seharusnya dapat menghasilkan prediksi yang berbeda dari average fusion.",
            "- Namun, sharpening tidak otomatis memperbaiki hasil; ia juga bisa memperbesar bias error dari view yang salah tetapi terlalu percaya diri.",
            "",
            "## Hasil Tiap Alpha",
            "",
            *alpha_lines,
            "",
            "## Apakah Ada Alpha Yang Benar-Benar Memperbaiki Macro F1",
            "",
            f"- Baseline average fusion: {baseline_avg_f1:.5f}.",
            f"- Baseline adaptive fusion lama: {baseline_adapt_f1:.5f}.",
            f"- {improvement_text}",
            "",
            "## Trade-off Safe Recall dan Phone Recall",
            "",
            "- Perubahan alpha perlu dibaca bersama perubahan `safe->phone` dan `phone->safe`, bukan hanya Macro F1.",
            "- Jika alpha yang lebih besar menurunkan `phone->safe` tetapi menaikkan `safe->phone`, maka peningkatan sensitivitas terhadap phone_use terjadi dengan trade-off false alarm pada safe_driving.",
            "- Sebaliknya, jika alpha tertentu menurunkan false alarm safe tetapi menaikkan `phone->safe`, maka sistem menjadi lebih konservatif terhadap deteksi phone_use.",
            "",
            "## Catatan Penting",
            "",
            "- Experiment 17 belum otomatis mengganti hasil utama penelitian.",
            "- Hasil utama tetap: Average Fusion = 0.78059 dan Adaptive Fusion lama = 0.78059.",
            "- Variasi adaptive sharpening pada experiment ini bersifat analisis fusion-level dan perlu direview sebelum dipertimbangkan lebih lanjut.",
        ]
    ) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Adaptive fusion analysis and sharpening sweep for Experiment 17.")
    parser.add_argument("--front-checkpoint", required=True, help="Path checkpoint front, e.g. checkpoints/front_best_exp14A.pt")
    parser.add_argument("--side-checkpoint", required=True, help="Path checkpoint side, e.g. checkpoints/side_best_exp13_backup.pt")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = run_analysis(args.front_checkpoint, args.side_checkpoint)

    print("=" * 70)
    print("EXPERIMENT 17 SUMMARY")
    print("=" * 70)
    print(f"Exp17A CSV   : {outputs['exp17a_csv']}")
    print(f"Exp17A JSON  : {outputs['exp17a_json']}")
    print(f"Exp17B CSV   : {outputs['exp17b_csv']}")
    print(f"Exp17B JSON  : {outputs['exp17b_json']}")
    print(f"Notes        : {outputs['notes'].resolve()}")
    print("-" * 70)
    print(
        "Legacy adaptive vs average different predictions:",
        outputs["summary_a"]["predictions_adaptive_different_from_average"],
    )
    print(
        "Weight front mean/std/min/max:",
        outputs["summary_a"]["mean_weight_front"],
        outputs["summary_a"]["std_weight_front"],
        outputs["summary_a"]["min_weight_front"],
        outputs["summary_a"]["max_weight_front"],
    )
    print("-" * 70)
    print("Sharpening sweep:")
    for alpha in ALPHAS:
        block = outputs["summary_b"]["alphas"][str(alpha)]
        print(
            f"alpha={alpha}: F1={block['macro_f1']:.5f}, Acc={block['accuracy']:.5f}, "
            f"safe->phone={block['safe_to_phone']}, phone->safe={block['phone_to_safe']}, "
            f"diff_vs_avg={block['predictions_different_from_average']}, "
            f"diff_vs_legacy={block['predictions_different_from_legacy_adaptive']}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
