import json
import logging

import pandas as pd

from src.final_v1.config_final_v1 import RESULTS_DIR_FINAL, THRESHOLD
from src.final_v1.metrics_final_v1 import compute_metrics
from src.fusion import adaptive_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def run_fusion_evaluation():
    predictions_path = RESULTS_DIR_FINAL / "final_v1_single_view_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"File prediksi single-view belum ada: {predictions_path}. "
            "Jalankan evaluate_single_view_final_v1 terlebih dahulu."
        )

    df = pd.read_csv(predictions_path)
    pivot = (
        df.pivot(
            index=["sample_id", "subject_id", "activity_id", "frame", "true_label"],
            columns="view",
            values=["prob_phone", "pred_label"],
        )
        .sort_index()
    )
    pivot.columns = [f"{left}_{right}" for left, right in pivot.columns]
    pivot = pivot.reset_index()

    prob_front = pivot["prob_phone_front"].to_numpy(dtype=float)
    prob_side = pivot["prob_phone_side"].to_numpy(dtype=float)
    labels = pivot["true_label"].to_numpy(dtype=int)

    prob_average = (prob_front + prob_side) / 2.0
    prob_adaptive = adaptive_fusion(prob_front, prob_side, threshold=THRESHOLD)

    pred_average = (prob_average >= THRESHOLD).astype(int)
    pred_adaptive = (prob_adaptive >= THRESHOLD).astype(int)
    diff_adaptive_vs_average = int((pred_adaptive != pred_average).sum())

    average_metrics = compute_metrics(labels.tolist(), pred_average.tolist())
    adaptive_metrics = compute_metrics(labels.tolist(), pred_adaptive.tolist())

    fusion_metrics = {
        "average_fusion": average_metrics,
        "adaptive_fusion": adaptive_metrics,
        "adaptive_different_from_average": diff_adaptive_vs_average,
    }

    df_out = pivot.copy()
    df_out["prob_safe_front"] = 1.0 - df_out["prob_phone_front"]
    df_out["prob_safe_side"] = 1.0 - df_out["prob_phone_side"]
    df_out["prob_phone_average"] = prob_average
    df_out["prob_safe_average"] = 1.0 - prob_average
    df_out["pred_average"] = pred_average
    df_out["prob_phone_adaptive"] = prob_adaptive
    df_out["prob_safe_adaptive"] = 1.0 - prob_adaptive
    df_out["pred_adaptive"] = pred_adaptive

    metrics_path = RESULTS_DIR_FINAL / "fusion_final_v1_metrics.json"
    predictions_out_path = RESULTS_DIR_FINAL / "fusion_final_v1_predictions.csv"

    metrics_path.write_text(json.dumps(fusion_metrics, indent=2), encoding="utf-8")
    df_out.to_csv(predictions_out_path, index=False)

    log.info("Saved final_v1 fusion metrics to: %s", metrics_path)
    log.info("Saved final_v1 fusion predictions to: %s", predictions_out_path)

    return {
        "fusion_metrics": fusion_metrics,
        "metrics_path": metrics_path,
        "predictions_path": predictions_out_path,
    }


def main():
    outputs = run_fusion_evaluation()
    print(json.dumps(outputs["fusion_metrics"], indent=2))


if __name__ == "__main__":
    main()
