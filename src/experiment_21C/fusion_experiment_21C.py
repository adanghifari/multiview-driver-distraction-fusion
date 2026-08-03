import json
import logging

import numpy as np
import pandas as pd

from src.experiment_21C.config_experiment_21C import RESULTS_DIR_EXP21C, THRESHOLD
from src.final_v1.metrics_final_v1 import compute_metrics
from src.fusion import adaptive_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def run_fusion_evaluation_exp21C():
    predictions_path = RESULTS_DIR_EXP21C / "single_view_predictions_exp21C.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions: {predictions_path}")

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

    prob_front = pivot["prob_phone_front_locked_exp21B"].to_numpy(dtype=float)
    prob_side = pivot["prob_phone_side"].to_numpy(dtype=float)
    labels = pivot["true_label"].to_numpy(dtype=int)
    prob_average = (prob_front + prob_side) / 2.0
    prob_adaptive = adaptive_fusion(prob_front, prob_side, threshold=THRESHOLD)
    pred_average = (prob_average >= THRESHOLD).astype(int)
    pred_adaptive = (prob_adaptive >= THRESHOLD).astype(int)

    fusion_metrics = {
        "average_fusion": compute_metrics(labels.tolist(), pred_average.tolist()),
        "adaptive_fusion": compute_metrics(labels.tolist(), pred_adaptive.tolist()),
        "adaptive_different_from_average": int((pred_adaptive != pred_average).sum()),
        "support": int(len(labels)),
        "front_checkpoint": "locked_exp21B",
        "side_checkpoint": "exp21C_val_loss_monitor",
    }
    df_out = pivot.copy()
    df_out["prob_phone_average"] = np.round(prob_average, 8)
    df_out["pred_average"] = pred_average
    df_out["prob_phone_adaptive"] = np.round(prob_adaptive, 8)
    df_out["pred_adaptive"] = pred_adaptive

    (RESULTS_DIR_EXP21C / "fusion_metrics_exp21C.json").write_text(json.dumps(fusion_metrics, indent=2), encoding="utf-8")
    df_out.to_csv(RESULTS_DIR_EXP21C / "fusion_predictions_exp21C.csv", index=False)
    log.info("Saved Experiment 21C fusion metrics and predictions to %s", RESULTS_DIR_EXP21C)
    return fusion_metrics


def main():
    print(json.dumps(run_fusion_evaluation_exp21C(), indent=2))


if __name__ == "__main__":
    main()

