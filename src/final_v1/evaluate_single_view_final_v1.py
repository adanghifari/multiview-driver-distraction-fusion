import json
import logging

import pandas as pd
import torch

from src.final_v1.config_final_v1 import CHECKPOINTS_DIR_FINAL, RESULTS_DIR_FINAL, THRESHOLD
from src.final_v1.dataset_final_v1 import get_paired_split_loader, load_paired_split_dataframe
from src.final_v1.metrics_final_v1 import compute_brier_score, compute_ece_binary, compute_metrics
from src.evaluate import load_trained_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


FRONT_CHECKPOINT = CHECKPOINTS_DIR_FINAL / "front_best_final_v1.pt"
SIDE_CHECKPOINT = CHECKPOINTS_DIR_FINAL / "side_best_final_v1.pt"


@torch.no_grad()
def collect_single_view_predictions(model, images, device):
    images = images.to(device)
    logits = model(images)
    probs_phone = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = (probs_phone >= THRESHOLD).astype(int)
    return probs_phone, preds


def run_single_view_evaluation():
    RESULTS_DIR_FINAL.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_test = load_paired_split_dataframe("test")
    loader = get_paired_split_loader("test")

    model_front, front_ckpt = load_trained_model("front", device, checkpoint_path=str(FRONT_CHECKPOINT))
    model_side, side_ckpt = load_trained_model("side", device, checkpoint_path=str(SIDE_CHECKPOINT))

    rows = []
    front_probs_all = []
    side_probs_all = []
    labels_all = []
    indices_all = []

    for img_front, img_side, labels, indices in loader:
        probs_front, preds_front = collect_single_view_predictions(model_front, img_front, device)
        probs_side, preds_side = collect_single_view_predictions(model_side, img_side, device)

        front_probs_all.extend(probs_front.tolist())
        side_probs_all.extend(probs_side.tolist())
        labels_all.extend(labels.tolist())
        indices_all.extend(indices.tolist())

        for batch_idx in range(len(labels)):
            sample_index = int(indices[batch_idx])
            rows.append(
                {
                    "sample_id": df_test.loc[sample_index, "sample_id"],
                    "subject_id": int(df_test.loc[sample_index, "subject_id"]),
                    "activity_id": int(df_test.loc[sample_index, "activity_id"]),
                    "frame": int(df_test.loc[sample_index, "frame"]),
                    "true_label": int(labels[batch_idx]),
                    "view": "front",
                    "prob_safe": round(float(1.0 - probs_front[batch_idx]), 8),
                    "prob_phone": round(float(probs_front[batch_idx]), 8),
                    "pred_label": int(preds_front[batch_idx]),
                }
            )
            rows.append(
                {
                    "sample_id": df_test.loc[sample_index, "sample_id"],
                    "subject_id": int(df_test.loc[sample_index, "subject_id"]),
                    "activity_id": int(df_test.loc[sample_index, "activity_id"]),
                    "frame": int(df_test.loc[sample_index, "frame"]),
                    "true_label": int(labels[batch_idx]),
                    "view": "side",
                    "prob_safe": round(float(1.0 - probs_side[batch_idx]), 8),
                    "prob_phone": round(float(probs_side[batch_idx]), 8),
                    "pred_label": int(preds_side[batch_idx]),
                }
            )

    if indices_all != list(range(len(df_test))):
        raise AssertionError("Urutan paired test loader final_v1 tidak sinkron dengan dataframe.")

    front_preds_all = [int(prob >= THRESHOLD) for prob in front_probs_all]
    side_preds_all = [int(prob >= THRESHOLD) for prob in side_probs_all]

    front_metrics = compute_metrics(labels_all, front_preds_all)
    side_metrics = compute_metrics(labels_all, side_preds_all)
    front_metrics["ece_binary"] = compute_ece_binary(labels_all, front_probs_all)
    front_metrics["brier_score"] = compute_brier_score(labels_all, front_probs_all)
    side_metrics["ece_binary"] = compute_ece_binary(labels_all, side_probs_all)
    side_metrics["brier_score"] = compute_brier_score(labels_all, side_probs_all)

    front_metrics.update(
        {
            "view": "front",
            "checkpoint_path": str(FRONT_CHECKPOINT),
            "best_epoch": int(front_ckpt["epoch"]),
            "val_macro_f1": round(float(front_ckpt["val_macro_f1"]), 5),
        }
    )
    side_metrics.update(
        {
            "view": "side",
            "checkpoint_path": str(SIDE_CHECKPOINT),
            "best_epoch": int(side_ckpt["epoch"]),
            "val_macro_f1": round(float(side_ckpt["val_macro_f1"]), 5),
        }
    )

    front_metrics_path = RESULTS_DIR_FINAL / "front_final_v1_metrics.json"
    side_metrics_path = RESULTS_DIR_FINAL / "side_final_v1_metrics.json"
    predictions_path = RESULTS_DIR_FINAL / "final_v1_single_view_predictions.csv"

    front_metrics_path.write_text(json.dumps(front_metrics, indent=2), encoding="utf-8")
    side_metrics_path.write_text(json.dumps(side_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(predictions_path, index=False)

    log.info("Saved final_v1 front metrics to: %s", front_metrics_path)
    log.info("Saved final_v1 side metrics to: %s", side_metrics_path)
    log.info("Saved final_v1 single-view predictions to: %s", predictions_path)

    return {
        "front_metrics": front_metrics,
        "side_metrics": side_metrics,
        "front_metrics_path": front_metrics_path,
        "side_metrics_path": side_metrics_path,
        "predictions_path": predictions_path,
    }


def main():
    outputs = run_single_view_evaluation()
    print(
        json.dumps(
            {
                "front_final_v1_metrics": outputs["front_metrics"],
                "side_final_v1_metrics": outputs["side_metrics"],
                "single_view_predictions_csv": str(outputs["predictions_path"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
