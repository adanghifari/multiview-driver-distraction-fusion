import json
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import MANIFEST_PAIRED_PATH, MANIFEST_SPLIT_PATH
from src.dataset import get_transforms
from src.evaluate import load_trained_model
from src.experiment_21.protocol import assert_checkpoint_frame_stride
from src.experiment_21E.config_experiment_21E import (
    FRAME_STRIDE_EXP21E,
    FRONT_CONFIG_LOCKED,
    RESULTS_DIR_EXP21E,
    SIDE_CONFIG,
    THRESHOLD,
)
from src.final_v1.dataset_final_v1 import PairedFinalV1Dataset
from src.final_v1.metrics_final_v1 import compute_brier_score, compute_ece_binary, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_paired_split_dataframe_exp21E(split_name: str) -> pd.DataFrame:
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be 'val' or 'test'.")
    df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
    subject_ids = set(df_split[df_split["split"] == split_name]["subject_id"].unique())
    df = pd.read_csv(MANIFEST_PAIRED_PATH)
    df = df[df["subject_id"].isin(subject_ids)].copy()
    df = df[(df["frame"] - 1) % FRAME_STRIDE_EXP21E == 0].copy()
    df = df.sort_values(["subject_id", "activity_id", "frame"]).reset_index(drop=True)
    df["sample_id"] = df.apply(
        lambda row: f"S{int(row['subject_id']):02d}_AC{int(row['activity_id']):02d}_F{int(row['frame']):05d}",
        axis=1,
    )
    return df


def get_paired_loader_exp21E(split_name: str) -> tuple[pd.DataFrame, DataLoader]:
    df = load_paired_split_dataframe_exp21E(split_name)
    dataset = PairedFinalV1Dataset(df, get_transforms("front", "test"))
    return df, DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def collect_predictions(model, images, device):
    logits = model(images.to(device))
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = (probs >= THRESHOLD).astype(int)
    return probs, preds


def run_single_view_evaluation_exp21E():
    RESULTS_DIR_EXP21E.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_test, loader = get_paired_loader_exp21E("test")

    if not FRONT_CONFIG_LOCKED["checkpoint_path"].exists():
        raise FileNotFoundError(
            f"Locked Exp21 front checkpoint not found: {FRONT_CONFIG_LOCKED['checkpoint_path']}. "
            "Run Experiment 21 stride30-best first or restore the checkpoint."
        )

    model_front, front_ckpt = load_trained_model("front", device, checkpoint_path=str(FRONT_CONFIG_LOCKED["checkpoint_path"]))
    model_side, side_ckpt = load_trained_model("side", device, checkpoint_path=str(SIDE_CONFIG["checkpoint_path"]))
    assert_checkpoint_frame_stride(front_ckpt, str(FRONT_CONFIG_LOCKED["checkpoint_path"]), FRAME_STRIDE_EXP21E, "Exp21 locked front")
    assert_checkpoint_frame_stride(side_ckpt, str(SIDE_CONFIG["checkpoint_path"]), FRAME_STRIDE_EXP21E, "Exp21E side")

    rows = []
    labels_all = []
    front_probs_all = []
    side_probs_all = []
    indices_all = []

    for img_front, img_side, labels, indices in loader:
        probs_front, preds_front = collect_predictions(model_front, img_front, device)
        probs_side, preds_side = collect_predictions(model_side, img_side, device)
        labels_all.extend(labels.tolist())
        front_probs_all.extend(probs_front.tolist())
        side_probs_all.extend(probs_side.tolist())
        indices_all.extend(indices.tolist())

        for batch_idx in range(len(labels)):
            sample_index = int(indices[batch_idx])
            base = {
                "sample_id": df_test.loc[sample_index, "sample_id"],
                "subject_id": int(df_test.loc[sample_index, "subject_id"]),
                "activity_id": int(df_test.loc[sample_index, "activity_id"]),
                "frame": int(df_test.loc[sample_index, "frame"]),
                "true_label": int(labels[batch_idx]),
            }
            rows.append({**base, "view": "front_locked_exp21", "prob_phone": round(float(probs_front[batch_idx]), 8), "pred_label": int(preds_front[batch_idx])})
            rows.append({**base, "view": "side_light_regularized", "prob_phone": round(float(probs_side[batch_idx]), 8), "pred_label": int(preds_side[batch_idx])})

    if indices_all != list(range(len(df_test))):
        raise AssertionError("Experiment 21E paired loader order is not synchronized with dataframe.")

    y_true = labels_all
    front_preds = [int(prob >= THRESHOLD) for prob in front_probs_all]
    side_preds = [int(prob >= THRESHOLD) for prob in side_probs_all]
    front_metrics = compute_metrics(y_true, front_preds)
    side_metrics = compute_metrics(y_true, side_preds)
    front_metrics.update(
        {
            "view": "front_locked_exp21",
            "checkpoint_path": str(FRONT_CONFIG_LOCKED["checkpoint_path"]),
            "best_epoch": int(front_ckpt["epoch"]),
            "val_macro_f1": round(float(front_ckpt["val_macro_f1"]), 5),
            "ece_binary": compute_ece_binary(y_true, front_probs_all),
            "brier_score": compute_brier_score(y_true, front_probs_all),
            "support": len(y_true),
        }
    )
    side_metrics.update(
        {
            "view": "side_light_regularized",
            "checkpoint_path": str(SIDE_CONFIG["checkpoint_path"]),
            "checkpoint_monitor": side_ckpt.get("checkpoint_monitor", "val_loss"),
            "best_epoch": int(side_ckpt["epoch"]),
            "val_loss": round(float(side_ckpt["val_loss"]), 5),
            "val_macro_f1": round(float(side_ckpt["val_macro_f1"]), 5),
            "ece_binary": compute_ece_binary(y_true, side_probs_all),
            "brier_score": compute_brier_score(y_true, side_probs_all),
            "support": len(y_true),
        }
    )

    (RESULTS_DIR_EXP21E / "front_locked_exp21_metrics_exp21E.json").write_text(json.dumps(front_metrics, indent=2), encoding="utf-8")
    (RESULTS_DIR_EXP21E / "side_metrics_exp21E.json").write_text(json.dumps(side_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(RESULTS_DIR_EXP21E / "single_view_predictions_exp21E.csv", index=False)
    log.info("Saved Experiment 21E single-view predictions and metrics to %s", RESULTS_DIR_EXP21E)
    return {"front_locked_metrics": front_metrics, "side_metrics": side_metrics}


def main():
    print(json.dumps(run_single_view_evaluation_exp21E(), indent=2))


if __name__ == "__main__":
    main()
