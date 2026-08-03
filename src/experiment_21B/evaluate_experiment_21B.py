import json
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import MANIFEST_PAIRED_PATH, MANIFEST_SPLIT_PATH
from src.dataset import get_transforms
from src.evaluate import load_trained_model
from src.experiment_21B.config_experiment_21B import (
    FRONT_CONFIG,
    FRAME_STRIDE_EXP21B,
    RESULTS_DIR_EXP21B,
    SIDE_CONFIG_LOCKED,
    THRESHOLD,
)
from src.experiment_21.protocol import assert_checkpoint_frame_stride
from src.final_v1.dataset_final_v1 import PairedFinalV1Dataset
from src.final_v1.metrics_final_v1 import compute_brier_score, compute_ece_binary, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_paired_split_dataframe_exp21B(split_name: str) -> pd.DataFrame:
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be 'val' or 'test'.")
    df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
    subject_ids = set(df_split[df_split["split"] == split_name]["subject_id"].unique())
    df = pd.read_csv(MANIFEST_PAIRED_PATH)
    df = df[df["subject_id"].isin(subject_ids)].copy()
    df = df[(df["frame"] - 1) % FRAME_STRIDE_EXP21B == 0].copy()
    df = df.sort_values(["subject_id", "activity_id", "frame"]).reset_index(drop=True)
    df["sample_id"] = df.apply(
        lambda row: f"S{int(row['subject_id']):02d}_AC{int(row['activity_id']):02d}_F{int(row['frame']):05d}",
        axis=1,
    )
    return df


def get_paired_loader_exp21B(split_name: str) -> tuple[pd.DataFrame, DataLoader]:
    df = load_paired_split_dataframe_exp21B(split_name)
    dataset = PairedFinalV1Dataset(df, get_transforms("front", "test"))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    return df, loader


@torch.no_grad()
def collect_predictions(model, images, device):
    logits = model(images.to(device))
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = (probs >= THRESHOLD).astype(int)
    return probs, preds


def run_single_view_evaluation_exp21B():
    RESULTS_DIR_EXP21B.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_test, loader = get_paired_loader_exp21B("test")

    if not SIDE_CONFIG_LOCKED["checkpoint_path"].exists():
        raise FileNotFoundError(
            f"Locked Exp21A side checkpoint not found: {SIDE_CONFIG_LOCKED['checkpoint_path']}. "
            "Run Experiment 21A first or restore the checkpoint."
        )

    model_front, front_ckpt = load_trained_model("front", device, checkpoint_path=str(FRONT_CONFIG["checkpoint_path"]))
    model_side, side_ckpt = load_trained_model("side", device, checkpoint_path=str(SIDE_CONFIG_LOCKED["checkpoint_path"]))
    assert_checkpoint_frame_stride(front_ckpt, str(FRONT_CONFIG["checkpoint_path"]), FRAME_STRIDE_EXP21B, "Exp21B front")
    assert_checkpoint_frame_stride(side_ckpt, str(SIDE_CONFIG_LOCKED["checkpoint_path"]), FRAME_STRIDE_EXP21B, "Exp21A locked side")

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
            rows.append({**base, "view": "front", "prob_phone": round(float(probs_front[batch_idx]), 8), "pred_label": int(preds_front[batch_idx])})
            rows.append({**base, "view": "side_locked_exp21A", "prob_phone": round(float(probs_side[batch_idx]), 8), "pred_label": int(preds_side[batch_idx])})

    if indices_all != list(range(len(df_test))):
        raise AssertionError("Experiment 21B paired loader order is not synchronized with dataframe.")

    y_true = labels_all
    front_preds = [int(prob >= THRESHOLD) for prob in front_probs_all]
    side_preds = [int(prob >= THRESHOLD) for prob in side_probs_all]
    front_metrics = compute_metrics(y_true, front_preds)
    side_metrics = compute_metrics(y_true, side_preds)
    front_metrics.update(
        {
            "view": "front",
            "checkpoint_path": str(FRONT_CONFIG["checkpoint_path"]),
            "best_epoch": int(front_ckpt["epoch"]),
            "val_macro_f1": round(float(front_ckpt["val_macro_f1"]), 5),
            "ece_binary": compute_ece_binary(y_true, front_probs_all),
            "brier_score": compute_brier_score(y_true, front_probs_all),
            "support": len(y_true),
        }
    )
    side_metrics.update(
        {
            "view": "side_locked_exp21A",
            "checkpoint_path": str(SIDE_CONFIG_LOCKED["checkpoint_path"]),
            "best_epoch": int(side_ckpt["epoch"]),
            "val_macro_f1": round(float(side_ckpt["val_macro_f1"]), 5),
            "ece_binary": compute_ece_binary(y_true, side_probs_all),
            "brier_score": compute_brier_score(y_true, side_probs_all),
            "support": len(y_true),
        }
    )

    (RESULTS_DIR_EXP21B / "front_metrics_exp21B.json").write_text(json.dumps(front_metrics, indent=2), encoding="utf-8")
    (RESULTS_DIR_EXP21B / "side_locked_exp21A_metrics_exp21B.json").write_text(json.dumps(side_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(RESULTS_DIR_EXP21B / "single_view_predictions_exp21B.csv", index=False)
    log.info("Saved Experiment 21B single-view predictions and metrics to %s", RESULTS_DIR_EXP21B)
    return {"front_metrics": front_metrics, "side_locked_metrics": side_metrics}


def main():
    print(json.dumps(run_single_view_evaluation_exp21B(), indent=2))


if __name__ == "__main__":
    main()
