"""
Per-sample error analysis for Experiment 15.

This script does not train or tune models. It loads fixed front/side
checkpoints, runs inference on the paired test set, and exports per-sample
prediction outcomes for single-view and decision-level fusion methods.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from src.config import (
    BATCH_SIZE,
    BINARY_LABEL_MAP,
    DECISION_THRESHOLD,
    FRAME_STRIDE,
    MANIFEST_PAIRED_PATH,
    MANIFEST_SPLIT_PATH,
    RESULTS_DIR,
)
from src.dataset import get_transforms
from src.evaluate import load_trained_model
from src.fusion import adaptive_fusion, average_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class PairedMetadataDataset(Dataset):
    """Paired front/side test dataset that also returns stable sample metadata."""

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_front = Image.open(row["filepath_front"]).convert("RGB")
        img_side = Image.open(row["filepath_side"]).convert("RGB")
        img_front = self.transform(img_front)
        img_side = self.transform(img_side)
        label = BINARY_LABEL_MAP[row["binary_label"]]
        return img_front, img_side, label, idx


def load_paired_test_dataframe() -> pd.DataFrame:
    """Load paired test rows using the same subject split and frame stride as fusion.py."""
    df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
    test_subjects = set(df_split[df_split["split"] == "test"]["subject_id"].unique())

    df_paired = pd.read_csv(MANIFEST_PAIRED_PATH)
    df_test = df_paired[df_paired["subject_id"].isin(test_subjects)].copy()
    if df_test.empty:
        raise ValueError("Tidak ada data test di manifest_paired.csv")

    if FRAME_STRIDE > 1:
        df_test = df_test[(df_test["frame"] - 1) % FRAME_STRIDE == 0].copy()
        log.info(
            "Frame subsampling diterapkan (stride=%d): %d pasangan test tersisa",
            FRAME_STRIDE,
            len(df_test),
        )

    sort_cols = [c for c in ["subject_id", "activity_id", "frame"] if c in df_test.columns]
    if sort_cols:
        df_test = df_test.sort_values(sort_cols)

    df_test = df_test.reset_index(drop=True)
    df_test["sample_id"] = df_test.apply(
        lambda r: f"S{int(r['subject_id']):02d}_AC{int(r['activity_id']):02d}_F{int(r['frame']):05d}",
        axis=1,
    )
    return df_test


def get_paired_metadata_loader(df_test: pd.DataFrame, batch_size: int = BATCH_SIZE) -> DataLoader:
    transform = get_transforms("front", "test")
    dataset = PairedMetadataDataset(df_test, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


@torch.no_grad()
def collect_scores(model_front, model_side, loader, device):
    """Collect front/side positive-class probabilities, labels, and row indices."""
    model_front.eval()
    model_side.eval()

    scores_front = []
    scores_side = []
    labels_all = []
    indices_all = []

    for img_front, img_side, labels, indices in loader:
        img_front = img_front.to(device)
        img_side = img_side.to(device)

        logits_front = model_front(img_front)
        logits_side = model_side(img_side)

        probs_front = torch.softmax(logits_front, dim=1)[:, 1].cpu().numpy()
        probs_side = torch.softmax(logits_side, dim=1)[:, 1].cpu().numpy()

        scores_front.extend(probs_front.tolist())
        scores_side.extend(probs_side.tolist())
        labels_all.extend(labels.tolist())
        indices_all.extend(indices.tolist())

    return (
        np.asarray(scores_front, dtype=float),
        np.asarray(scores_side, dtype=float),
        np.asarray(labels_all, dtype=int),
        np.asarray(indices_all, dtype=int),
    )


def compute_metric_block(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 5),
        "precision_macro": round(float(precision_score(labels, preds, average="macro", zero_division=0)), 5),
        "recall_macro": round(float(recall_score(labels, preds, average="macro", zero_division=0)), 5),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 5),
    }


def classify_case(front_correct: bool, side_correct: bool, average_correct: bool, adaptive_correct: bool) -> str:
    """Assign one readable case label with fusion-specific events taking priority."""
    any_fusion_correct = average_correct or adaptive_correct
    any_fusion_wrong = (not average_correct) or (not adaptive_correct)

    if (not front_correct) and any_fusion_correct:
        return "fusion_fixed_front_error"
    if front_correct and any_fusion_wrong:
        return "fusion_broke_front_correct"
    if front_correct and side_correct:
        return "both_correct"
    if front_correct and not side_correct:
        return "front_only_correct"
    if side_correct and not front_correct:
        return "side_only_correct"
    return "both_wrong"


def misclassification_counts(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "safe_as_phone": int(((labels == 0) & (preds == 1)).sum()),
        "phone_as_safe": int(((labels == 1) & (preds == 0)).sum()),
    }


def build_outputs(df_test, scores_front, scores_side, labels, output_prefix: str):
    scores_avg = average_fusion(scores_front, scores_side)
    scores_adapt = adaptive_fusion(scores_front, scores_side)

    front_preds = (scores_front >= DECISION_THRESHOLD).astype(int)
    side_preds = (scores_side >= DECISION_THRESHOLD).astype(int)
    avg_preds = (scores_avg >= DECISION_THRESHOLD).astype(int)
    adapt_preds = (scores_adapt >= DECISION_THRESHOLD).astype(int)

    front_correct = front_preds == labels
    side_correct = side_preds == labels
    avg_correct = avg_preds == labels
    adapt_correct = adapt_preds == labels

    rows = []
    for i in range(len(labels)):
        rows.append({
            "sample_id": df_test.loc[i, "sample_id"],
            "true_label": int(labels[i]),
            "front_prob_safe": round(float(1.0 - scores_front[i]), 8),
            "front_prob_phone": round(float(scores_front[i]), 8),
            "front_pred": int(front_preds[i]),
            "side_prob_safe": round(float(1.0 - scores_side[i]), 8),
            "side_prob_phone": round(float(scores_side[i]), 8),
            "side_pred": int(side_preds[i]),
            "average_prob_safe": round(float(1.0 - scores_avg[i]), 8),
            "average_prob_phone": round(float(scores_avg[i]), 8),
            "average_pred": int(avg_preds[i]),
            "adaptive_prob_safe": round(float(1.0 - scores_adapt[i]), 8),
            "adaptive_prob_phone": round(float(scores_adapt[i]), 8),
            "adaptive_pred": int(adapt_preds[i]),
            "front_correct": bool(front_correct[i]),
            "side_correct": bool(side_correct[i]),
            "average_correct": bool(avg_correct[i]),
            "adaptive_correct": bool(adapt_correct[i]),
            "case_type": classify_case(
                bool(front_correct[i]),
                bool(side_correct[i]),
                bool(avg_correct[i]),
                bool(adapt_correct[i]),
            ),
        })

    df_out = pd.DataFrame(rows)

    case_counts = df_out["case_type"].value_counts().to_dict()
    front_mis = misclassification_counts(labels, front_preds)
    side_mis = misclassification_counts(labels, side_preds)
    avg_mis = misclassification_counts(labels, avg_preds)
    adapt_mis = misclassification_counts(labels, adapt_preds)

    summary = {
        "total_samples": int(len(labels)),
        "front_correct_count": int(front_correct.sum()),
        "side_correct_count": int(side_correct.sum()),
        "average_correct_count": int(avg_correct.sum()),
        "adaptive_correct_count": int(adapt_correct.sum()),
        "both_correct_count": int(((front_correct) & (side_correct)).sum()),
        "front_only_correct_count": int(((front_correct) & (~side_correct)).sum()),
        "side_only_correct_count": int(((side_correct) & (~front_correct)).sum()),
        "both_wrong_count": int(((~front_correct) & (~side_correct)).sum()),
        "fusion_fixed_front_error_count": int((df_out["case_type"] == "fusion_fixed_front_error").sum()),
        "fusion_broke_front_correct_count": int((df_out["case_type"] == "fusion_broke_front_correct").sum()),
        "safe_as_phone_front": front_mis["safe_as_phone"],
        "phone_as_safe_front": front_mis["phone_as_safe"],
        "safe_as_phone_side": side_mis["safe_as_phone"],
        "phone_as_safe_side": side_mis["phone_as_safe"],
        "safe_as_phone_average": avg_mis["safe_as_phone"],
        "phone_as_safe_average": avg_mis["phone_as_safe"],
        "safe_as_phone_adaptive": adapt_mis["safe_as_phone"],
        "phone_as_safe_adaptive": adapt_mis["phone_as_safe"],
        "case_type_counts": {k: int(v) for k, v in case_counts.items()},
        "metrics": {
            "front": compute_metric_block(labels, front_preds),
            "side": compute_metric_block(labels, side_preds),
            "average_fusion": compute_metric_block(labels, avg_preds),
            "adaptive_fusion": compute_metric_block(labels, adapt_preds),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"error_analysis_{output_prefix}.csv"
    summary_path = RESULTS_DIR / f"error_summary_{output_prefix}.json"
    df_out.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return csv_path, summary_path, summary


def main():
    parser = argparse.ArgumentParser(description="Per-sample error analysis for Experiment 15")
    parser.add_argument("--front-checkpoint", required=True, help="Path checkpoint front, e.g. checkpoints/front_best_exp14A.pt")
    parser.add_argument("--side-checkpoint", required=True, help="Path checkpoint side, e.g. checkpoints/side_best_exp13_backup.pt")
    parser.add_argument("--output-prefix", default="exp15", help="Output suffix, default: exp15")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model_front, _ = load_trained_model("front", device, checkpoint_path=args.front_checkpoint)
    model_side, _ = load_trained_model("side", device, checkpoint_path=args.side_checkpoint)

    df_test = load_paired_test_dataframe()
    loader = get_paired_metadata_loader(df_test)
    log.info("Running per-sample inference on %d paired test samples...", len(df_test))
    scores_front, scores_side, labels, indices = collect_scores(model_front, model_side, loader, device)

    if not np.array_equal(indices, np.arange(len(df_test))):
        raise AssertionError("Urutan DataLoader tidak sesuai dengan dataframe test.")

    csv_path, summary_path, summary = build_outputs(
        df_test=df_test,
        scores_front=scores_front,
        scores_side=scores_side,
        labels=labels,
        output_prefix=args.output_prefix,
    )

    log.info("CSV per-sample disimpan ke: %s", csv_path)
    log.info("Summary JSON disimpan ke: %s", summary_path)
    log.info("Ringkasan metrik: %s", json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
