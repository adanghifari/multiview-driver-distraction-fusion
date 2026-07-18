"""
Experiment 18: validation-based threshold tuning for fusion outputs.

This script does not retrain any model. It reuses fixed front/side checkpoints,
collects fusion probabilities on validation and test paired splits, searches
the best phone_use threshold on validation only, then evaluates that threshold
once on the test set.

Run:
    python -m src.threshold_analysis --front-checkpoint checkpoints/front_best_exp14A.pt --side-checkpoint checkpoints/side_best_exp13_backup.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
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
from src.evaluate import compute_metrics, load_trained_model
from src.fusion import adaptive_fusion, average_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


THRESHOLDS = np.round(np.arange(0.30, 0.80 + 0.001, 0.01), 2)
FUSION_METHODS = ("average_fusion", "adaptive_fusion")


class PairedSplitDataset(Dataset):
    """Paired front/side dataset for one split with stable metadata order."""

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_front = Image.open(row["filepath_front"]).convert("RGB")
        img_side = Image.open(row["filepath_side"]).convert("RGB")
        img_front = self.transform(img_front)
        img_side = self.transform(img_side)
        label = BINARY_LABEL_MAP[row["binary_label"]]
        return img_front, img_side, label, idx


def build_sample_id(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: f"S{int(r['subject_id']):02d}_AC{int(r['activity_id']):02d}_F{int(r['frame']):05d}",
        axis=1,
    )


def load_paired_split_dataframe(split: str) -> pd.DataFrame:
    if split not in {"val", "test"}:
        raise ValueError("split harus 'val' atau 'test'.")

    df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
    subjects = set(df_split[df_split["split"] == split]["subject_id"].unique())

    df_paired = pd.read_csv(MANIFEST_PAIRED_PATH)
    df_subset = df_paired[df_paired["subject_id"].isin(subjects)].copy()
    if df_subset.empty:
        raise ValueError(f"Tidak ada paired data untuk split={split}.")

    if FRAME_STRIDE > 1:
        df_subset = df_subset[(df_subset["frame"] - 1) % FRAME_STRIDE == 0].copy()
        log.info(
            "Frame subsampling diterapkan untuk split=%s (stride=%d): %d pasangan tersisa",
            split,
            FRAME_STRIDE,
            len(df_subset),
        )

    sort_cols = [c for c in ["subject_id", "activity_id", "frame"] if c in df_subset.columns]
    if sort_cols:
        df_subset = df_subset.sort_values(sort_cols)

    df_subset = df_subset.reset_index(drop=True)
    df_subset["sample_id"] = build_sample_id(df_subset)
    return df_subset


def get_paired_split_loader(df_split: pd.DataFrame, split: str, batch_size: int = BATCH_SIZE) -> DataLoader:
    transform = get_transforms("front", "test" if split in {"val", "test"} else split)
    dataset = PairedSplitDataset(df_split, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


@torch.no_grad()
def collect_scores(model_front, model_side, loader, device):
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


def count_directional_errors(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "safe_to_phone": int(((labels == 0) & (preds == 1)).sum()),
        "phone_to_safe": int(((labels == 1) & (preds == 0)).sum()),
    }


def evaluate_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    metrics = compute_metrics(labels.tolist(), preds.tolist())
    directional = count_directional_errors(labels, preds)
    return {
        "threshold": round(float(threshold), 2),
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "macro_f1": metrics["f1_macro"],
        "confusion_matrix": metrics["confusion_matrix"],
        "safe_to_phone": directional["safe_to_phone"],
        "phone_to_safe": directional["phone_to_safe"],
        "predictions": preds,
    }


def threshold_search(scores: np.ndarray, labels: np.ndarray, method_name: str) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        result = evaluate_threshold(scores, labels, float(threshold))
        rows.append(
            {
                "method": method_name,
                "threshold": result["threshold"],
                "accuracy": result["accuracy"],
                "precision_macro": result["precision_macro"],
                "recall_macro": result["recall_macro"],
                "macro_f1": result["macro_f1"],
                "confusion_matrix": json.dumps(result["confusion_matrix"]),
                "safe_to_phone": result["safe_to_phone"],
                "phone_to_safe": result["phone_to_safe"],
            }
        )
    return pd.DataFrame(rows)


def select_best_threshold(df_method: pd.DataFrame) -> pd.Series:
    df_ranked = df_method.copy()
    df_ranked["distance_to_half"] = (df_ranked["threshold"] - 0.5).abs()
    df_ranked = df_ranked.sort_values(
        by=["macro_f1", "phone_to_safe", "distance_to_half"],
        ascending=[False, True, True],
    )
    return df_ranked.iloc[0]


def metrics_block_for_json(result: dict) -> dict:
    return {
        "threshold": result["threshold"],
        "accuracy": result["accuracy"],
        "precision_macro": result["precision_macro"],
        "recall_macro": result["recall_macro"],
        "macro_f1": result["macro_f1"],
        "confusion_matrix": result["confusion_matrix"],
        "safe_to_phone": result["safe_to_phone"],
        "phone_to_safe": result["phone_to_safe"],
    }


def build_notes(test_summary: dict) -> str:
    avg = test_summary["average_fusion"]
    adapt = test_summary["adaptive_fusion"]

    def method_text(name: str, block: dict) -> list[str]:
        baseline = block["baseline_threshold_0_5"]
        tuned = block["selected_threshold_test"]
        best_val = block["best_validation_threshold"]
        improved = tuned["macro_f1"] > baseline["macro_f1"]
        return [
            f"### {name}",
            "",
            f"- Threshold terbaik dari validation set: {best_val['threshold']:.2f} dengan Macro F1 validation {best_val['macro_f1']:.5f}.",
            f"- Hasil test baseline threshold 0.50: Macro F1 {baseline['macro_f1']:.5f}, accuracy {baseline['accuracy']:.5f}, safe->phone {baseline['safe_to_phone']}, phone->safe {baseline['phone_to_safe']}.",
            f"- Hasil test dengan threshold terpilih: Macro F1 {tuned['macro_f1']:.5f}, accuracy {tuned['accuracy']:.5f}, safe->phone {tuned['safe_to_phone']}, phone->safe {tuned['phone_to_safe']}.",
            f"- Apakah Macro F1 test tembus 0.80: {'Ya' if tuned['macro_f1'] >= 0.80 else 'Tidak'}.",
            f"- Apakah ada peningkatan Macro F1 dibanding baseline 0.50: {'Ya' if improved else 'Tidak'}.",
        ]

    return "\n".join(
        [
            "# Experiment 18 Notes",
            "",
            "## Tujuan Experiment",
            "",
            "- Mencoba menaikkan Macro F1 fusion ke kisaran 0.80 tanpa training ulang.",
            "- Analisis dilakukan hanya pada level output probabilitas decision-level fusion.",
            "- Fokus utamanya adalah menguji apakah threshold phone_use selain 0.5 dapat mengurangi false alarm `safe->phone` tanpa menaikkan `phone->safe` terlalu banyak.",
            "",
            "## Alasan Threshold Tuning Dilakukan",
            "",
            "- Baseline fusion saat ini memiliki Macro F1 0.78059 dengan threshold 0.50.",
            "- Masalah utamanya adalah jumlah false alarm `safe->phone` yang masih tinggi dibanding `phone->safe`.",
            "- Karena skor fusion sudah tersedia sebagai probabilitas, threshold tuning dapat dilakukan tanpa retraining dan tanpa mengubah model.",
            "",
            "## Threshold Terbaik Berdasarkan Validation Set",
            "",
            *method_text("Average Fusion", avg),
            "",
            *method_text("Adaptive Fusion Lama", adapt),
            "",
            "## Perbandingan Dengan Baseline Threshold 0.5",
            "",
            "- Perbandingan utama dilihat dari perubahan Macro F1 test, serta trade-off antara `safe->phone` dan `phone->safe`.",
            "- Jika threshold terpilih menaikkan Macro F1 tetapi menambah `phone->safe` secara tajam, maka peningkatan tersebut perlu dibaca secara hati-hati.",
            "- Jika threshold terpilih menurunkan `safe->phone` tanpa merusak `phone->safe` terlalu banyak, maka tuning threshold dapat dianggap berguna pada level fusion-output.",
            "",
            "## Catatan Penting",
            "",
            "- Experiment 18 tidak menggunakan training ulang.",
            "- Checkpoint front dan side tetap sama seperti eksperimen sebelumnya.",
            "- Average fusion dan adaptive fusion lama tidak diubah rumusnya; yang dianalisis hanya threshold keputusan pada probabilitas output.",
            "- Hasil experiment ini bersifat analisis lanjutan dan belum otomatis mengganti hasil utama sebelum direview.",
        ]
    ) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Validation-based threshold tuning for fusion outputs.")
    parser.add_argument("--front-checkpoint", required=True, help="Path checkpoint front, e.g. checkpoints/front_best_exp14A.pt")
    parser.add_argument("--side-checkpoint", required=True, help="Path checkpoint side, e.g. checkpoints/side_best_exp13_backup.pt")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model_front, _ = load_trained_model("front", device, checkpoint_path=args.front_checkpoint)
    model_side, _ = load_trained_model("side", device, checkpoint_path=args.side_checkpoint)

    df_val = load_paired_split_dataframe("val")
    df_test = load_paired_split_dataframe("test")
    loader_val = get_paired_split_loader(df_val, split="val")
    loader_test = get_paired_split_loader(df_test, split="test")

    scores_front_val, scores_side_val, labels_val, indices_val = collect_scores(model_front, model_side, loader_val, device)
    scores_front_test, scores_side_test, labels_test, indices_test = collect_scores(model_front, model_side, loader_test, device)

    if not np.array_equal(indices_val, np.arange(len(df_val))):
        raise AssertionError("Urutan loader validation tidak sinkron dengan dataframe validation.")
    if not np.array_equal(indices_test, np.arange(len(df_test))):
        raise AssertionError("Urutan loader test tidak sinkron dengan dataframe test.")

    scores = {
        "average_fusion": {
            "val": average_fusion(scores_front_val, scores_side_val),
            "test": average_fusion(scores_front_test, scores_side_test),
        },
        "adaptive_fusion": {
            "val": adaptive_fusion(scores_front_val, scores_side_val),
            "test": adaptive_fusion(scores_front_test, scores_side_test),
        },
    }

    validation_rows = []
    test_summary = {}

    for method_name in FUSION_METHODS:
        df_search = threshold_search(scores[method_name]["val"], labels_val, method_name)
        validation_rows.append(df_search)
        best_row = select_best_threshold(df_search)

        baseline_test = evaluate_threshold(scores[method_name]["test"], labels_test, DECISION_THRESHOLD)
        selected_test = evaluate_threshold(scores[method_name]["test"], labels_test, float(best_row["threshold"]))

        test_summary[method_name] = {
            "best_validation_threshold": {
                "threshold": round(float(best_row["threshold"]), 2),
                "accuracy": float(best_row["accuracy"]),
                "precision_macro": float(best_row["precision_macro"]),
                "recall_macro": float(best_row["recall_macro"]),
                "macro_f1": float(best_row["macro_f1"]),
                "confusion_matrix": json.loads(best_row["confusion_matrix"]),
                "safe_to_phone": int(best_row["safe_to_phone"]),
                "phone_to_safe": int(best_row["phone_to_safe"]),
            },
            "baseline_threshold_0_5": metrics_block_for_json(baseline_test),
            "selected_threshold_test": metrics_block_for_json(selected_test),
            "macro_f1_improved_vs_baseline": bool(selected_test["macro_f1"] > baseline_test["macro_f1"]),
        }

    df_validation = pd.concat(validation_rows, ignore_index=True)
    validation_csv = RESULTS_DIR / "threshold_search_exp18_validation.csv"
    test_json = RESULTS_DIR / "threshold_test_exp18.json"
    notes_path = Path("experiment_18_notes.md")

    df_validation.to_csv(validation_csv, index=False)
    test_json.write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
    notes_path.write_text(build_notes(test_summary), encoding="utf-8")

    print("=" * 70)
    print("EXPERIMENT 18 SUMMARY")
    print("=" * 70)
    for method_name in FUSION_METHODS:
        block = test_summary[method_name]
        best_val = block["best_validation_threshold"]
        baseline = block["baseline_threshold_0_5"]
        selected = block["selected_threshold_test"]
        print(f"Method: {method_name}")
        print(
            f"  Best threshold validation : {best_val['threshold']:.2f} "
            f"(val Macro F1={best_val['macro_f1']:.5f}, phone->safe={best_val['phone_to_safe']})"
        )
        print(
            f"  Test baseline 0.50       : F1={baseline['macro_f1']:.5f}, "
            f"safe->phone={baseline['safe_to_phone']}, phone->safe={baseline['phone_to_safe']}"
        )
        print(
            f"  Test selected threshold  : F1={selected['macro_f1']:.5f}, "
            f"safe->phone={selected['safe_to_phone']}, phone->safe={selected['phone_to_safe']}"
        )
        print(
            f"  Macro F1 improved        : {'Ya' if block['macro_f1_improved_vs_baseline'] else 'Tidak'}"
        )
        print("-" * 70)
    print(f"Saved validation CSV: {validation_csv}")
    print(f"Saved test JSON     : {test_json}")
    print(f"Saved notes         : {notes_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
