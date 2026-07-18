"""
Validation-based gamma selection for Experiment 19.

Tujuan:
1. Evaluasi average fusion pada validation set untuk setiap checkpoint gamma.
2. Pilih gamma terbaik berdasarkan validation Macro F1.
3. Evaluasi SATU KALI gamma terpilih pada test set.

Pemilihan gamma TIDAK menggunakan test set agar menghindari test leakage.

Contoh:
  python -m src.gamma_validation_selection --front-checkpoint checkpoints/front_best_exp14A.pt
"""

from __future__ import annotations

import argparse
import csv
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
    CHECKPOINT_DIR,
    DECISION_THRESHOLD,
    FRAME_STRIDE,
    MANIFEST_PAIRED_PATH,
    MANIFEST_SPLIT_PATH,
    RESULTS_DIR,
)
from src.dataset import get_transforms
from src.evaluate import compute_metrics, load_trained_model
from src.fusion import average_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


BASELINE_TEST_MACRO_F1 = 0.78059
BASELINE_TEST_CONFUSION = [[20, 20], [4, 176]]


class PairedSplitDataset(Dataset):
    """Dataset pasangan front-side untuk split tertentu."""

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
        return img_front, img_side, label


def get_paired_loader(split: str, batch_size: int = BATCH_SIZE, num_workers: int = 0) -> DataLoader:
    """Bangun DataLoader paired untuk split val atau test."""
    if split not in {"val", "test"}:
        raise ValueError(f"split harus 'val' atau 'test', dapat: {split}")

    df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
    split_subjects = set(df_split[df_split["split"] == split]["subject_id"].unique())

    df_paired = pd.read_csv(MANIFEST_PAIRED_PATH)
    df_target = df_paired[df_paired["subject_id"].isin(split_subjects)]
    if df_target.empty:
        raise ValueError(f"Tidak ada data paired untuk split '{split}'")

    if FRAME_STRIDE > 1:
        df_target = df_target[(df_target["frame"] - 1) % FRAME_STRIDE == 0]
        log.info(
            "Frame subsampling diterapkan pada paired %s loader (stride=%d): %d pasangan tersisa",
            split,
            FRAME_STRIDE,
            len(df_target),
        )

    transform = get_transforms("front", "test")
    dataset = PairedSplitDataset(df_target, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    log.info("Paired %s DataLoader: n=%d pairs, batch_size=%d", split, len(dataset), batch_size)
    return loader


@torch.no_grad()
def get_paired_scores(model_front, model_side, paired_loader: DataLoader, device: torch.device):
    """Inferensi paired pada split tertentu."""
    model_front.eval()
    model_side.eval()

    all_scores_front = []
    all_scores_side = []
    all_labels = []

    for img_front, img_side, labels in paired_loader:
        img_front = img_front.to(device)
        img_side = img_side.to(device)

        logits_front = model_front(img_front)
        logits_side = model_side(img_side)

        probs_front = torch.softmax(logits_front, dim=1)[:, 1].cpu().numpy()
        probs_side = torch.softmax(logits_side, dim=1)[:, 1].cpu().numpy()

        all_scores_front.extend(probs_front.tolist())
        all_scores_side.extend(probs_side.tolist())
        all_labels.extend(labels.tolist())

    return np.array(all_scores_front), np.array(all_scores_side), np.array(all_labels)


def evaluate_average_fusion(scores_front: np.ndarray, scores_side: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluasi average fusion dengan threshold default."""
    fused_scores = average_fusion(scores_front, scores_side)
    preds = (fused_scores >= DECISION_THRESHOLD).astype(int)
    metrics = compute_metrics(labels.tolist(), preds.tolist())
    cm = metrics["confusion_matrix"]
    metrics["safe_to_phone"] = cm[0][1]
    metrics["phone_to_safe"] = cm[1][0]
    return metrics


def discover_gamma_checkpoints() -> list[dict]:
    """Temukan checkpoint gamma yang tersedia tanpa retraining."""
    mapping = [
        (0.25, CHECKPOINT_DIR / "side_best_exp19_gamma025.pt"),
        (0.50, CHECKPOINT_DIR / "side_best_exp19_gamma050.pt"),
        (0.75, CHECKPOINT_DIR / "side_best_exp19_gamma075.pt"),
        (0.85, CHECKPOINT_DIR / "side_best_exp19_gamma085.pt"),
        (0.90, CHECKPOINT_DIR / "side_best_exp19_gamma090.pt"),
        (0.93, CHECKPOINT_DIR / "side_best_exp19_gamma093.pt"),
        (0.94, CHECKPOINT_DIR / "side_best_exp19_gamma094.pt"),
        (0.95, CHECKPOINT_DIR / "side_best_exp19_gamma095.pt"),
        (0.96, CHECKPOINT_DIR / "side_best_exp19_gamma096.pt"),
        (0.97, CHECKPOINT_DIR / "side_best_exp19_gamma097.pt"),
        (0.98, CHECKPOINT_DIR / "side_best_exp19_gamma098.pt"),
        (1.00, CHECKPOINT_DIR / "side_best_exp19_gamma100.pt"),
        (1.00, CHECKPOINT_DIR / "side_best_exp19.pt"),
    ]

    discovered = []
    seen_gamma = set()
    for gamma, path in mapping:
        if gamma in seen_gamma:
            continue
        if path.exists():
            discovered.append({"gamma": gamma, "checkpoint": path})
            seen_gamma.add(gamma)

    if not discovered:
        raise FileNotFoundError("Tidak ada checkpoint gamma Experiment 19 yang ditemukan.")

    return discovered


def load_checkpoint_metadata(checkpoint_path: Path, device: torch.device) -> dict:
    """Ambil metadata hyperparameter dari checkpoint side."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = checkpoint.get("hyperparameters", {})
    class_weights = hp.get("class_weights", checkpoint.get("class_weights", [None, None]))
    return {
        "class_weight_safe": float(class_weights[0]),
        "class_weight_phone": float(class_weights[1]),
    }


def select_best_gamma(rows: list[dict]) -> dict:
    """Pilih gamma terbaik berdasarkan validation metrics saja."""
    return sorted(
        rows,
        key=lambda r: (
            -r["validation_macro_f1"],
            r["validation_phone_to_safe"],
            r["validation_safe_to_phone"],
            abs(r["gamma"] - 1.0),
        ),
    )[0]


def build_reason(selected: dict) -> str:
    """Bangun alasan pemilihan gamma."""
    return (
        "Dipilih berdasarkan validation Macro F1 tertinggi; "
        "jika seri, diprioritaskan phone->safe lebih rendah, lalu safe->phone lebih rendah, "
        "lalu gamma yang lebih sederhana/dekat baseline balanced (lebih dekat ke 1.00)."
    )


def main():
    parser = argparse.ArgumentParser(description="Validation-based gamma selection for Experiment 19")
    parser.add_argument(
        "--front-checkpoint",
        type=str,
        required=True,
        help="Path checkpoint front, misal checkpoints/front_best_exp14A.pt",
    )
    parser.add_argument(
        "--validation-output",
        type=str,
        default="results/gamma_selection_exp19_validation.csv",
        help="Path CSV hasil validation semua gamma",
    )
    parser.add_argument(
        "--selected-test-output",
        type=str,
        default="results/gamma_selection_exp19_selected_test.json",
        help="Path JSON hasil test gamma terpilih",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/gamma_selection_exp19_summary.json",
        help="Path JSON ringkasan seleksi gamma",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    validation_output = Path(args.validation_output)
    selected_test_output = Path(args.selected_test_output)
    summary_output = Path(args.summary_output)
    if not validation_output.is_absolute():
        validation_output = Path.cwd() / validation_output
    if not selected_test_output.is_absolute():
        selected_test_output = Path.cwd() / selected_test_output
    if not summary_output.is_absolute():
        summary_output = Path.cwd() / summary_output

    discovered = discover_gamma_checkpoints()
    log.info("Checkpoint gamma tersedia: %s", ", ".join(f"{d['gamma']:.2f}" for d in discovered))

    model_front, _ = load_trained_model("front", device, checkpoint_path=args.front_checkpoint)
    val_loader = get_paired_loader("val")
    test_loader = get_paired_loader("test")

    validation_rows = []
    for item in discovered:
        gamma = item["gamma"]
        side_checkpoint = item["checkpoint"]
        metadata = load_checkpoint_metadata(side_checkpoint, device)
        model_side, _ = load_trained_model("side", device, checkpoint_path=str(side_checkpoint))
        scores_front_val, scores_side_val, labels_val = get_paired_scores(model_front, model_side, val_loader, device)
        metrics_val = evaluate_average_fusion(scores_front_val, scores_side_val, labels_val)

        row = {
            "gamma": gamma,
            "side_checkpoint": str(side_checkpoint.relative_to(Path.cwd())),
            "class_weight_safe": round(metadata["class_weight_safe"], 4),
            "class_weight_phone": round(metadata["class_weight_phone"], 4),
            "validation_accuracy": metrics_val["accuracy"],
            "validation_precision_macro": metrics_val["precision_macro"],
            "validation_recall_macro": metrics_val["recall_macro"],
            "validation_macro_f1": metrics_val["f1_macro"],
            "validation_confusion_matrix": metrics_val["confusion_matrix"],
            "validation_safe_to_phone": metrics_val["safe_to_phone"],
            "validation_phone_to_safe": metrics_val["phone_to_safe"],
        }
        validation_rows.append(row)
        log.info(
            "Gamma %.2f | val_F1=%.5f | val_cm=%s",
            gamma,
            row["validation_macro_f1"],
            row["validation_confusion_matrix"],
        )

    selected = select_best_gamma(validation_rows)
    selected_gamma = selected["gamma"]
    selected_side_checkpoint = selected["side_checkpoint"]

    log.info("Gamma terpilih dari validation: %.2f", selected_gamma)

    model_side_selected, _ = load_trained_model("side", device, checkpoint_path=selected_side_checkpoint)
    scores_front_test, scores_side_test, labels_test = get_paired_scores(
        model_front, model_side_selected, test_loader, device
    )
    metrics_test = evaluate_average_fusion(scores_front_test, scores_side_test, labels_test)

    validation_output.parent.mkdir(parents=True, exist_ok=True)
    with validation_output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gamma",
                "side_checkpoint",
                "class_weight_safe",
                "class_weight_phone",
                "validation_accuracy",
                "validation_precision_macro",
                "validation_recall_macro",
                "validation_macro_f1",
                "validation_confusion_matrix",
                "validation_safe_to_phone",
                "validation_phone_to_safe",
            ],
        )
        writer.writeheader()
        for row in validation_rows:
            writer.writerow(row)

    selected_test_payload = {
        "selected_gamma": selected_gamma,
        "selected_side_checkpoint": selected_side_checkpoint,
        "alasan_selected_gamma": build_reason(selected),
        "test_accuracy": metrics_test["accuracy"],
        "test_precision_macro": metrics_test["precision_macro"],
        "test_recall_macro": metrics_test["recall_macro"],
        "test_macro_f1": metrics_test["f1_macro"],
        "test_confusion_matrix": metrics_test["confusion_matrix"],
        "test_safe_to_phone": metrics_test["safe_to_phone"],
        "test_phone_to_safe": metrics_test["phone_to_safe"],
        "baseline_test_macro_f1": BASELINE_TEST_MACRO_F1,
        "baseline_test_confusion": BASELINE_TEST_CONFUSION,
        "comparison_with_baseline": {
            "macro_f1_delta": round(metrics_test["f1_macro"] - BASELINE_TEST_MACRO_F1, 5),
            "safe_to_phone_delta": metrics_test["safe_to_phone"] - BASELINE_TEST_CONFUSION[0][1],
            "phone_to_safe_delta": metrics_test["phone_to_safe"] - BASELINE_TEST_CONFUSION[1][0],
        },
    }
    with selected_test_output.open("w") as f:
        json.dump(selected_test_payload, f, indent=2)

    summary_payload = {
        "protocol": "validation-based gamma selection to avoid test set leakage",
        "available_gammas": [row["gamma"] for row in validation_rows],
        "validation_results": validation_rows,
        "selected_gamma": selected_gamma,
        "selected_side_checkpoint": selected_side_checkpoint,
        "selection_rule": (
            "validation Macro F1 tertinggi; jika seri pilih phone->safe lebih rendah; "
            "jika masih seri pilih safe->phone lebih rendah; jika masih seri pilih gamma lebih dekat ke 1.00"
        ),
        "selected_test_result": selected_test_payload,
    }
    with summary_output.open("w") as f:
        json.dump(summary_payload, f, indent=2)

    log.info("Validation CSV disimpan di: %s", validation_output)
    log.info("Selected test JSON disimpan di: %s", selected_test_output)
    log.info("Summary JSON disimpan di: %s", summary_output)


if __name__ == "__main__":
    main()
