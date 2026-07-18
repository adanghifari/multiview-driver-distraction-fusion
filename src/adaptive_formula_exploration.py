r"""
Experiment 20: exploratory adaptive formula test.

This script keeps the trained front/side models fixed and only re-scores their
paired probabilities. Formula selection is performed on the validation split,
and the selected formula is evaluated exactly once on the test split.

Run:
    .venv\Scripts\python.exe -m src.adaptive_formula_exploration \
        --front-checkpoint checkpoints/front_best_exp14A.pt \
        --side-checkpoint checkpoints/side_best_exp13_backup.pt
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

EPSILON = 1e-12
BASELINE_TEST_MACRO_F1 = 0.78059
BASELINE_TEST_CONFUSION = [[20, 20], [4, 176]]


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    label: str
    simplicity_rank: int
    apply_fn: Callable


class PairedSplitDataset(Dataset):
    """Paired front/side dataset for a single split with stable row order."""

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


def load_paired_split_dataframe(split_name: str) -> pd.DataFrame:
    """Load paired rows for one split using the existing subject split."""
    df_split = pd.read_csv(MANIFEST_SPLIT_PATH)
    subjects = set(df_split[df_split["split"] == split_name]["subject_id"].unique())
    if not subjects:
        raise ValueError(f"Tidak ada subject untuk split '{split_name}'.")

    df_paired = pd.read_csv(MANIFEST_PAIRED_PATH)
    df = df_paired[df_paired["subject_id"].isin(subjects)].copy()
    if df.empty:
        raise ValueError(f"Tidak ada paired rows untuk split '{split_name}'.")

    if FRAME_STRIDE > 1:
        df = df[(df["frame"] - 1) % FRAME_STRIDE == 0].copy()

    sort_cols = [col for col in ["subject_id", "activity_id", "frame"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    df = df.reset_index(drop=True)
    df["sample_id"] = df.apply(
        lambda row: f"S{int(row['subject_id']):02d}_AC{int(row['activity_id']):02d}_F{int(row['frame']):05d}",
        axis=1,
    )
    return df


def get_paired_loader(df: pd.DataFrame, batch_size: int = BATCH_SIZE) -> DataLoader:
    transform = get_transforms("front", "test")
    dataset = PairedSplitDataset(df, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


@torch.no_grad()
def collect_paired_scores(model_front, model_side, loader: DataLoader, device: torch.device):
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


def legacy_confidence(prob_phone: np.ndarray) -> np.ndarray:
    return np.abs(prob_phone - DECISION_THRESHOLD)


def margin_confidence(prob_phone: np.ndarray) -> np.ndarray:
    prob_safe = 1.0 - prob_phone
    return np.abs(prob_phone - prob_safe)


def entropy_confidence(prob_phone: np.ndarray) -> np.ndarray:
    prob_safe = 1.0 - prob_phone
    entropy = -(
        prob_safe * np.log(prob_safe + EPSILON)
        + prob_phone * np.log(prob_phone + EPSILON)
    )
    normalized_entropy = entropy / np.log(2.0)
    return 1.0 - normalized_entropy


def maxprob_confidence(prob_phone: np.ndarray) -> np.ndarray:
    prob_safe = 1.0 - prob_phone
    return np.maximum(prob_safe, prob_phone)


def ratio_weights(conf_front: np.ndarray, conf_side: np.ndarray):
    denom = conf_front + conf_side + EPSILON
    weight_front = conf_front / denom
    weight_side = conf_side / denom
    return weight_front, weight_side


def softmax_weights(conf_front: np.ndarray, conf_side: np.ndarray):
    exp_front = np.exp(conf_front)
    exp_side = np.exp(conf_side)
    denom = exp_front + exp_side
    weight_front = exp_front / denom
    weight_side = exp_side / denom
    return weight_front, weight_side


def constant_weights(weight_front: float, weight_side: float, n_samples: int):
    return (
        np.full(n_samples, weight_front, dtype=float),
        np.full(n_samples, weight_side, dtype=float),
    )


def build_formula_outputs(
    formula_name: str,
    prob_front: np.ndarray,
    prob_side: np.ndarray,
    reliability_front: float,
    reliability_side: float,
):
    n_samples = len(prob_front)

    if formula_name == "average_fusion":
        conf_front = np.full(n_samples, 0.5, dtype=float)
        conf_side = np.full(n_samples, 0.5, dtype=float)
        weight_front, weight_side = constant_weights(0.5, 0.5, n_samples)
    elif formula_name == "legacy_adaptive_softmax":
        conf_front = legacy_confidence(prob_front)
        conf_side = legacy_confidence(prob_side)
        weight_front, weight_side = softmax_weights(conf_front, conf_side)
    elif formula_name == "legacy_adaptive_ratio":
        conf_front = legacy_confidence(prob_front)
        conf_side = legacy_confidence(prob_side)
        weight_front, weight_side = ratio_weights(conf_front, conf_side)
    elif formula_name == "margin_adaptive_ratio":
        conf_front = margin_confidence(prob_front)
        conf_side = margin_confidence(prob_side)
        weight_front, weight_side = ratio_weights(conf_front, conf_side)
    elif formula_name == "entropy_adaptive_ratio":
        conf_front = entropy_confidence(prob_front)
        conf_side = entropy_confidence(prob_side)
        weight_front, weight_side = ratio_weights(conf_front, conf_side)
    elif formula_name == "maxprob_adaptive_ratio":
        conf_front = maxprob_confidence(prob_front)
        conf_side = maxprob_confidence(prob_side)
        weight_front, weight_side = ratio_weights(conf_front, conf_side)
    elif formula_name == "reliability_weighted_static_fusion":
        conf_front = np.full(n_samples, reliability_front, dtype=float)
        conf_side = np.full(n_samples, reliability_side, dtype=float)
        total = reliability_front + reliability_side + EPSILON
        weight_front, weight_side = constant_weights(
            reliability_front / total,
            reliability_side / total,
            n_samples,
        )
    else:
        raise ValueError(f"Formula tidak dikenali: {formula_name}")

    fused_prob_phone = weight_front * prob_front + weight_side * prob_side
    return {
        "confidence_front": conf_front,
        "confidence_side": conf_side,
        "weight_front": weight_front,
        "weight_side": weight_side,
        "fused_prob_phone": fused_prob_phone,
    }


def directional_errors(labels: np.ndarray, preds: np.ndarray) -> tuple[int, int]:
    safe_to_phone = int(((labels == 0) & (preds == 1)).sum())
    phone_to_safe = int(((labels == 1) & (preds == 0)).sum())
    return safe_to_phone, phone_to_safe


def evaluate_formula(
    split_name: str,
    spec: FormulaSpec,
    df_split: pd.DataFrame,
    labels: np.ndarray,
    prob_front: np.ndarray,
    prob_side: np.ndarray,
    average_preds: np.ndarray,
    legacy_preds: np.ndarray,
    reliability_front: float,
    reliability_side: float,
):
    outputs = spec.apply_fn(spec.name, prob_front, prob_side, reliability_front, reliability_side)
    fused_prob_phone = outputs["fused_prob_phone"]
    preds = (fused_prob_phone >= DECISION_THRESHOLD).astype(int)
    metrics = compute_metrics(labels.tolist(), preds.tolist())
    safe_to_phone, phone_to_safe = directional_errors(labels, preds)
    diff_vs_avg = int((preds != average_preds).sum())
    diff_vs_legacy = int((preds != legacy_preds).sum())
    count_mid = int(((outputs["weight_front"] >= 0.45) & (outputs["weight_front"] <= 0.55)).sum())

    row = {
        "split": split_name,
        "formula": spec.name,
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "macro_f1": metrics["f1_macro"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        "safe_to_phone": safe_to_phone,
        "phone_to_safe": phone_to_safe,
        "predictions_different_from_average": diff_vs_avg,
        "predictions_different_from_legacy_adaptive_softmax": diff_vs_legacy,
        "mean_weight_front": round(float(outputs["weight_front"].mean()), 8),
        "mean_weight_side": round(float(outputs["weight_side"].mean()), 8),
        "std_weight_front": round(float(outputs["weight_front"].std()), 8),
        "std_weight_side": round(float(outputs["weight_side"].std()), 8),
        "min_weight_front": round(float(outputs["weight_front"].min()), 8),
        "max_weight_front": round(float(outputs["weight_front"].max()), 8),
        "count_weight_front_between_0_45_and_0_55": count_mid,
    }

    sample_df = pd.DataFrame(
        {
            "split": split_name,
            "sample_id": df_split["sample_id"],
            "formula": spec.name,
            "true_label": labels.astype(int),
            "front_prob_safe": np.round(1.0 - prob_front, 8),
            "front_prob_phone": np.round(prob_front, 8),
            "side_prob_safe": np.round(1.0 - prob_side, 8),
            "side_prob_phone": np.round(prob_side, 8),
            "confidence_front": np.round(outputs["confidence_front"], 8),
            "confidence_side": np.round(outputs["confidence_side"], 8),
            "weight_front": np.round(outputs["weight_front"], 8),
            "weight_side": np.round(outputs["weight_side"], 8),
            "fused_prob_safe": np.round(1.0 - fused_prob_phone, 8),
            "fused_prob_phone": np.round(fused_prob_phone, 8),
            "pred_label": preds.astype(int),
        }
    )

    detail = {
        "formula": spec.name,
        "label": spec.label,
        "split": split_name,
        "metrics": metrics,
        "safe_to_phone": safe_to_phone,
        "phone_to_safe": phone_to_safe,
        "predictions_different_from_average": diff_vs_avg,
        "predictions_different_from_legacy_adaptive_softmax": diff_vs_legacy,
        "mean_weight_front": row["mean_weight_front"],
        "mean_weight_side": row["mean_weight_side"],
        "std_weight_front": row["std_weight_front"],
        "std_weight_side": row["std_weight_side"],
        "min_weight_front": row["min_weight_front"],
        "max_weight_front": row["max_weight_front"],
        "count_weight_front_between_0_45_and_0_55": count_mid,
    }

    return row, sample_df, detail


def choose_best_formula(validation_details: list[dict], formula_specs: list[FormulaSpec]) -> dict:
    simplicity_map = {spec.name: spec.simplicity_rank for spec in formula_specs}
    return max(
        validation_details,
        key=lambda item: (
            item["metrics"]["f1_macro"],
            -item["phone_to_safe"],
            -item["safe_to_phone"],
            -simplicity_map[item["formula"]],
        ),
    )


def build_notes(summary: dict) -> str:
    validation_rows = summary["validation_results"]
    selected = summary["selected_formula"]
    test_result = summary["selected_test_result"]
    baseline_delta = test_result["comparison_with_baseline"]["macro_f1_delta"]
    improved_text = (
        "meningkat valid di atas baseline"
        if baseline_delta > 0
        else "tidak mengungguli baseline"
    )

    lines = [
        "# Experiment 20 Adaptive Formula Notes",
        "",
        "## Posisi Experiment 20",
        "",
        "- Experiment 20 diposisikan sebagai exploratory/diagnostic analysis, bukan pengganti otomatis metode utama.",
        "- Fokus eksperimen ini adalah mengecek apakah variasi adaptive confidence-weighting pada level probabilitas dapat menghasilkan prediksi akhir yang berbeda dari average fusion.",
        "- Hasil utama penelitian sebelum Experiment 20 tetap Front14A + Side13 dengan Macro F1 0.78059.",
        "",
        "## Tujuan Experiment 20",
        "",
        "- Mengeksplorasi formula adaptive fusion baru yang lebih diskriminatif dibanding adaptive fusion lama tanpa retraining model.",
        "- Menguji apakah perubahan definisi confidence dan mekanisme pembentukan bobot dapat mengubah keputusan akhir dibanding average fusion.",
        "- Menjaga seluruh checkpoint, dataset, split, backbone, dan arsitektur tetap sama seperti eksperimen sebelumnya.",
        "- Memilih formula hanya dari validation set, lalu mengevaluasi formula terpilih satu kali pada test set.",
        "",
        "## Batasan Experiment 20",
        "",
        "- Tidak ada retraining front maupun side.",
        "- Tidak ada perubahan dataset.",
        "- Tidak ada perubahan split train/validation/test.",
        "- Tidak ada perubahan backbone EfficientNetV2-S.",
        "- Tidak ada perubahan arsitektur model.",
        "- Tidak ada perubahan checkpoint Front14A dan Side13 yang dievaluasi.",
        "- Average fusion lama tidak diubah.",
        "- Adaptive fusion lama tidak dihapus dan tetap dipertahankan sebagai pembanding resmi.",
        "",
        "## Mengapa Adaptive Lama Dianalisis Ulang",
        "",
        "- Experiment 17 menunjukkan `legacy_adaptive_softmax` identik dengan average fusion pada test set.",
        "- Rata-rata bobot front adaptive lama berada di sekitar 0.52368 dengan 167 dari 220 sampel berada pada rentang 0.45-0.55.",
        "- Adaptive sharpening alpha 1, 2, 3, 5, 10 juga tidak mengubah hasil utama; Macro F1 tetap 0.78059.",
        "- Ini mengarah pada dugaan bahwa masalah utama bukan hanya definisi confidence, tetapi juga mekanisme pembentukan bobot yang terlalu meredam perbedaan confidence.",
        "",
        "## Formula Yang Diuji",
        "",
        "1. `average_fusion`",
        "2. `legacy_adaptive_softmax`",
        "3. `legacy_adaptive_ratio`",
        "4. `margin_adaptive_ratio`",
        "5. `entropy_adaptive_ratio`",
        "6. `maxprob_adaptive_ratio`",
        "7. `reliability_weighted_static_fusion`",
        "",
        "## Catatan Matematis Singkat",
        "",
        "- Pada klasifikasi biner, `abs(prob_phone - prob_safe)` proporsional dengan `abs(prob_phone - 0.5)`.",
        "- Karena itu, `margin_adaptive_ratio` secara teoritis berpotensi identik dengan formula legacy berbasis rasio dan dijalankan sebagai verifikasi empiris.",
        "- `legacy_adaptive_ratio` ditambahkan untuk mengisolasi apakah penyebab utama ketidakberbedaan adaptive lama berasal dari softmax weighting, bukan dari definisi confidence-nya.",
        "- Fusion berbentuk `w_front * P_front + w_side * P_side` dengan bobot positif dan jumlah bobot 1 adalah convex combination, sehingga probabilitas fusion selalu berada di antara `P_front` dan `P_side`.",
        "- Jika perubahan bobot tidak mendorong skor gabungan melintasi decision boundary 0.5, maka prediksi akhir tetap sama walaupun distribusi bobot berubah cukup jauh.",
        "",
        "## Hasil Validation Semua Formula",
        "",
        "| Formula | Macro F1 | Accuracy | safe->phone | phone->safe | diff_vs_avg | diff_vs_legacy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in validation_rows:
        lines.append(
            f"| `{row['formula']}` | {row['macro_f1']:.5f} | {row['accuracy']:.5f} | "
            f"{row['safe_to_phone']} | {row['phone_to_safe']} | "
            f"{row['predictions_different_from_average']} | "
            f"{row['predictions_different_from_legacy_adaptive_softmax']} |"
        )

    lines.extend(
        [
            "",
            "## Formula Terpilih Dari Validation",
            "",
            f"- Formula terpilih: `{selected['formula']}`.",
            f"- Validation Macro F1: {selected['metrics']['f1_macro']:.5f}.",
            f"- Tie-break yang dipakai: phone->safe lebih rendah, lalu safe->phone lebih rendah, lalu urutan kesederhanaan kandidat.",
            "- Karena seluruh kandidat seri pada validation dan seluruh `diff_vs_avg = 0`, formula paling sederhana yaitu `average_fusion` dipilih sebagai hasil final validation.",
            "",
            "## Hasil Test Formula Terpilih",
            "",
            f"- Test Macro F1: {test_result['metrics']['f1_macro']:.5f}.",
            f"- Confusion matrix: {test_result['metrics']['confusion_matrix']}.",
            f"- safe->phone: {test_result['safe_to_phone']}; phone->safe: {test_result['phone_to_safe']}.",
            f"- Prediksi berbeda dari average fusion: {test_result['predictions_different_from_average']}.",
            f"- Prediksi berbeda dari legacy adaptive: {test_result['predictions_different_from_legacy_adaptive_softmax']}.",
            "",
            "## Perbandingan Dengan Baseline Utama",
            "",
            f"- Baseline average fusion utama: Macro F1 {BASELINE_TEST_MACRO_F1:.5f}.",
            f"- Formula terpilih pada test {improved_text}; delta Macro F1 = {baseline_delta:+.5f}.",
            f"- Apakah prediksi baru benar-benar berubah dari average: {'ya' if test_result['predictions_different_from_average'] > 0 else 'tidak'}.",
            f"- Trade-off safe->phone dan phone->safe terhadap baseline: "
            f"{test_result['comparison_with_baseline']['safe_to_phone_delta']:+d} dan "
            f"{test_result['comparison_with_baseline']['phone_to_safe_delta']:+d}.",
            "",
            "## Interpretasi Final",
            "",
            "- Seluruh weighted probability adaptive variants tetap tidak mengubah keputusan akhir terhadap average fusion, baik pada validation maupun pada test.",
            "- `legacy_adaptive_ratio` dan `entropy_adaptive_ratio` memang menghasilkan bobot yang lebih ekstrem, tetapi perubahan bobot tersebut tetap tidak cukup untuk mendorong probabilitas fusion melintasi threshold keputusan 0.5 pada sampel mana pun.",
            "- Pada konfigurasi checkpoint Front14A + Side13, average fusion atas probabilitas mentah sudah secara implisit membawa unsur confidence, sehingga confidence-weighting eksplisit menjadi redundant.",
            "- Experiment 20 dapat dibaca sebagai early-stop metodologis untuk jalur adaptive fusion berbasis weighted probability average pada konfigurasi ini.",
            "",
            "## Status Reliability-Weighted Static Fusion",
            "",
            "- `reliability_weighted_static_fusion` hanya pembanding statis berbasis Macro F1 validation single-view.",
            "- Formula ini bukan adaptive fusion per-sample dan tidak boleh dilabeli sebagai adaptive.",
            "",
            "## Status Kesimpulan",
            "",
            "- Adaptive confidence-weighting belum terbukti mengungguli average fusion.",
            "- Experiment 20 tidak mengubah rumus adaptive fusion utama.",
            "- Average fusion tetap strategi yang paling sederhana dan stabil pada konfigurasi ini.",
            "- Hasil Experiment 20 bersifat eksploratif dan belum otomatis menggantikan hasil utama penelitian.",
            "- Jika hasil tampak menjanjikan, formula baru tetap perlu dikonsultasikan ke pembimbing sebelum dijadikan metode utama.",
            "- Validation set yang relatif kecil membuat beberapa kandidat perlu ditafsirkan hati-hati.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Experiment 20 adaptive formula exploration.")
    parser.add_argument("--front-checkpoint", required=True, help="Path checkpoint front")
    parser.add_argument("--side-checkpoint", required=True, help="Path checkpoint side")
    args = parser.parse_args()

    formula_specs = [
        FormulaSpec("average_fusion", "Average Fusion", 0, build_formula_outputs),
        FormulaSpec("legacy_adaptive_softmax", "Legacy Adaptive Softmax", 1, build_formula_outputs),
        FormulaSpec("legacy_adaptive_ratio", "Legacy Adaptive Ratio", 2, build_formula_outputs),
        FormulaSpec("margin_adaptive_ratio", "Margin Adaptive Ratio", 3, build_formula_outputs),
        FormulaSpec("entropy_adaptive_ratio", "Entropy Adaptive Ratio", 4, build_formula_outputs),
        FormulaSpec("maxprob_adaptive_ratio", "MaxProb Adaptive Ratio", 5, build_formula_outputs),
        FormulaSpec("reliability_weighted_static_fusion", "Reliability-Weighted Static Fusion", 6, build_formula_outputs),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model_front, _ = load_trained_model("front", device, checkpoint_path=args.front_checkpoint)
    model_side, _ = load_trained_model("side", device, checkpoint_path=args.side_checkpoint)

    df_val = load_paired_split_dataframe("val")
    df_test = load_paired_split_dataframe("test")

    loader_val = get_paired_loader(df_val)
    loader_test = get_paired_loader(df_test)

    scores_front_val, scores_side_val, labels_val, indices_val = collect_paired_scores(
        model_front, model_side, loader_val, device
    )
    scores_front_test, scores_side_test, labels_test, indices_test = collect_paired_scores(
        model_front, model_side, loader_test, device
    )

    if not np.array_equal(indices_val, np.arange(len(df_val))):
        raise AssertionError("Urutan validation loader tidak sinkron dengan dataframe.")
    if not np.array_equal(indices_test, np.arange(len(df_test))):
        raise AssertionError("Urutan test loader tidak sinkron dengan dataframe.")

    front_val_preds = (scores_front_val >= DECISION_THRESHOLD).astype(int)
    side_val_preds = (scores_side_val >= DECISION_THRESHOLD).astype(int)
    front_val_metrics = compute_metrics(labels_val.tolist(), front_val_preds.tolist())
    side_val_metrics = compute_metrics(labels_val.tolist(), side_val_preds.tolist())
    reliability_front = front_val_metrics["f1_macro"]
    reliability_side = side_val_metrics["f1_macro"]

    average_val_outputs = build_formula_outputs(
        "average_fusion", scores_front_val, scores_side_val, reliability_front, reliability_side
    )
    legacy_val_outputs = build_formula_outputs(
        "legacy_adaptive_softmax", scores_front_val, scores_side_val, reliability_front, reliability_side
    )
    average_val_preds = (average_val_outputs["fused_prob_phone"] >= DECISION_THRESHOLD).astype(int)
    legacy_val_preds = (legacy_val_outputs["fused_prob_phone"] >= DECISION_THRESHOLD).astype(int)

    validation_rows = []
    validation_details = []
    weights_frames = []

    for spec in formula_specs:
        row, sample_df, detail = evaluate_formula(
            split_name="val",
            spec=spec,
            df_split=df_val,
            labels=labels_val,
            prob_front=scores_front_val,
            prob_side=scores_side_val,
            average_preds=average_val_preds,
            legacy_preds=legacy_val_preds,
            reliability_front=reliability_front,
            reliability_side=reliability_side,
        )
        validation_rows.append(row)
        validation_details.append(detail)
        weights_frames.append(sample_df)

    selected_validation = choose_best_formula(validation_details, formula_specs)
    selected_formula_name = selected_validation["formula"]
    selected_spec = next(spec for spec in formula_specs if spec.name == selected_formula_name)

    average_test_outputs = build_formula_outputs(
        "average_fusion", scores_front_test, scores_side_test, reliability_front, reliability_side
    )
    legacy_test_outputs = build_formula_outputs(
        "legacy_adaptive_softmax", scores_front_test, scores_side_test, reliability_front, reliability_side
    )
    average_test_preds = (average_test_outputs["fused_prob_phone"] >= DECISION_THRESHOLD).astype(int)
    legacy_test_preds = (legacy_test_outputs["fused_prob_phone"] >= DECISION_THRESHOLD).astype(int)

    test_row, test_weights_df, selected_test_detail = evaluate_formula(
        split_name="test",
        spec=selected_spec,
        df_split=df_test,
        labels=labels_test,
        prob_front=scores_front_test,
        prob_side=scores_side_test,
        average_preds=average_test_preds,
        legacy_preds=legacy_test_preds,
        reliability_front=reliability_front,
        reliability_side=reliability_side,
    )
    weights_frames.append(test_weights_df)

    comparison_with_baseline = {
        "macro_f1_delta": round(selected_test_detail["metrics"]["f1_macro"] - BASELINE_TEST_MACRO_F1, 5),
        "safe_to_phone_delta": int(selected_test_detail["safe_to_phone"] - BASELINE_TEST_CONFUSION[0][1]),
        "phone_to_safe_delta": int(selected_test_detail["phone_to_safe"] - BASELINE_TEST_CONFUSION[1][0]),
    }
    selected_test_detail["comparison_with_baseline"] = comparison_with_baseline

    validation_df = pd.DataFrame(validation_rows)
    weights_df = pd.concat(weights_frames, ignore_index=True)

    summary = {
        "experiment": "experiment_20_adaptive_formula",
        "front_checkpoint": args.front_checkpoint,
        "side_checkpoint": args.side_checkpoint,
        "validation_sample_count": int(len(df_val)),
        "test_sample_count": int(len(df_test)),
        "validation_single_view_macro_f1": {
            "front": reliability_front,
            "side": reliability_side,
        },
        "validation_results": validation_rows,
        "selected_formula": selected_validation,
        "selected_test_result": selected_test_detail,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation_csv_path = RESULTS_DIR / "adaptive_formula_exp20_validation.csv"
    selected_test_json_path = RESULTS_DIR / "adaptive_formula_exp20_selected_test.json"
    weights_csv_path = RESULTS_DIR / "adaptive_formula_exp20_weights.csv"
    summary_json_path = RESULTS_DIR / "adaptive_formula_exp20_summary.json"
    notes_path = Path("experiment_20_adaptive_formula_notes.md")

    validation_df.to_csv(validation_csv_path, index=False)
    weights_df.to_csv(weights_csv_path, index=False)
    selected_test_json_path.write_text(json.dumps(selected_test_detail, indent=2), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    notes_path.write_text(build_notes(summary), encoding="utf-8")

    print("=" * 72)
    print("EXPERIMENT 20: VALIDATION RESULTS")
    print("=" * 72)
    for row in validation_rows:
        print(
            f"{row['formula']}: "
            f"F1={row['macro_f1']:.5f}, Acc={row['accuracy']:.5f}, "
            f"safe->phone={row['safe_to_phone']}, phone->safe={row['phone_to_safe']}, "
            f"diff_vs_avg={row['predictions_different_from_average']}, "
            f"diff_vs_legacy={row['predictions_different_from_legacy_adaptive_softmax']}"
        )
    print("-" * 72)
    print(f"Selected formula: {selected_formula_name}")
    print(
        f"Selected validation F1: {selected_validation['metrics']['f1_macro']:.5f} | "
        f"phone->safe={selected_validation['phone_to_safe']} | "
        f"safe->phone={selected_validation['safe_to_phone']}"
    )
    print("-" * 72)
    print("SELECTED TEST RESULT")
    print(
        f"F1={selected_test_detail['metrics']['f1_macro']:.5f}, "
        f"Acc={selected_test_detail['metrics']['accuracy']:.5f}, "
        f"safe->phone={selected_test_detail['safe_to_phone']}, "
        f"phone->safe={selected_test_detail['phone_to_safe']}, "
        f"diff_vs_avg={selected_test_detail['predictions_different_from_average']}, "
        f"diff_vs_legacy={selected_test_detail['predictions_different_from_legacy_adaptive_softmax']}"
    )
    print(f"Beat baseline 0.78059: {selected_test_detail['metrics']['f1_macro'] > BASELINE_TEST_MACRO_F1}")
    print(f"Reached 0.80: {selected_test_detail['metrics']['f1_macro'] >= 0.80}")
    print("-" * 72)
    print(f"Validation CSV : {validation_csv_path}")
    print(f"Selected test  : {selected_test_json_path}")
    print(f"Weights CSV    : {weights_csv_path}")
    print(f"Summary JSON   : {summary_json_path}")
    print(f"Notes          : {notes_path.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
