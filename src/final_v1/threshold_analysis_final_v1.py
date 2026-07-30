import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.final_v1.config_final_v1 import RESULTS_DIR_FINAL, THRESHOLD
from src.fusion import adaptive_fusion, adaptive_linear_normalization_fusion, average_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
OUTPUT_DIR = RESULTS_DIR_FINAL / "threshold_analysis"

METHODS = {
    "average_fusion": "Average Fusion",
    "adaptive_softmax": "Adaptive Softmax",
    "adaptive_linear_normalization": "Adaptive Linear Normalization",
}

RANKING_COLUMNS = ["macro_f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc"]


def _round_float(value: float, ndigits: int = 5) -> float:
    return round(float(value), ndigits)


def load_paired_predictions() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    predictions_path = RESULTS_DIR_FINAL / "final_v1_single_view_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"File probabilitas single-view belum ada: {predictions_path}. "
            "Jalankan evaluate_single_view_final_v1 terlebih dahulu; jangan retraining."
        )

    df = pd.read_csv(predictions_path)
    required_columns = {
        "sample_id",
        "subject_id",
        "activity_id",
        "frame",
        "true_label",
        "view",
        "prob_phone",
        "pred_label",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise AssertionError(f"Kolom wajib tidak ada di single-view predictions: {sorted(missing)}")

    duplicate_keys = ["sample_id", "subject_id", "activity_id", "frame", "true_label", "view"]
    if df.duplicated(duplicate_keys).any():
        examples = df.loc[df.duplicated(duplicate_keys), duplicate_keys].head(5).to_dict(orient="records")
        raise AssertionError(f"Ada duplikasi prediksi per view: {examples}")

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

    expected_columns = {"prob_phone_front", "prob_phone_side", "pred_label_front", "pred_label_side"}
    missing_pivot = expected_columns.difference(pivot.columns)
    if missing_pivot:
        raise AssertionError(f"Pairing front/side tidak lengkap atau urutannya tidak sama: {sorted(missing_pivot)}")

    prob_front = pivot["prob_phone_front"].to_numpy(dtype=float)
    prob_side = pivot["prob_phone_side"].to_numpy(dtype=float)
    y_true = pivot["true_label"].to_numpy(dtype=int)

    validate_inputs(pivot, y_true, prob_front, prob_side)
    return pivot, y_true, prob_front, prob_side


def validate_inputs(pivot: pd.DataFrame, y_true: np.ndarray, prob_front: np.ndarray, prob_side: np.ndarray) -> None:
    if not (len(y_true) == len(prob_front) == len(prob_side) == len(pivot)):
        raise AssertionError("Panjang y_true, p_front, p_side, dan data pairing harus sama.")

    if pivot["sample_id"].duplicated().any():
        examples = pivot.loc[pivot["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise AssertionError(f"Ada sample_id duplikat setelah pairing: {examples}")

    for name, values in {"p_front": prob_front, "p_side": prob_side}.items():
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"{name} mengandung nilai non-finite.")
        if not np.all((values >= 0.0) & (values <= 1.0)):
            raise AssertionError(f"{name} memiliki probabilitas di luar [0, 1].")

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise AssertionError("y_true harus berupa label biner 0=safe_driving dan 1=phone_use.")


def compute_softmax_weights(prob_front: np.ndarray, prob_side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c_front = np.abs(prob_front - THRESHOLD)
    c_side = np.abs(prob_side - THRESHOLD)
    exp_front = np.exp(c_front)
    exp_side = np.exp(c_side)
    denom = exp_front + exp_side
    return exp_front / denom, exp_side / denom


def compute_linear_weights(prob_front: np.ndarray, prob_side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c_front = np.abs(prob_front - THRESHOLD)
    c_side = np.abs(prob_side - THRESHOLD)
    denom = c_front + c_side
    w_front = np.divide(c_front, denom, out=np.full_like(c_front, 0.5, dtype=float), where=denom != 0)
    w_side = np.divide(c_side, denom, out=np.full_like(c_side, 0.5, dtype=float), where=denom != 0)
    return w_front, w_side


def validate_fusion_outputs(fused_probs: dict[str, np.ndarray], weights: dict[str, np.ndarray]) -> None:
    for name, values in fused_probs.items():
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"Probabilitas fused {name} mengandung nilai non-finite.")
        if not np.all((values >= -1e-12) & (values <= 1.0 + 1e-12)):
            raise AssertionError(f"Probabilitas fused {name} berada di luar [0, 1].")

    for name, values in weights.items():
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"Bobot {name} mengandung nilai non-finite.")
        if not np.all((values >= -1e-12) & (values <= 1.0 + 1e-12)):
            raise AssertionError(f"Bobot {name} berada di luar [0, 1].")

    if not np.allclose(weights["softmax_front"] + weights["softmax_side"], 1.0):
        raise AssertionError("Jumlah bobot adaptive softmax tidak mendekati 1.")
    if not np.allclose(weights["linear_front"] + weights["linear_side"], 1.0):
        raise AssertionError("Jumlah bobot adaptive linear tidak mendekati 1.")


def compute_metric_row(
    method: str,
    threshold: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    prediction_differences: dict[str, int],
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if int(cm.sum()) != len(y_true):
        raise AssertionError("Jumlah confusion matrix tidak sama dengan total sampel.")

    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    predicted_safe = int((y_pred == 0).sum())
    predicted_phone = int((y_pred == 1).sum())

    return {
        "method": method,
        "threshold": f"{threshold:.2f}",
        "accuracy": _round_float(accuracy_score(y_true, y_pred)),
        "precision_macro": _round_float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": _round_float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": _round_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": _round_float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": _round_float(roc_auc_score(y_true, y_prob)),
        "pr_auc": _round_float(average_precision_score(y_true, y_prob)),
        "mcc": _round_float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": _round_float(cohen_kappa_score(y_true, y_pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "safe_to_phone": fp,
        "phone_to_safe": fn,
        "predicted_safe": predicted_safe,
        "predicted_phone": predicted_phone,
        "different_vs_average": prediction_differences["different_vs_average"],
        "different_vs_softmax": prediction_differences["different_vs_softmax"],
        "different_vs_linear": prediction_differences["different_vs_linear"],
    }


def build_threshold_rows(y_true: np.ndarray, fused_probs: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    rows = []
    diff_rows = []

    for threshold in THRESHOLDS:
        predictions = {
            method: (probs >= threshold).astype(int)
            for method, probs in fused_probs.items()
        }

        diff_softmax_vs_average = int((predictions["adaptive_softmax"] != predictions["average_fusion"]).sum())
        diff_linear_vs_average = int(
            (predictions["adaptive_linear_normalization"] != predictions["average_fusion"]).sum()
        )
        diff_linear_vs_softmax = int(
            (predictions["adaptive_linear_normalization"] != predictions["adaptive_softmax"]).sum()
        )

        diff_rows.append(
            {
                "threshold": f"{threshold:.2f}",
                "softmax_vs_average": diff_softmax_vs_average,
                "linear_vs_average": diff_linear_vs_average,
                "linear_vs_softmax": diff_linear_vs_softmax,
            }
        )

        for method, probs in fused_probs.items():
            rows.append(
                compute_metric_row(
                    method=method,
                    threshold=threshold,
                    y_true=y_true,
                    y_pred=predictions[method],
                    y_prob=probs,
                    prediction_differences={
                        "different_vs_average": int((predictions[method] != predictions["average_fusion"]).sum()),
                        "different_vs_softmax": int((predictions[method] != predictions["adaptive_softmax"]).sum()),
                        "different_vs_linear": int(
                            (predictions[method] != predictions["adaptive_linear_normalization"]).sum()
                        ),
                    },
                )
            )

    return rows, diff_rows


def rank_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = results_df.sort_values(RANKING_COLUMNS, ascending=[False] * len(RANKING_COLUMNS)).reset_index(drop=True)
    best_by_metric = sorted_df.copy()
    best_by_metric.insert(0, "rank", np.arange(1, len(best_by_metric) + 1))

    best_by_method = (
        sorted_df.groupby("method", as_index=False, group_keys=False)
        .head(1)
        .sort_values(RANKING_COLUMNS, ascending=[False] * len(RANKING_COLUMNS))
        .reset_index(drop=True)
    )
    best_by_method.insert(0, "rank", np.arange(1, len(best_by_method) + 1))
    return best_by_metric, best_by_method


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_summary_markdown(
    results_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    best_by_metric: pd.DataFrame,
    best_by_method: pd.DataFrame,
) -> str:
    changed_thresholds = diff_df[
        (diff_df["softmax_vs_average"] > 0)
        | (diff_df["linear_vs_average"] > 0)
        | (diff_df["linear_vs_softmax"] > 0)
    ]

    best_macro_f1 = results_df["macro_f1"].max()
    best_macro_rows = results_df[results_df["macro_f1"] == best_macro_f1].sort_values(
        RANKING_COLUMNS, ascending=[False] * len(RANKING_COLUMNS)
    )

    auc_rows = (
        results_df[["method", "roc_auc", "pr_auc"]]
        .drop_duplicates()
        .sort_values(["roc_auc", "pr_auc"], ascending=[False, False])
    )

    lines = [
        "# Threshold Analysis Final V1",
        "",
        "Eksperimen ini hanya menerapkan threshold berbeda pada probabilitas fused yang sudah tersedia. "
        "Tidak ada retraining, perubahan checkpoint, perubahan dataset, split, backbone, atau probabilitas dasar model.",
        "",
        f"- Threshold diuji: {', '.join(f'{x:.2f}' for x in THRESHOLDS)}",
        f"- Total sampel: {int(results_df.iloc[0]['tn'] + results_df.iloc[0]['fp'] + results_df.iloc[0]['fn'] + results_df.iloc[0]['tp'])}",
        "- Urutan confusion matrix: [[safe_as_safe, safe_as_phone], [phone_as_safe, phone_as_phone]]",
        "- ROC-AUC dan PR-AUC dihitung dari probabilitas, sehingga tetap sama untuk metode yang sama pada semua threshold.",
        "",
        "## Threshold Dengan Perbedaan Prediksi",
    ]

    if changed_thresholds.empty:
        lines.append("- Tidak ada threshold yang menghasilkan perbedaan label prediksi antar ketiga metode.")
    else:
        for row in changed_thresholds.to_dict(orient="records"):
            lines.append(
                f"- Threshold {row['threshold']}: "
                f"Softmax vs Average={row['softmax_vs_average']}, "
                f"Linear vs Average={row['linear_vs_average']}, "
                f"Linear vs Softmax={row['linear_vs_softmax']}"
            )

    lines.extend(["", "## Macro F1 Tertinggi"])
    for row in best_macro_rows.to_dict(orient="records"):
        lines.append(
            f"- {METHODS[row['method']]} threshold {row['threshold']}: "
            f"Macro F1={row['macro_f1']:.5f}, Balanced Accuracy={row['balanced_accuracy']:.5f}, "
            f"MCC={row['mcc']:.5f}, safe->phone={row['safe_to_phone']}, phone->safe={row['phone_to_safe']}"
        )

    lines.extend(["", "## ROC-AUC dan PR-AUC"])
    for row in auc_rows.to_dict(orient="records"):
        lines.append(f"- {METHODS[row['method']]}: ROC-AUC={row['roc_auc']:.5f}, PR-AUC={row['pr_auc']:.5f}")

    lines.extend(
        [
            "",
            "## Ranking Utama",
            dataframe_to_markdown(
                best_by_metric[["rank", "method", "threshold", *RANKING_COLUMNS, "safe_to_phone", "phone_to_safe"]]
            ),
            "",
            "## Terbaik Per Metode",
            dataframe_to_markdown(
                best_by_method[["rank", "method", "threshold", *RANKING_COLUMNS, "safe_to_phone", "phone_to_safe"]]
            ),
            "",
            "## Catatan Interpretasi",
            "- Hasil ini untuk analisis threshold sesuai arahan dosen, bukan otomatis mengganti threshold final 0.50.",
            "- Jangan memilih threshold final berdasarkan test set; gunakan hasil ini sebagai bukti analitis perilaku probabilitas dan label.",
            "- Perbedaan ROC-AUC/PR-AUC menunjukkan perbedaan ranking probabilitas antar metode, bukan efek threshold.",
        ]
    )
    return "\n".join(lines) + "\n"


def print_terminal_summary(results_df: pd.DataFrame, diff_df: pd.DataFrame) -> None:
    for threshold in [f"{x:.2f}" for x in THRESHOLDS]:
        block = results_df[results_df["threshold"] == threshold].set_index("method")
        diffs = diff_df[diff_df["threshold"] == threshold].iloc[0]

        print(f"\nThreshold {threshold}")
        print("Accuracy")
        for method, label in METHODS.items():
            print(f"- {label:<31}: {block.loc[method, 'accuracy']:.5f}")

        print("\nPrecision Macro")
        for method, label in METHODS.items():
            print(f"- {label:<31}: {block.loc[method, 'precision_macro']:.5f}")

        print("\nRecall Macro")
        for method, label in METHODS.items():
            print(f"- {label:<31}: {block.loc[method, 'recall_macro']:.5f}")

        print("\nMacro F1-Score")
        for method, label in METHODS.items():
            print(f"- {label:<31}: {block.loc[method, 'macro_f1']:.5f}")

        print("\nConfusion Matrix")
        for method, label in METHODS.items():
            cm = [
                [int(block.loc[method, "tn"]), int(block.loc[method, "fp"])],
                [int(block.loc[method, "fn"]), int(block.loc[method, "tp"])],
            ]
            print(f"- {label:<31}: {cm}")

        print("\nJumlah prediksi berbeda")
        print(f"- Softmax vs Average         : {int(diffs['softmax_vs_average'])}")
        print(f"- Linear vs Average          : {int(diffs['linear_vs_average'])}")
        print(f"- Linear vs Softmax          : {int(diffs['linear_vs_softmax'])}")


def run_threshold_analysis() -> dict:
    _, y_true, prob_front, prob_side = load_paired_predictions()

    w_front_softmax, w_side_softmax = compute_softmax_weights(prob_front, prob_side)
    w_front_linear, w_side_linear = compute_linear_weights(prob_front, prob_side)

    fused_probs = {
        "average_fusion": average_fusion(prob_front, prob_side),
        "adaptive_softmax": adaptive_fusion(prob_front, prob_side, threshold=THRESHOLD),
        "adaptive_linear_normalization": adaptive_linear_normalization_fusion(prob_front, prob_side, threshold=THRESHOLD),
    }

    validate_fusion_outputs(
        fused_probs=fused_probs,
        weights={
            "softmax_front": w_front_softmax,
            "softmax_side": w_side_softmax,
            "linear_front": w_front_linear,
            "linear_side": w_side_linear,
        },
    )

    rows, diff_rows = build_threshold_rows(y_true, fused_probs)
    results_df = pd.DataFrame(rows)
    diff_df = pd.DataFrame(diff_rows)
    best_by_metric, best_by_method = rank_results(results_df)
    summary_md = build_summary_markdown(results_df, diff_df, best_by_metric, best_by_method)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_csv_path = OUTPUT_DIR / "threshold_analysis_all.csv"
    all_json_path = OUTPUT_DIR / "threshold_analysis_all.json"
    summary_path = OUTPUT_DIR / "threshold_analysis_summary.md"
    differences_path = OUTPUT_DIR / "threshold_prediction_differences.csv"
    best_by_metric_path = OUTPUT_DIR / "threshold_best_by_metric.csv"
    best_by_method_path = OUTPUT_DIR / "threshold_best_by_method.csv"

    results_df.to_csv(all_csv_path, index=False)
    all_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary_path.write_text(summary_md, encoding="utf-8")
    diff_df.to_csv(differences_path, index=False)
    best_by_metric.to_csv(best_by_metric_path, index=False)
    best_by_method.to_csv(best_by_method_path, index=False)

    outputs = {
        "all_csv": all_csv_path,
        "all_json": all_json_path,
        "summary": summary_path,
        "prediction_differences": differences_path,
        "best_by_metric": best_by_metric_path,
        "best_by_method": best_by_method_path,
    }

    print_terminal_summary(results_df, diff_df)
    print("\nFile output")
    for path in outputs.values():
        print(f"- {path}")

    return outputs


def main() -> None:
    run_threshold_analysis()


if __name__ == "__main__":
    main()
