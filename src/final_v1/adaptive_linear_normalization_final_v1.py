import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
)

from src.final_v1.config_final_v1 import RESULTS_DIR_FINAL, THRESHOLD
from src.final_v1.metrics_final_v1 import compute_metrics
from src.fusion import adaptive_fusion, adaptive_linear_normalization_fusion, average_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


METHOD_AVERAGE = "average_fusion"
METHOD_SOFTMAX = "adaptive_softmax"
METHOD_LINEAR = "adaptive_linear_normalization"


def _round_float(value: float, ndigits: int = 8) -> float:
    return round(float(value), ndigits)


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


def compute_metric_block(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    metrics = compute_metrics(labels.tolist(), preds.tolist())
    metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(labels, preds)), 5)
    metrics["roc_auc"] = round(float(roc_auc_score(labels, probs)), 5)
    metrics["pr_auc"] = round(float(average_precision_score(labels, probs)), 5)
    metrics["mcc"] = round(float(matthews_corrcoef(labels, preds)), 5)
    metrics["cohen_kappa"] = round(float(cohen_kappa_score(labels, preds)), 5)
    return metrics


def describe_array(values: np.ndarray) -> dict:
    return {
        "mean": _round_float(np.mean(values)),
        "std": _round_float(np.std(values)),
        "min": _round_float(np.min(values)),
        "max": _round_float(np.max(values)),
    }


def validate_inputs(pivot: pd.DataFrame, prob_front: np.ndarray, prob_side: np.ndarray, labels: np.ndarray) -> None:
    if not (len(prob_front) == len(prob_side) == len(labels) == len(pivot)):
        raise AssertionError("Panjang data front, side, label, dan pairing tidak sama.")

    if pivot["sample_id"].duplicated().any():
        duplicated = pivot.loc[pivot["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise AssertionError(f"Ada sample_id duplikat setelah pairing: {duplicated}")

    for name, values in {"prob_front": prob_front, "prob_side": prob_side}.items():
        if not np.all((values >= 0.0) & (values <= 1.0)):
            raise AssertionError(f"{name} memiliki nilai di luar rentang [0, 1].")


def validate_outputs(probabilities: dict[str, np.ndarray], weights: dict[str, np.ndarray]) -> None:
    for name, values in probabilities.items():
        if not np.all((values >= -1e-12) & (values <= 1.0 + 1e-12)):
            raise AssertionError(f"Probabilitas {name} memiliki nilai di luar rentang [0, 1].")

    for name, values in weights.items():
        if not np.all((values >= -1e-12) & (values <= 1.0 + 1e-12)):
            raise AssertionError(f"Bobot {name} memiliki nilai di luar rentang [0, 1].")

    if not np.allclose(weights["softmax_front"] + weights["softmax_side"], 1.0):
        raise AssertionError("Jumlah bobot adaptive softmax tidak mendekati 1.")
    if not np.allclose(weights["linear_front"] + weights["linear_side"], 1.0):
        raise AssertionError("Jumlah bobot adaptive linear normalization tidak mendekati 1.")


def build_examples(df_out: pd.DataFrame) -> dict:
    example_columns = [
        "sample_id",
        "true_label",
        "prob_phone_front",
        "prob_phone_side",
        "confidence_front",
        "confidence_side",
        "w_front_softmax",
        "w_side_softmax",
        "w_front_linear",
        "w_side_linear",
        "prob_phone_average",
        "prob_phone_adaptive_softmax",
        "prob_phone_adaptive_linear",
        "pred_average",
        "pred_adaptive_softmax",
        "pred_adaptive_linear",
    ]

    by_linear_delta = df_out.sort_values("abs_prob_linear_vs_average", ascending=False).head(10)
    changed_preds = df_out[df_out["pred_adaptive_linear"] != df_out["pred_average"]].head(10)
    near_threshold = (
        df_out.assign(distance_to_threshold=(df_out["prob_phone_adaptive_linear"] - THRESHOLD).abs())
        .sort_values("distance_to_threshold")
        .head(10)
    )

    manual_rows = df_out.head(3).copy()
    manual_examples = []
    for _, row in manual_rows.iterrows():
        denom = row["confidence_front"] + row["confidence_side"]
        manual_examples.append(
            {
                "sample_id": row["sample_id"],
                "c_front": _round_float(row["confidence_front"]),
                "c_side": _round_float(row["confidence_side"]),
                "denominator": _round_float(denom),
                "w_front_linear": _round_float(row["w_front_linear"]),
                "w_side_linear": _round_float(row["w_side_linear"]),
                "p_adaptive_linear": _round_float(row["prob_phone_adaptive_linear"]),
                "calculation": (
                    f"(({row['w_front_linear']:.8f} * {row['prob_phone_front']:.8f}) + "
                    f"({row['w_side_linear']:.8f} * {row['prob_phone_side']:.8f}))"
                ),
            }
        )

    return {
        "largest_probability_differences": by_linear_delta[example_columns].to_dict(orient="records"),
        "changed_predictions": changed_preds[example_columns].to_dict(orient="records"),
        "near_threshold": near_threshold[example_columns].to_dict(orient="records"),
        "manual_validation_examples": manual_examples,
    }


def run_adaptive_linear_normalization_comparison() -> dict:
    predictions_path = RESULTS_DIR_FINAL / "final_v1_single_view_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"File prediksi single-view belum ada: {predictions_path}. "
            "Jalankan evaluate_single_view_final_v1 terlebih dahulu."
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
        raise AssertionError(f"Pairing front/side tidak lengkap: {sorted(missing_pivot)}")

    prob_front = pivot["prob_phone_front"].to_numpy(dtype=float)
    prob_side = pivot["prob_phone_side"].to_numpy(dtype=float)
    labels = pivot["true_label"].to_numpy(dtype=int)
    validate_inputs(pivot, prob_front, prob_side, labels)

    confidence_front = np.abs(prob_front - THRESHOLD)
    confidence_side = np.abs(prob_side - THRESHOLD)
    w_front_softmax, w_side_softmax = compute_softmax_weights(prob_front, prob_side)
    w_front_linear, w_side_linear = compute_linear_weights(prob_front, prob_side)

    prob_average = average_fusion(prob_front, prob_side)
    prob_softmax = adaptive_fusion(prob_front, prob_side, threshold=THRESHOLD)
    prob_linear = adaptive_linear_normalization_fusion(prob_front, prob_side, threshold=THRESHOLD)

    validate_outputs(
        {
            METHOD_AVERAGE: prob_average,
            METHOD_SOFTMAX: prob_softmax,
            METHOD_LINEAR: prob_linear,
        },
        {
            "softmax_front": w_front_softmax,
            "softmax_side": w_side_softmax,
            "linear_front": w_front_linear,
            "linear_side": w_side_linear,
        },
    )

    pred_average = (prob_average >= THRESHOLD).astype(int)
    pred_softmax = (prob_softmax >= THRESHOLD).astype(int)
    pred_linear = (prob_linear >= THRESHOLD).astype(int)

    metrics = {
        METHOD_AVERAGE: compute_metric_block(labels, pred_average, prob_average),
        METHOD_SOFTMAX: compute_metric_block(labels, pred_softmax, prob_softmax),
        METHOD_LINEAR: compute_metric_block(labels, pred_linear, prob_linear),
    }

    diff_softmax_vs_average = int((pred_softmax != pred_average).sum())
    diff_linear_vs_average = int((pred_linear != pred_average).sum())
    diff_linear_vs_softmax = int((pred_linear != pred_softmax).sum())

    df_out = pivot.copy()
    df_out["prob_safe_front"] = 1.0 - df_out["prob_phone_front"]
    df_out["prob_safe_side"] = 1.0 - df_out["prob_phone_side"]
    df_out["confidence_front"] = confidence_front
    df_out["confidence_side"] = confidence_side
    df_out["w_front_softmax"] = w_front_softmax
    df_out["w_side_softmax"] = w_side_softmax
    df_out["w_front_linear"] = w_front_linear
    df_out["w_side_linear"] = w_side_linear
    df_out["prob_phone_average"] = prob_average
    df_out["prob_phone_adaptive_softmax"] = prob_softmax
    df_out["prob_phone_adaptive_linear"] = prob_linear
    df_out["pred_average"] = pred_average
    df_out["pred_adaptive_softmax"] = pred_softmax
    df_out["pred_adaptive_linear"] = pred_linear
    df_out["abs_prob_softmax_vs_average"] = np.abs(prob_softmax - prob_average)
    df_out["abs_prob_linear_vs_average"] = np.abs(prob_linear - prob_average)
    df_out["abs_prob_linear_vs_softmax"] = np.abs(prob_linear - prob_softmax)

    changed_probability_linear = df_out["abs_prob_linear_vs_average"] > 1e-12
    same_label_linear = df_out["pred_adaptive_linear"] == df_out["pred_average"]

    comparison_rows = []
    for method_name, block in metrics.items():
        comparison_rows.append(
            {
                "method": method_name,
                "accuracy": block["accuracy"],
                "precision_macro": block["precision_macro"],
                "recall_macro": block["recall_macro"],
                "f1_macro": block["f1_macro"],
                "balanced_accuracy": block["balanced_accuracy"],
                "roc_auc": block["roc_auc"],
                "pr_auc": block["pr_auc"],
                "mcc": block["mcc"],
                "cohen_kappa": block["cohen_kappa"],
                "confusion_matrix": json.dumps(block["confusion_matrix"]),
                "safe_to_phone": block["safe_to_phone"],
                "phone_to_safe": block["phone_to_safe"],
            }
        )

    analysis = {
        "method_names": [METHOD_AVERAGE, METHOD_SOFTMAX, METHOD_LINEAR],
        "threshold": THRESHOLD,
        "sample_count": int(len(labels)),
        "metrics": metrics,
        "prediction_differences": {
            "softmax_vs_average": diff_softmax_vs_average,
            "linear_vs_average": diff_linear_vs_average,
            "linear_vs_softmax": diff_linear_vs_softmax,
        },
        "weight_summary": {
            "softmax": {
                "w_front": describe_array(w_front_softmax),
                "w_side": describe_array(w_side_softmax),
            },
            "linear_normalization": {
                "w_front": describe_array(w_front_linear),
                "w_side": describe_array(w_side_linear),
            },
        },
        "probability_difference_summary": {
            "abs_softmax_vs_average": {
                "mean": _round_float(df_out["abs_prob_softmax_vs_average"].mean()),
                "max": _round_float(df_out["abs_prob_softmax_vs_average"].max()),
            },
            "abs_linear_vs_average": {
                "mean": _round_float(df_out["abs_prob_linear_vs_average"].mean()),
                "max": _round_float(df_out["abs_prob_linear_vs_average"].max()),
            },
            "abs_linear_vs_softmax": {
                "mean": _round_float(df_out["abs_prob_linear_vs_softmax"].mean()),
                "max": _round_float(df_out["abs_prob_linear_vs_softmax"].max()),
            },
        },
        "probability_and_label_change_counts": {
            "linear_probability_changed_but_label_same_vs_average": int((changed_probability_linear & same_label_linear).sum()),
            "linear_label_changed_vs_average": diff_linear_vs_average,
        },
        "examples": build_examples(df_out),
        "validation": {
            "all_softmax_weights_sum_to_one": bool(np.allclose(w_front_softmax + w_side_softmax, 1.0)),
            "all_linear_weights_sum_to_one": bool(np.allclose(w_front_linear + w_side_linear, 1.0)),
            "all_fused_probabilities_in_range": bool(
                np.all((prob_average >= 0.0) & (prob_average <= 1.0))
                and np.all((prob_softmax >= 0.0) & (prob_softmax <= 1.0))
                and np.all((prob_linear >= 0.0) & (prob_linear <= 1.0))
            ),
            "pairing_source": str(predictions_path),
        },
        "notes": (
            "Adaptive softmax lama dipertahankan sesuai proposal. "
            "Adaptive linear normalization hanya dijalankan sebagai eksperimen pembanding tanpa retraining."
        ),
    }

    metrics_path = RESULTS_DIR_FINAL / "adaptive_linear_normalization_metrics.json"
    predictions_out_path = RESULTS_DIR_FINAL / "adaptive_linear_normalization_predictions.csv"
    comparison_path = RESULTS_DIR_FINAL / "fusion_comparison_additional.csv"
    analysis_path = RESULTS_DIR_FINAL / "adaptive_linear_normalization_analysis.json"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df_out.to_csv(predictions_out_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "analysis": analysis,
        "metrics_path": metrics_path,
        "predictions_path": predictions_out_path,
        "comparison_path": comparison_path,
        "analysis_path": analysis_path,
    }


def print_required_summary(outputs: dict) -> None:
    metrics = outputs["metrics"]
    diffs = outputs["analysis"]["prediction_differences"]

    labels = [
        (METHOD_AVERAGE, "Average Fusion"),
        (METHOD_SOFTMAX, "Adaptive Softmax"),
        (METHOD_LINEAR, "Adaptive Linear Normalization"),
    ]

    print("\nAccuracy")
    for key, label in labels:
        print(f"- {label:<31}: {metrics[key]['accuracy']:.5f}")

    print("\nPrecision Macro")
    for key, label in labels:
        print(f"- {label:<31}: {metrics[key]['precision_macro']:.5f}")

    print("\nRecall Macro")
    for key, label in labels:
        print(f"- {label:<31}: {metrics[key]['recall_macro']:.5f}")

    print("\nMacro F1-Score")
    for key, label in labels:
        print(f"- {label:<31}: {metrics[key]['f1_macro']:.5f}")

    print("\nConfusion Matrix")
    for key, label in labels:
        print(f"- {label:<31}: {metrics[key]['confusion_matrix']}")

    print("\nJumlah prediksi berbeda")
    print(f"- Softmax vs Average         : {diffs['softmax_vs_average']}")
    print(f"- Linear vs Average          : {diffs['linear_vs_average']}")
    print(f"- Linear vs Softmax          : {diffs['linear_vs_softmax']}")

    print("\nFile output")
    print(f"- {outputs['metrics_path']}")
    print(f"- {outputs['predictions_path']}")
    print(f"- {outputs['comparison_path']}")
    print(f"- {outputs['analysis_path']}")


def main() -> None:
    outputs = run_adaptive_linear_normalization_comparison()
    print_required_summary(outputs)


if __name__ == "__main__":
    main()
