import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.final_v2.config_final_v2 import RESULTS_DIR_FINAL, THRESHOLD
from src.final_v2.metrics_final_v2 import compute_brier_score, compute_ece_binary, compute_metrics
from src.final_v2.statistical_tests_final_v2_vs_final_v1 import delong_roc_test


PREDICTIONS_PATH = RESULTS_DIR_FINAL / "fusion_final_v2_predictions.csv"
OUTPUT_JSON = RESULTS_DIR_FINAL / "fusion_variant_analysis_final_v2.json"
OUTPUT_CSV = RESULTS_DIR_FINAL / "fusion_variant_analysis_final_v2.csv"
N_BOOTSTRAP = 2000
RANDOM_SEED = 42


def adaptive_linear_normalization(prob_front: np.ndarray, prob_side: np.ndarray) -> np.ndarray:
    d_front = np.abs(prob_front - THRESHOLD)
    d_side = np.abs(prob_side - THRESHOLD)
    denom = d_front + d_side
    w_front = np.divide(d_front, denom, out=np.full_like(d_front, 0.5), where=denom > 0)
    w_side = 1.0 - w_front
    return w_front * prob_front + w_side * prob_side


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    positive_idx = np.flatnonzero(y_true == 1)
    negative_idx = np.flatnonzero(y_true == 0)
    rows = []

    for _ in range(N_BOOTSTRAP):
        sampled = np.concatenate(
            [
                rng.choice(negative_idx, size=len(negative_idx), replace=True),
                rng.choice(positive_idx, size=len(positive_idx), replace=True),
            ]
        )
        y = y_true[sampled]
        prob = y_prob[sampled]
        pred = (prob >= THRESHOLD).astype(int)
        rows.append(
            {
                "macro_f1": f1_score(y, pred, average="macro"),
                "roc_auc": roc_auc_score(y, prob),
                "pr_auc": average_precision_score(y, prob),
            }
        )

    boot = pd.DataFrame(rows)
    return {
        metric: {
            "mean": round(float(boot[metric].mean()), 5),
            "ci95_low": round(float(boot[metric].quantile(0.025)), 5),
            "ci95_high": round(float(boot[metric].quantile(0.975)), 5),
        }
        for metric in boot.columns
    }


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    both_correct = int((correct_a & correct_b).sum())
    both_wrong = int((~correct_a & ~correct_b).sum())
    total_discordant = a_only + b_only
    p_value = 1.0
    if total_discordant:
        p_value = float(stats.binomtest(min(a_only, b_only), total_discordant, 0.5).pvalue)
    return {
        "table_both_correct_a_only_b_only_both_wrong": [[both_correct, a_only], [b_only, both_wrong]],
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "p_value": p_value,
        "significant_alpha_0_05": bool(p_value < 0.05),
    }


def method_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= THRESHOLD).astype(int)
    metrics = compute_metrics(y_true.tolist(), y_pred.tolist())
    metrics.update(
        {
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 5),
            "pr_auc": round(float(average_precision_score(y_true, y_prob)), 5),
            "ece_binary": compute_ece_binary(y_true, y_prob),
            "brier_score": compute_brier_score(y_true, y_prob),
        }
    )
    return metrics


def analyze_fusion_variants() -> dict:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Prediksi fusion Final v2 belum ada: {PREDICTIONS_PATH}")

    df = pd.read_csv(PREDICTIONS_PATH)
    y_true = df["true_label"].to_numpy(dtype=int)
    prob_front = df["prob_phone_front"].to_numpy(dtype=float)
    prob_side = df["prob_phone_side"].to_numpy(dtype=float)

    probabilities = {
        "Average Fusion": df["prob_phone_average"].to_numpy(dtype=float),
        "Adaptive Softmax": df["prob_phone_adaptive"].to_numpy(dtype=float),
        "Adaptive Linear Normalization": adaptive_linear_normalization(prob_front, prob_side),
    }
    predictions = {name: (prob >= THRESHOLD).astype(int) for name, prob in probabilities.items()}

    methods = {
        name: {
            **method_metrics(y_true, prob),
            "bootstrap_ci": bootstrap_ci(y_true, prob),
        }
        for name, prob in probabilities.items()
    }

    pairwise = {}
    for method_a, method_b in combinations(probabilities.keys(), 2):
        prob_a = probabilities[method_a]
        prob_b = probabilities[method_b]
        pred_a = predictions[method_a]
        pred_b = predictions[method_b]
        delong = delong_roc_test(y_true, prob_a, prob_b)
        prob_abs_diff = np.abs(prob_a - prob_b)
        pairwise[f"{method_a} vs {method_b}"] = {
            "hard_prediction_differences": int((pred_a != pred_b).sum()),
            "max_probability_abs_diff": round(float(prob_abs_diff.max()), 8),
            "mean_probability_abs_diff": round(float(prob_abs_diff.mean()), 8),
            "mcnemar_exact": mcnemar_exact(y_true, pred_a, pred_b),
            "delong_roc_auc": {
                "auc_method_a": delong["auc_final_v1"],
                "auc_method_b": delong["auc_final_v2"],
                "delta_auc_b_minus_a": delong["delta_auc_final_v2_minus_final_v1"],
                "z": delong["z"],
                "p_value": delong["p_value"],
                "significant_alpha_0_05": bool(delong["p_value"] < 0.05),
            },
        }

    result = {
        "experiment": "final_v2_retrain",
        "analysis": "Average fusion vs adaptive probability variants on the same Final v2 predictions",
        "threshold": THRESHOLD,
        "support": int(len(y_true)),
        "positive_support": int(y_true.sum()),
        "negative_support": int((y_true == 0).sum()),
        "bootstrap": {"n_bootstrap": N_BOOTSTRAP, "random_seed": RANDOM_SEED},
        "methods": methods,
        "pairwise": pairwise,
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for name, values in methods.items():
        rows.append(
            {
                "method": name,
                "accuracy": values["accuracy"],
                "precision_macro": values["precision_macro"],
                "recall_macro": values["recall_macro"],
                "f1_macro": values["f1_macro"],
                "roc_auc": values["roc_auc"],
                "pr_auc": values["pr_auc"],
                "ece_binary": values["ece_binary"],
                "brier_score": values["brier_score"],
                "safe_to_phone": values["safe_to_phone"],
                "phone_to_safe": values["phone_to_safe"],
                "confusion_matrix": values["confusion_matrix"],
                "bootstrap_macro_f1_ci95": f"[{values['bootstrap_ci']['macro_f1']['ci95_low']}, {values['bootstrap_ci']['macro_f1']['ci95_high']}]",
                "bootstrap_roc_auc_ci95": f"[{values['bootstrap_ci']['roc_auc']['ci95_low']}, {values['bootstrap_ci']['roc_auc']['ci95_high']}]",
                "bootstrap_pr_auc_ci95": f"[{values['bootstrap_ci']['pr_auc']['ci95_low']}, {values['bootstrap_ci']['pr_auc']['ci95_high']}]",
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    return result


def main() -> None:
    print(json.dumps(analyze_fusion_variants(), indent=2))


if __name__ == "__main__":
    main()
