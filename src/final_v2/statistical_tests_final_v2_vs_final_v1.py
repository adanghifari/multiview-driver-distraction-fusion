import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.config import PROJECT_ROOT


FINAL_V1_PREDICTIONS = PROJECT_ROOT / "results" / "final_v1" / "fusion_final_v1_predictions.csv"
FINAL_V2_PREDICTIONS = PROJECT_ROOT / "results" / "final_v2" / "fusion_final_v2_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_v2" / "statistical_tests_vs_final_v1"
THRESHOLD = 0.50
N_BOOTSTRAP = 5000
RANDOM_SEED = 42


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def fast_delong(predictions_sorted: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    positives = predictions_sorted[:, :label_1_count]
    negatives = predictions_sorted[:, label_1_count:]
    m = label_1_count
    n = predictions_sorted.shape[1] - m
    tx = np.array([compute_midrank(row) for row in positives])
    ty = np.array([compute_midrank(row) for row in negatives])
    tz = np.array([compute_midrank(row) for row in predictions_sorted])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(covariance)


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    predictions_sorted = np.vstack([prob_a, prob_b])[:, order]
    aucs, covariance = fast_delong(predictions_sorted, label_1_count)
    contrast = np.array([[1, -1]])
    variance = float((contrast @ covariance @ contrast.T).item())
    if variance <= 0:
        z_score = 0.0 if np.isclose(aucs[0], aucs[1]) else np.inf
        p_value = 1.0 if np.isclose(aucs[0], aucs[1]) else 0.0
    else:
        z_score = float((aucs[0] - aucs[1]) / np.sqrt(variance))
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))
    return {
        "auc_final_v1": round(float(aucs[0]), 5),
        "auc_final_v2": round(float(aucs[1]), 5),
        "delta_auc_final_v2_minus_final_v1": round(float(aucs[1] - aucs[0]), 5),
        "z": round(z_score, 5),
        "p_value": p_value,
    }


def bootstrap_paired_ci(y_true: np.ndarray, final_v1_prob: np.ndarray, final_v2_prob: np.ndarray) -> dict:
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
        final_v1_p = final_v1_prob[sampled]
        final_v2_p = final_v2_prob[sampled]
        final_v1_pred = (final_v1_p >= THRESHOLD).astype(int)
        final_v2_pred = (final_v2_p >= THRESHOLD).astype(int)
        rows.append(
            {
                "macro_f1_delta": f1_score(y, final_v2_pred, average="macro") - f1_score(y, final_v1_pred, average="macro"),
                "roc_auc_delta": roc_auc_score(y, final_v2_p) - roc_auc_score(y, final_v1_p),
                "pr_auc_delta": average_precision_score(y, final_v2_p) - average_precision_score(y, final_v1_p),
            }
        )
    boot = pd.DataFrame(rows)
    return {
        metric: {
            "mean_delta": round(float(boot[metric].mean()), 5),
            "ci95_low": round(float(boot[metric].quantile(0.025)), 5),
            "ci95_high": round(float(boot[metric].quantile(0.975)), 5),
            "p_bootstrap_two_sided": round(float(2 * min((boot[metric] <= 0).mean(), (boot[metric] >= 0).mean())), 5),
        }
        for metric in boot.columns
    }


def load_and_align_predictions() -> pd.DataFrame:
    final_v1_df = pd.read_csv(FINAL_V1_PREDICTIONS)
    final_v2_df = pd.read_csv(FINAL_V2_PREDICTIONS)
    cols = ["sample_id", "true_label", "prob_phone_average", "pred_average"]
    merged = final_v1_df[cols].merge(
        final_v2_df[cols],
        on="sample_id",
        suffixes=("_final_v1", "_final_v2"),
        validate="one_to_one",
    )
    if len(merged) != len(final_v1_df) or len(merged) != len(final_v2_df):
        raise ValueError(f"Prediction alignment mismatch: merged={len(merged)}, final_v1={len(final_v1_df)}, final_v2={len(final_v2_df)}")
    if not (merged["true_label_final_v1"] == merged["true_label_final_v2"]).all():
        raise ValueError("Aligned predictions have inconsistent true labels.")
    return merged


def run_statistics() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_and_align_predictions()
    y_true = merged["true_label_final_v1"].to_numpy(dtype=int)
    final_v1_prob = merged["prob_phone_average_final_v1"].to_numpy(dtype=float)
    final_v2_prob = merged["prob_phone_average_final_v2"].to_numpy(dtype=float)
    final_v1_pred = merged["pred_average_final_v1"].to_numpy(dtype=int)
    final_v2_pred = merged["pred_average_final_v2"].to_numpy(dtype=int)

    final_v1_correct = final_v1_pred == y_true
    final_v2_correct = final_v2_pred == y_true
    final_v1_only = int((final_v1_correct & ~final_v2_correct).sum())
    final_v2_only = int((~final_v1_correct & final_v2_correct).sum())
    both_correct = int((final_v1_correct & final_v2_correct).sum())
    both_wrong = int((~final_v1_correct & ~final_v2_correct).sum())
    mcnemar = stats.binomtest(min(final_v1_only, final_v2_only), final_v1_only + final_v2_only, 0.5)

    final_v1_f1 = f1_score(y_true, final_v1_pred, average="macro")
    final_v2_f1 = f1_score(y_true, final_v2_pred, average="macro")
    delong = delong_roc_test(y_true, final_v1_prob, final_v2_prob)
    bootstrap = bootstrap_paired_ci(y_true, final_v1_prob, final_v2_prob)

    result = {
        "comparison": "Final v1 average fusion vs Final v2 average fusion",
        "prediction_alignment": {
            "support": int(len(merged)),
            "positive_support": int(y_true.sum()),
            "negative_support": int((y_true == 0).sum()),
        },
        "metrics": {
            "final_v1_macro_f1": round(float(final_v1_f1), 5),
            "final_v2_macro_f1": round(float(final_v2_f1), 5),
            "delta_macro_f1_final_v2_minus_final_v1": round(float(final_v2_f1 - final_v1_f1), 5),
            "final_v1_roc_auc": round(float(roc_auc_score(y_true, final_v1_prob)), 5),
            "final_v2_roc_auc": round(float(roc_auc_score(y_true, final_v2_prob)), 5),
            "final_v1_pr_auc": round(float(average_precision_score(y_true, final_v1_prob)), 5),
            "final_v2_pr_auc": round(float(average_precision_score(y_true, final_v2_prob)), 5),
        },
        "mcnemar_exact": {
            "table_both_correct_final_v1_only_final_v2_only_both_wrong": [
                [both_correct, final_v1_only],
                [final_v2_only, both_wrong],
            ],
            "final_v1_only_correct": final_v1_only,
            "final_v2_only_correct": final_v2_only,
            "p_value": float(mcnemar.pvalue),
            "significant_alpha_0_05": bool(mcnemar.pvalue < 0.05),
        },
        "delong_roc_auc": {
            **delong,
            "significant_alpha_0_05": bool(delong["p_value"] < 0.05),
        },
        "paired_stratified_bootstrap": {
            "n_bootstrap": N_BOOTSTRAP,
            "random_seed": RANDOM_SEED,
            "delta_is_final_v2_minus_final_v1": True,
            **bootstrap,
        },
    }

    (OUTPUT_DIR / "final_v1_vs_final_v2_statistics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = [
        {
            "test": "McNemar exact",
            "metric": "decision correctness @ threshold 0.50",
            "final_v1": f"unique_correct={final_v1_only}",
            "final_v2": f"unique_correct={final_v2_only}",
            "delta_final_v2_minus_final_v1": final_v2_only - final_v1_only,
            "p_value": mcnemar.pvalue,
            "significant_alpha_0_05": mcnemar.pvalue < 0.05,
        },
        {
            "test": "DeLong",
            "metric": "ROC-AUC",
            "final_v1": delong["auc_final_v1"],
            "final_v2": delong["auc_final_v2"],
            "delta_final_v2_minus_final_v1": delong["delta_auc_final_v2_minus_final_v1"],
            "p_value": delong["p_value"],
            "significant_alpha_0_05": delong["p_value"] < 0.05,
        },
    ]
    for metric, values in bootstrap.items():
        rows.append(
            {
                "test": "Paired stratified bootstrap",
                "metric": metric,
                "final_v1": "",
                "final_v2": "",
                "delta_final_v2_minus_final_v1": values["mean_delta"],
                "ci95_low": values["ci95_low"],
                "ci95_high": values["ci95_high"],
                "p_value": values["p_bootstrap_two_sided"],
                "significant_alpha_0_05": values["p_bootstrap_two_sided"] < 0.05,
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "final_v1_vs_final_v2_statistics_summary.csv", index=False)
    return result


def main() -> None:
    print(json.dumps(run_statistics(), indent=2))


if __name__ == "__main__":
    main()
