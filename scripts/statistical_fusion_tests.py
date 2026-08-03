"""
Statistical comparison for decision-level fusion predictions.

Input CSV must contain one ground-truth label column and probability columns for
the phone class from each fusion method. Defaults match:
results/final_v1/adaptive_linear_normalization_predictions.csv

References:
- McNemar exact test: statsmodels.stats.contingency_tables.mcnemar
- DeLong ROC-AUC test: Sun & Xu (2014), "Fast Implementation of DeLong's
  Algorithm for Comparing the Areas Under Correlated Receiver Operating
  Characteristic Curves", IEEE Signal Processing Letters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

try:
    from statsmodels.stats.contingency_tables import mcnemar as statsmodels_mcnemar
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    statsmodels_mcnemar = None


@dataclass(frozen=True)
class Method:
    name: str
    prob_col: str


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for DeLong's method, using 1-based ranks."""
    order = np.argsort(x)
    sorted_x = x[order]
    midranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x), dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted: np.ndarray, n_positive: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong covariance for one or more correlated ROC curves.

    predictions_sorted shape: (n_classifiers, n_examples), ordered with all
    positive examples first, then all negative examples.
    """
    n_negative = predictions_sorted.shape[1] - n_positive
    positive_scores = predictions_sorted[:, :n_positive]
    negative_scores = predictions_sorted[:, n_positive:]
    n_classifiers = predictions_sorted.shape[0]

    tx = np.empty((n_classifiers, n_positive), dtype=float)
    ty = np.empty((n_classifiers, n_negative), dtype=float)
    tz = np.empty((n_classifiers, n_positive + n_negative), dtype=float)

    for r in range(n_classifiers):
        tx[r] = compute_midrank(positive_scores[r])
        ty[r] = compute_midrank(negative_scores[r])
        tz[r] = compute_midrank(predictions_sorted[r])

    aucs = tz[:, :n_positive].sum(axis=1) / n_positive / n_negative
    aucs -= (n_positive + 1.0) / (2.0 * n_negative)

    v01 = (tz[:, :n_positive] - tx) / n_negative
    v10 = 1.0 - (tz[:, n_positive:] - ty) / n_positive
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / n_positive + sy / n_negative

    if n_classifiers == 1:
        covariance = np.array([[float(covariance)]])
    return aucs, covariance


def normal_survival_two_sided(z: float) -> float:
    """Two-sided normal p-value without requiring scipy."""
    from math import erfc, sqrt

    return erfc(abs(z) / sqrt(2.0))


def exact_mcnemar_fallback(table: np.ndarray) -> tuple[int, float]:
    """
    Exact McNemar fallback using a two-sided binomial test with p=0.5.

    For discordant counts b and c, the exact statistic commonly reported by
    statsmodels is min(b, c), and the p-value is 2 * P[X <= min(b,c)] for
    X ~ Binomial(b+c, 0.5), capped at 1.0.
    """
    from math import comb

    b = int(table[0, 1])
    c = int(table[1, 0])
    n = b + c
    statistic = min(b, c)
    if n == 0:
        return statistic, 1.0
    tail = sum(comb(n, k) for k in range(statistic + 1)) / (2**n)
    return statistic, min(1.0, 2.0 * tail)


def delong_roc_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> tuple[float, float, float]:
    """Return auc_a, auc_b, and two-sided p-value for paired DeLong ROC-AUC test."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.vstack([score_a, score_b]).astype(float)
    order = np.argsort(-y_true)
    n_positive = int(y_true.sum())
    n_negative = int(len(y_true) - n_positive)

    if n_positive == 0 or n_negative == 0:
        raise ValueError("DeLong test needs both positive and negative labels.")

    aucs, covariance = fast_delong(scores[:, order], n_positive)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)

    if variance <= 0:
        p_value = 1.0 if np.isclose(aucs[0], aucs[1]) else 0.0
    else:
        z = float((aucs[0] - aucs[1]) / np.sqrt(variance))
        p_value = normal_survival_two_sided(z)
    return float(aucs[0]), float(aucs[1]), float(p_value)


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample with replacement within each class so class ratio stays fixed."""
    y_true = np.asarray(y_true)
    indices = []
    for label in np.unique(y_true):
        class_idx = np.flatnonzero(y_true == label)
        indices.append(rng.choice(class_idx, size=len(class_idx), replace=True))
    return np.concatenate(indices)


def metric_value(metric: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    if metric == "macro_f1":
        pred = (scores >= threshold).astype(int)
        return float(f1_score(y_true, pred, average="macro"))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, scores))
    if metric == "pr_auc":
        return float(average_precision_score(y_true, scores))
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str,
    threshold: float,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y_true, rng)
        values.append(metric_value(metric, y_true[idx], scores[idx], threshold))
    values = np.asarray(values)
    return float(values.mean()), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def ci_overlap(ci_a: tuple[float, float, float], ci_b: tuple[float, float, float]) -> bool:
    return max(ci_a[1], ci_b[1]) <= min(ci_a[2], ci_b[2])


def significance_text(p_value: float, alpha: float) -> str:
    return "Ya" if p_value < alpha else "Tidak"


def mcnemar_pair(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str,
    name_b: str,
    threshold: float,
    alpha: float,
) -> dict[str, str | float]:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    table = np.array(
        [
            [np.sum(correct_a & correct_b), np.sum(correct_a & ~correct_b)],
            [np.sum(~correct_a & correct_b), np.sum(~correct_a & ~correct_b)],
        ]
    )
    if statsmodels_mcnemar is not None:
        result = statsmodels_mcnemar(table, exact=True)
        statistic = result.statistic
        p_value = float(result.pvalue)
    else:
        statistic, p_value = exact_mcnemar_fallback(table)
    b = int(table[0, 1])
    c = int(table[1, 0])
    return {
        "Perbandingan": f"{name_a} vs {name_b}",
        "Metrik": f"McNemar exact @ threshold {threshold:.2f}",
        "Nilai Metode A": f"benar unik A={b}",
        "Nilai Metode B": f"benar unik B={c}",
        "Statistik Uji": f"statistic={statistic}; table={table.tolist()}",
        "P-value": p_value,
        "Signifikan (Ya/Tidak)": significance_text(p_value, alpha),
        "Interpretasi": "Tidak ada bukti perbedaan label keputusan yang signifikan."
        if p_value >= alpha
        else "Ada bukti perbedaan label keputusan yang signifikan.",
    }


def paired_methods(methods: list[Method]) -> Iterable[tuple[Method, Method]]:
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            yield methods[i], methods[j]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical tests for fusion predictions.")
    parser.add_argument(
        "--input",
        default="results/final_v1/adaptive_linear_normalization_predictions.csv",
        help="CSV containing true labels and phone probabilities.",
    )
    parser.add_argument("--true-col", default="true_label", help="Ground-truth label column.")
    parser.add_argument("--average-col", default="prob_phone_average", help="Average fusion phone probability column.")
    parser.add_argument(
        "--softmax-col",
        default="prob_phone_adaptive_softmax",
        help="Adaptive Softmax phone probability column.",
    )
    parser.add_argument(
        "--linear-col",
        default="prob_phone_adaptive_linear",
        help="Adaptive Linear Normalization phone probability column.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.50, 0.40, 0.45])
    parser.add_argument("--bootstrap", type=int, default=2000, help="Number of bootstrap resamples.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default="results/final_v1/statistical_fusion_tests_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    methods = [
        Method("Average Fusion", args.average_col),
        Method("Adaptive Softmax", args.softmax_col),
        Method("Adaptive Linear Normalization", args.linear_col),
    ]

    required_cols = [args.true_col] + [m.prob_col for m in methods]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    y_true = df[args.true_col].to_numpy(dtype=int)
    scores = {m.name: df[m.prob_col].to_numpy(dtype=float) for m in methods}
    rows: list[dict[str, str | float]] = []

    # 1) McNemar exact tests compare paired correctness of thresholded labels.
    for threshold in args.thresholds:
        preds = {name: (score >= threshold).astype(int) for name, score in scores.items()}
        for a, b in paired_methods(methods):
            rows.append(
                mcnemar_pair(
                    y_true,
                    preds[a.name],
                    preds[b.name],
                    a.name,
                    b.name,
                    threshold,
                    args.alpha,
                )
            )

    # 2) Stratified bootstrap confidence intervals for each method and metric.
    bootstrap_results: dict[tuple[str, str], tuple[float, float, float]] = {}
    for method in methods:
        for metric in ["macro_f1", "roc_auc", "pr_auc"]:
            ci = bootstrap_ci(
                y_true,
                scores[method.name],
                metric,
                threshold=0.50,
                n_bootstrap=args.bootstrap,
                random_state=args.random_state,
            )
            bootstrap_results[(method.name, metric)] = ci
            rows.append(
                {
                    "Perbandingan": method.name,
                    "Metrik": f"Bootstrap 95% CI {metric}",
                    "Nilai Metode A": f"mean={ci[0]:.5f}; CI95%=[{ci[1]:.5f}, {ci[2]:.5f}]",
                    "Nilai Metode B": "",
                    "Statistik Uji": f"stratified bootstrap n={args.bootstrap}",
                    "P-value": "",
                    "Signifikan (Ya/Tidak)": "",
                    "Interpretasi": "Interval kepercayaan menggambarkan ketidakpastian estimasi pada test set.",
                }
            )

    # 3) CI overlap checks. Overlap is descriptive, not a formal significance test.
    for adaptive in ["Adaptive Softmax", "Adaptive Linear Normalization"]:
        for metric in ["macro_f1", "roc_auc", "pr_auc"]:
            ci_avg = bootstrap_results[("Average Fusion", metric)]
            ci_adp = bootstrap_results[(adaptive, metric)]
            overlap = ci_overlap(ci_avg, ci_adp)
            rows.append(
                {
                    "Perbandingan": f"Average Fusion vs {adaptive}",
                    "Metrik": f"Overlap CI {metric}",
                    "Nilai Metode A": f"Average CI=[{ci_avg[1]:.5f}, {ci_avg[2]:.5f}]",
                    "Nilai Metode B": f"{adaptive} CI=[{ci_adp[1]:.5f}, {ci_adp[2]:.5f}]",
                    "Statistik Uji": "CI overlap descriptive check",
                    "P-value": "",
                    "Signifikan (Ya/Tidak)": "Tidak" if overlap else "Ya",
                    "Interpretasi": "CI tumpang tindih; perbedaan sebaiknya dianggap belum meyakinkan."
                    if overlap
                    else "CI tidak tumpang tindih; ada indikasi perbedaan praktis yang lebih kuat.",
                }
            )

    # 4) DeLong paired ROC-AUC tests for correlated predictions on the same samples.
    for a, b in paired_methods(methods):
        auc_a, auc_b, p_value = delong_roc_test(y_true, scores[a.name], scores[b.name])
        rows.append(
            {
                "Perbandingan": f"{a.name} vs {b.name}",
                "Metrik": "DeLong ROC-AUC",
                "Nilai Metode A": f"{auc_a:.5f}",
                "Nilai Metode B": f"{auc_b:.5f}",
                "Statistik Uji": f"delta_auc={auc_a - auc_b:.5f}",
                "P-value": p_value,
                "Signifikan (Ya/Tidak)": significance_text(p_value, args.alpha),
                "Interpretasi": "Perbedaan ROC-AUC signifikan secara statistik."
                if p_value < args.alpha
                else "Tidak ada bukti ROC-AUC berbeda signifikan; selisih dapat berasal dari variasi sampel.",
            }
        )

    out = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved summary to: {output_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
