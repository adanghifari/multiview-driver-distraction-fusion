from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "confusion_matrix": cm,
        "safe_to_phone": int(cm[0][1]),
        "phone_to_safe": int(cm[1][0]),
    }


def compute_ece_binary(y_true, y_prob, n_bins: int = 10) -> float:
    probs = np.asarray(y_prob, dtype=float)
    labels = np.asarray(y_true, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for idx in range(n_bins):
        lo = bin_edges[idx]
        hi = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return round(float(ece), 5)


def compute_brier_score(y_true, y_prob) -> float:
    probs = np.asarray(y_prob, dtype=float)
    labels = np.asarray(y_true, dtype=float)
    return round(float(np.mean((probs - labels) ** 2)), 5)
