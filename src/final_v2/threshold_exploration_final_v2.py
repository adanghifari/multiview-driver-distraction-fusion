import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluate import load_trained_model
from src.final_v2.config_final_v2 import CHECKPOINTS_DIR_FINAL, FRAME_STRIDE_FINAL, RESULTS_DIR_FINAL, THRESHOLD
from src.final_v2.dataset_final_v2 import get_paired_split_loader, load_paired_split_dataframe
from src.final_v2.fusion_variant_analysis_final_v2 import adaptive_linear_normalization
from src.final_v2.metrics_final_v2 import compute_metrics
from src.fusion import adaptive_fusion, average_fusion


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


FRONT_CHECKPOINT = CHECKPOINTS_DIR_FINAL / "front_best_final_v2.pt"
SIDE_CHECKPOINT = CHECKPOINTS_DIR_FINAL / "side_best_final_v2.pt"
OUTPUT_DIR = RESULTS_DIR_FINAL / "threshold_exploration"
THRESHOLDS = np.round(np.arange(0.01, 0.99 + 0.001, 0.01), 2)
FUSION_METHODS = ("average_fusion", "adaptive_fusion", "adaptive_linear_normalization")


def _assert_frame_stride(checkpoint: dict, checkpoint_path: Path, label: str) -> None:
    actual_stride = (checkpoint.get("hyperparameters") or {}).get("frame_stride")
    if actual_stride != FRAME_STRIDE_FINAL:
        raise ValueError(
            f"{label} checkpoint frame_stride mismatch for {checkpoint_path}: "
            f"expected {FRAME_STRIDE_FINAL}, got {actual_stride}."
        )


@torch.no_grad()
def collect_paired_scores(split: str, model_front, model_side, device: torch.device) -> dict:
    df_split = load_paired_split_dataframe(split)
    loader = get_paired_split_loader(split)
    model_front.eval()
    model_side.eval()

    front_probs = []
    side_probs = []
    labels_all = []
    indices_all = []

    for img_front, img_side, labels, indices in loader:
        logits_front = model_front(img_front.to(device))
        logits_side = model_side(img_side.to(device))
        probs_front = torch.softmax(logits_front, dim=1)[:, 1].cpu().numpy()
        probs_side = torch.softmax(logits_side, dim=1)[:, 1].cpu().numpy()

        front_probs.extend(probs_front.tolist())
        side_probs.extend(probs_side.tolist())
        labels_all.extend(labels.tolist())
        indices_all.extend(indices.tolist())

    if indices_all != list(range(len(df_split))):
        raise AssertionError(f"Urutan paired {split} loader final_v2 tidak sinkron dengan dataframe.")

    return {
        "df": df_split,
        "front": np.asarray(front_probs, dtype=float),
        "side": np.asarray(side_probs, dtype=float),
        "labels": np.asarray(labels_all, dtype=int),
    }


def evaluate_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    metrics = compute_metrics(labels.tolist(), preds.tolist())
    return {
        "threshold": round(float(threshold), 2),
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "macro_f1": metrics["f1_macro"],
        "confusion_matrix": metrics["confusion_matrix"],
        "safe_to_phone": metrics["safe_to_phone"],
        "phone_to_safe": metrics["phone_to_safe"],
    }


def threshold_search(scores: np.ndarray, labels: np.ndarray, method: str, split: str) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        result = evaluate_threshold(scores, labels, float(threshold))
        rows.append(
            {
                "split": split,
                "method": method,
                **{k: v for k, v in result.items() if k != "confusion_matrix"},
                "confusion_matrix": json.dumps(result["confusion_matrix"]),
            }
        )
    return pd.DataFrame(rows)


def select_best_threshold(df_method_validation: pd.DataFrame) -> pd.Series:
    ranked = df_method_validation.copy()
    ranked["distance_to_half"] = (ranked["threshold"] - THRESHOLD).abs()
    ranked = ranked.sort_values(
        by=["macro_f1", "phone_to_safe", "safe_to_phone", "distance_to_half"],
        ascending=[False, True, True, True],
    )
    return ranked.iloc[0]


def select_best_test_threshold_descriptive(df_method_test: pd.DataFrame) -> pd.Series:
    ranked = df_method_test.copy()
    ranked["distance_to_half"] = (ranked["threshold"] - THRESHOLD).abs()
    ranked = ranked.sort_values(
        by=["macro_f1", "phone_to_safe", "safe_to_phone", "distance_to_half"],
        ascending=[False, True, True, True],
    )
    return ranked.iloc[0]


def build_scores(split_scores: dict) -> dict:
    return {
        "average_fusion": average_fusion(split_scores["front"], split_scores["side"]),
        "adaptive_fusion": adaptive_fusion(split_scores["front"], split_scores["side"], threshold=THRESHOLD),
        "adaptive_linear_normalization": adaptive_linear_normalization(split_scores["front"], split_scores["side"]),
    }


def build_notes(summary: dict) -> str:
    lines = [
        "# Final v2 Threshold Exploration",
        "",
        "## Tujuan",
        "",
        "- Mengeksplorasi threshold `phone_use` pada output fusion Final v2 tanpa retraining.",
        "- Threshold grid penuh 0.01 sampai 0.99 dicoba untuk validation dan test.",
        "- Threshold protocol dipilih dari validation set, lalu dievaluasi satu kali pada test set.",
        "- Best test threshold disertakan sebagai analisis deskriptif/oracle, bukan protocol selection resmi.",
        "- Hasil utama Final v2 tetap threshold 0.50; eksplorasi ini adalah analisis tambahan.",
        "- Adaptive Linear Normalization disertakan sebagai analisis tambahan, bukan pengganti adaptive fusion proposal.",
        "",
        "## Ringkasan",
        "",
    ]

    for method, block in summary["methods"].items():
        best_val = block["best_validation_threshold"]
        baseline = block["test_baseline_threshold_0_50"]
        selected = block["test_selected_threshold"]
        lines.extend(
            [
                f"### {method}",
                "",
                f"- Best validation threshold: {best_val['threshold']:.2f} dengan Macro F1 validation {best_val['macro_f1']:.5f}.",
                f"- Test baseline 0.50: Macro F1 {baseline['macro_f1']:.5f}, accuracy {baseline['accuracy']:.5f}, safe->phone {baseline['safe_to_phone']}, phone->safe {baseline['phone_to_safe']}.",
                f"- Test selected threshold: Macro F1 {selected['macro_f1']:.5f}, accuracy {selected['accuracy']:.5f}, safe->phone {selected['safe_to_phone']}, phone->safe {selected['phone_to_safe']}.",
                f"- Delta Macro F1 test: {block['delta_macro_f1_selected_minus_baseline']:+.5f}.",
                f"- Best test threshold descriptive/oracle: {block['best_test_threshold_descriptive']['threshold']:.2f} dengan Macro F1 test {block['best_test_threshold_descriptive']['macro_f1']:.5f}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Catatan Interpretasi",
            "",
            "- Jika threshold terpilih menaikkan Macro F1 test, tetap baca sebagai hasil tuning berbasis validation, bukan retraining model.",
            "- Perhatikan trade-off `safe->phone` dan `phone->safe`; threshold lebih tinggi biasanya mengurangi false alarm safe->phone tetapi dapat menaikkan missed phone->safe.",
            "- Average fusion dan adaptive fusion tidak diubah rumusnya.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_threshold_exploration() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model_front, front_ckpt = load_trained_model("front", device, checkpoint_path=str(FRONT_CHECKPOINT))
    model_side, side_ckpt = load_trained_model("side", device, checkpoint_path=str(SIDE_CHECKPOINT))
    _assert_frame_stride(front_ckpt, FRONT_CHECKPOINT, "final_v2 front")
    _assert_frame_stride(side_ckpt, SIDE_CHECKPOINT, "final_v2 side")

    val_raw = collect_paired_scores("val", model_front, model_side, device)
    test_raw = collect_paired_scores("test", model_front, model_side, device)
    val_scores = build_scores(val_raw)
    test_scores = build_scores(test_raw)

    all_search_rows = []
    summary = {
        "experiment": "final_v2_threshold_exploration",
        "frame_stride": FRAME_STRIDE_FINAL,
        "baseline_threshold": THRESHOLD,
        "threshold_grid": [float(v) for v in THRESHOLDS],
        "validation_support": int(len(val_raw["labels"])),
        "test_support": int(len(test_raw["labels"])),
        "checkpoints": {
            "front": str(FRONT_CHECKPOINT),
            "side": str(SIDE_CHECKPOINT),
        },
            "selection_rule": "maximize validation Macro F1, then minimize phone_to_safe, safe_to_phone, and distance to 0.50",
            "best_test_threshold_descriptive_is_oracle": True,
            "methods": {},
        }

    for method in FUSION_METHODS:
        val_search = threshold_search(val_scores[method], val_raw["labels"], method, "validation")
        test_search = threshold_search(test_scores[method], test_raw["labels"], method, "test")
        all_search_rows.extend([val_search, test_search])

        best_val = select_best_threshold(val_search)
        best_test = select_best_test_threshold_descriptive(test_search)
        selected_threshold = float(best_val["threshold"])
        baseline_test = evaluate_threshold(test_scores[method], test_raw["labels"], THRESHOLD)
        selected_test = evaluate_threshold(test_scores[method], test_raw["labels"], selected_threshold)

        summary["methods"][method] = {
            "best_validation_threshold": {
                "threshold": round(selected_threshold, 2),
                "accuracy": float(best_val["accuracy"]),
                "precision_macro": float(best_val["precision_macro"]),
                "recall_macro": float(best_val["recall_macro"]),
                "macro_f1": float(best_val["macro_f1"]),
                "safe_to_phone": int(best_val["safe_to_phone"]),
                "phone_to_safe": int(best_val["phone_to_safe"]),
                "confusion_matrix": json.loads(best_val["confusion_matrix"]),
            },
            "test_baseline_threshold_0_50": baseline_test,
            "test_selected_threshold": selected_test,
            "best_test_threshold_descriptive": {
                "threshold": round(float(best_test["threshold"]), 2),
                "accuracy": float(best_test["accuracy"]),
                "precision_macro": float(best_test["precision_macro"]),
                "recall_macro": float(best_test["recall_macro"]),
                "macro_f1": float(best_test["macro_f1"]),
                "safe_to_phone": int(best_test["safe_to_phone"]),
                "phone_to_safe": int(best_test["phone_to_safe"]),
                "confusion_matrix": json.loads(best_test["confusion_matrix"]),
            },
            "delta_macro_f1_selected_minus_baseline": round(
                float(selected_test["macro_f1"] - baseline_test["macro_f1"]), 5
            ),
        }

    search_csv = OUTPUT_DIR / "threshold_search_final_v2.csv"
    summary_json = OUTPUT_DIR / "threshold_summary_final_v2.json"
    notes_md = OUTPUT_DIR / "threshold_notes_final_v2.md"
    pd.concat(all_search_rows, ignore_index=True).to_csv(search_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    notes_md.write_text(build_notes(summary), encoding="utf-8")
    return {
        "summary": summary,
        "search_csv": str(search_csv),
        "summary_json": str(summary_json),
        "notes_md": str(notes_md),
    }


def main() -> None:
    outputs = run_threshold_exploration()
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
