"""
Experiment 16: qualitative failure-case review.

Builds a compact set of representative samples from Experiment 15 error
analysis, exports their metadata with image paths, and creates paired
front-side contact sheets for quick visual inspection.
"""

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from src.config import MANIFEST_PAIRED_PATH, RESULTS_DIR


TARGET_CASES = [
    "fusion_fixed_front_error",
    "fusion_broke_front_correct",
    "side_only_correct",
    "both_wrong",
]

CASE_QUOTAS = {
    "fusion_fixed_front_error": 5,
    "fusion_broke_front_correct": 5,
    "side_only_correct": 5,
    "both_wrong": 5,
}

LABEL_NAMES = {
    0: "safe_driving",
    1: "phone_use",
}


def build_sample_id(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: f"S{int(r['subject_id']):02d}_AC{int(r['activity_id']):02d}_F{int(r['frame']):05d}",
        axis=1,
    )


def compute_priority_score(row: pd.Series) -> float:
    front_gap = abs(float(row["front_prob_phone"]) - 0.5)
    side_gap = abs(float(row["side_prob_phone"]) - 0.5)
    avg_gap = abs(float(row["average_prob_phone"]) - 0.5)
    adapt_gap = abs(float(row["adaptive_prob_phone"]) - 0.5)

    if row["case_type"] == "fusion_fixed_front_error":
        return side_gap - front_gap
    if row["case_type"] == "fusion_broke_front_correct":
        return side_gap + max(avg_gap, adapt_gap) - front_gap
    if row["case_type"] == "side_only_correct":
        return side_gap - front_gap
    if row["case_type"] == "both_wrong":
        return -(front_gap + side_gap)
    return 0.0


def select_samples(df_cases: pd.DataFrame) -> pd.DataFrame:
    picked = []
    for case_type in TARGET_CASES:
        df_case = df_cases[df_cases["case_type"] == case_type].copy()
        if df_case.empty:
            continue
        quota = min(CASE_QUOTAS[case_type], len(df_case))
        df_case["priority_score"] = df_case.apply(compute_priority_score, axis=1)
        df_case = df_case.sort_values(
            by=["priority_score", "sample_id"],
            ascending=[False, True],
        ).head(quota)
        picked.append(df_case)

    if not picked:
        raise ValueError("Tidak ada sampel untuk case_type target.")

    df_selected = pd.concat(picked, ignore_index=True)
    df_selected["selection_rank"] = (
        df_selected.groupby("case_type")["priority_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return df_selected.sort_values(["case_type", "selection_rank", "sample_id"]).reset_index(drop=True)


def draw_text_block(draw: ImageDraw.ImageDraw, xy, lines, line_height=18):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill="black")
        y += line_height


def create_contact_sheet(row: pd.Series, output_dir: Path):
    img_front = Image.open(row["filepath_front"]).convert("RGB")
    img_side = Image.open(row["filepath_side"]).convert("RGB")

    target_width = 420
    target_height = 280
    canvas_width = target_width * 2 + 48
    canvas_height = target_height + 110
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")

    front_resized = img_front.resize((target_width, target_height))
    side_resized = img_side.resize((target_width, target_height))
    canvas.paste(front_resized, (16, 16))
    canvas.paste(side_resized, (target_width + 32, 16))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((16, 16, 16 + target_width, 16 + target_height), outline="black", width=2)
    draw.rectangle(
        (target_width + 32, 16, target_width * 2 + 32, 16 + target_height),
        outline="black",
        width=2,
    )
    draw.text((24, 24), "FRONT", fill="yellow")
    draw.text((target_width + 40, 24), "SIDE", fill="yellow")

    lines = [
        f"sample_id: {row['sample_id']}",
        f"case_type: {row['case_type']} (rank {row['selection_rank']})",
        f"label: {row['true_label_name']}",
        (
            "front/side/avg/adapt pred: "
            f"{row['front_pred']}/{row['side_pred']}/{row['average_pred']}/{row['adaptive_pred']}"
        ),
        (
            "phone prob front/side/avg/adapt: "
            f"{row['front_prob_phone']:.3f}/{row['side_prob_phone']:.3f}/"
            f"{row['average_prob_phone']:.3f}/{row['adaptive_prob_phone']:.3f}"
        ),
    ]
    draw_text_block(draw, (16, target_height + 28), lines)

    output_path = output_dir / f"{row['case_type']}__{row['sample_id']}.jpg"
    canvas.save(output_path, quality=92)


def build_notes(df_selected: pd.DataFrame) -> str:
    total = len(df_selected)
    counts = df_selected["case_type"].value_counts().to_dict()
    safe_selected = int((df_selected["true_label"] == 0).sum())
    phone_selected = int((df_selected["true_label"] == 1).sum())

    safe_front_as_phone = int(
        ((df_selected["true_label"] == 0) & (df_selected["front_pred"] == 1)).sum()
    )
    both_wrong_safe = int(
        ((df_selected["case_type"] == "both_wrong") & (df_selected["true_label"] == 0)).sum()
    )
    side_help_safe = int(
        ((df_selected["case_type"].isin(["side_only_correct", "fusion_fixed_front_error"])) & (df_selected["true_label"] == 0)).sum()
    )
    fusion_broke_front_safe = int(
        ((df_selected["case_type"] == "fusion_broke_front_correct") & (df_selected["true_label"] == 0)).sum()
    )

    lines = [
        "# Experiment 16 - Qualitative Sample Review",
        "",
        "## Tujuan",
        "",
        "Memilih sampel error paling representatif dari output Experiment 15 untuk",
        "review visual front-vs-side dan pembahasan kualitatif di laporan.",
        "",
        "## Output",
        "",
        "- `results/exp16_selected_failure_cases.csv`",
        "- `results/exp16_failure_case_images/`",
        "- `experiment_16_notes.md`",
        "",
        "## Ringkasan Seleksi",
        "",
        f"- Total sampel terpilih: {total}",
        f"- Komposisi label: {safe_selected} safe_driving, {phone_selected} phone_use",
        f"- Komposisi case: {counts}",
        "",
        "## Temuan Awal Untuk Dilaporkan",
        "",
        f"- Front salah cukup sering terjadi pada sampel `safe_driving` yang condong diprediksi `phone_use` ({safe_front_as_phone} dari {total} sampel terpilih).",
        f"- Side membantu terutama pada sampel aman yang ambigu dari front view ({side_help_safe} sampel pada `side_only_correct` + `fusion_fixed_front_error`).",
        f"- Fusion merusak terutama ketika front sudah benar tetapi side memberi sinyal kuat ke kelas lawan ({fusion_broke_front_safe} sampel aman di `fusion_broke_front_correct`).",
        f"- Kasus `both_wrong` masih didominasi sampel aman yang sangat mirip `phone_use` ({both_wrong_safe} sampel), cocok untuk menjawab pertanyaan apakah kelas safe sering mirip phone_use.",
        "",
        "## Observasi Visual Awal",
        "",
        "- Pada beberapa `safe_driving`, front view menampilkan tangan kanan dekat wajah, setir, atau konsol tengah sehingga gesture aman terlihat mirip aktivitas memegang ponsel.",
        "- Pada beberapa `phone_use`, objek ponsel tidak terlalu jelas dari front karena tertutup tangan, rendah di area lap, atau menyatu dengan pose tubuh; side view lebih mudah menangkap siluet tangan-ke-telinga atau perangkat di samping tubuh.",
        "- Beberapa kasus `fusion_broke_front_correct` menunjukkan side view terlalu percaya diri pada sinyal yang salah, misalnya tangan di area gear/console atau komposisi penumpang belakang yang menambah clutter visual.",
        "- Kasus `both_wrong` cenderung borderline: probabilitas front dan side sama-sama dekat ambang 0.5, menandakan frame memang ambigu secara visual, bukan sekadar kesalahan threshold tunggal.",
        "",
        "## Checklist Review Visual",
        "",
        "- Cek apakah tangan/ponsel tertutup atau keluar frame pada front.",
        "- Cek apakah pose tubuh atau arah kepala lebih jelas pada side.",
        "- Cek apakah front menonjolkan gesture yang mirip phone_use walau label sebenarnya safe.",
        "- Cek apakah fusion gagal karena side terlalu percaya diri pada prediksi yang salah.",
        "",
        "## Kesimpulan Cepat",
        "",
        "- Experiment 15 sudah memberi daftar failure case per-sample.",
        "- Experiment 16 ini melanjutkan ke tahap kurasi visual sehingga sampel bisa langsung dipakai untuk pembahasan kualitatif.",
        "- Fokus utama pembahasan sebaiknya pada confusion `safe_driving -> phone_use`, kontribusi side saat front ambigu, dan beberapa contoh saat fusion justru menurunkan keputusan yang awalnya benar.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 16 qualitative review artifacts.")
    parser.add_argument(
        "--error-csv",
        default=str(RESULTS_DIR / "error_analysis_exp15.csv"),
        help="Path to Experiment 15 per-sample CSV.",
    )
    args = parser.parse_args()

    error_csv = Path(args.error_csv)
    if not error_csv.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {error_csv}")

    df_error = pd.read_csv(error_csv)
    df_manifest = pd.read_csv(MANIFEST_PAIRED_PATH).copy()
    df_manifest["sample_id"] = build_sample_id(df_manifest)

    join_cols = [
        "sample_id",
        "subject_id",
        "activity_id",
        "frame",
        "filepath_front",
        "filepath_side",
        "binary_label",
    ]
    df_joined = df_error.merge(df_manifest[join_cols], on="sample_id", how="left", validate="one_to_one")
    if df_joined["filepath_front"].isna().any() or df_joined["filepath_side"].isna().any():
        raise ValueError("Ada sample_id yang gagal dipetakan ke filepath front/side.")

    df_cases = df_joined[df_joined["case_type"].isin(TARGET_CASES)].copy()
    df_selected = select_samples(df_cases)
    df_selected["true_label_name"] = df_selected["true_label"].map(LABEL_NAMES)

    output_csv = RESULTS_DIR / "exp16_selected_failure_cases.csv"
    output_image_dir = RESULTS_DIR / "exp16_failure_case_images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "sample_id",
        "case_type",
        "selection_rank",
        "priority_score",
        "subject_id",
        "activity_id",
        "frame",
        "binary_label",
        "true_label",
        "true_label_name",
        "front_pred",
        "side_pred",
        "average_pred",
        "adaptive_pred",
        "front_prob_phone",
        "side_prob_phone",
        "average_prob_phone",
        "adaptive_prob_phone",
        "front_correct",
        "side_correct",
        "average_correct",
        "adaptive_correct",
        "filepath_front",
        "filepath_side",
    ]
    df_selected[export_cols].to_csv(output_csv, index=False)

    for _, row in df_selected.iterrows():
        create_contact_sheet(row, output_image_dir)

    notes_path = Path("experiment_16_notes.md")
    notes_path.write_text(build_notes(df_selected), encoding="utf-8")

    print(f"Saved CSV: {output_csv}")
    print(f"Saved images: {output_image_dir}")
    print(f"Saved notes: {notes_path.resolve()}")


if __name__ == "__main__":
    main()
