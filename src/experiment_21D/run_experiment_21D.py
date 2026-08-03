import argparse

from src.experiment_21D.evaluate_experiment_21D import run_single_view_evaluation_exp21D
from src.experiment_21D.fusion_experiment_21D import run_fusion_evaluation_exp21D
from src.experiment_21D.summarize_experiment_21D import run_summary_exp21D
from src.experiment_21D.train_experiment_21D import train_side_exp21D


def main():
    parser = argparse.ArgumentParser(description="End-to-end Experiment 21D side regularization pipeline.")
    parser.add_argument("--skip-train", action="store_true", help="Skip side training and use existing Exp 21D checkpoint.")
    args = parser.parse_args()

    if not args.skip_train:
        train_side_exp21D()

    run_single_view_evaluation_exp21D()
    run_fusion_evaluation_exp21D()
    run_summary_exp21D()


if __name__ == "__main__":
    main()
