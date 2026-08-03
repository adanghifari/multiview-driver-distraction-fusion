import argparse

from src.experiment_21E.evaluate_experiment_21E import run_single_view_evaluation_exp21E
from src.experiment_21E.fusion_experiment_21E import run_fusion_evaluation_exp21E
from src.experiment_21E.summarize_experiment_21E import run_summary_exp21E
from src.experiment_21E.train_experiment_21E import train_side_exp21E


def main():
    parser = argparse.ArgumentParser(description="End-to-end Experiment 21E light side regularization pipeline.")
    parser.add_argument("--skip-train", action="store_true", help="Skip side training and use existing Exp 21E checkpoint.")
    args = parser.parse_args()

    if not args.skip_train:
        train_side_exp21E()

    run_single_view_evaluation_exp21E()
    run_fusion_evaluation_exp21E()
    run_summary_exp21E()


if __name__ == "__main__":
    main()
