import argparse

from src.experiment_21.evaluate_experiment_21 import run_single_view_evaluation_exp21
from src.experiment_21.fusion_experiment_21 import run_fusion_evaluation_exp21
from src.experiment_21.summarize_experiment_21 import run_summary_exp21
from src.experiment_21.train_experiment_21 import train_front_exp21, train_side_exp21


def main():
    parser = argparse.ArgumentParser(description="End-to-end Experiment 21 pipeline.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and use existing Exp 21 checkpoints.")
    parser.add_argument("--view", choices=["front", "side"], default=None, help="Train only one view if needed.")
    args = parser.parse_args()

    if not args.skip_train:
        if args.view == "front":
            train_front_exp21()
        elif args.view == "side":
            train_side_exp21()
        else:
            train_front_exp21()
            train_side_exp21()

    run_single_view_evaluation_exp21()
    run_fusion_evaluation_exp21()
    run_summary_exp21()


if __name__ == "__main__":
    main()
