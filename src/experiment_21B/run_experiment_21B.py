import argparse

from src.experiment_21B.evaluate_experiment_21B import run_single_view_evaluation_exp21B
from src.experiment_21B.fusion_experiment_21B import run_fusion_evaluation_exp21B
from src.experiment_21B.summarize_experiment_21B import run_summary_exp21B
from src.experiment_21B.train_experiment_21B import train_front_exp21B


def main():
    parser = argparse.ArgumentParser(description="End-to-end Experiment 21B front-stabilization pipeline.")
    parser.add_argument("--skip-train", action="store_true", help="Skip front training and use existing Exp 21B checkpoint.")
    args = parser.parse_args()

    if not args.skip_train:
        train_front_exp21B()

    run_single_view_evaluation_exp21B()
    run_fusion_evaluation_exp21B()
    run_summary_exp21B()


if __name__ == "__main__":
    main()
