import argparse

from src.experiment_21C.evaluate_experiment_21C import run_single_view_evaluation_exp21C
from src.experiment_21C.fusion_experiment_21C import run_fusion_evaluation_exp21C
from src.experiment_21C.summarize_experiment_21C import run_summary_exp21C
from src.experiment_21C.train_experiment_21C import train_side_exp21C


def main():
    parser = argparse.ArgumentParser(description="End-to-end Experiment 21C side-stabilization pipeline.")
    parser.add_argument("--skip-train", action="store_true", help="Skip side training and use existing Exp 21C checkpoint.")
    args = parser.parse_args()

    if not args.skip_train:
        train_side_exp21C()

    run_single_view_evaluation_exp21C()
    run_fusion_evaluation_exp21C()
    run_summary_exp21C()


if __name__ == "__main__":
    main()
