import argparse

from src.final_v2.evaluate_single_view_final_v2 import run_single_view_evaluation
from src.final_v2.fusion_final_v2 import run_fusion_evaluation
from src.final_v2.summarize_final_v2 import run_summary
from src.final_v2.train_single_view_final_v2 import train_front_final_v2, train_side_final_v2


def main():
    parser = argparse.ArgumentParser(description="End-to-end final_v2 retrain pipeline.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and use existing final_v2 checkpoints.")
    parser.add_argument("--view", choices=["front", "side"], default=None, help="Train only one view if needed.")
    args = parser.parse_args()

    if not args.skip_train:
        if args.view == "front":
            train_front_final_v2()
        elif args.view == "side":
            train_side_final_v2()
        else:
            train_front_final_v2()
            train_side_final_v2()

    run_single_view_evaluation()
    run_fusion_evaluation()
    run_summary()


if __name__ == "__main__":
    main()
