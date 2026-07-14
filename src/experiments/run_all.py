from __future__ import annotations
import argparse
from src.experiments.scaling import run_experiment_1
from src.experiments.adaboost_scaling import run_experiment_2
from src.experiments.rf_scaling import run_experiment_3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML experiments for final project.")
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        help="Experiments to run: 1, 2, 3, or all (default: all)",
    )

    args = parser.parse_args()

    if args.exp in ("1", "all"):
        run_experiment_1()

    if args.exp in ("2", "all"):
        run_experiment_2()

    if args.exp in ("3", "all"):
        run_experiment_3()


if __name__ == "__main__":
    main()
