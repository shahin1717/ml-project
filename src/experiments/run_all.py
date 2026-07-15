from __future__ import annotations
import argparse
import os
import sys

# Ensure project root is in the Python search path to import from the root-level experiments directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all experiments and bonuses
from experiments.scaling import run_experiment_1  # type: ignore # noqa: E402
from experiments.adaboost_scaling import run_experiment_2  # type: ignore # noqa: E402
from experiments.rf_scaling import run_experiment_3  # type: ignore # noqa: E402
from experiments.head_to_head import run_experiment_4  # type: ignore # noqa: E402
from experiments.noise_robustness import run_experiment_5  # type: ignore # noqa: E402
from experiments.bias_variance import run_experiment_6  # type: ignore # noqa: E402
from experiments.unsupervised_analysis import run_experiment_7  # type: ignore # noqa: E402
from experiments.gbm_vs_adaboost import run_gbm_vs_adaboost_experiment  # type: ignore # noqa: E402
from experiments.sammer_comparison import run_sammer_comparison  # type: ignore # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML experiments for final project.")
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        help="Experiments to run: 1, 2, 3, 4, 5, 6, 7, gbm, sammer, or all (default: all)",
    )

    args = parser.parse_args()

    if args.exp in ("1", "all"):
        print("\n=== Running Experiment 1: Baseline ===")
        run_experiment_1()

    if args.exp in ("2", "all"):
        print("\n=== Running Experiment 2: AdaBoost Scaling ===")
        run_experiment_2()

    if args.exp in ("3", "all"):
        print("\n=== Running Experiment 3: Random Forest Scaling ===")
        run_experiment_3()

    if args.exp in ("4", "all"):
        print("\n=== Running Experiment 4: Head-to-Head Comparison ===")
        run_experiment_4()

    if args.exp in ("5", "all"):
        print("\n=== Running Experiment 5: Noise Robustness ===")
        run_experiment_5()

    if args.exp in ("6", "all"):
        print("\n=== Running Experiment 6: Bias-Variance Decomposition ===")
        run_experiment_6()

    if args.exp in ("7", "all"):
        print("\n=== Running Experiment 7: Unsupervised Analysis ===")
        run_experiment_7()

    if args.exp in ("gbm", "all"):
        print("\n=== Running Bonus: Gradient Boosting vs AdaBoost ===")
        run_gbm_vs_adaboost_experiment()

    if args.exp in ("sammer", "all"):
        print("\n=== Running Bonus: SAMME vs SAMME.R Calibration ===")
        run_sammer_comparison()


if __name__ == "__main__":
    main()
