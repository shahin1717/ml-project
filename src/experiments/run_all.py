from __future__ import annotations
import argparse
import os
import sys

# Ensure project root is in the Python search path to import from the root-level experiments directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.scaling import run_experiment_1  # noqa: E402
from experiments.adaboost_scaling import run_experiment_2  # noqa: E402
from experiments.rf_scaling import run_experiment_3  # noqa: E402


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
