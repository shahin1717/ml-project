"""Experiment 5: Noise robustness under label corruption.

Self-contained: label-flipping and plotting helpers are defined locally in
this file — no changes required elsewhere in the repo.

For each dataset:
  - Randomly flip a fraction eta in {0.05, 0.10, 0.20} of TRAINING labels
    (eta = 0.0 is also included as the clean baseline).
  - For each eta, train AdaBoost (100 stumps) and Random Forest (100 trees)
    on the corrupted training data.
  - Evaluate both models on the untouched, clean test set.
  - Plot accuracy-degradation curves (accuracy vs. eta) for both models and
    print a short automatic comparison of which model degrades faster.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from src.boosting.adaboost import AdaBoostClassifier
from src.bagging.random_forest import RandomForestClassifier
from src.metrics.evaluation import accuracy_score
from experiments.utils import get_dataset

N_ESTIMATORS = 100
RANDOM_STATE = 42
ETAS = [0.0, 0.05, 0.10, 0.20]


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _flip_labels(
    y: np.ndarray,
    eta: float,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Randomly corrupt a fraction eta of labels, reassigning each flipped
    label uniformly at random among the other classes."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).copy()
    n_samples = len(y)
    classes = np.unique(y)

    n_flip = int(round(eta * n_samples))
    if n_flip == 0:
        return y

    flip_idx = rng.choice(n_samples, size=n_flip, replace=False)
    for i in flip_idx:
        other_classes = classes[classes != y[i]]
        y[i] = rng.choice(other_classes)

    return y


def _plot_degradation_curves(
    etas: list[float],
    curves: dict[str, list[float]],
    ylabel: str,
    title: str,
    save_filename: str,
) -> None:
    """Plot accuracy-vs-noise-level degradation curves for multiple models."""
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(8, 5.5))
    markers = ["o", "s", "^", "D", "v"]
    for i, (label, vals) in enumerate(curves.items()):
        plt.plot(etas, vals, label=label, marker=markers[i % len(markers)], linewidth=2)

    plt.xlabel("Label noise fraction (η)", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join("figures", save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment_5() -> None:
    """Run Experiment 5: noise robustness of AdaBoost vs Random Forest."""
    datasets = ["breast_cancer", "adult", "mnist", "covertype"]
    print("================================================================================")
    print("Experiment 5: Noise Robustness (label flipping)")
    print("================================================================================")

    for name in datasets:
        print(f"\nEvaluating dataset: {name.upper()}")
        X_train, X_test, y_train, y_test = get_dataset(name, random_state=RANDOM_STATE)

        adaboost_accs: list[float] = []
        rf_accs: list[float] = []

        print(f"{'eta':<8} | {'AdaBoost acc.':<16} | {'Random Forest acc.':<20}")
        print("-" * 50)

        for eta in ETAS:
            y_train_noisy = _flip_labels(y_train, eta=eta, random_state=RANDOM_STATE)

            ada = AdaBoostClassifier(
                n_estimators=N_ESTIMATORS, criterion="gini", random_state=RANDOM_STATE
            )
            ada.fit(X_train, y_train_noisy)
            ada_pred = ada.predict(X_test)
            ada_acc = accuracy_score(y_test, ada_pred)

            rf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, n_jobs=4, random_state=RANDOM_STATE
            )
            rf.fit(X_train, y_train_noisy)
            rf_pred = rf.predict(X_test)
            rf_acc = accuracy_score(y_test, rf_pred)

            adaboost_accs.append(ada_acc)
            rf_accs.append(rf_acc)

            print(f"{eta:<8.2f} | {ada_acc:<16.4f} | {rf_acc:<20.4f}  (AdaBoost fitted: {len(ada.estimators_)} estimators)")

        # ------------------------------------------------------------------
        # Plot accuracy degradation curves
        # ------------------------------------------------------------------
        _plot_degradation_curves(
            etas=ETAS,
            curves={"AdaBoost": adaboost_accs, "Random Forest": rf_accs},
            ylabel="Test Accuracy (clean test set)",
            title=f"Noise Robustness on {name.upper()}",
            save_filename=f"noise_robustness_{name}.png",
        )
        print(f"Saved plot to figures/noise_robustness_{name}.png")

        # ------------------------------------------------------------------
        # Quantify sensitivity: accuracy drop from clean (eta=0) to eta=0.20
        # ------------------------------------------------------------------
        ada_drop = adaboost_accs[0] - adaboost_accs[-1]
        rf_drop = rf_accs[0] - rf_accs[-1]
        more_sensitive = "AdaBoost" if ada_drop > rf_drop else "Random Forest"
        print(
            f"Accuracy drop (eta=0 -> eta={ETAS[-1]:.2f}): "
            f"AdaBoost={ada_drop:.4f}, Random Forest={rf_drop:.4f} "
            f"=> {more_sensitive} degrades more on {name.upper()}."
        )

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    run_experiment_5()