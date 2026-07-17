"""Experiment 4: Head-to-head comparison (fixed resources, 5-fold CV).

Reuses the existing experiments/utils.py `get_dataset` loader (same one
used by scaling.py, adaboost_scaling.py, rf_scaling.py) instead of
re-implementing dataset loading. Only the CV-split and box-plot helpers,
which don't already exist in utils.py, are defined locally in this file.

For each dataset:
  - n_estimators = 100 for both ensembles (AdaBoost, Random Forest).
  - Stratified 5-fold cross-validation over the full (standardized) dataset.
    For each fold, each model is fit on the train fold and evaluated on the
    held-out fold: accuracy, macro F1, AUC-ROC.
  - Report mean +/- std across the 5 folds for:
      * single tree (custom DecisionTree, unpruned)
      * AdaBoost (custom, 100 stumps)
      * Random Forest (custom, 100 trees)
      * sklearn.ensemble.RandomForestClassifier (100 trees, reference)
  - Results are shown as a table and as box plots (accuracy / macro F1 / AUC-ROC).
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier  # type: ignore

from src.trees.decision_tree import DecisionTree
from src.boosting.adaboost import AdaBoostClassifier
from src.bagging.random_forest import RandomForestClassifier
from src.metrics.evaluation import accuracy_score, precision_recall_f1
from experiments.utils import get_dataset, safe_auc

N_ESTIMATORS = 100
N_FOLDS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Local helpers (not already present in experiments/utils.py)
# ---------------------------------------------------------------------------

def _stratified_kfold_indices(
    y: np.ndarray,
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified K-fold (train_idx, test_idx) index pairs."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    classes = np.unique(y)

    per_class_folds: list[list[np.ndarray]] = []
    for c in classes:
        idx_c = rng.permutation(np.where(y == c)[0])
        per_class_folds.append(np.array_split(idx_c, n_splits))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        test_idx = np.concatenate([per_class_folds[ci][k] for ci in range(len(classes))])
        train_idx = np.concatenate(
            [
                per_class_folds[ci][j]
                for ci in range(len(classes))
                for j in range(n_splits)
                if j != k
            ]
        )
        rng.shuffle(test_idx)
        rng.shuffle(train_idx)
        splits.append((train_idx, test_idx))

    return splits


def _plot_boxplot_comparison(
    data: dict[str, list[float]],
    ylabel: str,
    title: str,
    save_filename: str,
) -> None:
    """Draw a box plot comparing per-fold metric distributions across models."""
    os.makedirs("figures", exist_ok=True)

    labels = list(data.keys())
    values = [data[k] for k in labels]

    plt.figure(figsize=(8, 5.5))
    plt.boxplot(values, tick_labels=labels, showmeans=True)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    save_path = os.path.join("figures", save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _make_models(random_state: int) -> dict:
    """Factory: build fresh, unfitted model instances (fixed resources)."""
    return {
        "Single Tree": DecisionTree(criterion="gini", random_state=random_state),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=N_ESTIMATORS, criterion="gini", random_state=random_state
        ),
        "Random Forest (custom)": RandomForestClassifier(
            n_estimators=N_ESTIMATORS, n_jobs=4, random_state=random_state
        ),
        "Random Forest (sklearn)": SklearnRandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=random_state
        ),
    }


def _evaluate_fold(model, X_train, y_train, X_test, y_test) -> tuple[float, float, float]:
    """Fit model on a fold, return (accuracy, macro_f1, auc_roc) on the held-out fold."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    _, _, f1 = precision_recall_f1(y_test, y_pred, average="macro")
    auc = safe_auc(y_test, proba)

    return acc, f1, auc


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment_4() -> None:
    """Run Experiment 4: head-to-head comparison across datasets via 5-fold CV."""
    datasets = ["breast_cancer", "adult", "mnist", "covertype"]
    print("================================================================================")
    print("Experiment 4: Head-to-Head Comparison (100 estimators, 5-fold CV)")
    print("================================================================================")

    for name in datasets:
        print(f"\nEvaluating dataset: {name.upper()}")

        # Reuse the existing get_dataset() loader (load + 80/20 split + standardize),
        # then recombine into one standardized pool so we can carve our own 5 folds.
        X_train, X_test, y_train, y_test = get_dataset(name, random_state=RANDOM_STATE)
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        folds = _stratified_kfold_indices(y, n_splits=N_FOLDS, random_state=RANDOM_STATE)

        # metric_results[model_name] = {"accuracy": [...5 folds...], "f1": [...], "auc": [...]}
        metric_results: dict[str, dict[str, list[float]]] = {
            model_name: {"accuracy": [], "f1": [], "auc": []}
            for model_name in _make_models(RANDOM_STATE)
        }

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_fold_train, X_fold_test = X[train_idx], X[test_idx]
            y_fold_train, y_fold_test = y[train_idx], y[test_idx]

            models = _make_models(RANDOM_STATE + fold_idx)
            for model_name, model in models.items():
                acc, f1, auc = _evaluate_fold(
                    model, X_fold_train, y_fold_train, X_fold_test, y_fold_test
                )
                metric_results[model_name]["accuracy"].append(acc)
                metric_results[model_name]["f1"].append(f1)
                metric_results[model_name]["auc"].append(auc)

        # ------------------------------------------------------------------
        # Print mean +/- std table
        # ------------------------------------------------------------------
        print(f"\n{'Model':<26} | {'Accuracy':<18} | {'Macro F1':<18} | {'AUC-ROC':<18}")
        print("-" * 90)
        for model_name, metrics in metric_results.items():
            acc_arr = np.array(metrics["accuracy"])
            f1_arr = np.array(metrics["f1"])
            auc_arr = np.array(metrics["auc"])
            print(
                f"{model_name:<26} | "
                f"{f'{acc_arr.mean():.4f} ± {acc_arr.std():.4f}':<18} | "
                f"{f'{f1_arr.mean():.4f} ± {f1_arr.std():.4f}':<18} | "
                f"{f'{auc_arr.mean():.4f} ± {auc_arr.std():.4f}':<18}"
            )

        # ------------------------------------------------------------------
        # Box plots (one figure per metric per dataset)
        # ------------------------------------------------------------------
        for metric_key, metric_label, fname_suffix in [
            ("accuracy", "Accuracy", "accuracy"),
            ("f1", "Macro F1", "macro_f1"),
            ("auc", "AUC-ROC", "auc_roc"),
        ]:
            plot_data = {
                model_name: metrics[metric_key] for model_name, metrics in metric_results.items()
            }
            _plot_boxplot_comparison(
                data=plot_data,
                ylabel=metric_label,
                title=f"Head-to-Head {metric_label} (5-fold CV) on {name.upper()}",
                save_filename=f"head_to_head_{fname_suffix}_{name}.png",
            )
        print(f"Saved box plots to figures/head_to_head_{{accuracy,macro_f1,auc_roc}}_{name}.png")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    run_experiment_4()