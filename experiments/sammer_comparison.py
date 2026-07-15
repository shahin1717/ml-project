import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # type: ignore
from experiments.utils import standardize
from src.utils.preprocessing import train_test_split
from src.boosting.adaboost import AdaBoostClassifier
from src.metrics.evaluation import accuracy_score


def brier_score(y_true, y_prob):
    """Compute the multi-class Brier score."""
    y_onehot = np.zeros_like(y_prob)
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1.0
    return np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1))


def log_loss(y_true, y_prob):
    """Compute the multi-class log loss."""
    prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    y_onehot = np.zeros_like(prob_clipped)
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1.0
    return -np.mean(np.sum(y_onehot * np.log(prob_clipped), axis=1))


def run_sammer_comparison():
    print("Loading Iris dataset...")
    data = load_iris()
    X, y = data.data, data.target

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    X_train_s, X_test_s = standardize(X_train, X_test)

    n_estimators = 80
    print(f"Training SAMME and SAMME.R models with n_estimators={n_estimators}...")

    # Train SAMME
    clf_samme = AdaBoostClassifier(
        n_estimators=n_estimators, algorithm="SAMME", random_state=42
    )
    clf_samme.fit(X_train_s, y_train)

    # Train SAMME.R
    clf_sammer = AdaBoostClassifier(
        n_estimators=n_estimators, algorithm="SAMME.R", random_state=42
    )
    clf_sammer.fit(X_train_s, y_train)

    # Record staged metrics
    samme_train_acc = []
    samme_test_acc = []
    samme_test_brier = []
    samme_test_loss = []

    sammer_train_acc = []
    sammer_test_acc = []
    sammer_test_brier = []
    sammer_test_loss = []

    # SAMME staged evaluation
    for y_pred_train, y_prob_test in zip(
        clf_samme.staged_predict(X_train_s),
        clf_samme.staged_predict_proba(X_test_s)
    ):
        samme_train_acc.append(accuracy_score(y_train, y_pred_train))
        y_pred_test = clf_samme.classes_[np.argmax(y_prob_test, axis=1)]
        samme_test_acc.append(accuracy_score(y_test, y_pred_test))
        samme_test_brier.append(brier_score(y_test, y_prob_test))
        samme_test_loss.append(log_loss(y_test, y_prob_test))

    # SAMME.R staged evaluation
    for y_pred_train, y_prob_test in zip(
        clf_sammer.staged_predict(X_train_s),
        clf_sammer.staged_predict_proba(X_test_s)
    ):
        sammer_train_acc.append(accuracy_score(y_train, y_pred_train))
        y_pred_test = clf_sammer.classes_[np.argmax(y_prob_test, axis=1)]
        sammer_test_acc.append(accuracy_score(y_test, y_pred_test))
        sammer_test_brier.append(brier_score(y_test, y_prob_test))
        sammer_test_loss.append(log_loss(y_test, y_prob_test))

    # Summarize final results
    print("\n" + "=" * 50)
    print(f"Final Results (at {n_estimators} rounds):")
    print(f"SAMME (Discrete):")
    print(f"  Train Accuracy: {samme_train_acc[-1]:.4f}")
    print(f"  Test Accuracy : {samme_test_acc[-1]:.4f}")
    print(f"  Test Brier    : {samme_test_brier[-1]:.4f}")
    print(f"  Test Log-Loss : {samme_test_loss[-1]:.4f}")
    print("-" * 50)
    print(f"SAMME.R (Real):")
    print(f"  Train Accuracy: {sammer_train_acc[-1]:.4f}")
    print(f"  Test Accuracy : {sammer_test_acc[-1]:.4f}")
    print(f"  Test Brier    : {sammer_test_brier[-1]:.4f}")
    print(f"  Test Log-Loss : {sammer_test_loss[-1]:.4f}")
    print("=" * 50 + "\n")

    # Plot results
    os.makedirs("figures", exist_ok=True)
    rounds = np.arange(1, len(samme_test_acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Accuracy subplot
    axes[0].plot(rounds, samme_train_acc, label="SAMME (Train)", color="#1f77b4", linestyle="--", alpha=0.7)
    axes[0].plot(rounds, samme_test_acc, label="SAMME (Test)", color="#1f77b4", linewidth=2.5)
    axes[0].plot(rounds, sammer_train_acc, label="SAMME.R (Train)", color="#ff7f0e", linestyle="--", alpha=0.7)
    axes[0].plot(rounds, sammer_test_acc, label="SAMME.R (Test)", color="#ff7f0e", linewidth=2.5)
    axes[0].set_xlabel("Boosting Rounds", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_title("Training & Testing Accuracy", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Brier score (Calibration) subplot
    axes[1].plot(rounds, samme_test_brier, label="SAMME (Discrete)", color="#1f77b4", linewidth=2.5)
    axes[1].plot(rounds, sammer_test_brier, label="SAMME.R (Real)", color="#ff7f0e", linewidth=2.5)
    axes[1].set_xlabel("Boosting Rounds", fontsize=11)
    axes[1].set_ylabel("Brier Score (Lower is Better)", fontsize=11)
    axes[1].set_title("Test Probability Calibration (Brier Score)", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("AdaBoost Multiclass Comparison: SAMME vs SAMME.R", fontsize=15, fontweight="bold")
    plt.tight_layout()

    save_path = "figures/sammer_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved successfully to {save_path}")


if __name__ == "__main__":
    run_sammer_comparison()
