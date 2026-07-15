import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from experiments.utils import get_dataset
from src.metrics.evaluation import accuracy_score, precision_recall_f1
from src.boosting.adaboost import AdaBoostClassifier
from src.boosting.gradient_boosting import GradientBoostingClassifier


def run_gbm_vs_adaboost_experiment():
    datasets = ["breast_cancer", "adult", "mnist"]
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}

    for name in datasets:
        print(f"\nEvaluating models on dataset: {name} ...")
        X_train, X_test, y_train, y_test = get_dataset(name)
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        if name in ["adult", "mnist"]:
            # Downsample to 1000 samples for fast cross-validation
            n_sub = min(1000, len(X))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=n_sub, replace=False)
            X, y = X[idx], y[idx]

        if name == "mnist":
            # Project high-dimensional MNIST to 30 components for fast execution
            from src.unsupervised.pca import PCA
            pca = PCA(n_components=30)
            X = pca.fit_transform(X)

        results[name] = {
            "AdaBoost": {"acc": [], "f1": [], "auc": []},
            "GBM": {"acc": [], "f1": [], "auc": []},
        }

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # 1. AdaBoost
            ada = AdaBoostClassifier(n_estimators=100, random_state=42 + fold)
            ada.fit(X_tr, y_tr)
            ada_preds = ada.predict(X_te)
            ada_probs = ada.predict_proba(X_te)[:, 1]

            _, _, ada_f1 = precision_recall_f1(y_te, ada_preds, average="macro")
            results[name]["AdaBoost"]["acc"].append(accuracy_score(y_te, ada_preds))
            results[name]["AdaBoost"]["f1"].append(ada_f1)
            results[name]["AdaBoost"]["auc"].append(roc_auc_score(y_te, ada_probs))

            # 2. Gradient Boosting (GBM)
            gbm = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42 + fold
            )
            gbm.fit(X_tr, y_tr)
            gbm_preds = gbm.predict(X_te)
            gbm_probs = gbm.predict_proba(X_te)[:, 1]

            _, _, gbm_f1 = precision_recall_f1(y_te, gbm_preds, average="macro")
            results[name]["GBM"]["acc"].append(accuracy_score(y_te, gbm_preds))
            results[name]["GBM"]["f1"].append(gbm_f1)
            results[name]["GBM"]["auc"].append(roc_auc_score(y_te, gbm_probs))

        # Summarize results
        print(f"Results for {name}:")
        for clf_name in ["AdaBoost", "GBM"]:
            metrics = results[name][clf_name]
            acc_mean, acc_std = np.mean(metrics["acc"]), np.std(metrics["acc"])
            f1_mean, f1_std = np.mean(metrics["f1"]), np.std(metrics["f1"])
            auc_mean, auc_std = np.mean(metrics["auc"]), np.std(metrics["auc"])

            print(
                f"  {clf_name:<10}: "
                f"Accuracy = {acc_mean:.4f} ± {acc_std:.4f} | "
                f"Macro-F1 = {f1_mean:.4f} ± {f1_std:.4f} | "
                f"AUC-ROC  = {auc_mean:.4f} ± {auc_std:.4f}"
            )

        # Plot comparison figure
        plot_comparison(name, results[name])

    return results


def plot_comparison(dataset_name, dataset_results):
    os.makedirs("figures", exist_ok=True)

    metrics = ["acc", "f1", "auc"]
    metric_names = ["Accuracy", "Macro F1", "AUC-ROC"]

    ada_means = [np.mean(dataset_results["AdaBoost"][m]) for m in metrics]
    ada_stds = [np.std(dataset_results["AdaBoost"][m]) for m in metrics]

    gbm_means = [np.mean(dataset_results["GBM"][m]) for m in metrics]
    gbm_stds = [np.std(dataset_results["GBM"][m]) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(
        x - width/2,
        ada_means,
        width,
        yerr=ada_stds,
        capsize=5,
        label="AdaBoost (Stumps)",
        color="skyblue",
        edgecolor="black",
    )
    plt.bar(
        x + width/2,
        gbm_means,
        width,
        yerr=gbm_stds,
        capsize=5,
        label="GBM (Trees)",
        color="salmon",
        edgecolor="black",
    )

    plt.xticks(x, metric_names)
    plt.ylabel("Score")
    plt.title(f"AdaBoost vs GBM Comparison on {dataset_name.replace('_', ' ').title()}")
    plt.ylim(0.5, 1.02)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join("figures", f"gbm_vs_adaboost_{dataset_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {save_path}")


if __name__ == "__main__":
    run_gbm_vs_adaboost_experiment()
