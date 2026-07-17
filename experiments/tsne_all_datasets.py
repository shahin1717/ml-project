"""
Script to generate t-SNE plots for Breast Cancer, Adult Income, and MNIST datasets.
Saves plots under the figures/ directory.

Run:
    PYTHONPATH=. python experiments/tsne_all_datasets.py
"""

from __future__ import annotations
import os
import time
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from experiments.utils import get_dataset

def main() -> None:
    datasets = ["breast_cancer", "adult", "mnist"]
    n_samples_limit = 2000
    random_seed = 42

    os.makedirs("figures", exist_ok=True)
    
    # We will also create a combined plot with a 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    dataset_labels = {
        "breast_cancer": {
            "title": "Breast Cancer Wisconsin (t-SNE)",
            "classes": {0: "Benign", 1: "Malignant"},
            "cmap": "coolwarm"
        },
        "adult": {
            "title": "Adult Income (t-SNE, N=2000)",
            "classes": {0: "<=50K", 1: ">50K"},
            "cmap": "coolwarm"
        },
        "mnist": {
            "title": "MNIST 3 vs 8 (t-SNE, N=2000)",
            "classes": {0: "Digit 3", 1: "Digit 8"},
            "cmap": "coolwarm"
        }
    }

    for idx, name in enumerate(datasets):
        print(f"\nProcessing dataset: {name.upper()}")
        t0 = time.time()
        
        # 1. Load data
        X_train, X_test, y_train, y_test = get_dataset(name, random_state=random_seed)
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        
        # 2. Subsample if dataset is too large (t-SNE is O(N^2))
        if X.shape[0] > n_samples_limit:
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(X.shape[0], size=n_samples_limit, replace=False)
            X = X[indices]
            y = y[indices]
            print(f"Subsampled to {n_samples_limit} rows for performance.")

        # 3. Fit t-SNE
        print("Running t-SNE with perplexity=30...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, X.shape[0] - 1),
            learning_rate="auto",
            init="pca",
            random_state=random_seed,
            max_iter=1000
        )
        X_tsne = tsne.fit_transform(X)
        print(f"t-SNE completed in {time.time() - t0:.2f}s | Final KL divergence: {tsne.kl_divergence_:.3f}")

        # 4. Draw individual plot
        plt.figure(figsize=(8, 6.5))
        
        info = dataset_labels[name]
        # Plot class 0
        idx_0 = (y == 0)
        plt.scatter(
            X_tsne[idx_0, 0], X_tsne[idx_0, 1], 
            color="#3182bd", label=info["classes"][0], 
            s=20, alpha=0.7, edgecolors="none"
        )
        # Plot class 1
        idx_1 = (y == 1)
        plt.scatter(
            X_tsne[idx_1, 0], X_tsne[idx_1, 1], 
            color="#de2d26", label=info["classes"][1], 
            s=20, alpha=0.7, edgecolors="none"
        )
        
        plt.title(info["title"], fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("t-SNE Dimension 1", fontsize=11)
        plt.ylabel("t-SNE Dimension 2", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Class Labels", fontsize=10, loc="best")
        
        plt.savefig(f"figures/tsne_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figures/tsne_{name}.png")

        # 5. Add to combined plot
        ax = axes[idx]
        ax.scatter(
            X_tsne[idx_0, 0], X_tsne[idx_0, 1], 
            color="#3182bd", label=info["classes"][0], 
            s=12, alpha=0.6, edgecolors="none"
        )
        ax.scatter(
            X_tsne[idx_1, 0], X_tsne[idx_1, 1], 
            color="#de2d26", label=info["classes"][1], 
            s=12, alpha=0.6, edgecolors="none"
        )
        ax.set_title(info["title"], fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE Dim 1", fontsize=10)
        ax.set_ylabel("t-SNE Dim 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig("figures/tsne_all_datasets.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\nSaved combined plot: figures/tsne_all_datasets.png")

if __name__ == "__main__":
    main()
