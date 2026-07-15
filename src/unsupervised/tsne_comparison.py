"""
PCA vs t-SNE comparison on the MNIST dataset (2-class subset, 784-dim -> 2D).

Run:
    PYTHONPATH=. python src/unsupervised/tsne_comparison.py
Outputs:
    figures/pca_vs_tsne.png  -- side-by-side scatter plots
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from src.unsupervised.pca import PCA
from sklearn.manifold import TSNE  # type: ignore
from src.utils.preprocessing import load_mnist_subset, standardize

def run_tsne_comparison(n_samples: int = 2000, save_plot: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compare PCA and t-SNE on a subset of MNIST."""
    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    # Load 2-class MNIST subset
    X, y = load_mnist_subset(n_samples=n_samples, classes=("3", "8"), random_state=42)

    # Standardize features
    X_scaled, _ = standardize(X, X)

    # ---------------------------------------------------------------
    # 2. PCA: linear, deterministic, fast, preserves global variance
    # ---------------------------------------------------------------
    t0 = time.time()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    t_pca = time.time() - t0

    assert pca.explained_variance_ratio_ is not None
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA: {t_pca:.3f}s | explained variance (2 comps): {explained_var:.2%}")

    # ---------------------------------------------------------------
    # 3. t-SNE: non-linear, stochastic, preserves local neighborhoods
    # ---------------------------------------------------------------
    t0 = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, n_samples - 1),
        learning_rate="auto",
        init="pca",          
        random_state=42,
        max_iter=1000,
    )
    X_tsne = tsne.fit_transform(X_scaled)
    t_tsne = time.time() - t0

    print(f"t-SNE: {t_tsne:.3f}s | KL divergence: {tsne.kl_divergence_:.3f}")

    # ---------------------------------------------------------------
    # 4. Plot side by side
    # ---------------------------------------------------------------
    if save_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, X_2d, title in zip(
            axes,
            [X_pca, X_tsne],
            [f"PCA (explains {explained_var:.1%} variance)", f"t-SNE (perplexity={min(30, n_samples - 1)})"],
        ):
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm", s=15, alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        fig.colorbar(scatter, ax=axes, label="Digit class (0=3, 1=8)", ticks=[0, 1])
        os.makedirs("figures", exist_ok=True)
        plt.savefig("figures/pca_vs_tsne.png", dpi=150, bbox_inches="tight")
        print("Saved figures/pca_vs_tsne.png")

    return X_pca, X_tsne


if __name__ == "__main__":
    run_tsne_comparison()