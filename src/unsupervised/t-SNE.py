"""
PCA vs t-SNE comparison on the sklearn digits dataset (8x8 handwritten digits, 64-dim -> 2D).

Run:
    python pca_vs_tsne.py
Outputs:
    pca_vs_tsne.png  -- side-by-side scatter plots
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
digits = load_digits()
X, y = digits.data, digits.target  # X: (1797, 64), y: (1797,) labels 0-9

# Standardize features ;
X_scaled = StandardScaler().fit_transform(X)

# ---------------------------------------------------------------
# 2. PCA: linear, deterministic, fast, preserves global variance
# ---------------------------------------------------------------
t0 = time.time()
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
t_pca = time.time() - t0

explained_var = pca.explained_variance_ratio_.sum()
print(f"PCA: {t_pca:.3f}s | explained variance (2 comps): {explained_var:.2%}")

# ---------------------------------------------------------------
# 3. t-SNE: non-linear, stochastic, preserves local neighborhoods
# ---------------------------------------------------------------
t0 = time.time()
tsne = TSNE(
    n_components=2,
    perplexity=30,
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
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, X_2d, title in zip(
    axes,
    [X_pca, X_tsne],
    [f"PCA (explains {explained_var:.1%} variance)", "t-SNE (perplexity=30)"],
):
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=15, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

fig.colorbar(scatter, ax=axes, label="Digit class", ticks=range(10))
plt.savefig("pca_vs_tsne.png", dpi=150, bbox_inches="tight")
print("Saved pca_vs_tsne.png")