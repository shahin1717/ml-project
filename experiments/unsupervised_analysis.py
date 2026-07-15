# Experiment 7: Unsupervised analysis (PCA, K-Means, DBSCAN) – placeholder stub
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score  # type: ignore

from src.utils.preprocessing import load_breast_cancer, standardize
from src.unsupervised.pca import PCA
from src.unsupervised.kmeans import KMeans
from src.unsupervised.dbscan import DBSCAN


def scree_plot(X, max_components=10):
    pca = PCA(n_components=max_components)
    pca.fit(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_components + 1), cumulative, marker="o")
    plt.axhline(0.9, color="red", linestyle="--", label="90% variance")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Scree Plot")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/scree_plot.png")
    plt.close()

    n_for_90 = np.argmax(cumulative >= 0.9) + 1
    print(f"Components needed for >=90% variance: {n_for_90}")
    return n_for_90


def elbow_method(X, k_range=range(1, 11), n_restarts=10, random_state=42):
    inertias = []
    for k in k_range:
        best_inertia = np.inf
        for r in range(n_restarts):
            km = KMeans(n_clusters=k, random_state=random_state + r)
            km.fit(X)
            if km.inertia_ < best_inertia:
                best_inertia = km.inertia_
        inertias.append(best_inertia)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.savefig("figures/elbow_method.png")
    plt.close()


def k_distance_plot(X, min_samples=5):
    from numpy.linalg import norm
    n = X.shape[0]
    k_distances = np.zeros(n)
    for i in range(n):
        dists = norm(X - X[i], axis=1)
        sorted_dists = np.sort(dists)
        k_distances[i] = sorted_dists[min_samples]

    sorted_k_dist = np.sort(k_distances)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_k_dist)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_samples}-th nearest neighbor distance")
    plt.title("K-Distance Plot (for DBSCAN eps selection)")
    plt.grid(True)
    plt.savefig("figures/k_distance_plot.png")
    plt.close()


def pca_scatter_comparison(X, y_true, kmeans_labels, dbscan_labels):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="viridis")
    axes[0].set_title("True Class Labels")

    axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap="viridis")
    axes[1].set_title("K-Means Clusters")

    axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=dbscan_labels, cmap="viridis")
    axes[2].set_title("DBSCAN Clusters")

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig("figures/pca_scatter_comparison.png")
    plt.close()


def run_experiment_7():
    X, y = load_breast_cancer()
    X_scaled, _ = standardize(X, X)  # standardize whole dataset for unsupervised analysis

    print("=" * 55)
    print("1. Scree plot")
    print("=" * 55)
    scree_plot(X_scaled, max_components=10)

    print("=" * 55)
    print("2. Elbow method for K-Means")
    print("=" * 55)
    elbow_method(X_scaled)

    best_k = 2  # Breast cancer is binary; also confirmed via elbow
    km = KMeans(n_clusters=best_k, random_state=42)
    km.fit(X_scaled)
    ari_kmeans = adjusted_rand_score(y, km.labels_)
    print(f"K-Means (k={best_k}) ARI: {ari_kmeans:.4f}")

    print("=" * 55)
    print("3. K-distance plot for DBSCAN")
    print("=" * 55)
    k_distance_plot(X_scaled, min_samples=5)

    eps_choice = 3.5  # to be tuned after inspecting the k-distance plot
    db = DBSCAN(eps=eps_choice, min_samples=5)
    db.fit(X_scaled)
    ari_dbscan = adjusted_rand_score(y, db.labels_)
    noise_fraction = np.mean(db.labels_ == -1)
    print(f"DBSCAN (eps={eps_choice}) ARI: {ari_dbscan:.4f}, noise fraction: {noise_fraction:.4f}")

    print("=" * 55)
    print("4. PCA scatter comparison")
    print("=" * 55)
    pca_scatter_comparison(X_scaled, y, km.labels_, db.labels_)

    return {
        "ari_kmeans": ari_kmeans,
        "ari_dbscan": ari_dbscan,
        "noise_fraction": noise_fraction,
    }


if __name__ == "__main__":
    run_experiment_7()