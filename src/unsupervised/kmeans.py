from __future__ import annotations
import numpy as np


class KMeans:
    """K-Means clustering from scratch.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference in
        the cluster centers of two consecutive iterations to declare convergence.
    random_state : int or None, default=None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
    centroids_ : np.ndarray, shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : np.ndarray, shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = 0.0

    def fit(self, X: np.ndarray) -> KMeans:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
        n_samples, n_features = X.shape

        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} cannot exceed n_samples={n_samples}."
            )

        rng = np.random.default_rng(self.random_state)

        # Initialize centroids randomly choosing from X
        initial_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        self.centroids_ = X[initial_indices].copy()

        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids_[k]
                for k in range(self.n_clusters)
            ])

            # Check convergence using relative tolerance
            centroid_shift = np.linalg.norm(new_centroids - self.centroids_)
            if centroid_shift < self.tol:
                self.centroids_ = new_centroids
                break

            self.centroids_ = new_centroids

        self.labels_ = self._assign_clusters(X)

        # Compute inertia_: sum of squared distances to closest centroid
        distances = self._compute_distances(X)
        min_distances = np.min(distances, axis=1)
        self.inertia_ = float(np.sum(min_distances ** 2))

        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("KMeans instance is not fitted yet.")
        # Returns array of shape (n_samples, n_clusters)
        return np.array([
            np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids_
        ]).T

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
        return self._assign_clusters(X)