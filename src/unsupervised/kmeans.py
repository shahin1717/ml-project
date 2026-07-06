# K-Means – placeholder stub
import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        initial_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        self.centroids = X[initial_indices].copy()

        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            if np.allclose(new_centroids, self.centroids):
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

        self.labels = self._assign_clusters(X)

    def _assign_clusters(self, X):
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids
        ])
        return np.argmin(distances, axis=0)
    
    def predict(self, X):
        return self._assign_clusters(X)