"""DBSCAN – Density-Based Spatial Clustering of Applications with Noise.

Implements density-based clustering from scratch using NumPy.
Follows the requirements in §2.4.3 of ML_FINAL_PROJECT.tex.
"""

from __future__ import annotations
import numpy as np


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps : float
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered
        as a core point. This includes the point itself.
    """

    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray = np.empty(0, dtype=int)

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """Perform DBSCAN clustering on X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")

        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = self._find_neighbors(i, X)
            if len(neighbors) < self.min_samples:
                # Marked as noise (remains -1)
                continue
            else:
                self.labels_[i] = cluster_id
                self._expand_cluster(i, neighbors, X, visited, cluster_id)
                cluster_id += 1

        return self

    def _find_neighbors(self, i: int, X: np.ndarray) -> np.ndarray:
        """Find indices of neighbors within eps of point i."""
        dists = np.linalg.norm(X - X[i], axis=1)
        return np.where(dists <= self.eps)[0]

    def _expand_cluster(
        self,
        core_idx: int,
        neighbors: np.ndarray,
        X: np.ndarray,
        visited: np.ndarray,
        cluster_id: int,
    ) -> None:
        """Expand cluster from core point using BFS queue."""
        queue = list(neighbors)
        queued = set(neighbors)

        idx = 0
        while idx < len(queue):
            q = queue[idx]
            idx += 1

            if not visited[q]:
                visited[q] = True
                q_neighbors = self._find_neighbors(q, X)
                if len(q_neighbors) >= self.min_samples:
                    for n in q_neighbors:
                        if n not in queued:
                            queue.append(n)
                            queued.add(n)

            if self.labels_[q] == -1:
                self.labels_[q] = cluster_id
