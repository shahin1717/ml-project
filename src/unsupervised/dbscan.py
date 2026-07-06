# DBSCAN – placeholder stub
import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

        return self.labels

    def _region_query(self, X, idx):
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, idx, neighbors, cluster_id, visited):
        self.labels[idx] = cluster_id
        neighbors = list(neighbors)
        i = 0
        while i < len(neighbors):
            point = neighbors[i]
            if not visited[point]:
                visited[point] = True
                point_neighbors = self._region_query(X, point)
                if len(point_neighbors) >= self.min_samples:
                    neighbors.extend(list(point_neighbors))

            if self.labels[point] == -1:
                self.labels[point] = cluster_id

            i += 1