from __future__ import annotations
import numpy as np


class PCA:
    """
    Principal Component Analysis via eigen-decomposition of the covariance
    matrix.

    Attributes
    ----------
    n_components : int
        Number of principal components to retain.
    components_ : np.ndarray, shape (n_components, n_features)
        Top eigenvectors of the covariance matrix, sorted by descending
        eigenvalue.
    explained_variance_ratio_ : np.ndarray, shape (n_components,)
        Fraction of total variance explained by each retained component.
    """

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} cannot exceed "
                f"n_features={n_features}"
            )

        # Center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix (features x features)
        cov_matrix = np.atleast_2d(np.cov(X_centered, rowvar=False, ddof=1))

        # Eigen-decomposition (eigh: cov_matrix is symmetric)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # eigh returns ascending order -- flip to descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Guard against tiny negative eigenvalues from floating point error
        eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)

        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]

        total_variance = eigenvalues.sum()
        if total_variance <= 0:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        else:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call fit first.")
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)