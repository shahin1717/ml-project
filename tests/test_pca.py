import numpy as np
import pytest
from src.unsupervised.pca import PCA


def test_pca_fitting():
    # Simple synthetic dataset: 10 samples, 4 features
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 4))

    pca = PCA(n_components=2)
    pca.fit(X)

    # Check shapes
    assert pca.components_ is not None
    assert pca.components_.shape == (2, 4)
    assert pca.explained_variance_ is not None
    assert pca.explained_variance_.shape == (2,)
    assert pca.explained_variance_ratio_ is not None
    assert pca.explained_variance_ratio_.shape == (2,)
    assert pca.mean_ is not None
    assert pca.mean_.shape == (4,)

    # Check components are orthonormal
    # components_ shape is (n_components, n_features)
    # self.components_ @ self.components_.T should be close to identity matrix
    identity_approx = pca.components_ @ pca.components_.T
    assert np.allclose(identity_approx, np.eye(2))

    # Check eigenvalues are in descending order
    assert pca.explained_variance_[0] >= pca.explained_variance_[1]

    # Check explained variance ratio is in [0, 1]
    assert np.all(pca.explained_variance_ratio_ >= 0)
    assert np.sum(pca.explained_variance_ratio_) <= 1.0


def test_pca_transform():
    # Simple dataset where first feature has all variance
    X = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0]
    ])

    pca = PCA(n_components=1)
    X_trans = pca.fit_transform(X)

    # Output shape should be (4, 1)
    assert X_trans.shape == (4, 1)
    # The first PC should capture all the variance, so the ratio should be 1.0
    assert pca.explained_variance_ratio_ is not None
    assert np.isclose(pca.explained_variance_ratio_[0], 1.0)


def test_pca_invalid_inputs():
    pca = PCA(n_components=3)

    # Check 1D array raises error
    with pytest.raises(ValueError):
        pca.fit(np.array([1, 2, 3]))

    # Check n_components > n_features raises error
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        pca.fit(X)

    # Check n_components < 1 raises error
    with pytest.raises(ValueError):
        PCA(n_components=0).fit(X)


def test_pca_not_fitted():
    pca = PCA(n_components=2)
    # Check transform before fit raises error
    with pytest.raises(RuntimeError):
        pca.transform(np.array([[1.0, 2.0], [3.0, 4.0]]))
