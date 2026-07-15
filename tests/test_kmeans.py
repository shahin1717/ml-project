import numpy as np
import pytest
from src.unsupervised.kmeans import KMeans


def test_kmeans_clustering():
    # Construct three distinct blobs
    blob1 = np.array([[0.0, 0.0], [0.1, 0.1], [0.0, 0.1], [0.1, 0.0]])
    blob2 = np.array([[10.0, 10.0], [10.1, 10.1], [10.0, 10.1], [10.1, 10.0]])
    blob3 = np.array([[-10.0, -10.0], [-10.1, -10.1], [-10.0, -10.1], [-10.1, -10.0]])

    X = np.vstack([blob1, blob2, blob3])

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Check labels shape
    assert kmeans.labels_ is not None
    assert kmeans.labels_.shape == (12,)

    # Labels for the three blobs should be grouped into different clusters
    c1 = kmeans.labels_[0:4]
    c2 = kmeans.labels_[4:8]
    c3 = kmeans.labels_[8:12]

    assert len(np.unique(c1)) == 1
    assert len(np.unique(c2)) == 1
    assert len(np.unique(c3)) == 1
    assert c1[0] != c2[0]
    assert c2[0] != c3[0]
    assert c1[0] != c3[0]

    # Verify centroids shape
    assert kmeans.centroids_ is not None
    assert kmeans.centroids_.shape == (3, 2)

    # Verify predict works identically
    preds = kmeans.predict(X)
    assert np.array_equal(preds, kmeans.labels_)

    # Verify inertia is positive
    assert kmeans.inertia_ > 0.0


def test_kmeans_invalid_inputs():
    kmeans = KMeans(n_clusters=2)

    # Check 1D array raises error
    with pytest.raises(ValueError):
        kmeans.fit(np.array([1, 2, 3]))

    # Check n_clusters > n_samples raises error
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        KMeans(n_clusters=3).fit(X)

    # Check n_clusters < 1 raises error
    with pytest.raises(ValueError):
        KMeans(n_clusters=0).fit(X)


def test_kmeans_not_fitted():
    kmeans = KMeans(n_clusters=2)
    # Check predict before fit raises error
    with pytest.raises(RuntimeError):
        kmeans.predict(np.array([[1.0, 2.0]]))


def test_kmeans_reproducibility():
    X = np.random.default_rng(42).normal(size=(50, 2))

    kmeans1 = KMeans(n_clusters=3, random_state=42).fit(X)
    kmeans2 = KMeans(n_clusters=3, random_state=42).fit(X)

    assert kmeans1.centroids_ is not None
    assert kmeans2.centroids_ is not None
    assert np.allclose(kmeans1.centroids_, kmeans2.centroids_)
    assert np.array_equal(kmeans1.labels_, kmeans2.labels_)
    assert np.isclose(kmeans1.inertia_, kmeans2.inertia_)
