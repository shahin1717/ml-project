import numpy as np
import pytest
from src.unsupervised.dbscan import DBSCAN


def test_dbscan_clustering():
    # Construct two distinct blobs and a noise point
    # Blob 1: centered around [0, 0]
    blob1 = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.1],
        [0.1, 0.0],
        [-0.1, -0.1],
    ])
    # Blob 2: centered around [10, 10]
    blob2 = np.array([
        [10.0, 10.0],
        [10.1, 10.1],
        [10.0, 10.1],
        [10.1, 10.0],
        [9.9, 9.9],
    ])
    # Noise point: centered around [5, 5]
    noise = np.array([[5.0, 5.0]])

    X = np.vstack([blob1, blob2, noise])

    # With eps=0.5 and min_samples=3:
    # Blob 1 should form cluster 0, Blob 2 should form cluster 1, Noise should be -1.
    db = DBSCAN(eps=0.5, min_samples=3)
    db.fit(X)

    # Check labels shape
    assert db.labels_.shape == (11,)

    # Labels for Blob 1 and Blob 2 should be different clusters
    cluster1 = db.labels_[0:5]
    cluster2 = db.labels_[5:10]
    noise_label = db.labels_[10]

    assert len(np.unique(cluster1)) == 1
    assert len(np.unique(cluster2)) == 1
    assert cluster1[0] != -1
    assert cluster2[0] != -1
    assert cluster1[0] != cluster2[0]
    assert noise_label == -1


def test_dbscan_invalid_inputs():
    db = DBSCAN(eps=0.5, min_samples=2)
    # Check 1D array raises error
    with pytest.raises(ValueError):
        db.fit(np.array([1, 2, 3]))


def test_dbscan_empty_input():
    db = DBSCAN(eps=0.5, min_samples=2)
    X = np.empty((0, 2))
    db.fit(X)
    assert db.labels_.shape == (0,)
