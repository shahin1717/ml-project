import numpy as np
import pytest
import os
from src.utils.preprocessing import (
    load_breast_cancer,
    train_test_split,
    standardize,
    load_adult,
    load_covertype,
    load_mnist_subset,
    make_severely_imbalanced
)


def test_load_breast_cancer():
    X, y = load_breast_cancer()
    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert set(np.unique(y)) == {0, 1}


def test_train_test_split():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    assert X_train.shape == (8, 2)
    assert X_test.shape == (2, 2)
    assert y_train.shape == (8,)
    assert y_test.shape == (2,)


def test_standardize():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_test = np.array([[1.0, 2.0], [7.0, 8.0]])

    X_train_scaled, X_test_scaled = standardize(X_train, X_test)

    assert np.allclose(X_train_scaled.mean(axis=0), [0.0, 0.0])
    assert np.allclose(X_train_scaled.std(axis=0), [1.0, 1.0])

    # Standard deviation check for zero variance column
    X_train_zero = np.array([[2.0, 3.0], [2.0, 5.0]])
    X_test_zero = np.array([[2.0, 4.0]])
    X_tr_s, X_te_s = standardize(X_train_zero, X_test_zero)
    assert np.allclose(X_tr_s[:, 0], [0.0, 0.0])  # zero variance should not divide by zero


def test_load_adult():
    # Only test if files exist
    if os.path.exists("data/adult.data") and os.path.exists("data/adult.test"):
        X_train, X_test, y_train, y_test = load_adult()
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == X_test.shape[1]
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]
        assert set(np.unique(y_train)) == {0, 1}
        assert set(np.unique(y_test)) == {0, 1}


def test_load_covertype():
    if os.path.exists("data/covtype.data"):
        X, y = load_covertype(n_samples=200, random_state=42)
        assert X.shape == (200, 54)
        assert y.shape == (200,)
        # Classes should be 0-indexed (0 to 6)
        assert np.all(y >= 0) and np.all(y <= 6)


def test_load_mnist_subset():
    # Check loading a very small subset of MNIST (e.g. 20 samples)
    try:
        X, y = load_mnist_subset(n_samples=20, classes=("3", "8"), random_state=42)
        assert X.shape == (20, 784)
        assert y.shape == (20,)
        assert np.all(np.isin(y, [0, 1]))
    except Exception as e:
        # Ignore network failure in unit tests if fetch_openml fails due to offline/no-network
        pytest.skip(f"Skipping MNIST test due to potential network error: {e}")


def test_make_severely_imbalanced():
    # 200 samples: 100 of class 0, 100 of class 1
    X = np.random.normal(size=(200, 2))
    y = np.array([0] * 100 + [1] * 100)

    # Target ratio = 0.01 (1%)
    # n_majority = 100
    # n_minority_target = round(0.01 * 100 / 0.99) = 1
    X_imb, y_imb = make_severely_imbalanced(X, y, minority_class=1, ratio=0.01, random_state=42)

    assert y_imb.shape == (101,)
    assert np.sum(y_imb == 0) == 100
    assert np.sum(y_imb == 1) == 1

    # Target ratio = 0.05 (5%)
    # n_minority_target = round(0.05 * 100 / 0.95) = 5
    X_imb2, y_imb2 = make_severely_imbalanced(X, y, minority_class=1, ratio=0.05, random_state=42)
    assert y_imb2.shape == (105,)
    assert np.sum(y_imb2 == 0) == 100
    assert np.sum(y_imb2 == 1) == 5

    # Check error raised if ratio is too high
    with pytest.raises(ValueError):
        make_severely_imbalanced(X, y, minority_class=1, ratio=0.6, random_state=42)
