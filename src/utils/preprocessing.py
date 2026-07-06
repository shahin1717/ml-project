# Preprocessing utilities – placeholder stub
import numpy as np
import os


def load_breast_cancer(data_dir="data"):
    path = os.path.join(data_dir, "wdbc.data")
    raw = np.genfromtxt(path, delimiter=",", dtype=str)

    y_raw = raw[:, 1]
    y = np.where(y_raw == "M", 1, 0)

    X = raw[:, 2:].astype(float)

    return X, y

def train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    indices = rng.permutation(n_samples)

    n_test = int(n_samples * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled