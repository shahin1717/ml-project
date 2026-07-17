from __future__ import annotations
import numpy as np
import os
import pandas as pd  # type: ignore
from sklearn.datasets import fetch_openml  # type: ignore


def load_breast_cancer(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    """Load the Breast Cancer Wisconsin dataset from static wdbc.data file."""
    path = os.path.join(data_dir, "wdbc.data")
    raw = np.genfromtxt(path, delimiter=",", dtype=str)

    y_raw = raw[:, 1]
    y = np.where(y_raw == "M", 1, 0)

    X = raw[:, 2:].astype(float)

    return X, y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    stratify: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features and target matrices into random train and test subsets."""
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    if stratify is not None:
        y_strat = np.asarray(stratify)
        classes = np.unique(y_strat)
        train_indices_list = []
        test_indices_list = []

        for c in classes:
            idx_c = rng.permutation(np.where(y_strat == c)[0])
            n_test_c = max(1, int(np.round(len(idx_c) * test_size))) if len(idx_c) > 1 else 0
            if len(idx_c) == 1:
                train_indices_list.append(idx_c)
            else:
                test_indices_list.append(idx_c[:n_test_c])
                train_indices_list.append(idx_c[n_test_c:])

        train_idx = np.concatenate(train_indices_list)
        test_idx = np.concatenate(test_indices_list)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        indices = rng.permutation(n_samples)
        n_test = int(n_samples * test_size)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize continuous features to zero mean, unit variance."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


def load_adult(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, clean, and preprocess the Adult Income dataset from CSV files.

    Categorical columns are one-hot encoded and missing rows are discarded.
    """
    train_path = os.path.join(data_dir, "adult.data")
    test_path = os.path.join(data_dir, "adult.test")

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    # Clean rows containing "?" missing value string
    df_train = pd.read_csv(train_path, names=columns, na_values="?", sep=",", skipinitialspace=True)
    df_test = pd.read_csv(test_path, names=columns, na_values="?", sep=",", skipinitialspace=True, skiprows=1)

    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Strip dots from labels in test set
    df_train["income"] = df_train["income"].str.strip().str.rstrip(".")
    df_test["income"] = df_test["income"].str.strip().str.rstrip(".")

    n_train = len(df_train)
    df_all = pd.concat([df_train, df_test], axis=0)

    # Convert targets to binary classes
    df_all["income"] = (df_all["income"] == ">50K").astype(int)

    y_all = df_all["income"].values
    df_features = df_all.drop("income", axis=1)

    # One-hot encode categorical features
    categorical_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)

    X_all = df_features.astype(float).values

    X_train = X_all[:n_train]
    X_test = X_all[n_train:]
    y_train = y_all[:n_train]
    y_test = y_all[n_train:]

    return X_train, X_test, y_train, y_test


def load_covertype(
    data_dir: str = "data",
    n_samples: int | None = 30000,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a subset of the Covertype dataset and convert target to 0-indexed classes."""
    path = os.path.join(data_dir, "covtype.data")

    df = pd.read_csv(path, header=None)

    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_state)

    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)

    # Convert 1-indexed target classes to 0-indexed
    y = y - 1

    return X, y


def load_mnist_subset(
    n_samples: int | None = 6000,
    classes: tuple[str, str] = ("3", "8"),
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch the MNIST dataset, filter for a binary digit pair, and downsample."""
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="auto",
    )

    y_str = y.astype(str)
    mask = (y_str == classes[0]) | (y_str == classes[1])
    X_filtered = X[mask]
    y_filtered = y_str[mask]

    # Map selected classes to 0 and 1
    y_bin = np.where(y_filtered == classes[0], 0, 1)

    if n_samples and len(X_filtered) > n_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X_filtered), size=n_samples, replace=False)
        X_filtered = X_filtered[indices]
        y_bin = y_bin[indices]

    return X_filtered.astype(float), y_bin


def make_severely_imbalanced(
    X: np.ndarray,
    y: np.ndarray,
    minority_class: int = 1,
    ratio: float = 0.01,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the minority class of a binary classification dataset to a target ratio.

    Solve for n_minority: n_minority / (n_majority + n_minority) = ratio
    n_minority = ratio * n_majority / (1 - ratio)
    """
    rng = np.random.default_rng(random_state)

    idx_majority = np.where(y != minority_class)[0]
    idx_minority = np.where(y == minority_class)[0]

    n_majority = len(idx_majority)
    n_minority_target = int(np.round(ratio * n_majority / (1 - ratio)))

    if n_minority_target > len(idx_minority):
        raise ValueError(
            f"Cannot achieve target ratio {ratio} because there are only "
            f"{len(idx_minority)} minority class instances."
        )

    idx_minority_downsampled = rng.choice(
        idx_minority,
        size=n_minority_target,
        replace=False,
    )

    idx_combined = np.concatenate([idx_majority, idx_minority_downsampled])
    rng.shuffle(idx_combined)

    return X[idx_combined], y[idx_combined]