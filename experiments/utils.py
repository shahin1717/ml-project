from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from src.utils.preprocessing import (
    load_breast_cancer,
    load_adult,
    load_mnist_subset,
    train_test_split,
    standardize,
)


def get_dataset(
    name: str,
    data_dir: str = "data",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, split, and standardize a dataset by name.

    Parameters
    ----------
    name : {"breast_cancer", "adult", "mnist"}
        Name of the dataset to load.
    data_dir : str, default="data"
        Path to the data directory.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    if name == "breast_cancer":
        X, y = load_breast_cancer(data_dir=data_dir)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

    elif name == "adult":
        # load_adult returns pre-split train/test arrays
        X_train_full, X_test_full, y_train_full, y_test_full = load_adult(data_dir=data_dir)
        
        # Combine to downsample to 5,000 samples for scaling performance
        X_all = np.vstack([X_train_full, X_test_full])
        y_all = np.concatenate([y_train_full, y_test_full])

        n_samples = min(5000, len(X_all))
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X_all), size=n_samples, replace=False)
        X_sub = X_all[indices]
        y_sub = y_all[indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=random_state
        )

    elif name == "mnist":
        # Subsample to 2,000 samples for high-dim scaling performance
        X, y = load_mnist_subset(
            n_samples=2000,
            classes=("3", "8"),
            random_state=random_state,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # Standardize features
    X_train_s, X_test_s = standardize(X_train, X_test)

    return X_train_s, X_test_s, y_train, y_test


def plot_scaling_curve(
    x_vals: list[int] | np.ndarray,
    train_vals: list[float] | np.ndarray,
    test_vals: list[float] | np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    save_filename: str,
    legend_labels: tuple[str, str] = ("Train", "Test"),
) -> None:
    """Plot training and testing metrics vs a parameter and save to figures/."""
    os.makedirs("figures", exist_ok=True)
    
    plt.figure(figsize=(9, 5.5))
    plt.plot(x_vals, train_vals, label=legend_labels[0], marker="o", linewidth=2)
    plt.plot(x_vals, test_vals, label=legend_labels[1], marker="x", linewidth=2)
    
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    save_path = os.path.join("figures", save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
