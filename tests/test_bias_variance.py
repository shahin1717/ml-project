import numpy as np
from experiments.bias_variance import bias_variance_decomposition
from src.trees.decision_tree import DecisionTree


def test_bias_variance_decomposition_smoke():
    # Simple XOR-like dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    def model_fn():
        return DecisionTree(max_depth=2, random_state=42)

    bias_sq, variance = bias_variance_decomposition(
        model_fn,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        n_bootstraps=5,
        random_state=42,
    )

    assert isinstance(bias_sq, float)
    assert isinstance(variance, float)
    assert bias_sq >= 0.0
    assert variance >= 0.0
