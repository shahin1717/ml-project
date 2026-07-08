"""Random Forest classifier – bootstrap aggregation over Decision Trees.

Implements bagging with feature sub-sampling, as required by Module 3 of
ML_FINAL_PROJECT.tex.

Public API
----------
RandomForestClassifier(n_estimators, max_depth, max_features, min_samples_split,
                        bootstrap, oob_score, n_jobs, random_state)
    .fit(X, y)              -> RandomForestClassifier
    .predict(X)             -> np.ndarray
    .predict_proba(X)       -> np.ndarray   shape (n_samples, n_classes)
    .oob_score_              -> float        (if oob_score=True)
    .feature_importances_    -> np.ndarray   shape (n_features,)
"""

from __future__ import annotations

import time
from multiprocessing import Pool
from typing import Optional

import numpy as np

from src.trees.decision_tree import DecisionTree


# ---------------------------------------------------------------------------
# Worker (module-level so it is picklable for multiprocessing.Pool)
# ---------------------------------------------------------------------------

def _fit_one_tree(args: tuple) -> tuple["DecisionTree", np.ndarray]:
    """Bootstrap-sample the data, fit one DecisionTree, return (tree, oob_idx)."""
    X, y, seed, tree_params, bootstrap = args
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)

    if bootstrap:
        sample_idx = rng.integers(0, n_samples, size=n_samples)  # RF.1
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[np.unique(sample_idx)] = False
        oob_idx = np.where(oob_mask)[0]
    else:
        sample_idx = np.arange(n_samples)
        oob_idx = np.empty(0, dtype=int)

    # Derive the tree's own seed from the forest seed (RF.2): reproducible
    # per-tree feature sub-sampling independent of the bootstrap draw above.
    tree_seed = int(rng.integers(0, 2**31 - 1))
    tree = DecisionTree(random_state=tree_seed, **tree_params)
    tree.fit(X[sample_idx], y[sample_idx])
    return tree, oob_idx


# ---------------------------------------------------------------------------
# RandomForestClassifier
# ---------------------------------------------------------------------------

class RandomForestClassifier:
    """Random Forest classifier (bagging + feature sub-sampling).

    Parameters
    ----------
    n_estimators      : number of trees in the forest.
    max_depth         : max depth passed to each DecisionTree.
    max_features      : int, "sqrt", "log2", or None — features considered per split.
    min_samples_split : min samples required to split a node.
    bootstrap         : draw bootstrap samples per tree (RF.1). If False, every
                         tree trains on the full dataset (no OOB estimate possible).
    oob_score         : compute out-of-bag accuracy after fitting (RF.4).
    n_jobs            : if > 1, train trees in parallel via multiprocessing.Pool (RF.3).
    random_state      : seed for the master RNG; per-tree seeds are derived from it,
                         so results are reproducible regardless of n_jobs.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_features: int | str | None = "sqrt",
        min_samples_split: int = 2,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.estimators_: list[DecisionTree] = []
        self._oob_indices: list[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: int = 0
        self._oob_score_: Optional[float] = None
        self.fit_time_: float = 0.0  # wall-clock seconds, for RF.3 timing comparisons

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} entries.")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True.")

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        master_rng = np.random.default_rng(self.random_state)
        seeds = master_rng.integers(0, 2**31 - 1, size=self.n_estimators)

        tree_params = dict(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion="gini",
            max_features=self.max_features,
        )
        args = [(X, y, int(seed), tree_params, self.bootstrap) for seed in seeds]

        start = time.perf_counter()
        if self.n_jobs is not None and self.n_jobs > 1:
            with Pool(processes=self.n_jobs) as pool:
                results = pool.map(_fit_one_tree, args)
        else:
            results = [_fit_one_tree(a) for a in args]  # sequential baseline (RF.3)
        self.fit_time_ = time.perf_counter() - start

        self.estimators_ = [tree for tree, _ in results]
        self._oob_indices = [oob for _, oob in results]

        if self.oob_score:
            self._oob_score_ = self._compute_oob_score(X, y)

        return self

    # ------------------------------------------------------------------
    # OOB score (RF.4)
    # ------------------------------------------------------------------

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        vote_sums = np.zeros((n_samples, n_classes))
        was_oob = np.zeros(n_samples, dtype=bool)

        for tree, oob_idx in zip(self.estimators_, self._oob_indices):
            if len(oob_idx) == 0:
                continue
            vote_sums[oob_idx] += tree.predict_proba(X[oob_idx])
            was_oob[oob_idx] = True

        if not was_oob.any():
            return float("nan")  # e.g. too few trees / tiny dataset

        oob_pred = self.classes_[np.argmax(vote_sums[was_oob], axis=1)]
        return float(np.mean(oob_pred == y[was_oob]))

    @property
    def oob_score_(self) -> float:
        if not self.oob_score:
            raise AttributeError("oob_score_ is only available when oob_score=True.")
        if self._oob_score_ is None:
            raise AttributeError("Call fit() before accessing oob_score_.")
        return self._oob_score_

    # ------------------------------------------------------------------
    # predict / predict_proba (RF.5)
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average of each tree's predict_proba. Returns shape (n_samples, n_classes)."""
        X = np.asarray(X)
        proba_sum = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.estimators_:
            proba_sum += tree.predict_proba(X)
        return proba_sum / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Majority vote (argmax of averaged probabilities)."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------
    # Feature importances (RF.6)
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average of each tree's (already-normalized) MDI importances."""
        importances = np.zeros(self.n_features_, dtype=float)
        for tree in self.estimators_:
            importances += tree.feature_importances()
        return importances / len(self.estimators_)

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RandomForestClassifier(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, max_features={self.max_features!r}, "
            f"bootstrap={self.bootstrap}, oob_score={self.oob_score}, "
            f"n_jobs={self.n_jobs})"
        )