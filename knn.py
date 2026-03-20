"""
knn.py – K-Nearest Neighbours (KNN) classifier
================================================
A self-contained, pure-Python/NumPy implementation of the KNN algorithm
designed for learning purposes.

Key concepts illustrated
------------------------
* Distance metrics   – Euclidean distance between feature vectors
* Majority-vote      – choosing the most common label among k neighbours
* Decision boundary  – how k affects the shape of the boundary
"""

import numpy as np


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean distance between two 1-D arrays."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


# ---------------------------------------------------------------------------
# KNN Classifier
# ---------------------------------------------------------------------------

class KNNClassifier:
    """Simple K-Nearest Neighbours classifier.

    Parameters
    ----------
    k : int
        Number of neighbours to consider (default 3).

    Example
    -------
    >>> clf = KNNClassifier(k=3)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        self._X_train = np.asarray(X, dtype=float)
        self._y_train = np.asarray(y)
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in *X*.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    # ------------------------------------------------------------------
    def _predict_single(self, x: np.ndarray):
        """Return the predicted label for a single sample."""
        distances = [euclidean_distance(x, x_train) for x_train in self._X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self._y_train[k_indices]
        # Majority vote
        labels, counts = np.unique(k_labels, return_counts=True)
        return labels[np.argmax(counts)]

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given test data and labels."""
        predictions = self.predict(X)
        return float(np.mean(predictions == np.asarray(y)))


# ---------------------------------------------------------------------------
# Data generators (for the GUI demo)
# ---------------------------------------------------------------------------

def make_two_class_blobs(
    n_per_class: int = 40,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two Gaussian blobs, each representing a class.

    Returns
    -------
    X : np.ndarray, shape (2 * n_per_class, 2)
    y : np.ndarray, shape (2 * n_per_class,)  – labels 0 or 1
    """
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=[1.5, 1.5], scale=0.6, size=(n_per_class, 2))
    X1 = rng.normal(loc=[3.5, 3.5], scale=0.6, size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def make_decision_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a dense meshgrid for plotting decision boundaries.

    Returns
    -------
    xx, yy : 2-D arrays of grid coordinates
    grid   : 2-D array of shape (n_rows * n_cols, 2) – ready for predict()
    """
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid
