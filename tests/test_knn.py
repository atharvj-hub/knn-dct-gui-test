"""
tests/test_knn.py – Unit tests for the KNN classifier
"""

import numpy as np
import pytest

from knn import (
    euclidean_distance,
    KNNClassifier,
    make_two_class_blobs,
    make_decision_grid,
)


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------

def test_euclidean_distance_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert euclidean_distance(a, a) == pytest.approx(0.0)


def test_euclidean_distance_known():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean_distance(a, b) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# KNNClassifier
# ---------------------------------------------------------------------------

class TestKNNClassifier:
    def _simple_dataset(self):
        """Two perfectly separable clusters."""
        X_train = np.array([[0.0, 0.0], [0.5, 0.5], [5.0, 5.0], [5.5, 5.5]])
        y_train = np.array([0, 0, 1, 1])
        return X_train, y_train

    def test_fit_and_predict_correct(self):
        X, y = self._simple_dataset()
        clf = KNNClassifier(k=1)
        clf.fit(X, y)
        preds = clf.predict(np.array([[0.1, 0.1], [5.1, 5.1]]))
        assert list(preds) == [0, 1]

    def test_score_perfect(self):
        X, y = self._simple_dataset()
        clf = KNNClassifier(k=1)
        clf.fit(X, y)
        assert clf.score(X, y) == pytest.approx(1.0)

    def test_k3_majority_vote(self):
        """Majority vote with k=3 should still give the right answer."""
        X_train = np.array([[0.0, 0], [0.3, 0], [0.6, 0],
                             [5.0, 0], [5.3, 0], [5.6, 0]])
        y_train = np.array([0, 0, 0, 1, 1, 1])
        clf = KNNClassifier(k=3).fit(X_train, y_train)
        assert clf.predict(np.array([[0.1, 0]]))[0] == 0
        assert clf.predict(np.array([[5.1, 0]]))[0] == 1

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            KNNClassifier(k=0)

    def test_predict_before_fit_raises(self):
        clf = KNNClassifier(k=1)
        with pytest.raises(RuntimeError):
            clf.predict(np.array([[1.0, 2.0]]))

    def test_blobs_accuracy_above_90_percent(self):
        """End-to-end smoke test on generated data."""
        X, y = make_two_class_blobs(n_per_class=60, seed=42)
        split = 80
        clf = KNNClassifier(k=5).fit(X[:split], y[:split])
        acc = clf.score(X[split:], y[split:])
        assert acc >= 0.90


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def test_make_two_class_blobs_shape():
    X, y = make_two_class_blobs(n_per_class=30)
    assert X.shape == (60, 2)
    assert y.shape == (60,)
    assert set(y) == {0, 1}


def test_make_decision_grid_shape():
    xx, yy, grid = make_decision_grid(0, 2, 0, 2, resolution=0.5)
    assert xx.shape == yy.shape
    assert grid.shape[1] == 2
