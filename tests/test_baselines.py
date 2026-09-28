from __future__ import annotations

import numpy as np
import pytest

from b77.baselines import choose_c, knn_vote


def test_knn_vote_weights_by_similarity() -> None:
    labels = np.array([[1, 2, 2], [4, 4, 5]])
    sims = np.array([[0.9, 0.5, 0.5], [0.6, 0.1, 0.8]])
    pred, share = knn_vote(labels, sims, n_classes=6)
    assert pred.tolist() == [2, 5]
    assert share[0] == pytest.approx(1.0 / 1.9)
    assert share[1] == pytest.approx(0.8 / 1.5)


def test_knn_vote_breaks_exact_ties_toward_the_nearest_neighbour() -> None:
    pred, _ = knn_vote(np.array([[3, 1]]), np.array([[0.5, 0.5]]), n_classes=4)
    assert pred.tolist() == [3]


def test_choose_c_picks_the_best_dev_accuracy_and_prefers_small_c_on_ties() -> None:
    rng = np.random.default_rng(0)
    n_classes = 77
    centers = rng.normal(size=(n_classes, 8)) * 5
    y = np.repeat(np.arange(n_classes), 6)
    x = centers[y] + rng.normal(size=(y.size, 8))
    c, scores = choose_c(x, y, x, y, grid=(1.0, 10.0))
    assert set(scores) == {1.0, 10.0}
    assert scores[c] == max(scores.values())
    if scores[1.0] == scores[10.0]:
        assert c == 1.0
