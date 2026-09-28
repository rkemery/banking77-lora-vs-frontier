"""Cheap baselines on frozen bge-small embeddings: logistic regression and a kNN vote.

Logistic regression picks its inverse regularisation C on the dev split
(trained on train minus dev), then refits on the training set it is given.
The kNN row votes over the same 20 retrieved neighbours the few-shot prompts
show the LLM, so it answers "what does the LLM add over its own examples".
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from b77.data import N_CLASSES

C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class LogRegResult:
    c: float
    dev_accuracy: dict[float, float]
    probs: np.ndarray  # (n_test, 77)
    predict_ms: np.ndarray  # per-item time for predict_proba on one row
    fit_seconds: float


def fit_logreg(x: np.ndarray, y: np.ndarray, c: float) -> LogisticRegression:
    model = LogisticRegression(C=c, max_iter=5_000)
    model.fit(x, y)
    if list(model.classes_) != list(range(N_CLASSES)):
        raise ValueError("training data must contain every one of the 77 classes")
    return model


def choose_c(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    grid: Sequence[float] = C_GRID,
) -> tuple[float, dict[float, float]]:
    """The C with the best dev accuracy. Ties go to the smaller C (more regularisation)."""
    scores = {c: float(np.mean(fit_logreg(x_fit, y_fit, c).predict(x_dev) == y_dev)) for c in grid}
    best = max(sorted(scores), key=lambda c: scores[c])
    return best, scores


def logreg_run(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> LogRegResult:
    """Pick C on dev, refit on (x_train, y_train), and predict test one row at a time."""
    c, dev_scores = choose_c(x_fit, y_fit, x_dev, y_dev)
    start = time.perf_counter()
    model = fit_logreg(x_train, y_train, c)
    fit_seconds = time.perf_counter() - start
    probs = model.predict_proba(x_test)
    predict_ms = np.empty(x_test.shape[0])
    for i in range(x_test.shape[0]):
        t0 = time.perf_counter()
        model.predict_proba(x_test[i : i + 1])
        predict_ms[i] = (time.perf_counter() - t0) * 1000.0
    return LogRegResult(c, dev_scores, probs, predict_ms, fit_seconds)


def knn_vote(
    neighbor_labels: np.ndarray, neighbor_sims: np.ndarray, n_classes: int = N_CLASSES
) -> tuple[np.ndarray, np.ndarray]:
    """Similarity-weighted vote. Returns predicted labels and the winning share of the weight.

    Ties go to the label of the more similar neighbour, via argmax on scores
    built in neighbour order with a tiny rank bonus.
    """
    n, k = neighbor_labels.shape
    weights = np.clip(neighbor_sims, 0.0, None)
    scores = np.zeros((n, n_classes))
    rows = np.repeat(np.arange(n), k)
    np.add.at(scores, (rows, neighbor_labels.ravel()), weights.ravel())
    # Break exact ties toward the nearest neighbour's label.
    scores[np.arange(n), neighbor_labels[:, 0]] += 1e-9
    pred = scores.argmax(axis=1)
    total = scores.sum(axis=1)
    share = np.divide(scores[np.arange(n), pred], total, out=np.zeros(n), where=total > 0)
    return pred, np.clip(share, 0.0, 1.0)
