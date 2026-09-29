"""Classification metrics with bootstrap CIs, calibration error and latency percentiles.

Paired comparisons between runs (McNemar, paired bootstrap, MDE) come from
`llm_eval_harness.stats` and are not reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from llm_eval_harness.stats import Interval, percentile_interval

N_BOOT = 10_000
ECE_BINS = 15


def accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    gold, pred = _check(gold, pred)
    return float(np.mean(gold == pred))


def macro_f1(gold: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1 = 2TP / (2TP + FP + FN).

    Averages over the classes that occur in gold or pred, which is what
    `sklearn.metrics.f1_score(average="macro")` does with its default labels.
    A prediction of -1 (no valid answer) counts as a miss for the gold class
    and as a false positive for no class.
    """
    gold, pred = _check(gold, pred)
    return float(_macro_f1_from_counts(*_class_counts(gold, pred, n_classes)))


def _class_counts(
    gold: np.ndarray, pred: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hit = gold == pred
    tp = np.bincount(gold[hit], minlength=n_classes)
    support = np.bincount(gold, minlength=n_classes)
    valid = pred >= 0
    predicted = np.bincount(pred[valid], minlength=n_classes)
    return tp, support, predicted


def _macro_f1_from_counts(tp: np.ndarray, support: np.ndarray, predicted: np.ndarray) -> float:
    denom = support + predicted  # 2TP + FP + FN
    present = denom > 0
    return float(np.mean(2.0 * tp[present] / denom[present]))


@dataclass(frozen=True)
class ClassificationSummary:
    n: int
    accuracy: Interval
    macro_f1: Interval


def summarize(
    gold: np.ndarray,
    pred: np.ndarray,
    n_classes: int,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    confidence: float = 0.95,
) -> ClassificationSummary:
    """Accuracy and macro-F1 with percentile bootstrap CIs over items (same resamples for both)."""
    gold, pred = _check(gold, pred)
    n = gold.size
    rng = np.random.default_rng(seed)
    acc_reps = np.empty(n_boot)
    f1_reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        g, p = gold[idx], pred[idx]
        acc_reps[b] = np.mean(g == p)
        f1_reps[b] = _macro_f1_from_counts(*_class_counts(g, p, n_classes))
    method = f"percentile bootstrap over items ({n_boot} resamples, seed {seed})"
    return ClassificationSummary(
        n=n,
        accuracy=percentile_interval(acc_reps, accuracy(gold, pred), n, confidence, method),
        macro_f1=percentile_interval(
            f1_reps, macro_f1(gold, pred, n_classes), n, confidence, method
        ),
    )


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """Top-label ECE with equal-width bins (Guo et al. 2017, arXiv 1706.04599).

    ECE = sum over bins of (bin size / n) * |accuracy in bin - mean confidence in bin|.
    Bins are (0, 1/B], (1/B, 2/B], ..., so a confidence of exactly 1 lands in the last bin.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if conf.shape != hit.shape or conf.ndim != 1 or conf.size == 0:
        raise ValueError("confidence and correct must be 1-d arrays of the same non-zero length")
    if np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidence must be in [0, 1]")
    bins = np.clip(np.ceil(conf * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return float(total)


def latency_percentiles(latency_ms: np.ndarray) -> tuple[float, float]:
    """p50 and p95 in milliseconds (linear interpolation)."""
    values = np.asarray(latency_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no latencies")
    p50, p95 = np.percentile(values, [50, 95])
    return float(p50), float(p95)


def _check(gold: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(gold, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    if g.shape != p.shape or g.ndim != 1 or g.size == 0:
        raise ValueError("gold and pred must be 1-d arrays of the same non-zero length")
    if np.any(g < 0):
        raise ValueError("gold labels must be >= 0")
    return g, p
