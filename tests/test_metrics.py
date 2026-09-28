from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from b77.metrics import (
    accuracy,
    expected_calibration_error,
    latency_percentiles,
    macro_f1,
    summarize,
)


def _preds(n: int = 600, n_classes: int = 10, error_rate: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, n_classes, n)
    wrong = rng.random(n) < error_rate
    pred = np.where(wrong, rng.integers(0, n_classes, n), gold)
    return gold, pred


def test_macro_f1_matches_sklearn() -> None:
    gold, pred = _preds()
    assert macro_f1(gold, pred, 10) == pytest.approx(f1_score(gold, pred, average="macro"))


def test_macro_f1_counts_an_invalid_prediction_as_a_miss_only() -> None:
    gold = np.array([0, 1, 1])
    pred = np.array([0, 1, -1])
    # class 0: F1 1. class 1: TP 1, FN 1, FP 0 -> 2/3.
    assert macro_f1(gold, pred, 2) == pytest.approx((1 + 2 / 3) / 2)
    assert accuracy(gold, pred) == pytest.approx(2 / 3)


def test_summary_cis_contain_the_estimate_and_are_reproducible() -> None:
    gold, pred = _preds()
    a = summarize(gold, pred, 10, n_boot=2_000, seed=1)
    b = summarize(gold, pred, 10, n_boot=2_000, seed=1)
    assert a == b
    for interval in (a.accuracy, a.macro_f1):
        assert interval.low < interval.estimate < interval.high
    # Roughly the binomial width at n=600 and p=0.82.
    assert 0.05 < a.accuracy.high - a.accuracy.low < 0.08


def test_ece_is_zero_for_perfect_calibration_and_positive_for_overconfidence() -> None:
    conf = np.array([0.8] * 10)
    correct = np.array([1] * 8 + [0] * 2)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0)
    assert expected_calibration_error(np.array([0.99] * 10), correct) == pytest.approx(0.19)


def test_ece_puts_confidence_one_in_the_last_bin() -> None:
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1, 1])) == 0.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_calibration_error(np.array([1.2]), np.array([1]))


def test_latency_percentiles() -> None:
    p50, p95 = latency_percentiles(np.arange(1, 101, dtype=float))
    assert p50 == pytest.approx(50.5)
    assert p95 == pytest.approx(95.05)
