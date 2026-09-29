from __future__ import annotations

import numpy as np
import pytest

from b77.splits import (
    DEV_SIZE,
    LEARNING_CURVE_K,
    SWEEP_SIZE,
    dev_indices,
    per_class_subsets,
    read_splits,
    stratified_subset,
)


def _labels(n_classes: int = 77, per_class: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(np.repeat(np.arange(n_classes), per_class))


def test_dev_split_is_stratified_and_deterministic() -> None:
    labels = _labels()
    a, b = dev_indices(labels, size=770), dev_indices(labels, size=770)
    assert np.array_equal(a, b)
    assert np.bincount(labels[a], minlength=77).tolist() == [10] * 77


def test_stratified_subset_stays_inside_the_pool() -> None:
    labels = _labels()
    pool = np.arange(0, labels.size, 2)
    sub = stratified_subset(labels, pool, 500)
    assert sub.size == 500
    assert set(sub) <= set(pool)
    with pytest.raises(ValueError, match="smaller than the pool"):
        stratified_subset(labels, pool, pool.size)


def test_per_class_subsets_are_nested_and_balanced() -> None:
    labels = _labels()
    subsets = per_class_subsets(labels, (5, 10, 20), seed=3)
    for k, idx in subsets.items():
        assert np.bincount(labels[idx], minlength=77).tolist() == [k] * 77
    assert set(subsets[5]) <= set(subsets[10]) <= set(subsets[20])
    assert not np.array_equal(subsets[20], per_class_subsets(labels, (5, 10, 20), seed=4)[20])


def test_per_class_subsets_refuse_a_class_that_is_too_small() -> None:
    labels = np.repeat(np.arange(77), 4)
    with pytest.raises(ValueError, match="fewer than k"):
        per_class_subsets(labels, (5,), seed=0)


def test_committed_splits_are_consistent() -> None:
    splits = read_splits()
    dev = set(splits["dev"]["ids"])
    sweep = set(splits["sweep"]["ids"])
    assert len(dev) == DEV_SIZE
    assert len(sweep) == SWEEP_SIZE
    assert not dev & sweep
    for subsets in splits["learning_curve"]["subsets"].values():
        sizes = {int(k): len(ids) for k, ids in subsets.items()}
        assert sizes == {k: 77 * k for k in LEARNING_CURVE_K}
        assert not dev & set(subsets["20"])
        assert set(subsets["5"]) <= set(subsets["10"]) <= set(subsets["20"])
    d = splits["test_dedup"]
    assert d["kept"] + d["same_label_twins"] == 3_080
    assert d["same_label_twins"] + d["different_label_twins"] == d["near_twins"]
    assert len(d["removed_ids"]) == d["same_label_twins"]
