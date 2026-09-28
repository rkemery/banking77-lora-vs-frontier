"""Frozen splits: dev (from train), sweep subset, learning-curve subsets and the deduplicated test.

Everything is derived from seeded functions of the labels, written once to
`data/splits/splits.json` and committed. Every later step reads that file, so
the dev split used to pick hyperparameters is fixed before any test number
exists. `b77 splits --check` recomputes it and fails on any difference.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from b77.data import DATASET_ID, N_CLASSES, REVISION, Split

SPLITS_PATH = Path("data/splits/splits.json")
SEED = 0
DEV_SIZE = 1_000
SWEEP_SIZE = 2_000
LEARNING_CURVE_K = (5, 10, 20)
LEARNING_CURVE_SEEDS = (0, 1, 2)


def dev_indices(labels: np.ndarray, size: int = DEV_SIZE, seed: int = SEED) -> np.ndarray:
    """A class-stratified random sample of train, sorted. Class shares match train's."""
    idx = np.arange(labels.size)
    _, dev = train_test_split(idx, test_size=size, stratify=labels, random_state=seed)
    return np.sort(dev)


def stratified_subset(
    labels: np.ndarray, pool: np.ndarray, size: int, seed: int = SEED
) -> np.ndarray:
    """A class-stratified sample of `size` indices drawn from `pool`, sorted."""
    if size >= pool.size:
        raise ValueError(f"subset size {size} must be smaller than the pool ({pool.size})")
    chosen, _ = train_test_split(pool, train_size=size, stratify=labels[pool], random_state=seed)
    return np.sort(chosen)


def per_class_subsets(
    labels: np.ndarray, ks: Sequence[int], seed: int, n_classes: int = N_CLASSES
) -> dict[int, np.ndarray]:
    """k examples of every class for each k, nested: the k=5 set is inside the k=10 set.

    One random permutation per class, and each k takes its prefix, so adjacent
    points on the learning curve differ only by added examples, not by a fresh
    draw.
    """
    rng = np.random.default_rng(seed)
    kmax = max(ks)
    orders = []
    for c in range(n_classes):
        members = np.flatnonzero(labels == c)
        if members.size < kmax:
            raise ValueError(f"class {c} has {members.size} examples, fewer than k={kmax}")
        orders.append(rng.permutation(members))
    return {k: np.sort(np.concatenate([order[:k] for order in orders])) for k in ks}


def build_splits(train: Split, dedup: dict[str, Any]) -> dict[str, Any]:
    dev = dev_indices(train.labels)
    pool = np.setdiff1d(np.arange(len(train)), dev)
    sweep = stratified_subset(train.labels, pool, SWEEP_SIZE)
    curve = {
        f"seed{seed}": {
            str(k): [train.ids[pool[i]] for i in idx]
            for k, idx in per_class_subsets(train.labels[pool], LEARNING_CURVE_K, seed).items()
        }
        for seed in LEARNING_CURVE_SEEDS
    }
    return {
        "dataset": DATASET_ID,
        "revision": REVISION,
        "seed": SEED,
        "dev": {
            "description": "Stratified random sample of train, used only for hyperparameter "
            "choices. Final models are refit on all of train.",
            "size": int(dev.size),
            "ids": [train.ids[i] for i in dev],
        },
        "sweep": {
            "description": "Stratified random sample of train minus dev, used for the LoRA "
            "learning-rate sweep.",
            "size": int(sweep.size),
            "ids": [train.ids[i] for i in sweep],
        },
        "learning_curve": {
            "description": "k examples per class from train minus dev, nested across k, "
            "one draw per seed.",
            "subsets": curve,
        },
        "test_dedup": dedup,
    }


def write_splits(splits: dict[str, Any], path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=1) + "\n", encoding="utf-8", newline="\n")


def read_splits(path: Path = SPLITS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make splits` first.")
    splits = json.loads(path.read_text(encoding="utf-8"))
    if splits.get("revision") != REVISION:
        raise ValueError(f"{path} was built from revision {splits.get('revision')}, not {REVISION}")
    return splits


def select(split: Split, ids: Sequence[str], name: str) -> Split:
    """The items of `split` with these ids, in the order given."""
    position = {item_id: i for i, item_id in enumerate(split.ids)}
    missing = [i for i in ids if i not in position]
    if missing:
        raise KeyError(f"{len(missing)} ids not in {split.name}: {missing[:3]}")
    return split.subset([position[i] for i in ids], name=name)


def train_minus_dev(train: Split, splits: dict[str, Any]) -> Split:
    dev = set(splits["dev"]["ids"])
    return select(train, [i for i in train.ids if i not in dev], "train_pool")
