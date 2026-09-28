from __future__ import annotations

import numpy as np

from b77.data import Split
from b77.dedup import dedup_summary, find_twins, nearest_train

GREEK = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
]
FILLER = [f"Filler message number {i} about topic {w}" for i, w in enumerate(GREEK)]


def _train() -> Split:
    texts = [
        "When will my new card arrive?",
        "How do I change my PIN?",
        "Is there a fee for topping up by card?",
        *FILLER,
    ]
    labels = np.array([11, 21, 57] + [0] * len(FILLER))
    return Split("train", [f"train-{i}" for i in range(len(texts))], texts, labels)


def test_near_twins_are_found_and_split_by_label() -> None:
    test = Split(
        "test",
        ["test-0", "test-1", "test-2", "test-3"],
        [
            "when will my new card arrive?",  # same text up to case, same label: removed
            "How do I change my pin?",  # same text up to case, other label: label noise, kept
            "How do I change my PIN",  # similar (about 0.83) but under the threshold
            "Why was my transfer declined?",  # nothing close
        ],
        np.array([11, 49, 21, 27]),
    )
    _, sims = nearest_train(_train(), test)
    assert sims[0] > 0.99
    assert 0.5 < sims[2] < 0.9
    twins = find_twins(_train(), test)
    by_test = {t.test_id: t for t in twins}
    assert set(by_test) == {"test-0", "test-1"}
    assert by_test["test-0"].same_label
    assert not by_test["test-1"].same_label
    summary = dedup_summary(test, twins)
    assert summary["removed_ids"] == ["test-0"]
    assert summary["kept"] == 3
    assert summary["different_label_twins"] == 1
