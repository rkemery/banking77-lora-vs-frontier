"""Near-duplicate detection between test and train.

For every test item we record its nearest train item under character 3 to 5
gram TF-IDF cosine similarity, the same measure the ank018/lora-banking77
project used. A test item is a near twin when that similarity is at least
0.90. The deduplicated test subset drops near twins whose train neighbour has
the same label, since those are the items a fine-tuned model could answer by
recall. Twins with a different label stay in, and are counted as a sign of
label noise.

The vectorizer is fit on train and test together, so n-grams that appear only
in test still count against similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from b77.data import Split

THRESHOLD = 0.90
METHOD = (
    "Nearest train item by cosine similarity of character 3-5 gram TF-IDF vectors "
    "(scikit-learn TfidfVectorizer, analyzer='char_wb', lowercase, fit on train and test). "
    "A test item is removed when its nearest train item has similarity >= 0.90 and the same label."
)


@dataclass(frozen=True)
class Twin:
    test_id: str
    train_id: str
    similarity: float
    same_label: bool


def nearest_train(train: Split, test: Split, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """For each test item, the index of its most similar train item and that similarity."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    vectorizer.fit(list(train.texts) + list(test.texts))
    train_m = vectorizer.transform(train.texts)
    test_m = vectorizer.transform(test.texts)
    best_idx = np.empty(len(test), dtype=np.int64)
    best_sim = np.empty(len(test), dtype=np.float64)
    for start in range(0, len(test), chunk):
        sims = (test_m[start : start + chunk] @ train_m.T).toarray()
        best_idx[start : start + chunk] = sims.argmax(axis=1)
        best_sim[start : start + chunk] = sims.max(axis=1)
    return best_idx, best_sim


def find_twins(train: Split, test: Split, threshold: float = THRESHOLD) -> list[Twin]:
    idx, sim = nearest_train(train, test)
    return [
        Twin(test.ids[i], train.ids[j], round(float(s), 4), bool(test.labels[i] == train.labels[j]))
        for i, (j, s) in enumerate(zip(idx, sim, strict=True))
        if s >= threshold
    ]


def dedup_summary(test: Split, twins: list[Twin], threshold: float = THRESHOLD) -> dict[str, Any]:
    removed = sorted(t.test_id for t in twins if t.same_label)
    return {
        "method": METHOD,
        "threshold": threshold,
        "near_twins": len(twins),
        "same_label_twins": len(removed),
        "different_label_twins": sum(not t.same_label for t in twins),
        "kept": len(test) - len(removed),
        "removed_ids": removed,
    }
