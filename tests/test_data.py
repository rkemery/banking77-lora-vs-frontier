from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from b77 import data
from b77.data import (
    LABEL_NAMES,
    RAW_FILES,
    DataIntegrityError,
    Split,
    _label_names_from_metadata,
    fetch,
    load_split,
)


def test_label_names_are_the_77_upstream_names() -> None:
    assert len(LABEL_NAMES) == 77
    assert len(set(LABEL_NAMES)) == 77
    # Two names are spelled unusually upstream, and the prompts must match them exactly.
    assert "Refund_not_showing_up" in LABEL_NAMES
    assert "reverted_card_payment?" in LABEL_NAMES


def test_pinned_hashes_look_like_sha256() -> None:
    for raw in RAW_FILES.values():
        assert len(raw.sha256) == 64
        int(raw.sha256, 16)
    assert RAW_FILES["train"].rows == 10_003
    assert RAW_FILES["test"].rows == 3_080


def test_fetch_rejects_a_file_with_the_wrong_hash(tmp_path: Path) -> None:
    for split in RAW_FILES:
        (tmp_path / f"{split}.parquet").write_bytes(b"not the dataset")
    with pytest.raises(DataIntegrityError, match="sha256"):
        fetch(tmp_path)


def test_label_names_are_read_from_parquet_metadata(tmp_path: Path) -> None:
    features = {"label": {"names": list(LABEL_NAMES), "_type": "ClassLabel"}}
    meta = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
    table = pa.table({"text": ["hi"], "label": [0]}).replace_schema_metadata(meta)
    pq.write_table(table, tmp_path / "x.parquet")
    names = _label_names_from_metadata(pq.read_table(tmp_path / "x.parquet").schema.metadata)
    assert names == list(LABEL_NAMES)
    with pytest.raises(DataIntegrityError):
        _label_names_from_metadata({})


def test_split_subset_keeps_ids_texts_and_labels_aligned() -> None:
    split = Split("train", ["a", "b", "c"], ["x", "y", "z"], np.array([0, 1, 2]))
    sub = split.subset([2, 0])
    assert sub.ids == ["c", "a"]
    assert sub.texts == ["z", "x"]
    assert sub.labels.tolist() == [2, 0]


def test_download_url_is_pinned_to_the_revision() -> None:
    url = data.download_url("test")
    assert data.REVISION in url
    assert url.endswith("test-00000-of-00001.parquet")


@pytest.mark.network
def test_real_splits_load_and_match_the_pinned_shape() -> None:
    train, test = load_split("train"), load_split("test")
    assert len(train) == 10_003
    assert len(test) == 3_080
    assert np.bincount(test.labels).tolist() == [40] * 77
