from __future__ import annotations

from pathlib import Path

import numpy as np

from b77.runs import NO_PREDICTION, Prediction, load_run, write_run


def test_round_trip_through_harness_records(tmp_path: Path) -> None:
    preds = [
        Prediction("test-00000", 3, 3, confidence=0.9, latency_ms=12.5),
        Prediction("test-00001", 4, 7, confidence=0.4, latency_ms=10.0),
        Prediction("test-00002", 5, NO_PREDICTION, error="APIError: content filter"),
    ]
    write_run("r1", "cfg", "model-x", preds, {"arm": "demo"}, tmp_path)
    run = load_run("r1", tmp_path)
    assert run.info["arm"] == "demo"
    gold, pred = run.arrays()
    assert gold.tolist() == [3, 4, 5]
    assert pred.tolist() == [3, 7, NO_PREDICTION]
    assert run.n_errors == 1
    # The errored record has no score and no measured latency.
    assert run.records[2].scores == {}
    assert run.latencies().tolist() == [12.5, 10.0]
    # Confidence is missing on the errored item, so there is no calibration for the run.
    assert run.confidences() is None


def test_arrays_follow_the_requested_id_order(tmp_path: Path) -> None:
    preds = [Prediction(f"test-{i:05d}", i, i, confidence=1.0) for i in range(3)]
    write_run("r2", "cfg", "m", preds, {}, tmp_path)
    run = load_run("r2", tmp_path)
    gold, _ = run.arrays(["test-00002", "test-00000"])
    assert gold.tolist() == [2, 0]
    np.testing.assert_allclose(run.confidences(), [1.0, 1.0, 1.0])
