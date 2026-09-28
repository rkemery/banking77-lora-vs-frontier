"""Predictions to and from the harness JSONL results contract.

Each run is two files in `results/runs/`:

- `<run_id>.jsonl`: one `EvalRecord` per test item, `scores = {"correct": bool}`,
  with the gold and predicted label ids and the top-label probability in `meta`.
- `<run_id>.info.json`: what produced the run (arm, model, hyperparameters,
  training time, hardware, cost assumptions). The report reads both.

A record whose model call failed carries `error` and no score. The metrics
here count it as a wrong answer, so errors can only lower accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from llm_eval_harness import EvalRecord, read_records, write_records

RUNS_DIR = Path("results/runs")
NO_PREDICTION = -1


@dataclass(frozen=True)
class Prediction:
    item_id: str
    gold: int
    pred: int  # NO_PREDICTION when the model gave no valid label
    confidence: float | None = None
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def to_records(run_id: str, config: str, model: str, preds: list[Prediction]) -> list[EvalRecord]:
    records = []
    for p in preds:
        meta: dict[str, Any] = {"gold": p.gold, "pred": p.pred, **p.meta}
        if p.confidence is not None:
            meta["conf"] = round(p.confidence, 5)
        records.append(
            EvalRecord(
                run_id=run_id,
                item_id=p.item_id,
                config=config,
                model=model,
                scores={} if p.error else {"correct": p.pred == p.gold},
                tokens_in=p.tokens_in,
                tokens_out=p.tokens_out,
                reasoning_tokens=p.reasoning_tokens,
                cost_usd=p.cost_usd,
                latency_ms=round(p.latency_ms, 3),
                error=p.error,
                meta=meta,
            )
        )
    return records


def write_run(
    run_id: str,
    config: str,
    model: str,
    preds: list[Prediction],
    info: dict[str, Any],
    runs_dir: Path = RUNS_DIR,
) -> Path:
    path = runs_dir / f"{run_id}.jsonl"
    write_records(path, to_records(run_id, config, model, preds))
    info_path = runs_dir / f"{run_id}.info.json"
    info_path.write_text(
        json.dumps({"run_id": run_id, "config": config, "model": model, **info}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class Run:
    run_id: str
    info: dict[str, Any]
    records: list[EvalRecord]

    @property
    def item_ids(self) -> list[str]:
        return [r.item_id for r in self.records]

    def arrays(self, ids: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Gold and predicted label ids, in record order or in the order of `ids`."""
        records = self.records if ids is None else self.select(ids)
        gold = np.array([r.meta["gold"] for r in records], dtype=np.int64)
        pred = np.array(
            [NO_PREDICTION if r.error else r.meta["pred"] for r in records], dtype=np.int64
        )
        return gold, pred

    def select(self, ids: list[str]) -> list[EvalRecord]:
        by_id = {r.item_id: r for r in self.records}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(f"run {self.run_id} lacks {len(missing)} items, e.g. {missing[:3]}")
        return [by_id[i] for i in ids]

    def confidences(self) -> np.ndarray | None:
        if not all("conf" in r.meta for r in self.records):
            return None
        return np.array([r.meta["conf"] for r in self.records], dtype=np.float64)

    def latencies(self) -> np.ndarray:
        return np.array([r.latency_ms for r in self.records if r.error is None])

    @property
    def n_errors(self) -> int:
        return sum(r.error is not None for r in self.records)


def load_run(run_id: str, runs_dir: Path = RUNS_DIR) -> Run:
    records = read_records(runs_dir / f"{run_id}.jsonl")
    info_path = runs_dir / f"{run_id}.info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    return Run(run_id=run_id, info=info, records=records)


def run_exists(run_id: str, runs_dir: Path = RUNS_DIR) -> bool:
    return (runs_dir / f"{run_id}.jsonl").exists()
