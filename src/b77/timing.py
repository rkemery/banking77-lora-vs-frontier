"""Measure CPU throughput and latency, and turn them into time estimates for the long runs.

`b77 timing` trains each model for a few dozen steps on real Banking77
batches, times single-message inference, and writes `results/timing.json`.
The README results section quotes the estimates computed here. Throughput is
measured on whatever else the machine is doing at the time, so treat the
estimates as +/- 30%.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from b77 import plan
from b77.train import MODELS, TrainConfig, build_model, load_tokenizer, log, predict, train

TIMING_PATH = Path("results/timing.json")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def hardware() -> dict[str, Any]:
    import torch

    return {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def time_training(
    model_key: str, texts: Sequence[str], labels: np.ndarray, batch_size: int, steps: int
) -> dict[str, Any]:
    cfg = TrainConfig(
        model=model_key,
        lr=plan.DEFAULT_LR[model_key],
        epochs=1,
        batch_size=batch_size,
        max_steps=steps,
    )
    _, _, result = train(cfg, texts, labels, dev=None, log_every=10)
    return {
        "batch_size": batch_size,
        "steps": result.steps,
        "examples": result.examples_seen,
        "seconds": round(result.train_seconds, 2),
        "examples_per_second": round(result.examples_per_second, 2),
    }


def time_inference(model: Any, tokenizer: Any, texts: Sequence[str]) -> dict[str, float]:
    _, latency = predict(model, tokenizer, texts[:100], batch_size=1)
    assert latency is not None
    start = time.perf_counter()
    predict(model, tokenizer, texts[:512], batch_size=64)
    batched = 512 / (time.perf_counter() - start)
    return {
        "batch1_p50_ms": round(float(np.percentile(latency, 50)), 2),
        "batch1_mean_ms": round(float(np.mean(latency)), 2),
        "batch64_examples_per_second": round(batched, 2),
    }


def time_bge(texts: Sequence[str]) -> dict[str, float]:
    from b77.embed import Encoder

    encoder = Encoder()
    _, latency = encoder.encode_timed(list(texts[:100]))
    start = time.perf_counter()
    encoder.encode_batched(list(texts[:512]))
    return {
        "batch1_p50_ms": round(float(np.percentile(latency, 50)), 2),
        "batch1_mean_ms": round(float(statistics.mean(latency)), 2),
        "batch64_examples_per_second": round(512 / (time.perf_counter() - start), 2),
    }


def measure(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    steps: int = 40,
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    sample = rng.choice(len(train_texts), size=min(len(train_texts), 64 * steps), replace=False)
    texts = [train_texts[i] for i in sample]
    labels = np.asarray(train_labels)[sample]
    out: dict[str, Any] = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "load_average_1m_before": round(os.getloadavg()[0], 2),
        **hardware(),
    }
    for key in MODELS:
        log(f"timing {key}")
        batch = plan.BATCH_SIZE[key]
        entry: dict[str, Any] = {"train": time_training(key, texts, labels, batch, steps)}
        cfg = TrainConfig(model=key, lr=plan.DEFAULT_LR[key], epochs=1)
        tokenizer = load_tokenizer(cfg.spec)
        model = build_model(cfg, tokenizer).eval()
        entry["inference"] = time_inference(model, tokenizer, test_texts)
        if cfg.spec.method == "lora":
            merged = model.merge_and_unload()
            entry["inference_merged"] = time_inference(merged, tokenizer, test_texts)
        out[key] = entry
    out["bge-small"] = {"inference": time_bge(test_texts)}
    # Other jobs on the same CPUs slow every number here. The load average says how busy it was.
    out["load_average_1m_after"] = round(os.getloadavg()[0], 2)
    out["estimates_minutes"] = plan.estimates(out)
    return out


def write_timing(result: dict[str, Any], path: Path = TIMING_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def read_timing(path: Path = TIMING_PATH) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
