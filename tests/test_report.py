from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from llm_eval_harness.report import replace_section

from b77 import plan
from b77.report import build, headline, holm, results_table
from b77.runs import Prediction, write_run
from b77.splits import read_splits

TEST_IDS = [f"test-{i:05d}" for i in range(3080)]
GOLD = np.repeat(np.arange(77), 40)


def _fake_run(tmp: Path, run_id: str, accuracy: float, seed: int, api: bool = False) -> None:
    rng = np.random.default_rng(seed)
    wrong = rng.random(GOLD.size) > accuracy
    pred = np.where(wrong, (GOLD + 1) % 77, GOLD)
    preds = [
        Prediction(
            item_id=TEST_IDS[i],
            gold=int(GOLD[i]),
            pred=int(pred[i]),
            confidence=None if api else 0.9,
            latency_ms=float(100 + i % 50),
            tokens_in=1500 if api else 0,
            tokens_out=9 if api else 0,
            cost_usd=0.0002 if api else 0.0,
        )
        for i in range(GOLD.size)
    ]
    write_run(run_id, "cfg", "m", preds, {"train_seconds": 60.0} if not api else {}, tmp)


def test_report_shows_real_rows_and_pending_rows(tmp_path: Path) -> None:
    _fake_run(tmp_path, plan.RUN_LOGREG, 0.85, 0)
    _fake_run(tmp_path, plan.RUN_QWEN_FINAL.format(seed=0), 0.92, 1)
    _fake_run(tmp_path, plan.RUN_QWEN_FINAL.format(seed=1), 0.91, 2)
    _fake_run(tmp_path, plan.RUN_LUNA_FEW, 0.88, 3, api=True)
    body, summary = build(tmp_path)
    assert "pending live run: `make prompt-sol-fewshot`" in body
    assert "pending CPU run: `make train-modernbert`" in body
    assert "seed sd" in body
    assert "n/a (no logprobs)" in body
    qwen = summary["arms"][plan.RUN_QWEN_FINAL.format(seed=0)]
    assert qwen["n"] == 3080
    assert qwen["dedup_n"] == read_splits()["test_dedup"]["kept"]
    assert qwen["accuracy_ci"][0] < qwen["accuracy"] < qwen["accuracy_ci"][1]
    # The LoRA vs luna few-shot pair exists, so it gets a McNemar row and a break-even line.
    assert f"| {plan.RUN_QWEN_FINAL.format(seed=0)} | {plan.RUN_LUNA_FEW} | 9" in body
    assert "| primary (" in body
    assert f"vs `{plan.RUN_LUNA_FEW}` ($0.20): cheaper above" in body
    # Too few runs for the headline, so it says so instead of printing half the numbers.
    assert "once every arm has run" in headline(summary)
    # API cost per 1k is the mean record cost times 1,000.
    assert summary["arms"][plan.RUN_LUNA_FEW]["cost_per_1k_usd"] == pytest.approx(0.2)
    assert ";" not in body.replace("&", "")


def test_all_pending_table_has_a_row_per_arm() -> None:
    from b77.report import arms

    table = results_table([(spec, None) for spec in arms()], 2662)
    assert table.count("\n| ") == len(arms())
    assert "n=2,662" in table


def test_holm_matches_a_hand_computed_example() -> None:
    # Sorted p 0.01, 0.02, 0.04 times 3, 2, 1 is 0.03, 0.04, 0.04 (kept monotone).
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_committed_readme_section_is_current() -> None:
    """`make demo` must have been run after the last change to results/."""
    readme = Path("README.md").read_text(encoding="utf-8")
    body, summary = build()
    assert replace_section(readme, "results", body) == readme
    assert replace_section(readme, "headline", headline(summary)) == readme
