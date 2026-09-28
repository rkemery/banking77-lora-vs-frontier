from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from llm_eval_harness import (
    BudgetExceeded,
    CachedClient,
    CacheMiss,
    DollarCap,
    FakeClient,
    ModelRequest,
    ModelResponse,
)

from b77.data import LABEL_NAMES
from b77.prompting import (
    ARMS,
    INSTRUCTIONS,
    MAX_OUTPUT_TOKENS,
    Neighbors,
    Pacer,
    build_request,
    examples_for,
    parse_label,
    run_arm,
    user_turn,
)
from b77.runs import NO_PREDICTION

TRAIN_TEXTS = [f"train message {i}" for i in range(6)]
TRAIN_LABELS = np.array([11, 12, 11, 21, 21, 57])
TEST_IDS = ["test-00000", "test-00001"]
TEST_TEXTS = ["where is my card", "change pin please"]
TEST_LABELS = np.array([11, 21])
NEIGHBORS = Neighbors(
    TEST_IDS,
    train_idx=np.array([[0, 2, 1] + [5] * 17, [3, 4, 0] + [5] * 17]),
    sims=np.tile(np.linspace(0.9, 0.5, 20), (2, 1)),
)


def test_every_request_forces_one_of_the_77_labels_with_no_reasoning() -> None:
    for arm in ARMS.values():
        ex = [("m", "card_arrival")] if arm.few_shot else []
        req = build_request(arm, "hello", ex)
        assert req.reasoning_effort == "none"
        assert req.max_output_tokens == MAX_OUTPUT_TOKENS
        fmt = req.extra["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["strict"] is True
        assert fmt["schema"]["properties"]["label"]["enum"] == list(LABEL_NAMES)


def test_static_prefix_is_identical_across_arms_and_items() -> None:
    """Only the user turn varies, so the provider's prompt cache can serve the rest."""
    requests = [
        build_request(ARMS["luna-zeroshot"], "a"),
        build_request(ARMS["luna-zeroshot"], "b"),
        build_request(ARMS["luna-fewshot"], "c", [("x", "card_arrival")]),
        build_request(ARMS["sol-fewshot"], "d", [("y", "change_pin")]),
    ]
    assert {r.instructions for r in requests} == {INSTRUCTIONS}
    assert len({json.dumps(r.extra, sort_keys=True) for r in requests}) == 1
    for name in LABEL_NAMES:
        assert f"\n{name}: " in INSTRUCTIONS


def test_few_shot_examples_put_the_most_similar_last() -> None:
    ex = examples_for(0, NEIGHBORS, TRAIN_TEXTS, TRAIN_LABELS, k=3)
    assert [m for m, _ in ex] == ["train message 1", "train message 2", "train message 0"]
    turn = user_turn("query", ex)
    assert turn.index("train message 0") > turn.index("train message 1")
    assert turn.endswith("Message: query")


def test_arms_refuse_the_wrong_kind_of_prompt() -> None:
    with pytest.raises(ValueError, match="needs retrieved examples"):
        build_request(ARMS["sol-fewshot"], "q")
    with pytest.raises(ValueError, match="zero-shot"):
        build_request(ARMS["luna-zeroshot"], "q", [("x", "card_arrival")])


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"label": "card_arrival"}', LABEL_NAMES.index("card_arrival")),
        ('{"label": "reverted_card_payment?"}', LABEL_NAMES.index("reverted_card_payment?")),
        ('{"label": "not_a_label"}', NO_PREDICTION),
        ("card_arrival", NO_PREDICTION),
        ('["card_arrival"]', NO_PREDICTION),
        ('{"label": 3}', NO_PREDICTION),
    ],
)
def test_parse_label(reply: str, expected: int) -> None:
    assert parse_label(reply) == expected


def _nearest_label_client() -> FakeClient:
    """Answers with the label of the example shown last (the nearest), or card_arrival."""

    def answer(req: ModelRequest) -> ModelResponse:
        content = req.input[-1]["content"]
        labels = [
            line[len("Label: ") :] for line in content.splitlines() if line.startswith("Label: ")
        ]
        label = labels[-1] if labels else "card_arrival"
        return ModelResponse(
            text=json.dumps({"label": label}),
            model=req.model,
            input_tokens=1500,
            output_tokens=9,
            cached_input_tokens=1024,
            latency_ms=321.0,
        )

    return FakeClient(answer)


def test_run_arm_end_to_end_with_a_fake_model() -> None:
    preds = run_arm(
        ARMS["sol-fewshot"],
        _nearest_label_client(),
        TEST_IDS,
        TEST_TEXTS,
        TEST_LABELS,
        neighbors=NEIGHBORS,
        train_texts=TRAIN_TEXTS,
        train_labels=TRAIN_LABELS,
        log=lambda _: None,
    )
    assert [p.pred for p in preds] == [11, 21]
    # sol list price: 476 uncached in at $2/M, 1024 cached at $0.20/M, 9 out at $10/M.
    assert preds[0].cost_usd == pytest.approx((476 * 2.0 + 1024 * 0.2 + 9 * 10.0) / 1e6)
    assert preds[0].latency_ms == 321.0
    assert preds[0].meta["cached_in"] == 1024


def test_listed_item_errors_are_recorded_and_others_stop_the_run() -> None:
    class Refused(Exception):
        pass

    def refuse(req: ModelRequest) -> str:
        raise Refused("content filter")

    preds = run_arm(
        ARMS["luna-zeroshot"],
        FakeClient(refuse),
        TEST_IDS,
        TEST_TEXTS,
        TEST_LABELS,
        item_errors=(Refused,),
        log=lambda _: None,
    )
    assert all(p.error and p.pred == NO_PREDICTION for p in preds)
    with pytest.raises(RuntimeError, match="2 requests in a row failed"):
        run_arm(
            ARMS["luna-zeroshot"],
            FakeClient(refuse),
            TEST_IDS,
            TEST_TEXTS,
            TEST_LABELS,
            item_errors=(Refused,),
            max_consecutive_errors=2,
            log=lambda _: None,
        )
    with pytest.raises(Refused):
        run_arm(ARMS["luna-zeroshot"], FakeClient(refuse), TEST_IDS, TEST_TEXTS, TEST_LABELS)


def test_dollar_cap_stops_the_run() -> None:
    capped = DollarCap(FakeClient(lambda req: '{"label": "card_arrival"}'), cap_usd=0.0001)
    with pytest.raises(BudgetExceeded):
        run_arm(
            ARMS["sol-fewshot"],
            capped,
            TEST_IDS,
            TEST_TEXTS,
            TEST_LABELS,
            neighbors=NEIGHBORS,
            train_texts=TRAIN_TEXTS,
            train_labels=TRAIN_LABELS,
        )


def test_cached_run_replays_without_a_model(tmp_path: Path) -> None:
    kwargs = {
        "neighbors": NEIGHBORS,
        "train_texts": TRAIN_TEXTS,
        "train_labels": TRAIN_LABELS,
        "log": lambda _: None,
    }
    arm = ARMS["luna-fewshot"]
    live = CachedClient(_nearest_label_client(), tmp_path)
    first = run_arm(arm, live, TEST_IDS, TEST_TEXTS, TEST_LABELS, **kwargs)
    replay = CachedClient(None, tmp_path, replay_only=True)
    second = run_arm(arm, replay, TEST_IDS, TEST_TEXTS, TEST_LABELS, **kwargs)
    assert [p.pred for p in first] == [p.pred for p in second]
    assert [p.cost_usd for p in first] == [p.cost_usd for p in second]
    with pytest.raises(CacheMiss):
        run_arm(ARMS["luna-zeroshot"], replay, TEST_IDS, TEST_TEXTS, TEST_LABELS)


def test_pacer_waits_when_the_minute_budget_is_used_up() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    inner = FakeClient(
        lambda req: ModelResponse(text="{}", model=req.model, input_tokens=400, output_tokens=5)
    )
    pacer = Pacer(inner, tokens_per_minute=900, fraction=1.0, clock=lambda: now[0], sleep=sleep)
    req = ModelRequest(model="gpt-6-luna", input="x" * 300, max_output_tokens=32)
    pacer.complete(req)
    pacer.complete(req)  # 405 used + about 150 estimated fits under 900
    assert slept == []
    pacer.complete(req)  # 810 used, the next estimate does not fit: wait for the window
    assert len(slept) == 1
    assert now[0] >= 60.0
