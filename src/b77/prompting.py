"""Prompting arms: gpt-6-luna zero-shot, gpt-6-luna and gpt-6-sol with retrieved few-shot examples.

Every arm shares one static prefix: the instructions and the 77 intents with
one-line descriptions. The few-shot arms add the 20 most similar training
messages (bge-small cosine, most similar last, next to the query) in the user
turn, after the prefix, so the provider's prompt cache can serve the prefix.
Structured outputs with a JSON schema enum force the reply to be one of the 77
label names. Reasoning effort is `none` and `max_output_tokens` is 32, which
is about twice the longest label's JSON.

Calls go through the harness clients, outermost first:
`CachedClient(RetryingClient(Pacer(DollarCap(FoundryClient(), cap))))`.
The disk cache makes a rerun free, the pacer keeps requests under the
deployment's tokens-per-minute limit so 429s stay rare (each failed attempt is
charged its worst case against the cap), and `DollarCap` refuses any call that
could take spend past the cap.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from llm_eval_harness import (
    BudgetExceeded,
    CacheMiss,
    ModelClient,
    ModelRequest,
    ModelResponse,
)
from llm_eval_harness.client import DEFAULT_PRICES, cost_usd

from b77.data import LABEL_NAMES
from b77.labels import label_block
from b77.runs import NO_PREDICTION, Prediction

MAX_OUTPUT_TOKENS = 32
FEW_SHOT_K = 20
CACHE_DIR = Path("cache/llm")
NEIGHBORS_PATH = Path("data/splits/test_neighbors_bge-small.json")

INSTRUCTIONS = (
    "You classify messages that customers send to a bank's support chat. "
    "Each message has exactly one intent from the list below. "
    'Reply with JSON of the form {"label": "<intent>"}, using an intent name '
    "exactly as written. If several intents seem to fit, choose the most specific one.\n\n"
    "Intents (name: what it covers):\n" + label_block(with_descriptions=True)
)

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "banking77_intent",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": list(LABEL_NAMES)}},
        "required": ["label"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Arm:
    key: str
    run_id: str
    model: str
    few_shot: bool
    default_cap_usd: float
    tokens_per_minute: int  # the deployment's capacity, from the build plan


ARMS: dict[str, Arm] = {
    "luna-zeroshot": Arm("luna-zeroshot", "gpt-6-luna-zeroshot", "gpt-6-luna", False, 2.0, 20_000),
    "luna-fewshot": Arm("luna-fewshot", "gpt-6-luna-fewshot-k20", "gpt-6-luna", True, 2.0, 20_000),
    "sol-fewshot": Arm("sol-fewshot", "gpt-6-sol-fewshot-k20", "gpt-6-sol", True, 20.0, 10_000),
}


def user_turn(text: str, examples: Sequence[tuple[str, str]] = ()) -> str:
    """The dynamic part of the prompt. `examples` are (message, label), least similar first."""
    if not examples:
        return f"Message: {text}"
    shots = "\n\n".join(f"Message: {m}\nLabel: {label}" for m, label in examples)
    return (
        "Labeled messages from the training set, most similar to the new message last:\n\n"
        f"{shots}\n\nNow classify this message.\nMessage: {text}"
    )


def build_request(arm: Arm, text: str, examples: Sequence[tuple[str, str]] = ()) -> ModelRequest:
    if arm.few_shot and not examples:
        raise ValueError(f"arm {arm.key} needs retrieved examples")
    if not arm.few_shot and examples:
        raise ValueError(f"arm {arm.key} is zero-shot and takes no examples")
    return ModelRequest(
        model=arm.model,
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": user_turn(text, examples)}],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort="none",
        extra={"text": {"format": RESPONSE_FORMAT}},
    )


def parse_label(reply: str) -> int:
    """The label id in a reply, or NO_PREDICTION if the reply is not one of the 77 names."""
    try:
        value = json.loads(reply)
    except json.JSONDecodeError:
        return NO_PREDICTION
    if not isinstance(value, dict) or not isinstance(value.get("label"), str):
        return NO_PREDICTION
    try:
        return LABEL_NAMES.index(value["label"])
    except ValueError:
        return NO_PREDICTION


class Pacer:
    """Keeps the tokens sent in any 60-second window under a tokens-per-minute budget.

    Before a call it reserves the request's estimated tokens (UTF-8 bytes / 3,
    plus max_output_tokens, which Azure also counts when it rate limits), and
    sleeps until the reservation fits under `fraction` of the budget. After the
    call it replaces the estimate with the billed total from `usage`. Sits
    inside the retry wrapper, so every attempt is paced.
    """

    def __init__(
        self,
        inner: ModelClient,
        tokens_per_minute: int,
        *,
        fraction: float = 0.85,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
        self._inner = inner
        self.budget = tokens_per_minute * fraction
        self._clock = clock
        self._sleep = sleep
        self._window: deque[tuple[float, int]] = deque()
        self.waited_s = 0.0

    def _used(self, now: float) -> int:
        while self._window and now - self._window[0][0] >= 60.0:
            self._window.popleft()
        return sum(tokens for _, tokens in self._window)

    def complete(self, request: ModelRequest) -> ModelResponse:
        estimate = estimated_tokens(request)
        now = self._clock()
        while self._window and self._used(now) + estimate > self.budget:
            wait = 60.0 - (now - self._window[0][0]) + 0.05
            self._sleep(wait)
            self.waited_s += wait
            now = self._clock()
        self._window.append((now, estimate))
        # A failed call keeps its estimate in the window: the provider may have counted it.
        response = self._inner.complete(request)
        self._window[-1] = (now, response.input_tokens + response.output_tokens)
        return response


def estimated_tokens(request: ModelRequest) -> int:
    sent = json.dumps(
        {"instructions": request.instructions, "input": request.input, "extra": request.extra},
        ensure_ascii=False,
    )
    return len(sent.encode("utf-8")) // 3 + (request.max_output_tokens or 0)


@dataclass(frozen=True)
class Neighbors:
    """For each test item, its 20 nearest training items by bge-small cosine, most similar first."""

    test_ids: list[str]
    train_idx: np.ndarray  # (n_test, k) row indices into train
    sims: np.ndarray  # (n_test, k)

    @classmethod
    def load(cls, path: Path = NEIGHBORS_PATH) -> Neighbors:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            test_ids=[r["id"] for r in rows["items"]],
            train_idx=np.array([r["train_idx"] for r in rows["items"]], dtype=np.int64),
            sims=np.array([r["sims"] for r in rows["items"]], dtype=np.float64),
        )

    def save(self, path: Path = NEIGHBORS_PATH, meta: dict[str, Any] | None = None) -> None:
        items = [
            {"id": i, "train_idx": idx.tolist(), "sims": [round(float(s), 4) for s in sim]}
            for i, idx, sim in zip(self.test_ids, self.train_idx, self.sims, strict=True)
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**(meta or {}), "items": items}, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


def examples_for(
    row: int, neighbors: Neighbors, train_texts: Sequence[str], train_labels: np.ndarray, k: int
) -> list[tuple[str, str]]:
    """The k nearest training examples for one test row, least similar first."""
    idx = neighbors.train_idx[row, :k][::-1]
    return [(train_texts[i], LABEL_NAMES[int(train_labels[i])]) for i in idx]


def run_arm(
    arm: Arm,
    client: ModelClient,
    test_ids: Sequence[str],
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    *,
    neighbors: Neighbors | None = None,
    train_texts: Sequence[str] = (),
    train_labels: np.ndarray | None = None,
    k: int = FEW_SHOT_K,
    item_errors: tuple[type[Exception], ...] = (),
    max_consecutive_errors: int = 10,
    progress_every: int = 100,
    log: Callable[[str], None] = print,
) -> list[Prediction]:
    """Classify every test item.

    An exception whose type is in `item_errors` (for a live run, the OpenAI SDK's
    `APIError` family, raised after retries are used up or for a refused
    request such as a content filter) is recorded on that item as `error`, and
    the run goes on, unless `max_consecutive_errors` items in a row fail, which
    points at a broken request (a rejected schema, a wrong deployment name) and
    stops the run before it burns the budget on failures. Anything else stops the
    run too, including `BudgetExceeded` and a `CacheMiss` in replay mode.
    """
    if arm.few_shot and (neighbors is None or train_labels is None):
        raise ValueError("few-shot arms need neighbors and the training set")
    if neighbors is not None and list(neighbors.test_ids[: len(test_ids)]) != list(test_ids):
        raise ValueError("neighbor file does not match the test items")
    price = DEFAULT_PRICES[arm.model]
    preds: list[Prediction] = []
    start = time.monotonic()
    consecutive = 0
    for row, (item_id, text, gold) in enumerate(
        zip(test_ids, test_texts, test_labels, strict=True)
    ):
        examples = (
            examples_for(row, neighbors, train_texts, train_labels, k)  # type: ignore[arg-type]
            if arm.few_shot
            else []
        )
        request = build_request(arm, text, examples)
        try:
            response = client.complete(request)
        except (BudgetExceeded, CacheMiss):
            raise
        except item_errors as exc:
            log(f"[{arm.key}] {item_id}: {type(exc).__name__}: {exc}")
            consecutive += 1
            if consecutive >= max_consecutive_errors:
                raise RuntimeError(
                    f"{consecutive} requests in a row failed, the last with {exc!r}. Stopping."
                ) from exc
            preds.append(
                Prediction(item_id, int(gold), NO_PREDICTION, error=f"{type(exc).__name__}: {exc}")
            )
            continue
        consecutive = 0
        pred = parse_label(response.text)
        meta: dict[str, Any] = {"cached_in": response.cached_input_tokens}
        if pred == NO_PREDICTION:
            meta["raw"] = response.text[:200]
        if response.finish_reason != "stop":
            meta["finish"] = response.finish_reason
        preds.append(
            Prediction(
                item_id=item_id,
                gold=int(gold),
                pred=pred,
                latency_ms=response.latency_ms,
                tokens_in=response.input_tokens,
                tokens_out=response.output_tokens,
                reasoning_tokens=response.reasoning_tokens,
                cost_usd=cost_usd(price, response),
                meta=meta,
            )
        )
        if (row + 1) % progress_every == 0:
            spent = sum(p.cost_usd for p in preds)
            log(
                f"[{arm.key}] {row + 1}/{len(test_ids)} items, ${spent:.4f} at list price, "
                f"{(time.monotonic() - start) / 60:.1f} min"
            )
    return preds
