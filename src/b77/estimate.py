"""Offline estimates of tokens, dollars and wall-clock time for the prompting arms.

Token counts use the Qwen3 tokenizer as a stand-in when it is installed (a
151k-entry byte-level BPE, close to the o200k tokenizer on English) and
UTF-8 bytes / 4 otherwise. How the Responses API renders the JSON schema into
tokens is not public, so the schema counts as its JSON text. Treat the result
as +/- 25%.

Wall-clock assumes the pacer keeps usage at 85% of the deployment's
tokens-per-minute limit and that Azure counts the full prompt (cached or not)
plus max_output_tokens against it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import numpy as np
from llm_eval_harness.client import DEFAULT_PRICES, max_cost_usd

from b77.data import Split
from b77.prompting import (
    ARMS,
    INSTRUCTIONS,
    MAX_OUTPUT_TOKENS,
    RESPONSE_FORMAT,
    Neighbors,
    build_request,
    examples_for,
    user_turn,
)

OUTPUT_TOKENS = 14  # {"label":"card_arrival"} is about 8; the longest label about 18
CACHE_MIN_PREFIX = 1024  # OpenAI prompt caching starts at 1,024 identical prefix tokens


def token_counter() -> tuple[Callable[[str], int], str]:
    try:
        from transformers import AutoTokenizer

        from b77.train import MODELS

        spec = MODELS["qwen3-0.6b"]
        tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
        return (lambda s: len(tok(s)["input_ids"])), "Qwen3 tokenizer as a proxy"
    except (ImportError, OSError):
        return (lambda s: len(s.encode("utf-8")) // 4), "UTF-8 bytes / 4"


def estimate_all(train: Split, test: Split, sample: int = 300) -> str:
    count, how = token_counter()
    prefix = count(INSTRUCTIONS) + count(json.dumps(RESPONSE_FORMAT))
    rows = [
        f"Computed {time.strftime('%Y-%m-%d')} by `make estimate`. Token counts: {how}. "
        f"Static prefix (instructions + schema): about {prefix:,} tokens, "
        + ("enough to be cached." if prefix >= CACHE_MIN_PREFIX else "below the cache minimum."),
        "",
        "| Arm | Tokens in per call | Cost, no cache hits | Cost, prefix cached "
        "| Worst case per call (DollarCap) | Default cap | Wall clock at the TPM limit |",
        "|---|---|---|---|---|---|---|",
    ]
    rng = np.random.default_rng(0)
    rows_idx = rng.choice(len(test), size=min(sample, len(test)), replace=False)
    neighbors = Neighbors.load()
    for arm in ARMS.values():
        dyn, worst = [], []
        for r in rows_idx:
            ex = (
                examples_for(int(r), neighbors, train.texts, train.labels, 20)
                if arm.few_shot
                else []
            )
            dyn.append(count(user_turn(test.texts[r], ex)))
            worst.append(
                max_cost_usd(DEFAULT_PRICES[arm.model], build_request(arm, test.texts[r], ex))
            )
        tokens_in = prefix + float(np.mean(dyn))
        price = DEFAULT_PRICES[arm.model]
        cached_rate = price.cached_input_per_m or price.input_per_m
        n = len(test)
        out_cost = OUTPUT_TOKENS * price.output_per_m / 1e6
        no_cache = n * (tokens_in * price.input_per_m / 1e6 + out_cost)
        cached = n * (
            prefix * cached_rate / 1e6 + (tokens_in - prefix) * price.input_per_m / 1e6 + out_cost
        )
        per_minute = 0.85 * arm.tokens_per_minute / (tokens_in + MAX_OUTPUT_TOKENS)
        rows.append(
            f"| {arm.key} ({arm.model}) | {tokens_in:,.0f} | ${no_cache:.2f} | ${cached:.2f} "
            f"| ${float(np.mean(worst)):.4f} | ${arm.default_cap_usd:.2f} "
            f"| {n / per_minute / 60:.1f} h at {arm.tokens_per_minute:,} TPM |"
        )
    return "\n".join(rows)
