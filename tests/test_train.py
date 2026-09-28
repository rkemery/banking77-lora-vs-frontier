"""The training loop and LoRA wiring on a tiny randomly initialised Qwen3, offline."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from b77.train import (  # noqa: E402
    TrainConfig,
    apply_lora,
    collate,
    length_grouped_batches,
    predict,
    total_steps,
    train,
)


class TinyTokenizer:
    """Characters to ids, enough of the Hugging Face tokenizer interface for the loop."""

    pad_token_id = 0

    def __call__(self, texts, truncation=True, max_length=128, padding=False, return_tensors=None):
        ids = [[1 + ord(c) % 60 for c in t][:max_length] for t in texts]
        if return_tensors != "pt":
            return {"input_ids": ids}
        return collate(ids, self.pad_token_id)


def tiny_qwen(num_labels: int = 77):
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        num_labels=num_labels,
    )
    torch.manual_seed(0)
    return transformers.Qwen3ForSequenceClassification(config).to(torch.bfloat16)


def test_length_grouped_batches_cover_every_index_once() -> None:
    lengths = list(np.random.default_rng(0).integers(2, 60, size=1000))
    batches = length_grouped_batches(lengths, 16, np.random.default_rng(1))
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(range(1000))
    # Grouping keeps most batches narrow compared with random batching.
    spread = np.median([np.ptp(np.take(lengths, b)) for b in batches])
    assert spread < 10


def test_collate_pads_on_the_right() -> None:
    batch = collate([[5, 6, 7], [8]], pad_id=0)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_total_steps_respects_max_steps() -> None:
    cfg = TrainConfig(model="qwen3-0.6b", lr=1e-4, epochs=2, batch_size=16)
    assert total_steps(cfg, 100) == 14
    assert total_steps(TrainConfig("qwen3-0.6b", 1e-4, 2, max_steps=5), 100) == 5


def test_lora_wraps_every_linear_layer_and_trains_the_head_in_fp32() -> None:
    cfg = TrainConfig(model="qwen3-0.6b", lr=1e-3, epochs=1)
    model = apply_lora(tiny_qwen(), cfg)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert any(f"{proj}.lora_A" in n for n in trainable), proj
    assert any("score" in n for n in trainable)
    assert all(p.dtype == torch.float32 for p in trainable.values())
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert all(p.dtype == torch.bfloat16 for p in frozen)


def test_the_loop_learns_a_separable_toy_task() -> None:
    texts = ["aaaa bbb" * (1 + i % 3) for i in range(48)] + [
        "xyz qq" * (1 + i % 3) for i in range(48)
    ]
    labels = np.array([3] * 48 + [40] * 48)
    cfg = TrainConfig(model="qwen3-0.6b", lr=5e-3, epochs=4, batch_size=8, seed=0)
    tokenizer = TinyTokenizer()
    model = apply_lora(tiny_qwen(), cfg)
    model, _, result = train(
        cfg, texts, labels, dev=(texts, labels), model_and_tokenizer=(model, tokenizer), log_every=6
    )
    assert result.steps == 48
    losses = [h["loss"] for h in result.history if "loss" in h]
    assert losses[-1] < losses[0]
    assert result.history[-1]["dev_accuracy"] == 1.0
    probs, latency = predict(model, tokenizer, texts[:3], batch_size=1)
    assert probs.shape == (3, 77)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    assert latency is not None
    assert latency.shape == (3,)
