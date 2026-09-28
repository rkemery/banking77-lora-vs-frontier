"""Classifier fine-tuning on CPU: LoRA on Qwen3-0.6B-Base, full fine-tune of ModernBERT-base.

A plain PyTorch loop rather than the transformers Trainer, so every setting
that matters on CPU is visible here:

- bf16 autocast for matmuls (the container's Xeon has AMX, which made bf16
  the fastest option measured). LoRA keeps the frozen base weights in bf16
  and the trainable adapter and head in fp32. The full fine-tune keeps fp32
  master weights.
- Length-grouped batches: shuffle, cut into chunks of 50 batches, sort each
  chunk by length, then shuffle the batches. Banking77 messages run from 2 to
  over 100 tokens, and padding a batch to its longest message wasted about
  half the compute in the first timing run.
- AdamW, linear warm-up over the first 6% of steps then linear decay to 0,
  gradient clipping at 1.0.

LoRA follows "LoRA Without Regret" (Schulman and Thinking Machines Lab, 2025):
adapters on every linear layer including the MLP, alpha 32, rank 16, and a
learning rate about 10x what full fine-tuning would use. The classification
head (`score`) is trained in full, which PEFT does for task_type SEQ_CLS.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from b77.data import LABEL_NAMES, N_CLASSES

ARTIFACTS = Path("artifacts/models")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    method: Literal["lora", "full"]
    attn_implementation: str


MODELS: dict[str, ModelSpec] = {
    "qwen3-0.6b": ModelSpec(
        "qwen3-0.6b",
        "Qwen/Qwen3-0.6B-Base",
        "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "lora",
        "sdpa",
    ),
    # Eager attention trained 26% faster than sdpa for ModernBERT on this CPU.
    "modernbert-base": ModelSpec(
        "modernbert-base",
        "answerdotai/ModernBERT-base",
        "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "full",
        "eager",
    ),
}


@dataclass(frozen=True)
class TrainConfig:
    model: str
    lr: float
    epochs: float
    batch_size: int = 16
    warmup_ratio: float = 0.06
    weight_decay: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    max_length: int = 128
    seed: int = 0
    grad_clip: float = 1.0
    max_steps: int | None = None  # stop early (smoke and timing runs)
    eval_batch_size: int = 64

    @property
    def spec(self) -> ModelSpec:
        return MODELS[self.model]


@dataclass
class TrainResult:
    config: dict[str, Any]
    steps: int
    examples_seen: int
    train_seconds: float
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def examples_per_second(self) -> float:
        return self.examples_seen / self.train_seconds if self.train_seconds else 0.0


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
    if tokenizer.pad_token_id is None:
        raise ValueError(f"{spec.hf_id} has no pad token")
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(cfg: TrainConfig, tokenizer: Any) -> Any:
    """The classification model ready to train: LoRA-wrapped Qwen or full ModernBERT."""
    import torch
    from transformers import AutoModelForSequenceClassification

    spec = cfg.spec
    torch.manual_seed(cfg.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        num_labels=N_CLASSES,
        id2label=dict(enumerate(LABEL_NAMES)),
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
        dtype=torch.bfloat16 if spec.method == "lora" else torch.float32,
        attn_implementation=spec.attn_implementation,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model if spec.method == "full" else apply_lora(model, cfg)


def apply_lora(model: Any, cfg: TrainConfig) -> Any:
    """Wrap a classifier with LoRA on every linear layer. The head (`score`) trains in full."""
    from peft import LoraConfig, TaskType, get_peft_model

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    # Frozen base stays bf16. Adapters and the head train in fp32.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model


def load_trained(run_dir: Path, model_key: str) -> tuple[Any, Any]:
    """Load a model saved by `save_model` for inference."""
    import torch
    from transformers import AutoModelForSequenceClassification

    spec = MODELS[model_key]
    tokenizer = load_tokenizer(spec)
    if spec.method == "full":
        model = AutoModelForSequenceClassification.from_pretrained(
            run_dir, dtype=torch.bfloat16, attn_implementation=spec.attn_implementation
        )
        return model.eval(), tokenizer
    base = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        num_labels=N_CLASSES,
        dtype=torch.bfloat16,
        attn_implementation=spec.attn_implementation,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    return attach_adapter(base, run_dir), tokenizer


def attach_adapter(base: Any, run_dir: Path) -> Any:
    """Put a saved LoRA adapter and its classification head back on a fresh base model.

    Inference runs under bf16 autocast, which rounds every matmul input to bf16
    whether the stored weight is fp32 or bf16, so the adapter dtype after
    loading does not change the predictions.
    """
    from peft import PeftModel

    return PeftModel.from_pretrained(base, run_dir).eval()


def save_model(model: Any, tokenizer: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def tokenize(tokenizer: Any, texts: Sequence[str], max_length: int) -> list[list[int]]:
    return tokenizer(list(texts), truncation=True, max_length=max_length)["input_ids"]


def length_grouped_batches(
    lengths: Sequence[int], batch_size: int, rng: np.random.Generator, chunk_batches: int = 50
) -> list[np.ndarray]:
    """One epoch of batches of similar length, in random order. Every index appears once."""
    order = rng.permutation(len(lengths))
    lengths_arr = np.asarray(lengths)
    batches: list[np.ndarray] = []
    chunk = batch_size * chunk_batches
    for start in range(0, order.size, chunk):
        part = order[start : start + chunk]
        part = part[np.argsort(lengths_arr[part], kind="stable")]
        batches.extend(part[i : i + batch_size] for i in range(0, part.size, batch_size))
    return [batches[i] for i in rng.permutation(len(batches))]


def collate(ids: Sequence[list[int]], pad_id: int) -> dict[str, Any]:
    import torch

    width = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.long)
    for row, seq in enumerate(ids):
        input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, : len(seq)] = 1
    return {"input_ids": input_ids, "attention_mask": mask}


def total_steps(cfg: TrainConfig, n_train: int) -> int:
    steps = math.ceil(cfg.epochs * math.ceil(n_train / cfg.batch_size))
    return min(steps, cfg.max_steps) if cfg.max_steps else steps


def train(
    cfg: TrainConfig,
    texts: Sequence[str],
    labels: np.ndarray,
    *,
    dev: tuple[Sequence[str], np.ndarray] | None = None,
    eval_every_epoch: bool = True,
    log_every: int = 25,
    model_and_tokenizer: tuple[Any, Any] | None = None,
) -> tuple[Any, Any, TrainResult]:
    """Train and return (model, tokenizer, result). Dev accuracy is logged after each epoch.

    `model_and_tokenizer` replaces the Hugging Face download, which lets the
    tests run the real loop on a tiny randomly initialised model.
    """
    import torch
    from transformers import get_linear_schedule_with_warmup

    spec = cfg.spec
    if model_and_tokenizer is None:
        tokenizer = load_tokenizer(spec)
        model = build_model(cfg, tokenizer)
    else:
        model, tokenizer = model_and_tokenizer
    encoded = tokenize(tokenizer, texts, cfg.max_length)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    steps = total_steps(cfg, len(encoded))
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (decay if param.ndim >= 2 and "norm" not in name.lower() else no_decay).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, round(cfg.warmup_ratio * steps), steps)
    rng = np.random.default_rng(cfg.seed)
    result = TrainResult(config=asdict(cfg), steps=0, examples_seen=0, train_seconds=0.0)
    lengths = [len(x) for x in encoded]
    log(f"[{spec.key}] {len(encoded)} examples, {steps} steps, lr {cfg.lr}, bs {cfg.batch_size}")
    model.train()
    epoch = 0
    running, since_log = 0.0, 0
    while result.steps < steps:
        epoch += 1
        for batch_idx in length_grouped_batches(lengths, cfg.batch_size, rng):
            start = time.perf_counter()
            batch = collate([encoded[i] for i in batch_idx], tokenizer.pad_token_id)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model(**batch, labels=y[batch_idx])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            result.train_seconds += time.perf_counter() - start
            result.steps += 1
            result.examples_seen += len(batch_idx)
            running += float(out.loss.detach())
            since_log += 1
            if result.steps % log_every == 0 or result.steps == steps:
                rate = result.examples_per_second
                eta = (steps - result.steps) * cfg.batch_size / rate if rate else float("nan")
                log(
                    f"[{spec.key}] step {result.steps}/{steps} epoch {epoch} "
                    f"loss {running / since_log:.4f} {rate:.1f} ex/s eta {eta / 60:.1f} min"
                )
                result.history.append(
                    {"step": result.steps, "epoch": epoch, "loss": running / since_log}
                )
                running, since_log = 0.0, 0
            if result.steps >= steps:
                break
        if dev is not None and (eval_every_epoch or result.steps >= steps):
            probs, _ = predict(model, tokenizer, dev[0], cfg.max_length, cfg.eval_batch_size)
            acc = float(np.mean(probs.argmax(axis=1) == np.asarray(dev[1])))
            result.history.append({"step": result.steps, "epoch": epoch, "dev_accuracy": acc})
            log(f"[{spec.key}] epoch {epoch} dev accuracy {acc:.4f}")
            model.train()
    model.eval()
    return model, tokenizer, result


def predict(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int = 128,
    batch_size: int = 64,
    progress: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Class probabilities (n, 77). With batch_size=1 also wall-clock ms per item.

    The per-item time covers tokenisation, the forward pass and the softmax,
    which is what a single request to a CPU service would wait for.
    """
    import torch

    model.eval()
    probs = np.empty((len(texts), N_CLASSES), dtype=np.float32)
    latency = np.empty(len(texts)) if batch_size == 1 else None
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        if batch_size == 1 and texts:  # warm-up, not timed
            model(**tokenizer([texts[0]], return_tensors="pt"))
        for start in range(0, len(texts), batch_size):
            t0 = time.perf_counter()
            chunk = list(texts[start : start + batch_size])
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            p = torch.softmax(model(**enc).logits.float(), dim=-1).numpy()
            if latency is not None:
                latency[start] = (time.perf_counter() - t0) * 1000.0
            probs[start : start + len(chunk)] = p
            if progress is not None:
                progress(start + len(chunk))
    return probs, latency


def iter_progress(total: int, every: int, label: str) -> Callable[[int], None]:
    marks: Iterator[int] = iter(range(every, total + every, every))
    state = {"next": next(marks)}

    def report(done: int) -> None:
        if done >= state["next"] or done == total:
            log(f"[{label}] predicted {done}/{total}")
            state["next"] = next(marks, total + 1)

    return report


def save_result(result: TrainResult, path: Path, extra: dict[str, Any] | None = None) -> None:
    payload = {**asdict(result), "examples_per_second": result.examples_per_second, **(extra or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
