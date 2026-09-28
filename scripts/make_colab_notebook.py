"""Write notebooks/qwen3_8b_qlora_colab.ipynb from the cell sources below.

The notebook is generated so its code stays reviewable as plain Python in
this file. Run `uv run python scripts/make_colab_notebook.py` after editing.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("notebooks/qwen3_8b_qlora_colab.ipynb")

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Qwen3-8B-Base QLoRA on Banking77 (Colab T4, optional, not run)

**Status: not run.** This notebook is provided for anyone with a free Colab T4 who wants the
8B point. The repo reports no results from it.

Same recipe as the CPU run in this repo, scaled up: a sequence-classification head, LoRA
rank 16 and alpha 32 on every linear layer, trained on all 10,003 training messages and
scored on the 3,080 test messages. Differences, all forced by the T4:

- The base model is loaded in 4-bit NF4 with double quantisation (QLoRA, Dettmers et al.
  2023, arXiv 2305.14314) through bitsandbytes.
- fp16 autocast with a gradient scaler, because the T4 has no bf16.
- Gradient checkpointing to fit in 15 GB.

Plain transformers + PEFT + bitsandbytes. Unsloth is not used because it has no
sequence-classification head support.

Output: `qwen3-8b-qlora-r16-s0.jsonl`, one record per test item in the llm-eval-harness
results format. Copy it to `results/runs/` in the repo and run `make demo` to add the row.
Expect about 1.5 to 3 hours on a T4 (an estimate, not measured).""",
    ),
    (
        "code",
        """# Runtime > Change runtime type > T4 GPU, then run this cell.
!pip -q install "transformers>=5.17" "peft>=0.21" "bitsandbytes>=0.50" "accelerate>=1.15" pyarrow
!git clone -q https://github.com/rkemery/banking77-lora-vs-frontier.git repo""",
    ),
    (
        "code",
        """import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

assert torch.cuda.is_available(), "Select a GPU runtime first"
print(torch.cuda.get_device_name(0))

REPO = Path("repo")
DATASET = "legacy-datasets/banking77"
DATA_REV = "f54121560de48f2852f90be299010d1d6dc612ec"
MODEL = "Qwen/Qwen3-8B-Base"
MODEL_REV = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
RUN_ID = "qwen3-8b-qlora-r16-s0"
SEED = 0
EPOCHS = 2
BATCH = 16
MAX_LENGTH = 128

sweep = REPO / "results" / "sweep.json"
LR = json.loads(sweep.read_text())["chosen_lr"] if sweep.exists() else 2e-4
print("learning rate", LR)""",
    ),
    (
        "code",
        """def load(split):
    name = f"data/{split}-00000-of-00001.parquet"
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REV}/{name}"
    path = Path(f"{split}.parquet")
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    table = pq.read_table(path)
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["label"]
    return table.column("text").to_pylist(), np.array(table.column("label").to_pylist()), names


train_texts, train_labels, label_info = load("train")
test_texts, test_labels, _ = load("test")
LABELS = label_info["names"]
assert (len(train_texts), len(test_texts), len(LABELS)) == (10003, 3080, 77)""",
    ),
    (
        "code",
        """from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

torch.manual_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
tokenizer.padding_side = "right"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    revision=MODEL_REV,
    num_labels=77,
    quantization_config=bnb,
    dtype=torch.float16,
    device_map={"": 0},
)
model.config.pad_token_id = tokenizer.pad_token_id
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear"
)
model = get_peft_model(model, lora)
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()
model.print_trainable_parameters()""",
    ),
    (
        "code",
        """encoded = tokenizer(train_texts, truncation=True, max_length=MAX_LENGTH)["input_ids"]
labels_t = torch.tensor(train_labels)
steps = EPOCHS * math.ceil(len(encoded) / BATCH)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
scheduler = get_linear_schedule_with_warmup(optimizer, round(0.06 * steps), steps)
scaler = torch.amp.GradScaler("cuda")
rng = np.random.default_rng(SEED)


def collate(rows):
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), tokenizer.pad_token_id)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    return ids.cuda(), mask.cuda()


model.train()
step, start = 0, time.time()
for epoch in range(EPOCHS):
    order = rng.permutation(len(encoded))
    for s in range(0, len(order), BATCH):
        idx = order[s : s + BATCH]
        ids, mask = collate([encoded[i] for i in idx])
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels_t[idx].cuda()).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            minutes = (time.time() - start) / 60
            print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} {minutes:.1f} min")
train_seconds = time.time() - start
model.save_pretrained(RUN_ID)""",
    ),
    (
        "code",
        """model.eval()
probs = []
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    for s in range(0, len(test_texts), 32):
        enc = tokenizer(
            test_texts[s : s + 32],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to("cuda")
        probs.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
probs = np.concatenate(probs)
pred = probs.argmax(axis=1)
print("test accuracy", float(np.mean(pred == test_labels)))

with Path(f"{RUN_ID}.jsonl").open("w") as fh:
    for i in range(len(test_texts)):
        record = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "item_id": f"test-{i:05d}",
            "config": "qwen3-8b-qlora",
            "model": MODEL,
            "scores": {"correct": bool(pred[i] == test_labels[i])},
            "cluster": None,
            "tokens_in": 0,
            "tokens_out": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "error": None,
            "score_error": None,
            "meta": {
                "gold": int(test_labels[i]),
                "pred": int(pred[i]),
                "conf": round(float(probs[i].max()), 5),
            },
        }
        fh.write(json.dumps(record) + "\\n")
info = {
    "run_id": RUN_ID,
    "arm": "Qwen3-8B-Base QLoRA r16 on a Colab T4",
    "train_seconds": train_seconds,
    "gpu": torch.cuda.get_device_name(0),
    "latency": "not measured (GPU, batched)",
}
Path(f"{RUN_ID}.info.json").write_text(json.dumps(info, indent=2))
print("wrote", f"{RUN_ID}.jsonl")""",
    ),
]


def notebook() -> dict:
    cells = []
    for kind, source in CELLS:
        cell = {
            "cell_type": kind,
            "id": f"cell-{len(cells)}",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
