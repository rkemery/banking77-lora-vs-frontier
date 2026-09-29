# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "accelerate>=1.15",
#     "bitsandbytes>=0.50",
#     "numpy>=1.26",
#     "peft>=0.21.0",
#     "pyarrow>=25.0",
#     "torch>=2.14.0",
#     "transformers>=5.17.0",
# ]
# ///
"""Qwen3-8B-Base QLoRA on Banking77, run as a Hugging Face Job on one A10G.

Same recipe as the CPU LoRA run (classification head, LoRA r16 alpha 32 on every
linear layer, all 10,003 training messages, 2 epochs, bf16), with the base model
in 4-bit NF4 (QLoRA, Dettmers et al. 2023, arXiv 2305.14314) so it fits in 24 GB.
Two differences from the CPU run: batches come in random order, not length-grouped,
and the parquet files come from the pinned revision without the sha256 check that
`b77.data` does.
Test predictions are made one message at a time, so latency is batch-1 like the
CPU rows. Writes to --out:

- `qwen3-8b-qlora-r16-s0.jsonl.gz` and `.info.json` in the results/runs format
- `adapter/`, the trained LoRA adapter and classification head

Launch with `scripts/launch_hf_job.py`. `--smoke` trains 20 steps and scores 64
test items, to check the job end to end for a few cents.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

DATASET = "legacy-datasets/banking77"
DATA_REV = "f54121560de48f2852f90be299010d1d6dc612ec"
MODEL = "Qwen/Qwen3-8B-Base"
MODEL_REV = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
RUN_ID = "qwen3-8b-qlora-r16-s0"
CONFIG = "qwen3-8b-qlora"
SEED = 0
EPOCHS = 2
BATCH = 16
MAX_LENGTH = 128
WARMUP = 0.06


def load(split: str, workdir: Path) -> tuple[list[str], np.ndarray]:
    name = f"data/{split}-00000-of-00001.parquet"
    path = workdir / f"{split}.parquet"
    if not path.exists():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REV}/{name}"
        urllib.request.urlretrieve(url, path)
    table = pq.read_table(path)
    return table.column("text").to_pylist(), np.array(table.column("label").to_pylist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True, help="the CPU sweep's chosen learning rate")
    ap.add_argument("--out", type=Path, default=Path("/output"))
    ap.add_argument("--usd-per-hour", type=float, required=True, help="the job flavor's list price")
    ap.add_argument("--flavor", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU flavor"
    args.out.mkdir(parents=True, exist_ok=True)
    workdir = Path("/tmp/b77")
    workdir.mkdir(exist_ok=True)
    train_texts, train_labels = load("train", workdir)
    test_texts, test_labels = load("test", workdir)
    assert (len(train_texts), len(test_texts)) == (10003, 3080)
    n_test = 64 if args.smoke else len(test_texts)

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    tokenizer.padding_side = "right"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        revision=MODEL_REV,
        num_labels=77,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    encoded = tokenizer(train_texts, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    labels_t = torch.tensor(train_labels)
    steps = 20 if args.smoke else EPOCHS * math.ceil(len(encoded) / BATCH)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, round(WARMUP * steps), steps)
    rng = np.random.default_rng(SEED)

    def collate(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
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
            if step >= steps:
                break
            idx = order[s : s + BATCH]
            ids, mask = collate([encoded[i] for i in idx])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels_t[idx].cuda()).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == steps:
                minutes = (time.time() - start) / 60
                print(
                    f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} {minutes:.1f} min",
                    flush=True,
                )
    torch.cuda.synchronize()
    train_seconds = time.time() - start
    model.save_pretrained(args.out / "adapter")

    model.eval()
    preds, confs, lat_ms = [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(n_test):
            t0 = time.perf_counter()
            enc = tokenizer(
                test_texts[i], truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            ).to("cuda")
            probs = torch.softmax(model(**enc).logits.float(), dim=-1)[0].cpu().numpy()
            lat_ms.append((time.perf_counter() - t0) * 1000)
            preds.append(int(probs.argmax()))
            confs.append(float(probs.max()))
            if (i + 1) % 500 == 0:
                print(f"test {i + 1}/{n_test}", flush=True)
    acc = float(np.mean(np.array(preds) == test_labels[:n_test]))
    print(f"test accuracy {acc:.4f} on {n_test} items", flush=True)

    with gzip.open(args.out / f"{RUN_ID}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for i in range(n_test):
            record = {
                "schema_version": 1,
                "run_id": RUN_ID,
                "item_id": f"test-{i:05d}",
                "config": CONFIG,
                "model": MODEL,
                "scores": {"correct": preds[i] == int(test_labels[i])},
                "cluster": None,
                "tokens_in": 0,
                "tokens_out": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": round(lat_ms[i], 3),
                "error": None,
                "score_error": None,
                "meta": {"gold": int(test_labels[i]), "pred": preds[i], "conf": round(confs[i], 5)},
            }
            fh.write(json.dumps(record) + "\n")
    info = {
        "run_id": RUN_ID,
        "config": CONFIG,
        "model": MODEL,
        "model_revision": MODEL_REV,
        "arm": "Qwen3-8B-Base, 4-bit NF4 QLoRA r16 alpha 32, all linear layers, class head",
        "train_size": len(encoded),
        "epochs": EPOCHS,
        "steps": steps,
        "batch_size": BATCH,
        "lr": args.lr,
        "seed": SEED,
        "train_seconds": round(train_seconds, 1),
        "trainable_params": trainable,
        "smoke": args.smoke,
        "latency": "tokenize plus one forward pass per message on the GPU, batch 1",
        "usd_per_hour": args.usd_per_hour,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "hf_jobs_flavor": args.flavor,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
    }
    (args.out / f"{RUN_ID}.info.json").write_text(json.dumps(info, indent=2) + "\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
