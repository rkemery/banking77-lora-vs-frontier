"""`b77` command line. One subcommand per pipeline step, wired together by the Makefile."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from b77 import plan
from b77.data import DATASET_ID, REVISION, Split, fetch, load_split, sha256_file
from b77.runs import RUNS_DIR, Prediction, write_run
from b77.splits import SPLITS_PATH, read_splits, select, train_minus_dev

TWINS_PATH = Path("data/splits/test_near_twins.jsonl")
SMOKE_DIR = Path("results/smoke")
SWEEP_PATH = Path("results/sweep.json")
FAKE_RUNS_DIR = Path("artifacts/fake-runs")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- data and splits


def cmd_data(_: argparse.Namespace) -> int:
    for split, path in fetch().items():
        print(f"{split}: {path} sha256 {sha256_file(path)} ok ({DATASET_ID}@{REVISION[:12]})")
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    from b77.dedup import dedup_summary, find_twins
    from b77.splits import build_splits, write_splits

    train, test = load_split("train"), load_split("test")
    twins = find_twins(train, test)
    splits = build_splits(train, dedup_summary(test, twins))
    twin_lines = "".join(
        json.dumps(
            {
                "test_id": t.test_id,
                "train_id": t.train_id,
                "similarity": t.similarity,
                "same_label": t.same_label,
            }
        )
        + "\n"
        for t in twins
    )
    if args.check:
        current = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
        same = current == splits and TWINS_PATH.read_text(encoding="utf-8") == twin_lines
        print("splits match the committed files" if same else "splits DIFFER from committed files")
        return 0 if same else 1
    write_splits(splits)
    TWINS_PATH.write_text(twin_lines, encoding="utf-8", newline="\n")
    d = splits["test_dedup"]
    print(
        f"dev {splits['dev']['size']}, sweep {splits['sweep']['size']}, dedup test keeps "
        f"{d['kept']} of {len(test)} ({d['same_label_twins']} same-label and "
        f"{d['different_label_twins']} different-label near twins)"
    )
    return 0


def load_all() -> tuple[Split, Split, dict[str, Any]]:
    return load_split("train"), load_split("test"), read_splits()


# ---------------------------------------------------------------- embeddings and baselines


def cmd_embed(args: argparse.Namespace) -> int:
    from b77.embed import BGE_ID, BGE_REVISION, Encoder, cached_embeddings, top_k
    from b77.prompting import FEW_SHOT_K, NEIGHBORS_PATH, Neighbors

    train, test, _ = load_all()
    encoder = Encoder()
    train_emb, _ = cached_embeddings(train, encoder)
    test_emb, latency = cached_embeddings(test, encoder, timed=True)
    assert latency is not None
    idx, sims = top_k(test_emb, train_emb, FEW_SHOT_K)
    print(
        f"embedded {len(train)} train and {len(test)} test items, "
        f"test p50 {np.percentile(latency, 50):.1f} ms per message"
    )
    if NEIGHBORS_PATH.exists() and not args.force:
        # The committed file is what the prompts were built from. Float rounding on another
        # machine can reorder near-ties, so compare instead of overwriting.
        committed = Neighbors.load(NEIGHBORS_PATH)
        same = float(np.mean(np.all(committed.train_idx == idx, axis=1)))
        print(f"{NEIGHBORS_PATH} kept. {same:.1%} of test items have identical top-20 lists here.")
        return 0
    Neighbors(test.ids, idx, sims).save(
        NEIGHBORS_PATH,
        meta={
            "encoder": f"{BGE_ID}@{BGE_REVISION}",
            "pooling": "CLS, L2-normalised, no instruction prefix",
            "keys": "all 10,003 train items (train_idx is the row in train)",
            "k": FEW_SHOT_K,
        },
    )
    print(f"neighbours written to {NEIGHBORS_PATH}")
    return 0


def _hardware_info() -> dict[str, Any]:
    from b77.timing import hardware

    return {**hardware(), "load_average_1m": round(os.getloadavg()[0], 2)}


def cmd_baselines(args: argparse.Namespace) -> int:
    from b77.baselines import knn_vote, logreg_run
    from b77.embed import cached_embeddings
    from b77.prompting import Neighbors

    train, test, splits = load_all()
    train_emb, _ = cached_embeddings(train)
    test_emb, embed_ms = cached_embeddings(test, timed=True)
    assert embed_ms is not None
    position = {item_id: i for i, item_id in enumerate(train.ids)}
    dev_idx = np.array([position[i] for i in splits["dev"]["ids"]])
    fit_idx = np.setdiff1d(np.arange(len(train)), dev_idx)
    hw = _hardware_info()

    def logreg(run_id: str, train_idx: np.ndarray, fit_part: np.ndarray, note: str) -> None:
        res = logreg_run(
            train_emb[fit_part],
            train.labels[fit_part],
            train_emb[dev_idx],
            train.labels[dev_idx],
            train_emb[train_idx],
            train.labels[train_idx],
            test_emb,
        )
        preds = [
            Prediction(
                item_id=test.ids[i],
                gold=int(test.labels[i]),
                pred=int(res.probs[i].argmax()),
                confidence=float(res.probs[i].max()),
                latency_ms=float(embed_ms[i] + res.predict_ms[i]),
            )
            for i in range(len(test))
        ]
        info = {
            "arm": "logistic regression on frozen bge-small embeddings",
            "train_size": int(train_idx.size),
            "c": res.c,
            "dev_accuracy_by_c": {str(k): v for k, v in res.dev_accuracy.items()},
            "train_seconds": res.fit_seconds,
            "latency": "bge-small embedding of one message plus predict_proba, batch 1",
            "note": note,
            "hardware": hw,
        }
        write_run(run_id, "logreg", "BAAI/bge-small-en-v1.5", preds, info)
        acc = float(np.mean(res.probs.argmax(axis=1) == test.labels))
        print(f"{run_id}: C={res.c:g} test accuracy {acc:.4f}")

    if not args.curve_only:
        logreg(plan.RUN_LOGREG, np.arange(len(train)), fit_idx, "C chosen on dev, refit on all")
        neighbors = Neighbors.load()
        if neighbors.test_ids != test.ids:
            raise ValueError("neighbour file does not match the test split. Run `make embed`.")
        pred, share = knn_vote(train.labels[neighbors.train_idx], neighbors.sims)
        search_ms = _time_search(test_emb, train_emb)
        preds = [
            Prediction(
                item_id=test.ids[i],
                gold=int(test.labels[i]),
                pred=int(pred[i]),
                confidence=float(share[i]),
                latency_ms=float(embed_ms[i] + search_ms),
            )
            for i in range(len(test))
        ]
        write_run(
            plan.RUN_KNN,
            "knn-k20",
            "BAAI/bge-small-en-v1.5",
            preds,
            {
                "arm": "similarity-weighted vote over the 20 retrieved neighbours",
                "train_size": len(train),
                "confidence": "winning share of the similarity weight (not a probability)",
                "latency": "bge-small embedding of one message plus brute-force search",
                "hardware": hw,
            },
        )
        print(f"{plan.RUN_KNN}: test accuracy {float(np.mean(pred == test.labels)):.4f}")
    curve = splits["learning_curve"]["subsets"]
    for seed_key, subsets in curve.items():
        seed = int(seed_key.removeprefix("seed"))
        for k, ids in subsets.items():
            idx = np.array([position[i] for i in ids])
            logreg(
                plan.run_logreg_curve(int(k), seed),
                idx,
                idx,
                f"learning curve, {k} per class, draw {seed}. C chosen on dev with this subset.",
            )
    return 0


def _time_search(queries: np.ndarray, keys: np.ndarray, n: int = 200) -> float:
    import time

    from b77.embed import top_k

    start = time.perf_counter()
    for i in range(n):
        top_k(queries[i : i + 1], keys, 20)
    return (time.perf_counter() - start) * 1000.0 / n


# ---------------------------------------------------------------- training


def _train_cfg(model: str, lr: float, epochs: float, seed: int, max_steps: int | None = None):
    from b77.train import TrainConfig

    return TrainConfig(
        model=model,
        lr=lr,
        epochs=epochs,
        batch_size=plan.BATCH_SIZE[model],
        weight_decay=plan.WEIGHT_DECAY[model],
        seed=seed,
        max_steps=max_steps,
    )


def cmd_timing(args: argparse.Namespace) -> int:
    from b77.timing import measure, write_timing

    train, test, _ = load_all()
    result = measure(train.texts, train.labels, test.texts, steps=args.steps)
    write_timing(result)
    print(json.dumps(result["estimates_minutes"], indent=2))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from b77.train import train

    train_split, _, splits = load_all()
    pool = train_minus_dev(train_split, splits)
    dev = select(train_split, splits["dev"]["ids"], "dev")
    cfg = _train_cfg(args.model, plan.DEFAULT_LR[args.model], 1, 0, max_steps=args.steps)
    _, _, result = train(cfg, pool.texts, pool.labels, dev=(dev.texts, dev.labels))
    dev_acc = next(h["dev_accuracy"] for h in reversed(result.history) if "dev_accuracy" in h)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": cfg.spec.hf_id,
        "revision": cfg.spec.revision,
        "config": result.config,
        "steps": result.steps,
        "train_size": len(pool),
        "examples_seen": result.examples_seen,
        "train_seconds": round(result.train_seconds, 1),
        "examples_per_second": round(result.examples_per_second, 2),
        "dev_size": len(dev),
        "dev_accuracy": dev_acc,
        "history": result.history,
        "hardware": _hardware_info(),
    }
    path = SMOKE_DIR / f"{args.model}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{args.model} smoke: {result.steps} steps, dev accuracy {dev_acc:.4f} -> {path}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from b77.train import train

    train_split, _, splits = load_all()
    sub = select(train_split, splits["sweep"]["ids"], "sweep")
    dev = select(train_split, splits["dev"]["ids"], "dev")
    runs = []
    for lr in plan.SWEEP_LRS:
        cfg = _train_cfg(plan.QWEN, lr, plan.SWEEP_EPOCHS, seed=0)
        _, _, result = train(cfg, sub.texts, sub.labels, dev=(dev.texts, dev.labels))
        accs = [h["dev_accuracy"] for h in result.history if "dev_accuracy" in h]
        runs.append(
            {
                "lr": lr,
                "dev_accuracy_by_epoch": accs,
                "train_seconds": round(result.train_seconds, 1),
                "examples_per_second": round(result.examples_per_second, 2),
                "config": result.config,
            }
        )
        log(f"sweep lr {lr:g}: dev accuracy by epoch {accs}")
    best = max(sorted(runs, key=lambda r: r["lr"]), key=lambda r: r["dev_accuracy_by_epoch"][-1])
    payload = {
        "model": plan.QWEN,
        "train_size": len(sub),
        "dev_size": len(dev),
        "epochs": plan.SWEEP_EPOCHS,
        "seed": 0,
        "selection": "highest dev accuracy after the last epoch, ties to the smaller lr",
        "chosen_lr": best["lr"],
        "runs": runs,
        "hardware": _hardware_info(),
    }
    SWEEP_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"chosen lr {best['lr']:g} -> {SWEEP_PATH}")
    return 0


def chosen_lr() -> float:
    if not SWEEP_PATH.exists():
        raise FileNotFoundError(f"{SWEEP_PATH} not found. Run `make sweep` first or pass --lr.")
    return float(json.loads(SWEEP_PATH.read_text(encoding="utf-8"))["chosen_lr"])


def _fit_and_score(
    model: str,
    lr: float,
    epochs: float,
    seed: int,
    train_set: Split,
    test: Split,
    run_id: str,
    note: str,
    test_batch_size: int,
    save: bool,
) -> None:
    from b77.train import ARTIFACTS, iter_progress, predict, save_model, train

    cfg = _train_cfg(model, lr, epochs, seed)
    fitted, tokenizer, result = train(cfg, train_set.texts, train_set.labels, dev=None)
    if save:
        save_model(fitted, tokenizer, ARTIFACTS / run_id)
    if cfg.spec.method == "full":
        import torch

        fitted = fitted.to(dtype=torch.bfloat16)
    probs, latency = predict(
        fitted,
        tokenizer,
        test.texts,
        cfg.max_length,
        batch_size=test_batch_size,
        progress=iter_progress(len(test), 500, run_id),
    )
    preds = [
        Prediction(
            item_id=test.ids[i],
            gold=int(test.labels[i]),
            pred=int(probs[i].argmax()),
            confidence=float(probs[i].max()),
            latency_ms=float(latency[i]) if latency is not None else 0.0,
        )
        for i in range(len(test))
    ]
    info = {
        "arm": note,
        "hf_id": cfg.spec.hf_id,
        "revision": cfg.spec.revision,
        "method": cfg.spec.method,
        "train_size": len(train_set),
        "train_seconds": round(result.train_seconds, 1),
        "examples_per_second": round(result.examples_per_second, 2),
        "steps": result.steps,
        "config": result.config,
        "loss_history": result.history,
        "test_batch_size": test_batch_size,
        "latency": "tokenise, forward pass and softmax for one message, batch 1, bf16 autocast"
        if latency is not None
        else "not measured (batched evaluation)",
        "adapter_dir": str(ARTIFACTS / run_id) if save else None,
        "hardware": _hardware_info(),
    }
    write_run(run_id, f"{model}-{cfg.spec.method}", cfg.spec.hf_id, preds, info)
    acc = float(np.mean(probs.argmax(axis=1) == test.labels))
    print(f"{run_id}: test accuracy {acc:.4f}, trained {result.train_seconds / 60:.1f} min")


def cmd_train_final(args: argparse.Namespace) -> int:
    train_split, test, _ = load_all()
    lr = args.lr if args.lr is not None else chosen_lr()
    _fit_and_score(
        plan.QWEN,
        lr,
        plan.FINAL_EPOCHS[plan.QWEN],
        args.seed,
        train_split,
        test,
        plan.RUN_QWEN_FINAL.format(seed=args.seed),
        "Qwen3-0.6B-Base, LoRA r16 alpha 32 on all linear layers, classification head, "
        f"lr {lr:g} from the dev sweep, all of train",
        test_batch_size=1,
        save=True,
    )
    return 0


def cmd_train_modernbert(args: argparse.Namespace) -> int:
    train_split, test, _ = load_all()
    _fit_and_score(
        plan.MODERNBERT,
        plan.DEFAULT_LR[plan.MODERNBERT],
        plan.FINAL_EPOCHS[plan.MODERNBERT],
        args.seed,
        train_split,
        test,
        plan.RUN_MODERNBERT if args.seed == 0 else f"{plan.RUN_MODERNBERT}-s{args.seed}",
        "ModernBERT-base full fine-tune, lr 5e-5, all of train",
        test_batch_size=1,
        save=True,
    )
    return 0


def cmd_learning_curve(args: argparse.Namespace) -> int:
    train_split, test, splits = load_all()
    lr = args.lr if args.lr is not None else chosen_lr()
    subsets = splits["learning_curve"]["subsets"][f"seed{args.draw}"]
    for k in args.k:
        sub = select(train_split, subsets[str(k)], f"curve-k{k}")
        _fit_and_score(
            plan.QWEN,
            lr,
            plan.LEARNING_CURVE_EPOCHS,
            0,
            sub,
            test,
            plan.run_qwen_curve(k, args.draw),
            f"Qwen3-0.6B-Base LoRA r16, {k} examples per class (draw {args.draw}), lr {lr:g}",
            test_batch_size=64,
            save=False,
        )
    return 0


# ---------------------------------------------------------------- prompting


def build_client(args: argparse.Namespace, arm: Any) -> tuple[Any, Any]:
    """The client stack for a prompting run, and the DollarCap (None when not live)."""
    from llm_eval_harness import CachedClient, DollarCap, FakeClient, RetryingClient

    from b77.prompting import CACHE_DIR, Pacer

    if args.fake:
        return FakeClient(lambda req: '{"label": "card_arrival"}'), None
    if not args.live:
        return CachedClient(None, CACHE_DIR, replay_only=True), None
    from llm_eval_harness.azure import FoundryClient, retryable_errors

    cap = DollarCap(FoundryClient(), cap_usd=args.cap if args.cap else arm.default_cap_usd)
    paced = Pacer(cap, args.tpm or arm.tokens_per_minute)
    retrying = RetryingClient(paced, retryable_errors(), max_attempts=6, max_delay_s=60.0)
    return CachedClient(retrying, CACHE_DIR), cap


def cmd_prompt(args: argparse.Namespace) -> int:
    from llm_eval_harness import BudgetExceeded, CacheMiss

    from b77.prompting import ARMS, Neighbors, run_arm

    arm = ARMS[args.arm]
    train, test, _ = load_all()
    if args.limit:
        test = test.subset(np.arange(args.limit))
    client, cap = build_client(args, arm)
    item_errors: tuple[type[Exception], ...] = ()
    if args.live:
        import openai

        item_errors = (openai.APIError,)
    neighbors = Neighbors.load() if arm.few_shot else None
    try:
        preds = run_arm(
            arm,
            client,
            test.ids,
            test.texts,
            test.labels,
            neighbors=neighbors,
            train_texts=train.texts,
            train_labels=train.labels,
            item_errors=item_errors,
            log=log,
        )
    except BudgetExceeded as exc:
        log(f"stopped: {exc}. Completed calls are cached, so a rerun with a higher --cap resumes.")
        return 2
    except CacheMiss as exc:
        log(f"stopped: {exc}. Pass --live to call the model.")
        return 2
    run_id = arm.run_id if not args.limit else f"{arm.run_id}-first{args.limit}"
    runs_dir = FAKE_RUNS_DIR if args.fake else RUNS_DIR
    info = {
        "arm": arm.key,
        "few_shot_k": 20 if arm.few_shot else 0,
        "max_output_tokens": 32,
        "reasoning_effort": "none",
        "cap_usd": None if cap is None else cap.cap_usd,
        "spent_usd_per_cap": None if cap is None else round(cap.spent_usd, 6),
        "charged_for_errors_usd": None if cap is None else round(cap.charged_for_errors_usd, 6),
        "cost_basis": "list price from measured usage (harness DEFAULT_PRICES, 2026-09-28)",
        "latency": "one Responses API call over the network, measured by the client",
        "fake": bool(args.fake),
    }
    write_run(run_id, arm.key, arm.model, preds, info, runs_dir)
    acc = float(np.mean([p.pred == p.gold for p in preds]))
    spent = sum(p.cost_usd for p in preds)
    print(f"{run_id}: accuracy {acc:.4f} on {len(preds)} items, ${spent:.4f} at list price")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from llm_eval_harness.report import write_section

    from b77.estimate import estimate_all

    train, test, _ = load_all()
    body = estimate_all(train, test)
    print(body)
    if args.write:
        write_section(Path("README.md"), "estimate", body)
        print("README estimate section updated")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from b77.report import write_report

    changed = write_report()
    print("README results section " + ("updated" if changed else "unchanged"))
    return 0


# ---------------------------------------------------------------- entry point


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="b77", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="download and verify the pinned Banking77 parquet files")
    s = sub.add_parser("splits", help="build dev, sweep, learning-curve and dedup splits")
    s.add_argument("--check", action="store_true", help="recompute and compare, write nothing")
    s = sub.add_parser("embed", help="bge-small embeddings and the retrieved-neighbour file")
    s.add_argument("--force", action="store_true", help="overwrite the committed neighbour file")
    s = sub.add_parser("baselines", help="logistic regression and kNN on bge-small embeddings")
    s.add_argument("--curve-only", action="store_true")
    s = sub.add_parser("timing", help="measure CPU throughput and write results/timing.json")
    s.add_argument("--steps", type=int, default=40)
    s = sub.add_parser("smoke", help="short training run scored on dev")
    s.add_argument("--model", choices=list(plan.BATCH_SIZE), required=True)
    s.add_argument("--steps", type=int, default=300)
    sub.add_parser("sweep", help="LoRA learning-rate sweep on the stratified subset")
    s = sub.add_parser("train-final", help="Qwen3-0.6B LoRA on all of train, scored on test")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--lr", type=float, default=None, help="default: the sweep's choice")
    s = sub.add_parser("train-modernbert", help="ModernBERT-base on all of train, scored on test")
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("learning-curve", help="Qwen3-0.6B LoRA on k examples per class")
    s.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    s.add_argument("--draw", type=int, default=0)
    s.add_argument("--lr", type=float, default=None)
    s = sub.add_parser("prompt", help="run one prompting arm on the test set")
    s.add_argument("--arm", required=True, choices=["luna-zeroshot", "luna-fewshot", "sol-fewshot"])
    mode = s.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="call Azure (needs env, costs money)")
    mode.add_argument("--fake", action="store_true", help="offline fake model, for plumbing")
    s.add_argument("--cap", type=float, default=None, help="hard dollar cap for this run")
    s.add_argument("--tpm", type=int, default=None, help="deployment tokens-per-minute limit")
    s.add_argument("--limit", type=int, default=None, help="only the first N test items")
    s = sub.add_parser("estimate", help="token, cost and wall-clock estimates for the API arms")
    s.add_argument("--write", action="store_true", help="also update the README section")
    sub.add_parser("report", help="rewrite the README results section from results/")
    return p


COMMANDS = {
    "data": cmd_data,
    "splits": cmd_splits,
    "embed": cmd_embed,
    "baselines": cmd_baselines,
    "timing": cmd_timing,
    "smoke": cmd_smoke,
    "sweep": cmd_sweep,
    "train-final": cmd_train_final,
    "train-modernbert": cmd_train_modernbert,
    "learning-curve": cmd_learning_curve,
    "prompt": cmd_prompt,
    "estimate": cmd_estimate,
    "report": cmd_report,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
