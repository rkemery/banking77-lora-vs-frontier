"""Build the README results section from committed run files. Offline, no keys, no models.

Every number comes from `results/runs/*.jsonl` (harness records),
`results/sweep.json` and `results/timing.json`. A run that has not happened
yet shows as a "pending" row with the command that produces it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from llm_eval_harness.report import write_section
from llm_eval_harness.stats import mde_paired_binary, paired_bootstrap

from b77 import plan
from b77.data import N_CLASSES
from b77.metrics import expected_calibration_error, latency_percentiles, summarize
from b77.runs import RUNS_DIR, Run, load_run, run_exists
from b77.splits import LEARNING_CURVE_K, LEARNING_CURVE_SEEDS, read_splits

README = Path("README.md")
SUMMARY_PATH = Path("results/summary.json")
SWEEP_PATH = Path("results/sweep.json")
SMOKE_DIR = Path("results/smoke")

# CPU price used to turn local latency into dollars: Azure D4s v6 (4 vCPU, 16 GiB,
# 5th gen Xeon with AMX, like the container these runs used), Linux pay-as-you-go,
# East US 2, from the Azure Retail Prices API on 2026-09-28.
CPU_USD_PER_HOUR = 0.202
CPU_PRICE_NOTE = (
    "Local CPU cost assumes one Azure D4s v6 VM (4 vCPU, 16 GiB, 5th gen Xeon with AMX, "
    "the same CPU class these runs used) at the $0.202/hour Linux pay-as-you-go list price "
    "in East US 2 (Azure Retail Prices API, 2026-09-28), serving one message at a time "
    "with no batching and no idle time. The GPU row uses the Hugging Face Jobs a10g-large "
    "list price its run recorded ($1.50/hour, huggingface.co/docs/hub/jobs-pricing, "
    "2026-09-28) on the same one-message-at-a-time basis."
)


@dataclass(frozen=True)
class ArmSpec:
    label: str
    run_ids: tuple[str, ...]  # several ids = several seeds of one arm
    kind: str  # "local" or "api"
    pending: str  # what produces it


def arms() -> list[ArmSpec]:
    qwen = tuple(plan.RUN_QWEN_FINAL.format(seed=s) for s in range(5))
    return [
        ArmSpec(
            "LogReg on bge-small embeddings",
            (plan.RUN_LOGREG,),
            "local",
            "pending: `make baselines`",
        ),
        ArmSpec(
            "kNN vote over the same 20 retrieved neighbours",
            (plan.RUN_KNN,),
            "local",
            "pending: `make baselines`",
        ),
        ArmSpec(
            "ModernBERT-base, full fine-tune",
            (plan.RUN_MODERNBERT,),
            "local",
            "pending CPU run: `make train-modernbert`",
        ),
        ArmSpec(
            "Qwen3-0.6B-Base + LoRA r16, classification head",
            qwen,
            "local",
            "pending CPU run: `make sweep`, then `make train-final`",
        ),
        ArmSpec(
            "gpt-6-luna zero-shot, label descriptions",
            (plan.RUN_LUNA_ZERO,),
            "api",
            "pending live run: `make prompt-luna-zeroshot`",
        ),
        ArmSpec(
            "gpt-6-luna, 20 retrieved examples",
            (plan.RUN_LUNA_FEW,),
            "api",
            "pending live run: `make prompt-luna-fewshot`",
        ),
        ArmSpec(
            "gpt-6-sol, 20 retrieved examples",
            (plan.RUN_SOL_FEW,),
            "api",
            "pending live run: `make prompt-sol-fewshot`",
        ),
        ArmSpec(
            "Qwen3-8B-Base QLoRA r16 (one A10G, Hugging Face Jobs)",
            (plan.RUN_QWEN8B,),
            "gpu",
            "pending GPU run: `scripts/launch_hf_job.py`",
        ),
    ]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(low: float, high: float) -> str:
    return f"{100 * low:.1f} to {100 * high:.1f}"


def usd_per_hour(run: Run, kind: str) -> float | None:
    """The CPU VM price for local runs, the job flavor's price recorded by a GPU run."""
    if kind == "gpu":
        return run.info.get("usd_per_hour")
    return CPU_USD_PER_HOUR


def cost_per_1k(run: Run, kind: str) -> float | None:
    if kind == "api":
        return float(sum(r.cost_usd for r in run.records)) / len(run.records) * 1000
    price = usd_per_hour(run, kind)
    lat = run.latencies()
    if price is None or not lat.size:
        return None
    return price * float(np.mean(lat)) * 1000 / 3_600_000


def fmt_cost(value: float) -> str:
    return f"${value:.4f}" if value < 0.1 else f"${value:.2f}"


def arm_metrics(spec: ArmSpec, runs: list[Run], dedup_ids: list[str]) -> dict[str, Any]:
    """Metrics for the first seed, plus the spread over seeds when there are several."""
    run = runs[0]
    gold, pred = run.arrays()
    full = summarize(gold, pred, N_CLASSES)
    dg, dp = run.arrays(dedup_ids)
    dedup = summarize(dg, dp, N_CLASSES)
    conf = run.confidences()
    out: dict[str, Any] = {
        "run_id": run.run_id,
        "n": full.n,
        "accuracy": full.accuracy.estimate,
        "accuracy_ci": [full.accuracy.low, full.accuracy.high],
        "macro_f1": full.macro_f1.estimate,
        "macro_f1_ci": [full.macro_f1.low, full.macro_f1.high],
        "dedup_n": dedup.n,
        "dedup_accuracy": dedup.accuracy.estimate,
        "dedup_accuracy_ci": [dedup.accuracy.low, dedup.accuracy.high],
        "ece": None
        if conf is None or spec.kind == "api"
        else expected_calibration_error(conf, gold == pred),
        "errors": run.n_errors,
        "invalid": int(np.sum(pred < 0)) - run.n_errors,
        "cost_per_1k_usd": cost_per_1k(run, spec.kind),
        "total_cost_usd": float(sum(r.cost_usd for r in run.records)),
    }
    lat = run.latencies()
    if lat.size:
        out["latency_p50_ms"], out["latency_p95_ms"] = latency_percentiles(lat)
    if len(runs) > 1:
        accs = [float(np.mean(np.equal(*r.arrays()))) for r in runs]
        out["seed_accuracies"] = accs
        out["seed_sd"] = float(np.std(accs, ddof=1))
    train_seconds = run.info.get("train_seconds")
    price = usd_per_hour(run, spec.kind)
    if train_seconds is not None and price is not None:
        out["train_minutes"] = train_seconds / 60
        out["train_cost_usd"] = price * train_seconds / 3600
        out["train_on"] = spec.kind
    return out


def load_arm_runs(spec: ArmSpec, runs_dir: Path) -> list[Run]:
    return [load_run(r, runs_dir) for r in spec.run_ids if run_exists(r, runs_dir)]


def results_table(rows: list[tuple[ArmSpec, dict[str, Any] | None]], dedup_n: int) -> str:
    head = (
        f"| Arm | Accuracy (95% CI) | Macro-F1 (95% CI) | Accuracy, dedup test (n={dedup_n:,}) "
        "| ECE | Latency p50 / p95 | Cost per 1k predictions |\n|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for spec, m in rows:
        if m is None:
            lines.append(f"| {spec.label} | {spec.pending} | | | | | |")
            continue
        seeds = ""
        if "seed_sd" in m:
            seeds = f", seed sd {100 * m['seed_sd']:.1f} pts over {len(m['seed_accuracies'])}"
        if spec.kind == "api":
            ece = "n/a (no logprobs)"
        else:
            ece = "n/a" if m["ece"] is None else f"{m['ece']:.3f}"
        lat = f"{m['latency_p50_ms']:.0f} / {m['latency_p95_ms']:.0f} ms"
        if spec.kind == "local":
            lat += " (CPU, batch 1)"
        if spec.kind == "gpu":
            lat += " (A10G GPU, batch 1)"
        cost = "n/a" if m["cost_per_1k_usd"] is None else fmt_cost(m["cost_per_1k_usd"])
        lines.append(
            f"| {spec.label} | {pct(m['accuracy'])} ({ci(*m['accuracy_ci'])}){seeds} "
            f"| {pct(m['macro_f1'])} ({ci(*m['macro_f1_ci'])}) "
            f"| {pct(m['dedup_accuracy'])} ({ci(*m['dedup_accuracy_ci'])}) "
            f"| {ece} | {lat} | {cost} |"
        )
    return "\n".join(lines)


Q06 = plan.RUN_QWEN_FINAL.format(seed=0)

# The primary family answers the repo's question, fine-tuned local model vs retrieval-augmented
# prompting: {0.6B, 8B} x {luna few-shot, sol few-shot}. It was named after the results were in.
PRIMARY_PAIRS: list[tuple[str, str]] = [
    (Q06, plan.RUN_LUNA_FEW),
    (Q06, plan.RUN_SOL_FEW),
    (plan.RUN_QWEN8B, plan.RUN_LUNA_FEW),
    (plan.RUN_QWEN8B, plan.RUN_SOL_FEW),
]
SECONDARY_PAIRS: list[tuple[str, str]] = [
    (plan.RUN_LOGREG, plan.RUN_KNN),
    (plan.RUN_LOGREG, plan.RUN_MODERNBERT),
    (plan.RUN_LOGREG, Q06),
    (plan.RUN_LOGREG, plan.RUN_QWEN8B),
    (plan.RUN_MODERNBERT, Q06),
    (Q06, plan.RUN_QWEN8B),
    (Q06, plan.RUN_LUNA_ZERO),
    (plan.RUN_QWEN8B, plan.RUN_LUNA_ZERO),
    (plan.RUN_KNN, plan.RUN_LUNA_FEW),
    (plan.RUN_LUNA_ZERO, plan.RUN_LUNA_FEW),
    (plan.RUN_LUNA_FEW, plan.RUN_SOL_FEW),
]
PAIRS: list[tuple[str, str]] = PRIMARY_PAIRS + SECONDARY_PAIRS


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: pvalues[j])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def compare_pairs(runs: dict[str, Run]) -> dict[tuple[str, str], dict[str, Any]]:
    """Paired bootstrap CI and exact McNemar for every pair in PAIRS whose runs both exist.

    Adds Holm-adjusted p within the primary family, within the secondary pairs, and over
    every pair at once, so the report can show the conclusions don't hinge on the family.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for base_id, cand_id in PAIRS:
        if base_id not in runs or cand_id not in runs:
            continue
        base, cand = runs[base_id], runs[cand_id]
        ids = base.item_ids
        bg, bp = base.arrays(ids)
        cg, cp = cand.arrays(ids)
        if not np.array_equal(bg, cg):
            raise ValueError(f"{base_id} and {cand_id} disagree on gold labels")
        comp = paired_bootstrap(bp == bg, cp == cg, seed=0)
        assert comp.mcnemar is not None
        rate = comp.mcnemar.discordant / comp.n
        out[(base_id, cand_id)] = {
            "baseline": base_id,
            "candidate": cand_id,
            "family": "primary" if (base_id, cand_id) in PRIMARY_PAIRS else "secondary",
            "n": comp.n,
            "baseline_accuracy": comp.baseline_mean,
            "candidate_accuracy": comp.candidate_mean,
            "diff": comp.diff,
            "diff_ci": [comp.low, comp.high],
            "p": comp.pvalue if comp.pvalue is not None else float("nan"),
            "base_only": comp.mcnemar.a_only,
            "cand_only": comp.mcnemar.b_only,
            "mde": mde_paired_binary(comp.n, rate) if rate > 0 else None,
        }
    for key, family in (("holm_p", "primary"), ("holm_p_secondary", "secondary")):
        members = [c for c in out.values() if c["family"] == family]
        for c, adj in zip(members, holm([c["p"] for c in members]), strict=True):
            c[key] = adj
    for c, adj in zip(out.values(), holm([c["p"] for c in out.values()]), strict=True):
        c["holm_p_all"] = adj
    return out


def paired_rows(comps: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    lines = [
        "| Baseline | Candidate | Baseline acc. | Candidate acc. | Difference (95% CI) "
        "| McNemar p | Discordant (base only / cand. only) | MDE | n | Family (Holm p) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pair in PAIRS:
        if pair not in comps:
            lines.append(f"| {pair[0]} | {pair[1]} | pending | | | | | | | |")
            continue
        c = comps[pair]
        low, high = c["diff_ci"]
        family = f"primary ({_fmt_p(c['holm_p'])})" if "holm_p" in c else "secondary"
        mde = "n/a" if c["mde"] is None else f"{100 * c['mde']:.1f} pts"
        lines.append(
            f"| {c['baseline']} | {c['candidate']} | {pct(c['baseline_accuracy'])} "
            f"| {pct(c['candidate_accuracy'])} "
            f"| {100 * c['diff']:+.1f} pts ({100 * low:+.1f} to {100 * high:+.1f}) "
            f"| {_fmt_p(c['p'])} | {c['base_only']} / {c['cand_only']} "
            f"| {mde} | {c['n']:,} | {family} |"
        )
    return lines


def family_lines(comps: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    """How the primary result holds up when the family changes."""
    primary = [comps[p] for p in PRIMARY_PAIRS if p in comps]
    if len(primary) < len(PRIMARY_PAIRS):
        return []
    parts = [f"{c['baseline']} vs {c['candidate']} {_fmt_p(c['holm_p_all'])}" for c in primary]
    same = all((c["holm_p"] < 0.05) == (c["holm_p_all"] < 0.05) for c in primary)
    lines = [
        f"Holm over all {len(comps)} pairs at once gives the primary pairs "
        + ", ".join(parts)
        + (
            ". The same pairs clear 0.05 either way, so the conclusions don't depend on which "
            "family is named."
            if same
            else ". Some primary pairs change sides at 0.05 under the wider family."
        )
    ]
    lf_sol = comps.get((plan.RUN_LUNA_FEW, plan.RUN_SOL_FEW))
    if lf_sol is not None:
        lines.append(
            f"sol over luna few-shot (p={_fmt_p(lf_sol['p'])}) is suggestive only: Holm gives "
            f"{_fmt_p(lf_sol['holm_p_secondary'])} over the secondary pairs and "
            f"{_fmt_p(lf_sol['holm_p_all'])} over all of them."
        )
    return [" ".join(lines)]


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def curve_data(runs_dir: Path) -> dict[int, dict[str, Any]]:
    """Learning-curve accuracies per k: logreg over its draws, LoRA for draw 0."""
    out: dict[int, dict[str, Any]] = {}
    for k in LEARNING_CURVE_K:
        accs = []
        for seed in LEARNING_CURVE_SEEDS:
            rid = plan.run_logreg_curve(k, seed)
            if run_exists(rid, runs_dir):
                g, p = load_run(rid, runs_dir).arrays()
                accs.append(float(np.mean(g == p)))
        row: dict[str, Any] = {"logreg": None, "logreg_sd": None, "lora": None}
        if len(accs) == len(LEARNING_CURVE_SEEDS):
            row["logreg"] = float(np.mean(accs))
            row["logreg_sd"] = float(np.std(accs, ddof=1))
        qid = plan.run_qwen_curve(k, 0)
        if run_exists(qid, runs_dir):
            g, p = load_run(qid, runs_dir).arrays()
            row["lora"] = float(np.mean(g == p))
        out[k] = row
    return out


def curve_table(curve: dict[int, dict[str, Any]], full: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Examples per class | Train size | LogReg on bge-small (mean of 3 draws, sd) "
        "| Qwen3-0.6B LoRA (draw 0) |",
        "|---|---|---|---|",
    ]
    for k, row in curve.items():
        if row["logreg"] is None:
            lr_cell = "pending: `make baselines`"
        else:
            lr_cell = f"{pct(row['logreg'])} (sd {100 * row['logreg_sd']:.1f})"
        q_cell = "pending CPU run: `make learning-curve`"
        if row["lora"] is not None:
            q_cell = pct(row["lora"])
        lines.append(f"| {k} | {77 * k:,} | {lr_cell} | {q_cell} |")
    lr_full = full.get(plan.RUN_LOGREG)
    q_full = full.get(Q06)
    lr_cell = f"{pct(lr_full['accuracy'])} (one fit)" if lr_full else "pending"
    q_cell = pct(q_full["accuracy"]) if q_full else "pending"
    lines.append(f"| all (about 130) | {plan.N_TRAIN:,} | {lr_cell} | {q_cell} |")
    return "\n".join(lines)


def conditions_line(runs: dict[str, Run]) -> list[str]:
    """How busy the CPU was when each local run's latency was measured."""
    parts = []
    loaded = []
    for run_id, run in runs.items():
        hw = run.info.get("hardware")
        if not hw or "gpu" in hw:
            continue
        embed = hw.get("embedding_latency_measured_with") or {}
        threads = embed.get("torch_threads", hw.get("torch_threads"))
        load = embed.get("load_average_1m", hw.get("load_average_1m"))
        parts.append(f"`{run_id}` with {threads} torch thread(s) at load {load}")
        if load is not None and load > hw.get("logical_cpus", 4):
            loaded.append(f"`{run_id}`")
    if not parts:
        return []
    line = (
        "Local latency depends on how busy the machine was (load 4 means all 4 vCPUs busy, "
        "and the D4s v6 has 2 physical cores under them): " + ", ".join(parts) + "."
    )
    if loaded:
        line += (
            f" {' and '.join(loaded)} {'was' if len(loaded) == 1 else 'were'} timed on a "
            "shared machine with more work queued than it had vCPUs, not the idle D4s v6, but "
            "priced at the D4s v6 rate, so their latency and cost are upper bounds."
        )
    return [line]


def spend_lines(full: dict[str, dict[str, Any]]) -> list[str]:
    """One-off costs: training time for local runs, total spend for API runs."""
    lines = []
    for run_id, m in full.items():
        if "train_minutes" in m:
            took = (
                f"{m['train_minutes'] * 60:.0f} seconds"
                if m["train_minutes"] < 1
                else f"{m['train_minutes']:.1f} minutes"
            )
            where = (
                "on one A10G (about ${cost:.4f} at the job's list price)"
                if m.get("train_on") == "gpu"
                else "on the CPU (about ${cost:.4f} at the VM price above)"
            )
            lines.append(
                f"- `{run_id}` trained in {took} {where.format(cost=m['train_cost_usd'])}."
            )
        elif m["total_cost_usd"] > 0:
            lines.append(
                f"- `{run_id}` cost ${m['total_cost_usd']:.2f} for {m['n']:,} calls at list price, "
                f"with {m['errors']} failed calls and {m['invalid']} invalid labels."
            )
    return lines


BREAK_EVEN_PAIRS: list[tuple[str, str]] = [
    (Q06, plan.RUN_LUNA_FEW),
    (Q06, plan.RUN_SOL_FEW),
    (plan.RUN_QWEN8B, plan.RUN_LUNA_FEW),
    (plan.RUN_QWEN8B, plan.RUN_SOL_FEW),
]


def break_even(full: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The share of the time a local box must be busy to cost less per prediction than an API.

    A rented box costs the same per hour busy or idle, and the API bills per call. The
    per-1k local cost assumes the box is busy all the time, so at utilization u it is
    cost / u, and it matches the API at u = local cost / API cost.
    """
    out = []
    for local_id, api_id in BREAK_EVEN_PAIRS:
        local, api = full.get(local_id), full.get(api_id)
        if local is None or api is None:
            continue
        lc, ac = local["cost_per_1k_usd"], api["cost_per_1k_usd"]
        if lc is None or ac is None or ac <= 0:
            continue
        out.append(
            {
                "local": local_id,
                "api": api_id,
                "local_on": "A10G" if local.get("train_on") == "gpu" else "CPU",
                "local_cost_per_1k_usd": lc,
                "api_cost_per_1k_usd": ac,
                "utilization": lc / ac,
            }
        )
    return out


def break_even_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "",
        "**Break-even utilization.** The local costs above assume a box that is busy all the "
        "time, but a rented box costs the same per hour busy or idle and the API bills only "
        "per call. So a local arm is only cheaper per prediction while its box is busy for "
        "more than (local cost / API cost) of the time, with hardware at the prices above and "
        "training excluded:",
        "",
    ]
    for r in rows:
        u = r["utilization"]
        when = f"above {100 * u:.1f}% utilization" if u < 1 else "never, even fully busy"
        lines.append(
            f"- `{r['local']}` ({r['local_on']}, {fmt_cost(r['local_cost_per_1k_usd'])} per 1k "
            f"when fully busy) vs `{r['api']}` ({fmt_cost(r['api_cost_per_1k_usd'])}): "
            f"cheaper {when}."
        )
    return lines


def dedup_line(full: dict[str, dict[str, Any]]) -> list[str]:
    """How much each arm loses when the near-twin test items are dropped."""
    drops = {rid: 100 * (m["accuracy"] - m["dedup_accuracy"]) for rid, m in full.items()}
    if len(drops) < 2:
        return []
    low, high = min(drops.values()), max(drops.values())
    if low <= 0:
        return [f"Dedup minus full accuracy ranges from {-high:+.1f} to {-low:+.1f} points."]
    line = f"Every arm scores {low:.1f} to {high:.1f} points lower on the dedup subset."
    if plan.RUN_LUNA_ZERO in drops:
        line += (
            f" That includes `{plan.RUN_LUNA_ZERO}` ({drops[plan.RUN_LUNA_ZERO]:.1f}), which has "
            "no training messages in its prompt, so the dropped twins are mostly easier "
            "messages, not answers the fine-tuned models memorised."
        )
    return [line]


def headline(summary: dict[str, Any]) -> str:
    """The README's opening answer, filled from the same numbers as the results section.

    The wording fits the committed results. If a rerun changes which way a comparison
    goes, revisit the sentences here.
    """
    arms_ = summary["arms"]
    pairs = {(c["baseline"], c["candidate"]): c for c in summary["paired"]}
    be = {(r["local"], r["api"]): r for r in summary["break_even"]}
    curve = summary["learning_curve"]
    lr, mb, q8 = plan.RUN_LOGREG, plan.RUN_MODERNBERT, plan.RUN_QWEN8B
    lf, sol = plan.RUN_LUNA_FEW, plan.RUN_SOL_FEW
    needed_pairs = [(Q06, sol), (Q06, lf), (Q06, q8), (lr, Q06), (lr, mb)]
    needed_be = [(Q06, lf), (Q06, sol), (q8, lf)]
    k_small = [str(k) for k in LEARNING_CURVE_K]
    if (
        any(p not in pairs for p in needed_pairs)
        or any(p not in be for p in needed_be)
        or any(curve.get(k, {}).get("lora") is None for k in k_small)
        or any(curve.get(k, {}).get("logreg") is None for k in k_small)
    ):
        return "The short answer goes here once every arm has run (`make demo` writes it)."

    def at_most(pair: tuple[str, str]) -> str:
        return f"{100 * pairs[pair]['diff_ci'][1]:.1f}"

    def util(pair: tuple[str, str]) -> str:
        return f"{100 * be[pair]['utilization']:.1f}%"

    s6, lf6, q86 = pairs[(Q06, sol)], pairs[(Q06, lf)], pairs[(Q06, q8)]
    return (
        f"**The short answer.** gpt-6-sol with 20 retrieved examples beats the Qwen3-0.6B LoRA "
        f"by {100 * s6['diff']:.1f} points ({pct(s6['candidate_accuracy'])} vs "
        f"{pct(s6['baseline_accuracy'])}, McNemar p={_fmt_p(s6['p'])}, Holm-adjusted "
        f"{_fmt_p(s6['holm_p'])}), at about "
        f"{arms_[sol]['cost_per_1k_usd'] / arms_[lf]['cost_per_1k_usd']:.0f}x the "
        f"per-prediction cost of gpt-6-luna with the same examples. Neither luna few-shot "
        f"(p={_fmt_p(lf6['p'])}) nor the Qwen3-8B QLoRA (p={_fmt_p(q86['p'])}, "
        f"{_fmt_p(q86['holm_p_secondary'])} after Holm) can be separated from the 0.6B, and "
        f"logistic regression on frozen bge-small embeddings "
        f"({fmt_cost(arms_[lr]['cost_per_1k_usd'])} per 1k, "
        f"{arms_[lr]['train_minutes'] * 60:.0f} s to train) is at most {at_most((lr, Q06))} "
        f"points below it and beats it at {k_small[0]} to {k_small[-1]} examples per class. "
        f"A local model is only cheaper than the API while it stays busy: above "
        f"{util((Q06, lf))} utilization for the 0.6B's CPU box and {util((q8, lf))} for the "
        f"8B's A10G, both against luna few-shot."
    )


def sweep_table(path: Path = SWEEP_PATH) -> str:
    if not path.exists():
        return "LoRA learning-rate sweep: pending CPU run (`make sweep`)."
    sweep = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        f"LoRA learning-rate sweep on the {sweep['train_size']:,}-example stratified subset, "
        f"one seed, scored on the {sweep['dev_size']:,}-item dev split. "
        f"Chosen: {sweep['chosen_lr']:g}.",
        "",
        "| Learning rate | Dev accuracy by epoch | Train minutes |",
        "|---|---|---|",
    ]
    for entry in sweep["runs"]:
        accs = ", ".join(pct(a) for a in entry["dev_accuracy_by_epoch"])
        lines.append(f"| {entry['lr']:g} | {accs} | {entry['train_seconds'] / 60:.1f} |")
    return "\n".join(lines)


def smoke_lines(smoke_dir: Path = SMOKE_DIR) -> list[str]:
    out = []
    for path in sorted(smoke_dir.glob("*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        out.append(
            f"- Smoke run `{path.stem}`: {s['steps']} steps ({s['examples_seen']:,} examples "
            f"from train minus dev), dev accuracy {pct(s['dev_accuracy'])} "
            f"(n={s['dev_size']:,}), {s['examples_per_second']:.1f} training examples/s on "
            f"{s['hardware']['torch_threads']} threads at load {s['hardware']['load_average_1m']}. "
            "A check that training learns, not a result."
        )
    return out


def timing_lines(timing: dict[str, Any] | None) -> list[str]:
    if timing is None:
        return ["CPU timing: not measured yet (`make timing`)."]
    q, m = timing[plan.QWEN], timing[plan.MODERNBERT]
    est = timing["estimates_minutes"]
    bge = timing["bge-small"]["inference"]
    return [
        f"Measured on {timing['cpu']} ({timing['torch_threads']} threads, "
        f"torch {timing['torch']}), load average {timing['load_average_1m_before']} before and "
        f"{timing['load_average_1m_after']} after (4 means all 4 vCPUs busy):",
        "",
        "| Model | Training examples/s | Inference, batch 1 (p50) | Inference, batch 64 |",
        "|---|---|---|---|",
        f"| Qwen3-0.6B-Base + LoRA (bs {q['train']['batch_size']}) "
        f"| {q['train']['examples_per_second']:.1f} "
        f"| {q['inference']['batch1_p50_ms']:.0f} ms unmerged, "
        f"{q['inference_merged']['batch1_p50_ms']:.0f} ms merged "
        f"| {q['inference']['batch64_examples_per_second']:.0f}/s |",
        f"| ModernBERT-base (bs {m['train']['batch_size']}) "
        f"| {m['train']['examples_per_second']:.1f} | {m['inference']['batch1_p50_ms']:.0f} ms "
        f"| {m['inference']['batch64_examples_per_second']:.0f}/s |",
        f"| bge-small encoder | n/a (frozen) | {bge['batch1_p50_ms']:.0f} ms "
        f"| {bge['batch64_examples_per_second']:.0f}/s |",
        "",
        *(
            [
                "Other jobs shared the CPU during this measurement, so an idle machine with 4 "
                "threads will be faster. `make timing` re-measures.",
                "",
            ]
            if timing["load_average_1m_before"] > timing["logical_cpus"] / 2
            else []
        ),
        f"Estimated wall-clock for the long targets: `make sweep` {est['sweep']:.0f} min, "
        f"`make train-final` {est['train_final_per_seed']:.0f} min per seed, "
        f"`make train-modernbert` {est['train_modernbert']:.0f} min, "
        f"`make learning-curve` {est['learning_curve_lora']:.0f} min.",
    ]


def build(runs_dir: Path = RUNS_DIR) -> tuple[str, dict[str, Any]]:
    splits = read_splits()
    dedup = splits["test_dedup"]
    removed = set(dedup["removed_ids"])
    all_runs: dict[str, Run] = {}
    rows: list[tuple[ArmSpec, dict[str, Any] | None]] = []
    full: dict[str, dict[str, Any]] = {}
    dedup_ids: list[str] | None = None
    for spec in arms():
        runs = load_arm_runs(spec, runs_dir)
        if not runs:
            rows.append((spec, None))
            continue
        if dedup_ids is None:
            dedup_ids = [i for i in runs[0].item_ids if i not in removed]
        metrics = arm_metrics(spec, runs, dedup_ids)
        rows.append((spec, metrics))
        full[runs[0].run_id] = metrics
        all_runs[runs[0].run_id] = runs[0]
    n_dedup = dedup["kept"]
    from b77.timing import read_timing

    timing = read_timing()
    comps = compare_pairs(all_runs)
    curve = curve_data(runs_dir)
    be = break_even(full)
    body = [
        results_table(rows, n_dedup),
        "",
        "n = 3,080 test items (40 per class) for every row. CIs are percentile bootstraps over "
        "items (10,000 resamples). The dedup column drops the "
        f"{dedup['same_label_twins']} test items whose nearest training message has the same "
        f"label at character n-gram cosine >= {dedup['threshold']:.2f}. ECE is top-label "
        "expected calibration error with 15 bins (the kNN row's confidence is its winning "
        "vote share, not a probability). API latency is one request over the network "
        "and API cost is from measured token usage at list price. " + CPU_PRICE_NOTE,
        "",
        *dedup_line(full),
        "",
        *conditions_line(all_runs),
        "",
        *spend_lines(full),
        *break_even_lines(be),
        "",
        "**Paired comparisons** on the same 3,080 items (accuracy, candidate minus baseline). "
        "CI from a paired bootstrap, p from the exact McNemar test, MDE is the smallest "
        "difference this pair could detect with 80% power (harness `stats`). The CIs and "
        "p-values treat each trained model and each API run as fixed: every fine-tune here has "
        "one seed and every API arm ran once, so seed-to-seed and run-to-run variance isn't in "
        "them. The primary family is {0.6B LoRA, 8B QLoRA} x {luna few-shot, sol few-shot}, "
        "the four pairs that answer this repo's question, with Holm-adjusted p in the last "
        "column. It was named after the results were in. The other pairs are secondary and "
        "their p-values are unadjusted in the table.",
        "",
        *paired_rows(comps),
        "",
        *family_lines(comps),
        "",
        "**Learning curve** (accuracy on the full test set). Subsets are nested across k and "
        "drawn from train minus dev. The LoRA curve runs train for "
        f"{plan.LEARNING_CURVE_EPOCHS} epochs at the sweep's learning rate, the full-data run "
        f"for {plan.FINAL_EPOCHS[plan.QWEN]}. "
        "See What didn't work for why this curve can't test the small-data claim.",
        "",
        curve_table(curve, full),
        "",
        sweep_table(),
        "",
        *smoke_lines(),
        "",
        *timing_lines(timing),
    ]
    summary = {
        "arms": full,
        "dedup": {k: v for k, v in dedup.items() if k != "removed_ids"},
        "paired": list(comps.values()),
        "learning_curve": {str(k): row for k, row in curve.items()},
        "break_even": be,
    }
    return "\n".join(body).replace("\n\n\n", "\n\n"), summary


def write_report(readme: Path = README, runs_dir: Path = RUNS_DIR) -> bool:
    body, summary = build(runs_dir)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    changed = write_section(readme, "headline", headline(summary))
    return write_section(readme, "results", body) or changed
