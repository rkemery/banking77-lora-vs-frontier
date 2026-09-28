"""The experiment plan in one place: hyperparameters, run ids and time estimates.

Choices and where they come from:

- Qwen3-0.6B-Base LoRA: rank 16, alpha 32, all linear layers, batch 16.
  "LoRA Without Regret" (Schulman and Thinking Machines Lab, 2025) finds the
  best LoRA learning rate is about 10x the full fine-tuning one and nearly
  independent of rank, and that LoRA loses more than full fine-tuning at
  large batch sizes. Full fine-tunes of models this size usually sit at 1e-5
  to 3e-5, so the sweep tries 1e-4 and 3e-4, on a 2,000-example stratified
  subset with one seed, scored on dev. The final run uses the winner on all of train.
- ModernBERT-base: full fine-tune, lr 5e-5 and 3 epochs, both inside the grid
  the ModernBERT authors swept for GLUE (lr 1e-5 to 8e-5, 1 to 3 epochs for
  SST-2, MNLI and RTE, Warner et al. 2024, arXiv 2412.13663, appendix).
  Batch 32, weight decay 0.01. Not swept here, to save CPU time.
- Epochs are fixed in advance, not picked on test.
"""

from __future__ import annotations

from typing import Any

from b77.splits import DEV_SIZE, LEARNING_CURVE_K, SWEEP_SIZE

N_TRAIN = 10_003
N_TEST = 3_080

QWEN = "qwen3-0.6b"
MODERNBERT = "modernbert-base"

BATCH_SIZE = {QWEN: 16, MODERNBERT: 32}
DEFAULT_LR = {QWEN: 2e-4, MODERNBERT: 5e-5}
WEIGHT_DECAY = {QWEN: 0.0, MODERNBERT: 0.01}

SWEEP_LRS = (1e-4, 3e-4)
SWEEP_EPOCHS = 2
FINAL_EPOCHS = {QWEN: 2, MODERNBERT: 3}
LEARNING_CURVE_EPOCHS = 10
MODEL_LOAD_MINUTES = 0.5

# Run ids. The report looks for these files in results/runs/.
RUN_LOGREG = "logreg-bge-small"
RUN_KNN = "knn-bge-small-k20"
RUN_MODERNBERT = "modernbert-base-full"
RUN_QWEN_FINAL = "qwen3-0.6b-lora-r16-s{seed}"
RUN_LUNA_ZERO = "gpt-6-luna-zeroshot"
RUN_LUNA_FEW = "gpt-6-luna-fewshot-k20"
RUN_SOL_FEW = "gpt-6-sol-fewshot-k20"
RUN_QWEN8B = "qwen3-8b-qlora-r16-s0"  # written by scripts/hf_job_qwen3_8b_qlora.py


def run_logreg_curve(k: int, seed: int) -> str:
    return f"logreg-bge-small-k{k}-s{seed}"


def run_qwen_curve(k: int, seed: int) -> str:
    return f"qwen3-0.6b-lora-r16-k{k}-s{seed}"


def estimates(timing: dict[str, Any]) -> dict[str, float]:
    """Wall-clock minutes for each long target, from measured throughput."""
    q, m = timing[QWEN], timing[MODERNBERT]
    q_train = q["train"]["examples_per_second"]
    q_eval = q["inference"]["batch64_examples_per_second"]
    q_b1 = q["inference"]["batch1_mean_ms"] / 1000.0
    m_train = m["train"]["examples_per_second"]
    m_b1 = m["inference"]["batch1_mean_ms"] / 1000.0
    load = MODEL_LOAD_MINUTES * 60

    sweep_one = SWEEP_EPOCHS * (SWEEP_SIZE / q_train + DEV_SIZE / q_eval) + load
    final = FINAL_EPOCHS[QWEN] * N_TRAIN / q_train + N_TEST * q_b1 + load
    modernbert = FINAL_EPOCHS[MODERNBERT] * N_TRAIN / m_train + N_TEST * m_b1 + load
    curve = sum(
        LEARNING_CURVE_EPOCHS * 77 * k / q_train + N_TEST / q_eval + load for k in LEARNING_CURVE_K
    )
    return {
        "sweep": round(len(SWEEP_LRS) * sweep_one / 60, 1),
        "train_final_per_seed": round(final / 60, 1),
        "train_modernbert": round(modernbert / 60, 1),
        "learning_curve_lora": round(curve / 60, 1),
    }
