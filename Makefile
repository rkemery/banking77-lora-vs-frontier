.PHONY: install lint format test demo data splits embed baselines timing smoke dry-runs sweep \
	train-final train-modernbert learning-curve estimate notebook \
	prompt-luna-zeroshot prompt-luna-fewshot prompt-sol-fewshot

SEED ?= 0

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -q

# Offline, no keys, no models: rebuilds the README results section from results/.
demo:
	uv run b77 report

# ---- data (seconds) -------------------------------------------------------------
data:
	uv run b77 data

splits: data
	uv run b77 splits

# ---- cheap local steps (minutes) ------------------------------------------------
# bge-small embeddings for train and test, and the retrieved-neighbour file.
embed: data
	uv run b77 embed

# Logistic regression (full data and the learning curve) and the kNN vote.
baselines: embed
	uv run b77 baselines

# Throughput and latency on this CPU, written to results/timing.json.
timing: data
	uv run b77 timing

smoke: data
	uv run b77 smoke --model qwen3-0.6b --steps 200
	uv run b77 smoke --model modernbert-base --steps 200

# Every long target for 3 steps on a few items, under artifacts/dry-runs/. A few minutes.
dry-runs: data
	uv run b77 sweep --dry-run
	uv run b77 train-final --dry-run
	uv run b77 train-modernbert --dry-run
	uv run b77 learning-curve --k 5 --dry-run

# ---- long CPU runs (see results/timing.json for the measured estimates) ---------
# Qwen3-0.6B LoRA at lr 1e-4 and 3e-4 on the 2,000-example subset, scored on dev.
sweep: data
	uv run b77 sweep

# Qwen3-0.6B LoRA on all 10,003 training messages with the sweep's lr, then test at batch 1.
# More seeds: make train-final SEED=1
train-final: data
	uv run b77 train-final --seed $(SEED)

# ModernBERT-base full fine-tune on all of train, then test at batch 1.
train-modernbert: data
	uv run b77 train-modernbert --seed $(SEED)

# Qwen3-0.6B LoRA at 5, 10 and 20 examples per class (draw 0), scored on test.
learning-curve: data
	uv run b77 learning-curve

# ---- live prompting arms (cost money, need AZURE_OPENAI_BASE_URL) ---------------
# Token, cost and wall-clock estimates for the three arms. Offline.
estimate: data
	uv run b77 estimate --write

prompt-luna-zeroshot: data
	uv run b77 prompt --arm luna-zeroshot --live --cap 2

prompt-luna-fewshot: data
	uv run b77 prompt --arm luna-fewshot --live --cap 2

prompt-sol-fewshot: data
	uv run b77 prompt --arm sol-fewshot --live --cap 20

notebook:
	uv run python scripts/make_colab_notebook.py
