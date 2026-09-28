# Data and model sources

Every download is pinned to a revision. Licenses were checked on 2026-09-28 against the
Hugging Face API (`license:` tag and model or dataset card) and, for Banking77, the
upstream repository's LICENSE file.

## Dataset

| Name | Revision | License | URL |
|---|---|---|---|
| `legacy-datasets/banking77` (parquet mirror of PolyAI's Banking77) | `f54121560de48f2852f90be299010d1d6dc612ec` | CC-BY-4.0 (HF card and tag. The upstream `PolyAI-LDN/task-specific-datasets` LICENSE is "Attribution 4.0 International".) | https://huggingface.co/datasets/legacy-datasets/banking77 |

Files and sha256, checked on every load by `b77 data`:

| File | Rows | sha256 |
|---|---|---|
| `data/train-00000-of-00001.parquet` | 10,003 | `3c648a31689f4ab3acbd4f4f4d120944bb521cf6ba57da77aa87c57e8979be81` |
| `data/test-00000-of-00001.parquet` | 3,080 | `318da70fb77a0e01bcfaecc97ef6e3645313aab98c429f0f0630c9d48e703ecc` |

The raw files are not committed. `make data` downloads them into `data/raw/` and checks the
hashes. The derived files in `data/splits/` (dev, sweep and learning-curve ids, the
deduplicated test ids, near-twin evidence and the retrieved neighbours) are committed and
contain item ids, label ids and similarity scores, not message text.

`PolyAI/banking77` itself is still a loading-script repo, which current `datasets` releases
no longer run, so this repo reads the parquet mirror above.

Attribution: Casanueva, Temcinas, Gerz, Henderson and Vulic (2020), "Efficient Intent
Detection with Dual Sentence Encoders", arXiv 2003.04807.

## Models

| Model | Revision | License | Used for |
|---|---|---|---|
| `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | Apache-2.0 | LoRA with a classification head, CPU |
| `answerdotai/ModernBERT-base` | `8949b909ec900327062f0ebf497f51aef5e6f0c8` | Apache-2.0 | Full fine-tune baseline, CPU |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | MIT | Frozen embeddings (logistic regression, kNN) and few-shot retrieval |
| `Qwen/Qwen3-8B-Base` | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` | Apache-2.0 | Optional Colab QLoRA notebook, not run |

## Hosted models (Azure AI Foundry, pay per token)

`gpt-6-luna` and `gpt-6-sol`, called through the Responses API by deployment name. Prices are
the list prices in `llm-eval-harness` (`DEFAULT_PRICES`, 2026-09-28): luna $0.10 / $0.01 /
$0.50 and sol $2.00 / $0.20 / $10.00 per million input / cached input / output tokens.
