"""Frozen `BAAI/bge-small-en-v1.5` sentence embeddings on CPU.

CLS pooling and L2 normalisation, as the model card specifies. No query
instruction prefix: every use here compares a customer message with other
customer messages, which the card calls a symmetric task.

Embeddings are cached as .npy files under `artifacts/embeddings/` (not
committed). Test embeddings are computed one message at a time so the cache
also holds a real per-item latency for the logistic-regression and kNN rows.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from b77.data import Split

BGE_ID = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBED_DIR = Path("artifacts/embeddings")
MAX_LENGTH = 128


class Encoder:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(BGE_ID, revision=BGE_REVISION)
        self.model = AutoModel.from_pretrained(BGE_ID, revision=BGE_REVISION).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        with torch.inference_mode():
            cls = self.model(**enc).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=-1)
        return cls.numpy().astype(np.float32)

    def encode_batched(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        parts = [self.encode(texts[s : s + batch_size]) for s in range(0, len(texts), batch_size)]
        return np.concatenate(parts)

    def encode_timed(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """One message at a time, returning embeddings and wall-clock ms per message."""
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        latency = np.empty(len(texts))
        self.encode(texts[:1])  # warm-up, not timed
        for i, text in enumerate(texts):
            start = time.perf_counter()
            out[i] = self.encode([text])[0]
            latency[i] = (time.perf_counter() - start) * 1000.0
        return out, latency


def cached_embeddings(
    split: Split, encoder: Encoder | None = None, embed_dir: Path = EMBED_DIR, timed: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Embeddings for a split, computed once and cached. `timed` also returns ms per item."""
    embed_dir.mkdir(parents=True, exist_ok=True)
    path = embed_dir / f"bge-small-{split.name}.npy"
    lat_path = embed_dir / f"bge-small-{split.name}-latency-ms.npy"
    if path.exists() and (not timed or lat_path.exists()):
        emb = np.load(path)
        if emb.shape[0] != len(split):
            raise ValueError(f"{path} has {emb.shape[0]} rows, split has {len(split)}")
        return emb, (np.load(lat_path) if timed else None)
    encoder = encoder or Encoder()
    latency = None
    if timed:
        emb, latency = encoder.encode_timed(split.texts)
        np.save(lat_path, latency)
    else:
        emb = encoder.encode_batched(split.texts)
    np.save(path, emb)
    return emb, latency


def top_k(queries: np.ndarray, keys: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine similarities of the k most similar keys, most similar first.

    Both inputs are L2-normalised, so the dot product is the cosine. Ties break
    by lower key index, which keeps the result deterministic.
    """
    sims = queries @ keys.T
    # Stable descending sort: negate, then argsort with a stable algorithm.
    order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(sims, order, axis=1)
