"""Local embeddings and reranking (ONNX models via fastembed, no API key, no GPU)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DIM = 384
CACHE = str(Path.home() / ".cache" / "fastembed")


@lru_cache(maxsize=1)
def _embedder() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL, cache_dir=CACHE)


@lru_cache(maxsize=1)
def _reranker() -> TextCrossEncoder:
    return TextCrossEncoder(RERANK_MODEL, cache_dir=CACHE)


def embed_passages(texts: list[str]) -> list[np.ndarray]:
    return list(_embedder().passage_embed(texts, batch_size=64))


def embed_query(text: str) -> np.ndarray:
    return next(iter(_embedder().query_embed(text)))


def rerank(query: str, docs: list[str]) -> list[float]:
    return list(_reranker().rerank(query, docs))
