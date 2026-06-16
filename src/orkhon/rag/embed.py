"""Embedders — map text to dense vectors.

Two implementations:
- :class:`HashingEmbedder` — deterministic, dependency-free, used for offline tests and
  as a zero-install fallback. Quality is low but retrieval is exact & reproducible.
- :class:`SentenceTransformerEmbedder` — real semantic embeddings via
  ``sentence-transformers`` (lazy import; add the ``rag`` extra). Default model is a
  small multilingual model so Turkish + English both work.

Both expose ``embed_documents(list[str]) -> np.ndarray [N, D]`` and
``embed_query(str) -> np.ndarray [D]``. For SentenceTransformer, queries are
prefixed with ``"query: "`` per the e5 convention.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_WORD = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """A deterministic bag-of-hashed-words embedder (no deps, for tests/fallback).

    Each token is hashed to one of ``dim`` buckets; the vector is the normalized
    bucket-count histogram. Cosine similarity ≈ lexical overlap, so this is NOT
    semantic — but it's exact, fast, and reproducible for unit tests.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokens:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "big")
            v[h % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(_tokenize(t)) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(_tokenize(text))


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers (lazy; ``rag`` extra)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", *, device: str = "auto") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder needs the 'rag' extra (sentence-transformers); "
                "install it or use HashingEmbedder."
            ) from e
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return self._model.encode(["passage: " + t for t in texts], normalize_embeddings=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(["query: " + text], normalize_embeddings=True)[0]


def build_embedder(name: str = "hashing", **kw) -> Embedder:
    """Factory: ``"hashing"`` (default, offline) or a sentence-transformers model name."""
    if name == "hashing":
        return HashingEmbedder(**kw)
    return SentenceTransformerEmbedder(model_name=name, **kw)
