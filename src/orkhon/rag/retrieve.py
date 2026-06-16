"""Retrieve + format: turn a query into cited context for the model."""

from __future__ import annotations

from typing import Iterable

from orkhon.rag.embed import Embedder
from orkhon.rag.store import VectorStore
from orkhon.rag.types import Hit


def retrieve(query: str, store: VectorStore, embedder: Embedder, *,
             top_k: int = 5, min_score: float | None = None) -> list[Hit]:
    hits = store.search(embedder.embed_query(query), top_k=top_k)
    if min_score is not None:
        hits = [h for h in hits if h.score >= min_score]
    return hits


def format_hits(hits: Iterable[Hit], *, max_chars: int = 6000) -> str:
    """Render hits as ``[citation] text`` blocks, capped at ``max_chars`` total."""
    parts: list[str] = []
    used = 0
    for h in hits:
        body = h.chunk.text.strip()
        block = f"{h.chunk.cite()}\n{body}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()
