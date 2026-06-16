"""Retrieve tool — lets the tool-loop model query a RAG index for cited context."""

from __future__ import annotations

from orkhon.rag.embed import build_embedder
from orkhon.rag.retrieve import format_hits, retrieve
from orkhon.rag.store import VectorStore
from orkhon.serve.tools.base import ToolResult


class Retrieve:
    """Search a persisted RAG index and return cited chunks to the model."""

    name = "retrieve"
    description = (
        "Search the document index for a query and return cited passages. "
        "Use this to answer questions about the indexed files."
    )
    parameters = {"query": {"type": "string"}, "top_k": {"type": "integer"}}

    def __init__(self, index_dir: str, *, embed_model: str = "hashing", top_k: int = 5) -> None:
        self.store = VectorStore.load(index_dir)
        # If the index was built with hashing, honor it; otherwise use the same model.
        self.embedder = build_embedder(self.store.embed_model or embed_model)
        self.default_top_k = top_k

    def __call__(self, *, query: str, top_k: int | None = None, **_) -> ToolResult:
        try:
            hits = retrieve(query, self.store, self.embedder, top_k=top_k or self.default_top_k)
            if not hits:
                return ToolResult(output="(no matching documents found)")
            return ToolResult(output=format_hits(hits))
        except Exception as e:
            return ToolResult(output=f"retrieve error: {e}", error=True)
