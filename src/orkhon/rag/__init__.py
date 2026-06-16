"""Retrieval-Augmented Generation: ingest, embed, retrieve, cite."""

from orkhon.rag.chunk import chunk_document
from orkhon.rag.embed import HashingEmbedder, build_embedder
from orkhon.rag.loaders import iter_files, load_text
from orkhon.rag.pipeline import ingest
from orkhon.rag.retrieve import format_hits, retrieve
from orkhon.rag.store import VectorStore
from orkhon.rag.types import Chunk, Document, Hit

__all__ = [
    "ingest", "retrieve", "format_hits", "chunk_document",
    "VectorStore", "HashingEmbedder", "build_embedder",
    "iter_files", "load_text",
    "Chunk", "Document", "Hit",
]
