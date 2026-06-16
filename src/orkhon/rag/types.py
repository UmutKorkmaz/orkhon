"""RAG data types: documents, chunks, and retrieval hits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    path: str
    text: str
    metadata: dict


@dataclass
class Chunk:
    id: int
    path: str
    text: str
    start_line: int
    end_line: int
    metadata: dict = None  # type: ignore[assignment]

    def cite(self) -> str:
        """A compact citation string: ``[doc:path:Lstart-Lend]``."""
        return f"[doc:{self.path}:L{self.start_line}-L{self.end_line}]"


@dataclass
class Hit:
    chunk: Chunk
    score: float
