"""Chunk documents into overlapping windows with line-span metadata for citations."""

from __future__ import annotations

from orkhon.rag.types import Chunk, Document

_DEFAULT_CHARS = 1200
_DEFAULT_OVERLAP = 160


def chunk_document(
    doc: Document,
    *,
    chunk_chars: int = _DEFAULT_CHARS,
    overlap: int = _DEFAULT_OVERLAP,
    start_id: int = 0,
) -> list[Chunk]:
    """Split ``doc`` into chunks of ~``chunk_chars`` with ``overlap`` between them.

    Each chunk records the line span it covers (for citations). Chunks never split a
    line in the middle — they advance by ``chunk_chars - overlap`` character steps.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must be in [0, chunk_chars)")

    text = doc.text
    chunks: list[Chunk] = []
    cid = start_id
    n = len(text)
    pos = 0
    step = chunk_chars - overlap
    while pos < n:
        end = min(pos + chunk_chars, n)
        slice_ = text[pos:end]
        if slice_.strip():  # skip empty/whitespace-only chunks
            # Line span from the FIRST to the LAST included character, inclusive.
            # A trailing newline (no line-N content) does not bump the end line.
            start_line = text.count("\n", 0, pos) + 1
            last_char = end - 1
            end_line = text.count("\n", 0, last_char) + 1
            chunks.append(Chunk(id=cid, path=doc.path, text=slice_,
                                start_line=start_line, end_line=end_line,
                                metadata={"chars": len(slice_)}))
            cid += 1
        pos += step
    return chunks
