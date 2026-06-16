"""RAG tests — offline (HashingEmbedder, no model downloads)."""

from __future__ import annotations

from pathlib import Path

from orkhon.rag import (
    VectorStore,
    chunk_document,
    format_hits,
    ingest,
    retrieve,
)
from orkhon.rag.embed import HashingEmbedder
from orkhon.rag.types import Document
from orkhon.serve.tools.retrieve import Retrieve


def _doc(path: str, text: str) -> Document:
    return Document(path=path, text=text, metadata={})


def test_chunk_preserves_line_spans_and_overlap():
    text = "\n".join(f"line {i}" for i in range(40))
    chunks = chunk_document(_doc("f.txt", text), chunk_chars=60, overlap=20)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.start_line >= 1 and c.end_line >= c.start_line
        assert "[doc:f.txt:L" in c.cite()
    # overlap: the second chunk starts before the first ends (in characters)
    assert chunks[1].start_line <= chunks[0].end_line + 1


def test_ingest_and_retrieve_roundtrip(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# C3 RAG\n\nOrkhon can chat with files via retrieval.\n", encoding="utf-8")
    (root / "b.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    summary = ingest([root], tmp_path / "idx", embed_model="hashing", chunk_chars=200, overlap=20)
    assert summary["docs"] == 2 and summary["chunks"] >= 2

    store = VectorStore.load(tmp_path / "idx")
    emb = HashingEmbedder()
    hits = retrieve("retrieval files", store, emb, top_k=3)
    assert hits, "expected at least one hit"
    # the top hit is the RAG doc, not the python file (lexical match)
    assert "retrieval" in hits[0].chunk.text.lower() or "rag" in hits[0].chunk.text.lower()
    # every citation maps to a stored chunk path
    paths = {c.path for c in store.chunks}
    assert all(h.chunk.path in paths for h in hits)


def test_retrieve_tool_returns_cited_context(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "x.md").write_text("The secret code is 4242. Keep it safe.\n" * 3, encoding="utf-8")
    ingest([root], tmp_path / "idx", embed_model="hashing", chunk_chars=400, overlap=0)
    tool = Retrieve(str(tmp_path / "idx"))
    res = tool(query="secret code")
    assert not res.error
    assert "[doc:" in res.output and "4242" in res.output


def test_format_hits_caps_length(tmp_path):
    from orkhon.rag.types import Chunk, Hit
    chunks = [Chunk(id=i, path="f", text="x" * 100, start_line=1, end_line=2, metadata={})
              for i in range(50)]
    hits = [Hit(chunk=c, score=1.0) for c in chunks]
    out = format_hits(hits, max_chars=300)
    assert len(out) <= 300


def test_zero_norm_query_returns_no_hits(tmp_path):
    # An empty/punctuation-only query (zero-norm vector) must not return arbitrary chunks.
    ingest([_write(tmp_path, "d.md", "some content here")], tmp_path / "idx",
           embed_model="hashing", chunk_chars=200, overlap=0)
    store = VectorStore.load(tmp_path / "idx")
    hits = retrieve("!!! ???", store, HashingEmbedder(), top_k=3)
    assert hits == []


def test_line_span_no_trailing_newline_inflation(tmp_path):
    # "a\n" cited from "a\nb" should be L1-L1, not L1-L2 (no line-2 content).
    from orkhon.rag.chunk import chunk_document
    from orkhon.rag.types import Document
    chunks = chunk_document(Document("f", "a\nb", {}), chunk_chars=2, overlap=0)
    first = chunks[0]
    assert first.text == "a\n"
    assert first.start_line == 1 and first.end_line == 1


def test_corrupt_index_is_rejected(tmp_path):
    # Vector/chunk count mismatch must raise on load (citation-integrity guard).
    import json as _json
    import numpy as np
    d = tmp_path / "idx"
    d.mkdir()
    (d / "chunks.jsonl").write_text(_json.dumps(
        {"id": 0, "path": "f", "text": "x", "start_line": 1, "end_line": 1, "metadata": {}}) + "\n")
    np.save(d / "vectors.npy", np.zeros((3, 4), np.float32))  # 3 vectors, 1 chunk
    (d / "meta.json").write_text(_json.dumps({"dim": 4, "embed_model": "hashing", "n_chunks": 3}))
    import pytest
    with pytest.raises(ValueError):
        VectorStore.load(d)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p
