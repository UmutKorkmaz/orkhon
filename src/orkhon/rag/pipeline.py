"""The RAG ingest pipeline: files -> chunks -> embeddings -> persisted store."""

from __future__ import annotations

from pathlib import Path

from orkhon.rag.chunk import chunk_document
from orkhon.rag.embed import Embedder, build_embedder
from orkhon.rag.loaders import iter_files, load_text
from orkhon.rag.store import VectorStore


def ingest(
    inputs,
    out_dir: str | Path,
    *,
    embed_model: str = "hashing",
    chunk_chars: int = 1200,
    overlap: int = 160,
    batch_size: int = 64,
    device: str = "auto",
) -> dict:
    """Ingest files/dirs into a persisted vector store; return a summary dict.

    ``embed_model="hashing"`` (default) is offline/deterministic; pass a
    sentence-transformers model name for real semantic embeddings.
    """
    out_dir = Path(out_dir)
    inputs = [Path(p) for p in inputs]

    embedder = (build_embedder(embed_model, device=device)
                if embed_model != "hashing" else build_embedder("hashing"))

    all_chunks = []
    cid = 0
    n_docs = 0
    for fp in iter_files(inputs):
        # root = the input that contains it, for relative paths
        root = next((i for i in inputs if i.is_dir() and fp.is_relative_to(i)), fp.parent)
        doc = load_text(fp, root=root)
        if doc is None or not doc.text.strip():
            continue
        n_docs += 1
        chs = chunk_document(doc, chunk_chars=chunk_chars, overlap=overlap, start_id=cid)
        all_chunks.extend(chs)
        cid += len(chs)

    if not all_chunks:
        raise ValueError(f"no ingestible text found under {inputs}")

    texts = [c.text for c in all_chunks]
    vectors = []
    for i in range(0, len(texts), batch_size):
        vectors.append(embedder.embed_documents(texts[i : i + batch_size]))
    import numpy as np
    vectors = np.concatenate(vectors) if vectors else np.zeros((0, embedder.dim), np.float32)

    store = VectorStore.create(out_dir, all_chunks, vectors, embed_model=embed_model)
    return {"docs": n_docs, "chunks": len(store), "dim": store.dim,
            "embed_model": embed_model, "out_dir": str(out_dir)}
