"""A persisted vector store over document chunks.

Layout (under ``index_dir``):
    meta.json     {dim, embed_model, n_chunks, created}
    chunks.jsonl  one Chunk (as dict) per line
    vectors.npy   [N, D] float32 document embeddings

Search is cosine similarity (vectors are assumed normalized by the embedder; we
normalize defensively at search time). No external vector DB — a flat numpy matrix
is plenty for thousands of chunks and keeps the dependency surface minimal.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from orkhon.rag.types import Chunk, Hit


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, *, dim: int,
                 embed_model: str) -> None:
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.dim = dim
        self.embed_model = embed_model

    @classmethod
    def create(cls, index_dir: str | Path, chunks: list[Chunk], vectors: np.ndarray, *,
               embed_model: str) -> "VectorStore":
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        # Citation-integrity invariant: every vector row maps to exactly one chunk.
        if vectors.size:
            if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
                raise ValueError(
                    f"vectors {vectors.shape} must be [n_chunks={len(chunks)}, D]; "
                    "a mismatch would return the wrong chunk per citation"
                )
        dim = int(vectors.shape[1]) if vectors.size else 0
        with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as w:
            for c in chunks:
                w.write(json.dumps({
                    "id": c.id, "path": c.path, "text": c.text,
                    "start_line": c.start_line, "end_line": c.end_line, "metadata": c.metadata,
                }, ensure_ascii=False) + "\n")
        if vectors.size:
            np.save(index_dir / "vectors.npy", vectors)
        meta = {"dim": dim, "embed_model": embed_model, "n_chunks": len(chunks),
                "created": datetime.now().isoformat(timespec="seconds")}
        (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return cls(chunks, vectors, dim=dim, embed_model=embed_model)

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        chunks: list[Chunk] = []
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                chunks.append(Chunk(id=d["id"], path=d["path"], text=d["text"],
                                    start_line=d["start_line"], end_line=d["end_line"],
                                    metadata=d.get("metadata")))
        vec_path = index_dir / "vectors.npy"
        vectors = np.load(vec_path) if vec_path.exists() else np.zeros((0, meta.get("dim", 0)), np.float32)
        # Citation-integrity check: vector rows must line up with chunks + meta.
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"corrupt index: {vectors.shape[0]} vectors vs {len(chunks)} chunks"
            )
        if meta.get("n_chunks") is not None and meta["n_chunks"] != len(chunks):
            raise ValueError(
                f"corrupt index: meta.n_chunks={meta['n_chunks']} vs {len(chunks)} chunks"
            )
        return cls(chunks, vectors, dim=int(meta.get("dim", 0)), embed_model=meta.get("embed_model", ""))

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vec: np.ndarray, *, top_k: int = 5) -> list[Hit]:
        if self.vectors.shape[0] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        qn = np.linalg.norm(q)
        if qn == 0:  # empty/punctuation-only query -> no meaningful signal
            return []
        q = q / qn
        vn = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        vn[vn == 0] = 1.0
        normed = self.vectors / vn
        scores = normed @ q  # cosine (both normalized)
        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [Hit(chunk=self.chunks[i], score=float(scores[i])) for i in idx]
