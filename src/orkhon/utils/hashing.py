"""Stable content hashing for deterministic splits and artifact manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_unit_interval(key: str) -> float:
    """Map a string to a deterministic float in [0, 1).

    Used for hash-based train/val splitting so the split is identical across runs
    and a document never lands in both train and validation.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # First 8 bytes -> unsigned 64-bit int -> [0, 1)
    value = int.from_bytes(digest[:8], "big")
    return value / float(1 << 64)
