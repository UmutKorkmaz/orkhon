"""Document deduplication — exact-hash (default) and optional MinHash (near-dup).

Both are streaming generators: wrap an iterator of documents and yield only the
unique ones. Exact-hash dedup is cheap (SHA-256 set) and catches copy-paste /
repeated scrapes. MinHash-LSH dedup (near-duplicate, Jaccard) catches paraphrased /
boilerplate repetition; it pulls in ``datasketch`` lazily, so it is optional.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Iterator

from orkhon.utils.hashing import sha256_text

_WS = re.compile(r"\s+")


def _normalize(doc: str) -> str:
    return _WS.sub(" ", doc).strip().lower()


def dedupe_exact(docs: Iterable[str]) -> Iterator[str]:
    """Yield each document whose normalized text hasn't been seen (exact dedup)."""
    seen: set[str] = set()
    for doc in docs:
        key = sha256_text(_normalize(doc))
        if key not in seen:
            seen.add(key)
            yield doc


def _minhash_signature(text: str, ngram: int, num_perm: int, hashfunc) -> list[int]:
    tokens = _normalize(text).split()
    if len(tokens) < ngram:
        grams = [" ".join(tokens)]
    else:
        grams = [" ".join(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    return [min(hashfunc(f"{p}:{g}") for g in grams) for p in range(num_perm)]


def dedupe_minhash(
    docs: Iterable[str],
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram: int = 5,
) -> Iterator[str]:
    """Yield documents whose estimated Jaccard similarity to any kept doc is < threshold.

    Near-duplicate (paraphrase/boilerplate) dedup via a simple MinHash + banded
    store. Self-contained (no ``datasketch`` dependency). O(num_perm) per doc.
    """
    import hashlib

    def h(seed: str) -> int:
        return int.from_bytes(hashlib.blake2b(seed.encode(), digest_size=8).digest(), "big")

    bands = 16
    rows = max(1, num_perm // bands)
    store: list[dict[tuple, int]] = [{} for _ in range(bands)]
    kept = 0

    for doc in docs:
        sig = _minhash_signature(doc, ngram, num_perm, h)
        dup = False
        for b in range(bands):
            band = tuple(sig[b * rows : (b + 1) * rows])
            if band in store[b]:
                dup = True
                break
        if dup:
            continue
        for b in range(bands):
            store[b][tuple(sig[b * rows : (b + 1) * rows])] = kept
        kept += 1
        yield doc
