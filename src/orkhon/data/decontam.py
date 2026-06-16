"""Benchmark decontamination — keep training docs free of eval-set text.

If a held-out benchmark (HellaSwag, ARC, MMLU, TR-MMLU, the built-in eval) appears
verbatim in the pretraining corpus, the model can memorize it and the benchmark
becomes meaningless. This builds an n-gram index (default 13-gram, the standard
length from the GPT-3 / FineWeb decontamination recipe) from benchmark text and
filters out any training document containing a match.

Usage:
    idx = build_ngram_index(benchmark_texts, n=13)
    clean = list(decontaminate(corpus_docs, idx, n=13))
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

_WS = re.compile(r"\s+")
_DEFAULT_N = 13


def _ngrams(text: str, n: int) -> Iterator[str]:
    toks = _WS.sub(" ", text).strip().lower().split()
    if len(toks) < n:
        return
    for i in range(len(toks) - n + 1):
        yield " ".join(toks[i : i + n])


def build_ngram_index(benchmark_texts: Iterable[str], *, n: int = _DEFAULT_N) -> set[str]:
    """Return the set of n-grams across all benchmark texts."""
    idx: set[str] = set()
    for t in benchmark_texts:
        idx.update(_ngrams(t, n))
    return idx


def is_contaminated(doc: str, index: set[str], *, n: int = _DEFAULT_N) -> bool:
    """True if ``doc`` contains any benchmark n-gram."""
    return any(g in index for g in _ngrams(doc, n))


def decontaminate(
    docs: Iterable[str], index: set[str], *, n: int = _DEFAULT_N
) -> Iterator[str]:
    """Yield documents with NO benchmark n-gram (drops contaminated docs)."""
    for doc in docs:
        if not is_contaminated(doc, index, n=n):
            yield doc
