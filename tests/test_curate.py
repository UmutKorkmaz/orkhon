"""Tests for dedup + decontam filters."""

from __future__ import annotations

from orkhon.data.dedupe import dedupe_exact, dedupe_minhash
from orkhon.data.decontam import (
    build_ngram_index,
    decontaminate,
    is_contaminated,
)


def test_dedupe_exact_drops_repeats_and_is_whitespace_insensitive():
    docs = [
        "The quick brown fox",
        "the quick   brown fox",  # same after normalize
        "A totally different document",
        "The quick brown fox",  # exact repeat
    ]
    out = list(dedupe_exact(docs))
    assert out == ["The quick brown fox", "A totally different document"]


def test_dedupe_minhash_keeps_distinct_drops_near_dup():
    a = "the cat sat on the mat and the cat sat on the mat again today"
    b = "the cat sat on the mat and the cat sat on the mat again now"  # near-identical (1 word)
    c = "meanwhile in a galaxy far far away a completely different story unfolded"
    out = list(dedupe_minhash([a, b, c], threshold=0.5, num_perm=64, ngram=4))
    assert a in out and c in out
    assert b not in out  # dropped as a near-duplicate of a


def test_decontam_builds_index_and_filters():
    benchmarks = ["the only thing we have to fear is fear itself said the president"]
    idx = build_ngram_index(benchmarks, n=6)
    assert is_contaminated("the only thing we have to fear is fear itself end", idx, n=6)
    assert not is_contaminated("a completely unrelated sentence about bicycles", idx, n=6)
    clean = list(
        decontaminate(
            ["the only thing we have to fear is fear itself end",
             "an innocent document with no benchmark overlap at all here"],
            idx, n=6,
        )
    )
    assert clean == ["an innocent document with no benchmark overlap at all here"]


def test_decontam_too_short_doc_is_clean():
    idx = build_ngram_index(["some benchmark sentence here"], n=13)
    assert not is_contaminated("short doc", idx, n=13)
