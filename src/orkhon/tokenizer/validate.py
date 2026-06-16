"""Validation helpers for trained tokenizers.

These are cheap structural checks used by training scripts and tests to fail fast
if a tokenizer is malformed (wrong special ids) or lossy on ASCII round-trips.
"""

from __future__ import annotations

from orkhon.tokenizer.special_tokens import SPECIAL_TOKENS
from orkhon.tokenizer.tokenizer import OrkhonTokenizer

# A representative ASCII sample exercising letters, digits, punctuation, spaces.
_ROUNDTRIP_SAMPLES: tuple[str, ...] = (
    "Hello, world!",
    "The quick brown fox jumps over 13 lazy dogs.",
    "count: 1 2 3 4 5 -> done.",
    "Q: what is 2 + 2? A: 4",
)


def check_special_ids(tok: OrkhonTokenizer) -> None:
    """Assert the special tokens occupy ids 0..7 in declared order."""
    for expected_id, name in enumerate(SPECIAL_TOKENS):
        got = tok.token_to_id(name)
        if got != expected_id:
            raise AssertionError(
                f"special token {name!r} has id {got}, expected {expected_id}"
            )


def check_roundtrip(tok: OrkhonTokenizer, samples: tuple[str, ...] = _ROUNDTRIP_SAMPLES) -> None:
    """Assert ASCII samples survive encode -> decode unchanged.

    ByteLevel BPE is lossless on bytes, so decode(encode(s)) == s for ASCII text.
    """
    for s in samples:
        ids = tok.encode(s)
        out = tok.decode(ids, skip_special=True)
        if out != s:
            raise AssertionError(f"round-trip mismatch: {s!r} -> {out!r}")


def check_no_bos_prefix(tok: OrkhonTokenizer) -> None:
    """Assert ``encode`` does not prepend a bos/eos special id."""
    ids = tok.encode("hello")
    if ids and ids[0] in (tok.special.bos, tok.special.eos):
        raise AssertionError("encode() must not prepend bos/eos special tokens")


def validate_tokenizer(tok: OrkhonTokenizer) -> None:
    """Run all structural checks; raises AssertionError on the first failure."""
    check_special_ids(tok)
    check_roundtrip(tok)
    check_no_bos_prefix(tok)
