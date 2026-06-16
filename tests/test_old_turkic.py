"""Tests for the Old Turkic transliteration layer (deterministic, no training)."""

from __future__ import annotations

from orkhon.data.old_turkic import (
    OLD_TURKIC_RANGE,
    contains_old_turkic,
    rune_inventory,
    rune_to_latin,
)
from orkhon.data.old_turkic.synth import make_translit_sft


def test_unicode_block_present():
    inv = rune_inventory()
    assert len(inv) >= 60  # the Old Turkic block has 73 named runes
    # every key is in the block
    lo, hi = OLD_TURKIC_RANGE
    assert all(lo <= ord(ch) <= hi for ch in inv)


def test_rune_to_latin_is_deterministic_and_value_based():
    a = chr(0x10C00)  # OLD TURKIC LETTER ORKHON A -> 'a'
    assert rune_to_latin(a) == "a"
    # deterministic
    assert rune_to_latin(a + a) == rune_to_latin(a + a)
    # non-rune chars pass through
    assert rune_to_latin("x " + a + "!") == "x a!"
    # separator only between runes
    assert rune_to_latin(a + a, sep="·") == "a·a"


def test_contains_old_turkic():
    assert contains_old_turkic("merhaba " + chr(0x10C00))
    assert not contains_old_turkic("merhaba dünya")


def test_synthetic_sft_is_correct_by_construction(tmp_path):
    import json

    out = tmp_path / "t.jsonl"
    n = make_translit_sft(out, n=50, seed=1)
    assert n == 50
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        u, a = r["messages"][0]["content"], r["messages"][1]["content"]
        runic = u.split(": ", 1)[1]
        # the assistant answer must equal the deterministic transliteration of the runes
        assert a == rune_to_latin(runic)
