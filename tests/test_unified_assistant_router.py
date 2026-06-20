"""Tests for deterministic unified-assistant routing."""

from __future__ import annotations

from orkhon.assistant import deterministic_reply
from orkhon.data.old_turkic import rune_to_latin


def test_router_exactly_transliterates_old_turkic_runes():
    runes = "𐰃𐰡𐰞𐰜 𐰚𐰇𐰶"
    out = deterministic_reply(f"Transliterate this Old Turkic text into Latin: {runes}")

    assert out == rune_to_latin(runes)


def test_router_declines_unsourced_translation_but_gives_transliteration():
    runes = "𐰃𐰡𐰞𐰜"
    out = deterministic_reply(f"Translate this Old Turkic inscription to modern Turkish: {runes}")

    assert out is not None
    assert "needs sourced inscription data" in out
    assert rune_to_latin(runes) in out


def test_router_answers_simple_addition_in_english_and_turkish():
    assert deterministic_reply("What is 7 + 5?") == "The answer is 12."
    assert deterministic_reply("7 + 5 kactir?") == "Cevap 12."


def test_router_leaves_open_ended_prompts_to_the_model():
    assert deterministic_reply("What can you help me with?") is None
