"""Tests for the tokenizer fertility gate."""

from __future__ import annotations

from orkhon.tokenizer.fertility import evaluate_fertility_gate, measure_fertility
from orkhon.tokenizer.train import train_tokenizer


def _write_corpus(path) -> None:
    lines = []
    for _ in range(250):
        lines.append("Orkhon learns English web text and code patterns.")
        lines.append("Orkhon Türkçe ekleri, ünlüleri ve anlamlı kökleri öğrenir.")
        lines.append("𐰋𐰃𐰠𐰏𐰀 𐰴𐰍𐰣 𐱅𐰇𐰼𐰰")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_measure_fertility_reports_core_metrics(tmp_path):
    corpus = tmp_path / "corpus.txt"
    _write_corpus(corpus)
    tok = tmp_path / "tok"
    train_tokenizer([corpus], tok, vocab_size=600)

    metrics = measure_fertility(tok)
    assert metrics.english_bytes_per_token > 0
    assert metrics.turkish_bytes_per_token > 0
    assert metrics.old_turkic_tokens_per_rune > 0
    assert metrics.old_turkic_runes > 0


def test_evaluate_fertility_gate_can_pass_with_explicit_targets(tmp_path):
    corpus = tmp_path / "corpus.txt"
    _write_corpus(corpus)
    tok = tmp_path / "tok"
    train_tokenizer([corpus], tok, vocab_size=600)

    gate = evaluate_fertility_gate(
        tok,
        min_turkish_bytes_per_token=1.0,
        max_old_turkic_tokens_per_rune=10.0,
    )
    assert gate.passed
    assert gate.checks == {"turkish_target": True, "old_turkic_target": True}
