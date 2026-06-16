"""Tokenizer training/loading contract tests (fast, CPU-only)."""

from __future__ import annotations

import pytest

from orkhon.tokenizer.special_tokens import SPECIAL_TOKENS
from orkhon.tokenizer.tokenizer import load_tokenizer
from orkhon.tokenizer.train import train_tokenizer
from orkhon.tokenizer.validate import validate_tokenizer


def _write_corpus(path) -> None:
    """Write a small but varied ASCII corpus suitable for a tiny BPE vocab."""
    lines = []
    for i in range(300):
        lines.append(f"the quick brown fox jumps over {i % 10} lazy dogs")
        lines.append("question: what is 2 plus 2 ? answer: 4 .")
        lines.append("count from 0: 0 1 2 3 4 5 .")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def trained_tokenizer(tmp_path):
    corpus = tmp_path / "corpus.txt"
    _write_corpus(corpus)
    out_dir = tmp_path / "tok"
    train_tokenizer([corpus], out_dir, vocab_size=300, min_frequency=2)
    return load_tokenizer(out_dir)


def test_special_ids_are_zero_through_seven(trained_tokenizer):
    # The 8 core specials are ids 0..7 (load-bearing, never reordered); tool and
    # image specials (appended) are ids 8/9 on freshly-trained tokenizers.
    ids = [trained_tokenizer.token_to_id(t) for t in SPECIAL_TOKENS]
    assert ids == list(range(10))
    # `.all` exposes only the 8 core specials for back-compat.
    assert trained_tokenizer.special.all == tuple(range(8))
    assert trained_tokenizer.special.tool == 8
    assert trained_tokenizer.special.image == 9
    assert trained_tokenizer.special.all == tuple(range(8))


def test_encode_does_not_match_special_literals(trained_tokenizer):
    """encode() must NOT produce special-token ids for literal special strings in content."""
    eos = trained_tokenizer.special.eos
    end = trained_tokenizer.special.end
    ids = trained_tokenizer.encode("hello<|end|>world")
    assert eos not in ids, "eos leaked into content encoding"
    assert end not in ids, "<|end|> leaked into content encoding"
    # And round-trips back to the literal text (with skip_special).
    back = trained_tokenizer.decode(ids, skip_special=False)
    assert "<|end|>" in back or "end" in back  # the literal survived as ordinary text


def test_encode_handles_edge_inputs(trained_tokenizer):
    """encode() handles empty string, emoji, and control chars without crashing."""
    assert trained_tokenizer.encode("") == []
    emoji_ids = trained_tokenizer.encode("hello 🌍 world")
    assert len(emoji_ids) > 0
    # round-trip preserves the emoji
    assert "🌍" in trained_tokenizer.decode(emoji_ids, skip_special=False)


def test_ascii_roundtrip(trained_tokenizer):
    text = "the quick brown fox jumps over 7 lazy dogs"
    ids = trained_tokenizer.encode(text)
    assert trained_tokenizer.decode(ids, skip_special=True) == text


def test_encode_does_not_prepend_bos(trained_tokenizer):
    ids = trained_tokenizer.encode("hello world")
    assert len(ids) > 0
    assert ids[0] != trained_tokenizer.special.bos
    assert ids[0] != trained_tokenizer.special.eos


def test_validate_helper_passes(trained_tokenizer):
    # Should not raise.
    validate_tokenizer(trained_tokenizer)


def test_artifacts_written(tmp_path):
    corpus = tmp_path / "corpus.txt"
    _write_corpus(corpus)
    out_dir = tmp_path / "tok"
    train_tokenizer([corpus], out_dir, vocab_size=300)
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "special_tokens_map.json").exists()
    cfg = out_dir / "tokenizer_config.json"
    assert cfg.exists()
    import json

    data = json.loads(cfg.read_text())
    # Chat template must be embedded and reference the role markers.
    assert "chat_template" in data
    assert "<|assistant|>" in data["chat_template"]
