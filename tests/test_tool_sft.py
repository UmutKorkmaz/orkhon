"""Tool-use SFT tests: tokenizer retrofit, embedding resize, trace synthesis."""

from __future__ import annotations

import torch

from orkhon.data.tool_synth import make_tool_sft
from orkhon.model import Transformer
from orkhon.model.config import ModelConfig
from orkhon.model.resize import resize_token_embeddings
from orkhon.serve.tool_protocol import parse_tool_call
from orkhon.tokenizer.retrofit import ensure_tool_tokens
from orkhon.tokenizer.train import train_tokenizer


def _tokenizer(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(["the cat sat on the mat"] * 40 + ["a dog ran fast"] * 40))
    d = tmp_path / "tok"
    train_tokenizer(corpus_paths=str(corpus), out_dir=str(d), vocab_size=300)
    return d


def test_retrofit_preserves_old_ids_and_appends_tool(tmp_path, monkeypatch):
    # Simulate an OLD tokenizer (8 specials, no <|tool|>/<image>) by patching the
    # SPECIAL_TOKENS binding that train.py actually uses.
    from orkhon.tokenizer import special_tokens, train as train_mod
    old8 = special_tokens.SPECIAL_TOKENS[:8]
    monkeypatch.setattr(train_mod, "SPECIAL_TOKENS", old8)

    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(["the cat sat on the mat"] * 40 + ["a dog ran fast"] * 40))
    src = tmp_path / "tok"
    train_mod.train_tokenizer(corpus_paths=str(corpus), out_dir=str(src), vocab_size=300)

    from orkhon.tokenizer import load_tokenizer
    old = load_tokenizer(src)
    old_ids = {w: old.token_to_id(w) for w in ["the", "cat", "sat"]}
    assert old.special.tool is None  # old tokenizer: no tool token

    res = ensure_tool_tokens(src, tmp_path / "tok_tool")
    assert res["vocab_after"] == res["vocab_before"] + 2  # +tool +image
    new = load_tokenizer(tmp_path / "tok_tool")
    # EVERY old token id is unchanged.
    for w, i in old_ids.items():
        assert new.token_to_id(w) == i
    # The tool/image tokens are appended at the end.
    assert new.special.tool == res["vocab_before"]
    assert new.special.image == res["vocab_before"] + 1


def test_retrofit_is_idempotent_on_modern_tokenizer(tmp_path):
    # A tokenizer that already has the tool tokens -> no-op (adds nothing).
    src = _tokenizer(tmp_path)  # freshly trained with current 10 specials
    res = ensure_tool_tokens(src, tmp_path / "tok_tool2")
    assert res["added"] == []
    assert res["vocab_after"] == res["vocab_before"]


def test_resize_preserves_old_rows_and_grows():
    cfg = ModelConfig(vocab_size=64, block_size=32, n_layers=2, d_model=32, n_heads=4,
                      n_kv_heads=2, intermediate_size=64, tie_word_embeddings=True)
    m = Transformer(cfg)
    old_row0 = m.embed_tokens.weight.data[0].clone()
    old_row63 = m.embed_tokens.weight.data[63].clone()
    m, cfg2 = resize_token_embeddings(m, 66)
    assert cfg2.vocab_size == 66
    assert m.embed_tokens.weight.shape == (66, 32)
    # Old rows preserved verbatim.
    assert torch.allclose(m.embed_tokens.weight.data[0], old_row0)
    assert torch.allclose(m.embed_tokens.weight.data[63], old_row63)
    # New rows exist and are finite.
    assert torch.isfinite(m.embed_tokens.weight.data[64]).all()
    # Tied lm_head shares storage with the resized embedding.
    assert m.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr()
    # forward works at the new vocab.
    import torch as T
    logits, _ = m(T.randint(0, 66, (1, 8)))
    assert logits.shape == (1, 8, 66)


def test_resize_rejects_shrink():
    import pytest
    cfg = ModelConfig(vocab_size=64, block_size=32, n_layers=2, d_model=32, n_heads=4,
                      n_kv_heads=2, intermediate_size=64)
    with pytest.raises(ValueError):
        resize_token_embeddings(Transformer(cfg), 60)


def test_resize_untied_preserves_rows_and_device():
    # The untied lm_head branch + device/dtype preservation (codex HIGH).
    cfg = ModelConfig(vocab_size=64, block_size=32, n_layers=2, d_model=32, n_heads=4,
                      n_kv_heads=2, intermediate_size=64, tie_word_embeddings=False)
    m = Transformer(cfg)
    old_emb0 = m.embed_tokens.weight.data[0].clone()
    old_head5 = m.lm_head.weight.data[5].clone()
    dev = m.embed_tokens.weight.device
    m, cfg2 = resize_token_embeddings(m, 68)
    assert cfg2.vocab_size == 68
    assert m.embed_tokens.weight.shape == (68, 32)
    assert m.lm_head.weight.shape == (68, 32)  # untied head also grew
    assert torch.allclose(m.embed_tokens.weight.data[0], old_emb0)
    assert torch.allclose(m.lm_head.weight.data[5], old_head5)
    # New modules stay on the same device as the original weights.
    assert m.embed_tokens.weight.device == dev
    assert m.lm_head.weight.device == dev
    # lm_head is NOT tied (separate storage).
    assert m.lm_head.weight.data_ptr() != m.embed_tokens.weight.data_ptr()


def test_tool_synth_traces_parse(tmp_path):
    n = make_tool_sft(tmp_path / "sft.jsonl", out_val=tmp_path / "val.jsonl", n=200, seed=1)
    assert n["train"] + n["val"] == 200
    rows = [eval or _ for _ in open(tmp_path / "sft.jsonl")]  # noqa
    import json
    rows = [json.loads(l) for l in open(tmp_path / "sft.jsonl")]
    # Every assistant turn containing <|tool|> must parse into a valid tool call.
    parsed = 0
    for r in rows:
        for msg in r["messages"]:
            if msg["role"] == "assistant" and "<|tool|>" in msg["content"]:
                c = parse_tool_call(msg["content"])
                assert c is not None and c.name in {"calculator", "retrieve"}
                parsed += 1
    assert parsed > 0  # at least some tool-call traces
