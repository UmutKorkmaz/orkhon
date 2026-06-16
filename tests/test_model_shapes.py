"""Shape, param-count, and weight-tying tests for the Transformer."""

from __future__ import annotations

from dataclasses import replace

import torch

from orkhon.model.config import ModelConfig
from orkhon.model.transformer import Transformer, build_model


def _tiny_cfg(**overrides) -> ModelConfig:
    base = ModelConfig(
        vocab_size=64,
        block_size=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        dropout=0.0,
        attn_impl="manual",
    )
    return replace(base, **overrides) if overrides else base


def test_forward_output_shape():
    cfg = _tiny_cfg()
    model = build_model(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (3, 7))
    with torch.no_grad():
        logits, past = model(x)
    assert logits.shape == (3, 7, cfg.vocab_size)
    assert past is None


def test_param_count_within_5pct_of_estimate():
    cfg = _tiny_cfg()
    model = Transformer(cfg)
    actual = sum(p.numel() for p in model.parameters())
    estimate = cfg.estimate_params()
    rel_err = abs(actual - estimate) / estimate
    assert rel_err < 0.05, f"actual={actual} estimate={estimate} rel_err={rel_err:.4f}"


def test_param_count_untied_within_5pct():
    cfg = _tiny_cfg(tie_word_embeddings=False)
    model = Transformer(cfg)
    actual = sum(p.numel() for p in model.parameters())
    estimate = cfg.estimate_params()
    rel_err = abs(actual - estimate) / estimate
    assert rel_err < 0.05, f"actual={actual} estimate={estimate} rel_err={rel_err:.4f}"


def test_tied_embedding_shares_storage():
    cfg = _tiny_cfg(tie_word_embeddings=True)
    model = Transformer(cfg)
    # Same Parameter object -> shared storage.
    assert model.lm_head.weight is model.embed_tokens.weight
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()


def test_untied_embedding_separate_storage():
    cfg = _tiny_cfg(tie_word_embeddings=False)
    model = Transformer(cfg)
    assert model.lm_head.weight is not model.embed_tokens.weight
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()
