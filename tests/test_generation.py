"""Tests for autoregressive generation."""

from __future__ import annotations

import torch

from orkhon.model.config import ModelConfig
from orkhon.model.generation import generate
from orkhon.model.transformer import Transformer


def _tiny_model() -> Transformer:
    cfg = ModelConfig(
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
    torch.manual_seed(0)
    return Transformer(cfg).eval()


def test_generate_returns_max_new_tokens():
    model = _tiny_model()
    prompt = [1, 2, 3]
    out = generate(model, prompt, max_new_tokens=5, temperature=0.0)
    assert isinstance(out, list)
    assert len(out) == 5
    # Returns only NEW ids (prompt not echoed).
    assert all(isinstance(i, int) for i in out)


def test_greedy_is_deterministic():
    model = _tiny_model()
    prompt = [4, 5, 6, 7]
    out1 = generate(model, prompt, max_new_tokens=6, temperature=0.0)
    out2 = generate(model, prompt, max_new_tokens=6, temperature=0.0)
    assert out1 == out2


def test_generate_stops_at_eos():
    model = _tiny_model()
    prompt = [1, 2]
    # First greedy token is deterministic; use it as the eos to force an early stop.
    first = generate(model, prompt, max_new_tokens=1, temperature=0.0)[0]

    out = generate(model, prompt, max_new_tokens=10, temperature=0.0, eos_ids=(first,))
    # Should stop immediately after producing the eos token.
    assert out == [first]


def test_generate_accepts_tensor_prompt():
    model = _tiny_model()
    prompt = torch.tensor([1, 2, 3])
    out = generate(model, prompt, max_new_tokens=3, temperature=0.0)
    assert len(out) == 3


def test_sampling_runs_with_topk_topp():
    model = _tiny_model()
    torch.manual_seed(42)
    out = generate(
        model, [1, 2, 3], max_new_tokens=4, temperature=1.0, top_k=10, top_p=0.9
    )
    assert len(out) == 4
    assert all(0 <= i < model.cfg.vocab_size for i in out)
