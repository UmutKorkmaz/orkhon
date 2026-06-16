"""L0 long-context RoPE scaling: a model can infer beyond its trained block_size."""

from __future__ import annotations

import torch

from orkhon.model import Transformer, generate
from orkhon.model.config import ModelConfig
from orkhon.model.rope import _yarn_mscale, precompute_rope


def test_yarn_mscale():
    assert abs(_yarn_mscale(1.0) - 1.0) < 1e-6
    assert _yarn_mscale(4.0) > 1.0  # larger factor -> larger mscale


def test_scaled_rope_table_extends_beyond_block_size():
    """With scaling, the RoPE table is sized to block_size * factor, not block_size."""
    cfg = ModelConfig(vocab_size=64, block_size=16, n_layers=2, d_model=32, n_heads=4,
                      n_kv_heads=2, intermediate_size=64, rope_scaling_type="ntk",
                      rope_scaling_factor=4.0)
    m = Transformer(cfg)
    # The table has 16*4 = 64 positions, not 16.
    assert m.rope_cos.shape[0] == 64
    assert m._rope_max_seq == 64


def test_model_decodes_past_trained_block_size():
    """A model trained at block_size=16 can generate to position 40+ with yarn scaling."""
    cfg = ModelConfig(vocab_size=64, block_size=16, n_layers=2, d_model=32, n_heads=4,
                      n_kv_heads=2, intermediate_size=64, rope_scaling_type="ntk",
                      rope_scaling_factor=4.0)
    m = Transformer(cfg).eval()
    # Prompt of 8 tokens + 35 new = position 43 > trained block_size 16.
    out = generate(m, [1, 2, 3, 4, 5, 6, 7, 8], max_new_tokens=35,
                   temperature=0.0, eos_ids=(), device="cpu")
    assert len(out) == 35  # did not crash at block_size 16


def test_linear_scaling_divides_positions():
    """Linear scaling: positions are divided by the factor (position interpolation)."""
    cos_base, _ = precompute_rope(8, max_seq=4, theta=10000.0)
    cos_lin, _ = precompute_rope(8, max_seq=4, theta=10000.0, scaling_type="linear", scaling_factor=2.0)
    # Position 2 with linear/factor=2 == position 1 without scaling.
    assert torch.allclose(cos_lin[2], cos_base[1], atol=1e-5)
