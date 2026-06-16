"""Tests for grouped-query attention: repeat_kv, sdpa/manual parity, causal mask."""

from __future__ import annotations

from dataclasses import replace

import torch

from orkhon.model.attention import Attention, repeat_kv
from orkhon.model.config import ModelConfig
from orkhon.model.rope import precompute_rope


def _tiny_cfg(attn_impl: str = "manual") -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        block_size=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        dropout=0.0,
        attn_impl=attn_impl,
    )


def test_repeat_kv_maps_head_to_kv_head():
    # n_kv_heads=2, n_rep=2 -> query head h uses kv head h // n_rep.
    b, t, n_kv, hd = 1, 3, 2, 4
    n_rep = 2
    x = torch.arange(b * t * n_kv * hd, dtype=torch.float32).view(b, t, n_kv, hd)
    out = repeat_kv(x, n_rep)

    assert out.shape == (b, t, n_kv * n_rep, hd)
    # Heads 0,1 -> kv head 0; heads 2,3 -> kv head 1 (contiguous expansion).
    assert torch.allclose(out[:, :, 0], x[:, :, 0])
    assert torch.allclose(out[:, :, 1], x[:, :, 0])
    assert torch.allclose(out[:, :, 2], x[:, :, 1])
    assert torch.allclose(out[:, :, 3], x[:, :, 1])


def test_repeat_kv_identity_when_n_rep_one():
    x = torch.randn(1, 2, 3, 4)
    assert torch.equal(repeat_kv(x, 1), x)


def test_manual_matches_sdpa():
    """The manual softmax path is the reference; SDPA must match it closely."""
    torch.manual_seed(123)
    cfg_manual = _tiny_cfg("manual")
    cfg_sdpa = replace(cfg_manual, attn_impl="sdpa")

    attn_manual = Attention(cfg_manual, layer_idx=0).eval()
    attn_sdpa = Attention(cfg_sdpa, layer_idx=0).eval()
    # Share weights so only the backend differs.
    attn_sdpa.load_state_dict(attn_manual.state_dict())

    cos, sin = precompute_rope(cfg_manual.hd(), cfg_manual.block_size, cfg_manual.rope_theta)

    x = torch.randn(2, 6, cfg_manual.d_model)
    with torch.no_grad():
        out_manual = attn_manual(x, cos, sin)
        out_sdpa = attn_sdpa(x, cos, sin)

    assert torch.allclose(out_manual, out_sdpa, atol=1e-4)


def test_causal_mask_blocks_future():
    """Changing a future token must not affect an earlier query's output."""
    torch.manual_seed(7)
    cfg = _tiny_cfg("manual")
    attn = Attention(cfg, layer_idx=0).eval()
    cos, sin = precompute_rope(cfg.hd(), cfg.block_size, cfg.rope_theta)

    x = torch.randn(1, 5, cfg.d_model)
    with torch.no_grad():
        out1 = attn(x, cos, sin)

    # Perturb only the last position; earlier outputs must be identical.
    x2 = x.clone()
    x2[:, -1, :] += 10.0
    with torch.no_grad():
        out2 = attn(x2, cos, sin)

    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
    assert not torch.allclose(out1[:, -1], out2[:, -1], atol=1e-3)


def test_padding_mask_ignores_padded_keys():
    """A padded key position must not influence the output."""
    torch.manual_seed(11)
    cfg = _tiny_cfg("manual")
    attn = Attention(cfg, layer_idx=0).eval()
    cos, sin = precompute_rope(cfg.hd(), cfg.block_size, cfg.rope_theta)

    x = torch.randn(1, 4, cfg.d_model)
    mask = torch.tensor([[True, True, True, False]])  # last token is padding

    with torch.no_grad():
        out1 = attn(x, cos, sin, attention_mask=mask)

    # Change the padded position; only the (also-padded) last query may differ.
    x2 = x.clone()
    x2[:, -1, :] += 5.0
    with torch.no_grad():
        out2 = attn(x2, cos, sin, attention_mask=mask)

    # Real query positions 0..2 must be unaffected by the padded key.
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5)
