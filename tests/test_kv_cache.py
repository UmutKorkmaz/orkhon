"""KV-cache parity: cached decode must match a full no-cache re-forward."""

from __future__ import annotations

import torch

from orkhon.model.config import ModelConfig
from orkhon.model.kv_cache import KVCache
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


def test_cache_reset_and_length():
    cache = KVCache(n_layers=2)
    assert cache.length == 0
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    cache.append(0, k, v)
    cache.append(1, k, v)
    assert cache.length == 3
    cache.reset()
    assert cache.length == 0


# Parity tolerance. The cached single-token decode and the full re-forward are
# mathematically identical; the only residual is shape-dependent BLAS rounding
# (T==1 vs T==N matmul dispatch), which sits around ~1e-5 here. We keep the bar at
# 1e-4 deliberately: a looser 1e-3 previously MASKED a real RoPE position bug
# (per-layer past.length read shifted layers >= 1 by ~7e-4 at default init). The
# offset is now snapshotted once per forward pass; see
# tests/test_kv_cache_rope_regression.py for the high-init falsification guard.
PARITY_ATOL = 1e-4


def test_cached_decode_matches_full_reforward():
    model = _tiny_model()
    torch.manual_seed(1)

    prompt = torch.randint(0, model.cfg.vocab_size, (1, 3))

    # Cached path: prefill, then decode one token at a time.
    cache = KVCache(model.cfg.n_layers)
    with torch.no_grad():
        logits, cache = model(prompt, past=cache, use_cache=True)
    cached_next = logits[0, -1, :]

    seq = prompt.clone()
    steps = 5
    for step in range(steps):
        # No-cache reference over the full growing sequence.
        with torch.no_grad():
            full_logits, _ = model(seq, use_cache=False)
        ref_next = full_logits[0, -1, :]

        diff = (cached_next - ref_next).abs().max().item()
        assert diff < PARITY_ATOL, (
            f"cached logits diverged from full re-forward at step {step}: "
            f"max abs diff {diff:.2e} >= {PARITY_ATOL:.0e}"
        )

        # Pick a deterministic next token and advance both paths.
        next_id = int(torch.argmax(ref_next).item())
        seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)

        step_in = torch.tensor([[next_id]])
        with torch.no_grad():
            logits, cache = model(step_in, past=cache, use_cache=True)
        cached_next = logits[0, -1, :]


def test_cached_decode_argmax_matches_full():
    """Stronger functional guarantee: the greedy NEXT-TOKEN chosen by the cached
    path is identical to the no-cache path at every step (parity that matters for
    generation)."""
    model = _tiny_model()
    torch.manual_seed(3)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 2))

    cache = KVCache(model.cfg.n_layers)
    with torch.no_grad():
        logits, cache = model(prompt, past=cache, use_cache=True)
    cached_next = logits[0, -1, :]

    seq = prompt.clone()
    for step in range(6):
        with torch.no_grad():
            full_logits, _ = model(seq, use_cache=False)
        ref_next = full_logits[0, -1, :]

        assert int(torch.argmax(cached_next)) == int(torch.argmax(ref_next)), (
            f"cached/full argmax disagree at step {step}"
        )

        next_id = int(torch.argmax(ref_next).item())
        seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)
        with torch.no_grad():
            logits, cache = model(torch.tensor([[next_id]]), past=cache, use_cache=True)
        cached_next = logits[0, -1, :]
