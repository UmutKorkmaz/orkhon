"""Regression guard for the RoPE cached-decode position bug.

History: ``Attention.forward`` read ``past.length`` per layer to set RoPE
positions. ``KVCache.length`` reports layer 0's cache length, and layer 0 appends
before later layers read it, so layers >= 1 rotated at positions shifted by +T
during a cached forward. RoPE's relative-position invariance hid this in pure
prefill or pure step-by-step decode, but prefill-then-incremental-decode mixed
correctly-positioned cached keys with mis-positioned decode queries and diverged.

The fix snapshots ``pos_offset`` once per forward pass and threads it to every
layer. These tests use a HIGH ``init_std`` and a MULTI-LAYER model — the regime
where the bug produced a visible divergence (~2.0) and changed greedy tokens.
"""

from __future__ import annotations

import torch

from orkhon.model import KVCache, Transformer, generate
from orkhon.model.config import ModelConfig


def _cfg(attn_impl: str) -> ModelConfig:
    return ModelConfig(
        vocab_size=64, block_size=64, n_layers=3, d_model=64, n_heads=4,
        n_kv_heads=2, intermediate_size=128, init_std=0.5,
        tie_word_embeddings=False, attn_impl=attn_impl,
    )


def test_prefill_then_decode_matches_full_reforward() -> None:
    """Cached prefill+decode must match a full no-cache forward at every step."""
    for impl in ("manual", "auto"):
        torch.manual_seed(0)
        model = Transformer(_cfg(impl)).eval()
        prompt = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            past = KVCache(model.cfg.n_layers)
            logits, past = model(prompt, past=past, use_cache=True)
            running = prompt.clone()
            worst = 0.0
            for _ in range(6):
                nxt = logits[:, -1:].argmax(-1)
                running = torch.cat([running, nxt], dim=1)
                step_logits, past = model(nxt, past=past, use_cache=True)
                full_logits, _ = model(running, use_cache=False)
                worst = max(
                    worst,
                    (step_logits[:, -1] - full_logits[:, -1]).abs().max().item(),
                )
                logits = step_logits
        assert worst < 1e-3, f"[{impl}] cached vs full diverged: {worst:.2e}"


def test_cached_generate_matches_no_cache_greedy() -> None:
    """generate() (with KV-cache) must equal a no-cache greedy loop token-for-token."""
    torch.manual_seed(0)
    model = Transformer(_cfg("manual")).eval()
    prompt = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        cached = generate(
            model, prompt[0].tolist(), max_new_tokens=12, temperature=0.0,
            eos_ids=(), device="cpu",
        )
        ref: list[int] = []
        cur = prompt.clone()
        for _ in range(12):
            lg, _ = model(cur, use_cache=False)
            nx = int(lg[0, -1].argmax())
            ref.append(nx)
            cur = torch.cat([cur, torch.tensor([[nx]])], dim=1)
    assert cached == ref, f"cached {cached} != no-cache {ref}"
