"""Resize a model's token embeddings to a larger vocab WITHOUT disturbing old rows.

When a tokenizer is retrofitted with appended tokens (see :mod:`orkhon.tokenizer.retrofit`),
every existing id is preserved but the vocab grows. The model's ``embed_tokens`` (and the
tied ``lm_head``) must grow to match: the old rows are copied verbatim and the new rows
are initialized small (so the new tokens start near-uniform and learn during SFT).

This is the inverse of HF ``resize_token_embeddings``; it is correct because the appended
ids are strictly greater than the old vocab, so old weights stay at their old indices.
"""

from __future__ import annotations

import torch
from torch import nn

from orkhon.model.config import ModelConfig


def resize_token_embeddings(model: nn.Module, new_vocab_size: int) -> tuple[nn.Module, ModelConfig]:
    """Grow ``model``'s embedding (and tied lm_head) to ``new_vocab_size`` in place.

    Args:
        model: an :class:`~orkhon.model.Transformer`.
        new_vocab_size: must be >= the current vocab; old rows are preserved.

    Returns:
        ``(model, updated_cfg)`` — the model's ``cfg.vocab_size`` is updated too.
    """
    cfg: ModelConfig = model.cfg
    old = cfg.vocab_size
    if new_vocab_size < old:
        raise ValueError(f"resize only grows vocab: {new_vocab_size} < current {old}")
    if new_vocab_size == old:
        return model, cfg
    d = cfg.d_model
    old_emb = model.embed_tokens.weight.data  # [old, d]
    dev, dt = old_emb.device, old_emb.dtype
    # New rows allocated from the SAME device/dtype as the existing weights (so a
    # model on CUDA/MPS resizes on-device, not CPU).
    n_new = new_vocab_size - old
    new_rows = old_emb.new_empty(n_new, d).normal_(0, cfg.init_std)
    new_emb = torch.cat([old_emb, new_rows], dim=0)
    model.embed_tokens = nn.Embedding(new_vocab_size, d, device=dev, dtype=dt)
    model.embed_tokens.weight.data.copy_(new_emb)

    # The lm_head: if tied, it shares storage with embed_tokens (already resized). If not
    # tied, grow it the same way (copy old rows + init new) on the same device/dtype.
    if not cfg.tie_word_embeddings:
        old_head = model.lm_head.weight.data  # [old, d]
        head_new = torch.cat([old_head, old_head.new_empty(n_new, d).normal_(0, cfg.init_std)], 0)
        model.lm_head = nn.Linear(d, new_vocab_size, bias=False, device=dev, dtype=dt)
        model.lm_head.weight.data.copy_(head_new)
    else:
        model.lm_head.weight = model.embed_tokens.weight  # re-tie to the resized embedding

    # Update the frozen cfg in place (create a new frozen dataclass with the new vocab).
    from dataclasses import replace
    model.cfg = replace(cfg, vocab_size=new_vocab_size)
    return model, model.cfg
