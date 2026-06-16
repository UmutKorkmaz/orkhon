"""The Orkhon decoder-only Transformer.

Architecture:
  token embedding [vocab, d_model]
  -> n_layers x pre-norm TransformerBlock (GQA + SwiGLU)
  -> final RMSNorm
  -> lm_head Linear(d_model, vocab, bias=False), weight-tied to the embedding
     when ``cfg.tie_word_embeddings``.

forward(input_ids, attention_mask=None, past=None, use_cache=False)
    -> (logits [B, T, vocab], new_past | None)

The RoPE cos/sin tables are precomputed once and registered as buffers so they
follow the model across ``.to(device)`` / ``.to(dtype)`` moves.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from orkhon.model.block import TransformerBlock
from orkhon.model.config import ModelConfig
from orkhon.model.init import init_weights
from orkhon.model.kv_cache import KVCache
from orkhon.model.rmsnorm import RMSNorm
from orkhon.model.rope import precompute_rope


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, i) for i in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Precompute RoPE tables. With rope_scaling, the table extends beyond
        # block_size so the model can infer at longer context than it was trained at.
        rope_max_seq = math.ceil(cfg.block_size * cfg.rope_scaling_factor) if cfg.rope_scaling_type else cfg.block_size
        cos, sin = precompute_rope(
            head_dim=cfg.hd(),
            max_seq=rope_max_seq,
            theta=cfg.rope_theta,
            dtype=torch.float32,
            scaling_type=cfg.rope_scaling_type,
            scaling_factor=cfg.rope_scaling_factor,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._rope_max_seq = rope_max_seq

        # Initialize before tying so the tied weight keeps the embedding's init.
        init_weights(self, cfg)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        b, t = input_ids.shape

        if use_cache and past is None:
            past = KVCache(self.cfg.n_layers)

        x = self.embed_dropout(self.embed_tokens(input_ids))

        # Snapshot the RoPE position offset ONCE, before any layer appends to the
        # cache, so every layer rotates at the same absolute positions. (KVCache
        # reports layer 0's length, which grows mid-pass as layer 0 appends.)
        pos_offset = past.length if (use_cache and past is not None) else 0

        cos = self.rope_cos
        sin = self.rope_sin
        for layer in self.layers:
            x = layer(
                x,
                cos,
                sin,
                attention_mask=attention_mask,
                past=past,
                use_cache=use_cache,
                pos_offset=pos_offset,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)

        new_past = past if use_cache else None
        return logits, new_past

    @torch.no_grad()
    def num_params(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_word_embeddings:
            total -= self.lm_head.weight.numel()
        return total


def build_model(cfg: ModelConfig) -> Transformer:
    """Factory mirroring ``Transformer(cfg)`` for callers that prefer a function."""
    return Transformer(cfg)
