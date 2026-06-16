"""Pre-norm Transformer block.

    x = x + attn(rmsnorm1(x))
    x = x + mlp(rmsnorm2(x))
"""

from __future__ import annotations

import torch
from torch import nn

from orkhon.model.attention import Attention
from orkhon.model.config import ModelConfig
from orkhon.model.kv_cache import KVCache
from orkhon.model.mlp import SwiGLUMLP
from orkhon.model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past: KVCache | None = None,
        use_cache: bool = False,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            cos,
            sin,
            attention_mask=attention_mask,
            past=past,
            use_cache=use_cache,
            pos_offset=pos_offset,
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x
