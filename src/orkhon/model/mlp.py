"""SwiGLU feed-forward network.

``down(silu(gate(x)) * up(x))`` where ``gate`` and ``up`` project
``d_model -> intermediate`` and ``down`` projects ``intermediate -> d_model``.
Biases are included only when ``cfg.use_bias`` is set.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from orkhon.model.config import ModelConfig


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        inter = cfg.intermediate()
        bias = cfg.use_bias

        self.gate_proj = nn.Linear(d, inter, bias=bias)
        self.up_proj = nn.Linear(d, inter, bias=bias)
        self.down_proj = nn.Linear(inter, d, bias=bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
