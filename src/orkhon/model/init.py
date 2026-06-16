"""Weight initialization for the Orkhon Transformer.

Rules:
- Linear / Embedding weights: ``normal_(0, init_std)``.
- Biases: zeros.
- RMSNorm weights: ones.
- Output projections (attention ``o_proj``, MLP ``down_proj``) are additionally
  scaled by ``1 / sqrt(2 * n_layers)`` to keep residual-stream variance stable
  with depth (GPT-2 style residual scaling).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from orkhon.model.attention import Attention
from orkhon.model.config import ModelConfig
from orkhon.model.mlp import SwiGLUMLP
from orkhon.model.rmsnorm import RMSNorm


def init_weights(model: nn.Module, cfg: ModelConfig) -> None:
    std = cfg.init_std

    def _basic_init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    model.apply(_basic_init)

    # Scale the residual output projections by 1/sqrt(2 * n_layers).
    scale = 1.0 / math.sqrt(2 * cfg.n_layers)
    for module in model.modules():
        if isinstance(module, Attention):
            module.o_proj.weight.data.mul_(scale)
        elif isinstance(module, SwiGLUMLP):
            module.down_proj.weight.data.mul_(scale)
