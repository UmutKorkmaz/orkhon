"""Root-Mean-Square LayerNorm (RMSNorm).

Computes the normalization in float32 regardless of the input dtype, then casts
back. Weight is initialized to ones; there is no bias term.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        normed = self._norm(x.to(torch.float32)).to(orig_dtype)
        return normed * self.weight
