"""Key/Value cache for incremental decoding.

Holds one ``(k, v)`` pair per transformer layer. Each tensor has layout
``[B, n_kv_heads, T, head_dim]`` so it concatenates cleanly along the time axis
(dim=2). The cache is growable: each layer appends its freshly computed k/v and
reads back the full history.

The cache tracks a single shared length (all layers advance in lockstep during a
forward pass). ``current_length`` therefore reflects how many positions have been
appended *before* the current step — exactly the RoPE position offset the
attention module needs.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        # Per-layer (k, v) or None until first append.
        self._cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers

    def reset(self) -> None:
        """Drop all stored keys/values."""
        self._cache = [None] * self.n_layers

    @property
    def length(self) -> int:
        """Number of cached time steps (0 when empty)."""
        first = self._cache[0]
        if first is None:
            return 0
        return first[0].shape[2]

    def __len__(self) -> int:
        return self.length

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the stored ``(k, v)`` for a layer, or ``None`` if empty."""
        return self._cache[layer_idx]

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new ``k, v`` (``[B, n_kv_heads, T_new, hd]``) for a layer.

        Returns the full cached ``(k, v)`` after concatenation, which the caller
        uses as the attention keys/values for this step.
        """
        prev = self._cache[layer_idx]
        if prev is None:
            new_k, new_v = k, v
        else:
            prev_k, prev_v = prev
            new_k = torch.cat((prev_k, k), dim=2)
            new_v = torch.cat((prev_v, v), dim=2)
        self._cache[layer_idx] = (new_k, new_v)
        return new_k, new_v
