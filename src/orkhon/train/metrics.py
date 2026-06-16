"""Small training-loop metric helpers (throughput and gradient norms)."""

from __future__ import annotations

import torch
from torch import nn


def grad_global_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """Global gradient norm across all parameters that have a gradient.

    Computed the same way :func:`torch.nn.utils.clip_grad_norm_` measures it, but
    read-only (does not rescale). Returns ``0.0`` when no parameter has a grad.

    Args:
        model: model whose ``.grad`` tensors are aggregated.
        norm_type: p-norm order (default L2).
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    total = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
        norm_type,
    )
    return float(total.item())


def tokens_per_second(tokens: int, elapsed_seconds: float) -> float:
    """Throughput in tokens/sec, guarding against a zero/negative interval."""
    if elapsed_seconds <= 0:
        return 0.0
    return tokens / elapsed_seconds
