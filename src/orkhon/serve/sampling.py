"""Single-token sampling from a logits vector.

``sample_next`` is the serving-side mirror of the model's internal sampler. It
applies temperature, then optional top-k and top-p (nucleus) filtering, then draws
one token. ``temperature == 0`` is greedy (argmax) and deterministic.

Kept separate from :mod:`orkhon.model.generation` so the serving layer can sample
from arbitrary logits (e.g. for streaming token-by-token) without pulling in the
generation/KV-cache machinery, while preserving identical semantics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    """Coerce logits of shape ``[V]``, ``[1, V]`` or ``[B, T, V]`` to ``[V]``.

    For multi-position inputs the LAST position is used (next-token logits).
    """
    if logits.dim() == 1:
        return logits
    # Flatten any leading dims, take the final row.
    return logits.reshape(-1, logits.shape[-1])[-1]


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Sample the next token id from ``logits``.

    Args:
        logits: a logits tensor; ``[V]`` or any shape whose last dim is the vocab
            (the final position is used).
        temperature: softmax temperature; ``0`` -> greedy argmax.
        top_k: keep only the ``top_k`` highest-logit tokens before sampling.
        top_p: nucleus sampling; keep the smallest set whose cumulative
            probability >= ``top_p``.

    Returns:
        The chosen token id (int).
    """
    logits = _as_1d_logits(logits).float()

    if temperature == 0:
        return int(torch.argmax(logits).item())

    logits = logits / max(temperature, 1e-8)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth_val = torch.topk(logits, k).values[-1]
        logits = torch.where(
            logits < kth_val, torch.full_like(logits, float("-inf")), logits
        )

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # Keep tokens up to and including the one that crosses the threshold.
        remove = cumprobs - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            0, sorted_idx, sorted_logits
        )

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
