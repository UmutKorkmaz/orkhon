"""Rotary Position Embeddings (RoPE).

We precompute the inverse frequencies from ``rope_theta`` over an even
``head_dim`` and build a ``(max_seq, head_dim)`` cos/sin table. RoPE is applied
to the query and key tensors only (never to values).

CRITICAL contract points:
- A position *offset* is supported so that during cached decode, step ``t`` uses
  absolute position ``past_len`` rather than ``0``. The caller passes the
  absolute ``positions`` (or an offset) for the current query block.
- The rotation math is done in float32 for numerical stability, then cast back
  to the input dtype.
"""

from __future__ import annotations

import torch


def _yarn_mscale(factor: float, mscale: float = 0.1) -> float:
    """YaRN attention temperature correction: ``mscale * ln(factor) + 1``."""
    import math
    return mscale * math.log(factor) + 1 if factor > 1 else 1.0


def precompute_rope(
    head_dim: int,
    max_seq: int,
    theta: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    *,
    scaling_type: str | None = None,
    scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin tables of shape ``(max_seq, head_dim)`` with optional RoPE scaling.

    Without scaling (the default), this is standard RoPE. With scaling, the table
    extends BEYOND the trained ``block_size`` so a model can infer at longer context
    than it was trained at (L0 long-context extension — $0, no GPU):

    - ``scaling_type="linear"`` (Position Interpolation): positions are divided by
      ``scaling_factor``. Simplest and most robust.
    - ``scaling_type="ntk"``: NTK-aware base-frequency scaling (theta *
      factor^(hd/(hd-2))) + a YaRN mscale temperature correction on cos/sin.
      (Full YaRN with the dimension-wise ramp is a follow-up; this is the simpler
      NTK-aware variant which works well for moderate stretch factors.)
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    half = head_dim // 2
    exponent = torch.arange(0, half, dtype=torch.float32, device=device) / half
    inv_freq = 1.0 / (theta ** exponent)

    if scaling_type == "ntk" and scaling_factor > 1:
        ntk_theta = theta * (scaling_factor ** (head_dim / (head_dim - 2)))
        inv_freq = 1.0 / (ntk_theta ** exponent)

    positions = torch.arange(max_seq, dtype=torch.float32, device=device)
    if scaling_type == "linear" and scaling_factor > 1:
        positions = positions / scaling_factor

    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()
    sin = emb.sin()
    # YaRN mscale: a temperature correction on cos/sin VALUES (not the angle).
    if scaling_type == "ntk" and scaling_factor > 1:
        ms = _yarn_mscale(scaling_factor)
        cos, sin = cos * ms, sin * ms

    return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: ``[a, b] -> [-b, a]``."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to ``q`` and ``k`` (not values).

    Args:
        q: query tensor ``[B, T, n_heads, head_dim]``.
        k: key tensor ``[B, T, n_kv_heads, head_dim]``.
        cos, sin: precomputed tables ``[max_seq, head_dim]``.
        positions: absolute positions for the ``T`` query/key slots, shape ``[T]``
            (or broadcastable). For decode at step ``t`` this should be
            ``[past_len]``; for prefill it is ``arange(T)``.

    Returns:
        Rotated ``(q, k)`` cast back to their original dtype.
    """
    orig_dtype = q.dtype

    # Gather the cos/sin rows for these absolute positions: (T, head_dim).
    cos_t = cos.index_select(0, positions).to(torch.float32)
    sin_t = sin.index_select(0, positions).to(torch.float32)

    # Reshape for broadcasting against [B, T, H, hd]: (1, T, 1, hd).
    cos_b = cos_t[None, :, None, :]
    sin_b = sin_t[None, :, None, :]

    qf = q.to(torch.float32)
    kf = k.to(torch.float32)

    q_rot = qf * cos_b + rotate_half(qf) * sin_b
    k_rot = kf * cos_b + rotate_half(kf) * sin_b

    return q_rot.to(orig_dtype), k_rot.to(orig_dtype)
