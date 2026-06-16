"""Grouped-Query Attention (GQA) with RoPE and a growable KV-cache.

Heads:
  q_proj -> [B, T, n_heads,    head_dim]
  k_proj -> [B, T, n_kv_heads, head_dim]
  v_proj -> [B, T, n_kv_heads, head_dim]

GQA expands each KV head ``n_rep`` times (``n_rep = n_heads // n_kv_heads``) so
the key/value head count matches the query head count.

Masking:
  - Prefill (T > 1): a causal mask blocks each query from future keys.
  - Cached decode (T == 1): the single query attends to ALL cached keys, so no
    causal drop is applied for that one row.
  - A key-padding mask (``attention_mask``: True = real, False = pad) is honored
    in both paths.

Backends:
  - ``cfg.attn_impl in {"auto", "sdpa"}`` -> ``F.scaled_dot_product_attention``.
  - ``cfg.attn_impl == "manual"``         -> explicit softmax (reference path).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from orkhon.model.config import ModelConfig
from orkhon.model.kv_cache import KVCache
from orkhon.model.rope import apply_rotary


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA.

    Input ``x`` has shape ``[B, T, n_kv_heads, head_dim]``. Each KV head is
    repeated ``n_rep`` times *contiguously*, so query head ``h`` maps to KV head
    ``h // n_rep``. Output shape: ``[B, T, n_kv_heads * n_rep, head_dim]``.
    """
    if n_rep == 1:
        return x
    b, t, n_kv, hd = x.shape
    # Insert a repeat axis right after the head axis, then flatten it in.
    x = x[:, :, :, None, :].expand(b, t, n_kv, n_rep, hd)
    out = x.reshape(b, t, n_kv * n_rep, hd).contiguous()
    assert out.shape == (b, t, n_kv * n_rep, hd)
    return out


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_rep()
        self.hd = cfg.hd()
        self.attn_impl = cfg.attn_impl
        self.dropout_p = cfg.dropout

        q_out = self.n_heads * self.hd
        kv_out = self.n_kv_heads * self.hd
        bias = cfg.use_bias

        self.q_proj = nn.Linear(cfg.d_model, q_out, bias=bias)
        self.k_proj = nn.Linear(cfg.d_model, kv_out, bias=bias)
        self.v_proj = nn.Linear(cfg.d_model, kv_out, bias=bias)
        self.o_proj = nn.Linear(q_out, cfg.d_model, bias=bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

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
        b, t, _ = x.shape

        # Project and split into heads.
        q = self.q_proj(x).view(b, t, self.n_heads, self.hd)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.hd)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.hd)

        # RoPE positions start at pos_offset, which is snapshotted ONCE per
        # forward pass (before any layer appends to the cache) and threaded in
        # from Transformer.forward. Reading past.length here per-layer would be a
        # bug: layer 0 appends before later layers read it, so layers >= 1 would
        # see a length already grown by t and rotate at shifted positions.
        positions = torch.arange(pos_offset, pos_offset + t, device=x.device)
        q, k = apply_rotary(q, k, cos, sin, positions)

        # Move to [B, H, T, hd] for attention.
        q = q.transpose(1, 2)  # [B, n_heads, T, hd]
        k = k.transpose(1, 2)  # [B, n_kv_heads, T, hd]
        v = v.transpose(1, 2)

        # Update / read the KV-cache (stored as [B, n_kv_heads, T, hd]).
        if use_cache and past is not None:
            k, v = past.append(self.layer_idx, k, v)

        # Expand KV heads to match query heads. repeat_kv works on [B, T, H, hd],
        # so transpose around it.
        k = repeat_kv(k.transpose(1, 2), self.n_rep).transpose(1, 2)
        v = repeat_kv(v.transpose(1, 2), self.n_rep).transpose(1, 2)

        kv_len = k.shape[2]
        is_decode = (t == 1) and (kv_len > 1)

        # Build an additive/boolean mask over [B, 1, T_q, kv_len] when needed.
        bool_mask = self._build_mask(
            b=b, t_q=t, kv_len=kv_len, is_decode=is_decode,
            attention_mask=attention_mask, device=x.device,
        )

        if self.attn_impl == "manual":
            out = self._manual_attention(q, k, v, bool_mask, is_decode)
        else:
            out = self._sdpa_attention(q, k, v, bool_mask, is_decode)

        # [B, n_heads, T, hd] -> [B, T, n_heads*hd]
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.hd)
        return self.resid_dropout(self.o_proj(out))

    def _build_mask(
        self,
        b: int,
        t_q: int,
        kv_len: int,
        is_decode: bool,
        attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return a boolean *allow* mask of shape [B, 1, T_q, kv_len], or None.

        ``True`` means the (query, key) pair is allowed to attend. Returns None
        only when there is nothing to mask (single-step decode, no padding), in
        which case callers may rely on full attention.
        """
        need_causal = (t_q > 1) and (not is_decode)
        need_padding = attention_mask is not None

        if not need_causal and not need_padding:
            return None

        # Start fully allowed.
        allow = torch.ones(t_q, kv_len, dtype=torch.bool, device=device)

        if need_causal:
            # Query i (absolute pos kv_len - t_q + i) may attend to keys [0, abs_i].
            offset = kv_len - t_q
            q_idx = torch.arange(t_q, device=device).view(t_q, 1) + offset
            k_idx = torch.arange(kv_len, device=device).view(1, kv_len)
            allow = allow & (k_idx <= q_idx)

        allow = allow.view(1, 1, t_q, kv_len).expand(b, 1, t_q, kv_len)

        if need_padding:
            # attention_mask: [B, kv_len] True=real. Broadcast over query axis.
            pad = attention_mask.view(b, 1, 1, kv_len).to(torch.bool)
            allow = allow & pad

        return allow

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bool_mask: torch.Tensor | None,
        is_decode: bool,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0
        if bool_mask is not None:
            # SDPA boolean mask: True = keep. Use our allow-mask directly.
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=bool_mask, dropout_p=dropout_p, is_causal=False
            )
        # No explicit mask: rely on built-in causal for multi-token prefill.
        use_causal = (q.shape[2] > 1) and (not is_decode)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=use_causal
        )

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bool_mask: torch.Tensor | None,
        is_decode: bool,
    ) -> torch.Tensor:
        """Reference softmax attention; the parity target for SDPA.

        The whole computation is carried out in float32 (regardless of the input
        dtype) so that cached single-token decode and full-sequence prefill agree
        tightly — the KV-cache parity guarantee. The result is cast back to the
        input dtype before returning.
        """
        orig_dtype = q.dtype
        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        vf = v.to(torch.float32)

        scale = 1.0 / math.sqrt(self.hd)
        # [B, H, T_q, kv_len]
        scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale

        if bool_mask is not None:
            neg = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~bool_mask, neg)

        attn = torch.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.matmul(attn, vf)
        return out.to(orig_dtype)
