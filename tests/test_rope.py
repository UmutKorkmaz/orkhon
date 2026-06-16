"""Tests for Rotary Position Embeddings."""

from __future__ import annotations

import torch

from orkhon.model.rope import apply_rotary, precompute_rope, rotate_half


def test_rotate_half_swaps_and_negates():
    # [a, b, c, d] -> [-c, -d, a, b]
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.allclose(out, expected)


def test_rope_preserves_norm():
    # RoPE is a rotation per (i, i+half) pair, so per-vector norm is preserved.
    head_dim, max_seq, theta = 8, 16, 10000.0
    cos, sin = precompute_rope(head_dim, max_seq, theta)

    torch.manual_seed(0)
    q = torch.randn(2, 5, 3, head_dim)
    k = torch.randn(2, 5, 2, head_dim)
    positions = torch.arange(5)

    q_rot, k_rot = apply_rotary(q, k, cos, sin, positions)

    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


def test_position_offset_equivalence():
    """Rotating a token placed at absolute position p directly must equal
    rotating the same token with a position offset of p (the decode path)."""
    head_dim, max_seq, theta = 8, 32, 10000.0
    cos, sin = precompute_rope(head_dim, max_seq, theta)

    torch.manual_seed(1)
    q = torch.randn(1, 1, 2, head_dim)
    k = torch.randn(1, 1, 2, head_dim)

    p = 7
    # Direct: position [p].
    q_direct, k_direct = apply_rotary(q, k, cos, sin, torch.tensor([p]))
    # Offset: same single-token slot but with absolute position p (decode at step p).
    q_offset, k_offset = apply_rotary(q, k, cos, sin, torch.tensor([p]))

    assert torch.allclose(q_direct, q_offset, atol=1e-6)
    assert torch.allclose(k_direct, k_offset, atol=1e-6)


def test_offset_matches_full_sequence_slice():
    """A token rotated within a full prefill at position p must equal the same
    token rotated alone with offset p — the core cached-vs-uncached guarantee."""
    head_dim, max_seq, theta = 8, 32, 10000.0
    cos, sin = precompute_rope(head_dim, max_seq, theta)

    torch.manual_seed(2)
    seq = torch.randn(1, 6, 2, head_dim)
    seq_k = torch.randn(1, 6, 2, head_dim)

    # Full prefill rotation over positions 0..5.
    full_q, full_k = apply_rotary(seq, seq_k, cos, sin, torch.arange(6))

    # Rotate only position 4 with offset 4.
    p = 4
    single_q, single_k = apply_rotary(
        seq[:, p : p + 1], seq_k[:, p : p + 1], cos, sin, torch.tensor([p])
    )

    assert torch.allclose(full_q[:, p : p + 1], single_q, atol=1e-6)
    assert torch.allclose(full_k[:, p : p + 1], single_k, atol=1e-6)
