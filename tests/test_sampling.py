"""Tests for orkhon.serve.sampling.sample_next."""

from __future__ import annotations

import torch

from orkhon.serve.sampling import sample_next


def test_temperature_zero_is_argmax():
    logits = torch.tensor([0.1, 5.0, -2.0, 3.0, 0.0])
    expected = int(torch.argmax(logits).item())
    for _ in range(5):  # deterministic regardless of repetition
        assert sample_next(logits, temperature=0.0) == expected


def test_top_k_one_is_argmax():
    logits = torch.tensor([1.0, 2.0, 9.0, 0.5, -1.0])
    expected = int(torch.argmax(logits).item())
    torch.manual_seed(0)
    for _ in range(20):
        assert sample_next(logits, temperature=1.0, top_k=1) == expected


def test_top_p_keeps_only_probability_mass():
    """With a peaky distribution and small top_p, only the top token survives."""
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])  # token 0 dominates
    expected = 0
    torch.manual_seed(0)
    for _ in range(20):
        out = sample_next(logits, temperature=1.0, top_p=0.5)
        assert out == expected


def test_top_p_in_unit_interval_is_valid_token():
    logits = torch.randn(32)
    torch.manual_seed(1)
    for top_p in (0.1, 0.5, 0.9, 1.0):
        out = sample_next(logits, temperature=1.0, top_p=top_p)
        assert 0 <= out < logits.numel()


def test_accepts_multidim_logits_uses_last_position():
    # [B, T, V]: last position should drive the choice.
    logits = torch.zeros(1, 3, 4)
    logits[0, -1] = torch.tensor([0.0, 0.0, 9.0, 0.0])
    assert sample_next(logits, temperature=0.0) == 2


def test_combined_top_k_top_p_runs():
    logits = torch.randn(50)
    torch.manual_seed(2)
    out = sample_next(logits, temperature=0.8, top_k=10, top_p=0.9)
    assert 0 <= out < 50
