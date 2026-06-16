"""Tests for the GRPO loss core (no rollouts — fast, CPU)."""

from __future__ import annotations

import torch

from orkhon.model import Transformer
from orkhon.model.config import ModelConfig
from orkhon.train.grpo import grpo_loss, _group_advantages


def _model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=48, block_size=32, n_layers=2, d_model=32, n_heads=4,
                      n_kv_heads=2, intermediate_size=64, init_std=0.2,
                      tie_word_embeddings=False)
    return Transformer(cfg)


def test_group_advantages_normalize_and_degenerate():
    adv = _group_advantages([[0.0, 1.0, 1.0, 0.0]], eps=1e-6)
    # two 0s and two 1s: mean 0.5, std 0.5 -> [-1, 1, 1, -1]
    assert all(abs(a - b) < 1e-4 for a, b in zip(adv[0], [-1.0, 1.0, 1.0, -1.0]))
    # all-equal rewards -> zero advantages (degenerate)
    assert _group_advantages([[1.0, 1.0]], eps=1e-6) == [[0.0, 0.0]]


def test_grpo_loss_finite_grads_and_kl_nonneg():
    pol, ref = _model(0), _model(1)
    B, T = 3, 8
    input_ids = torch.randint(8, 48, (B, T))
    amask = torch.ones((B, T), dtype=torch.bool)
    cmask = torch.zeros((B, T), dtype=torch.bool)
    cmask[:, 4:] = True  # last 4 tokens are completion
    adv = torch.tensor([1.0, -1.0, 0.5])
    loss, m = grpo_loss(pol, ref, input_ids, amask, cmask, adv, beta=0.02)
    assert torch.isfinite(loss) and loss.requires_grad
    assert m["kl"] >= -1e-6  # Schulman estimator is non-negative in expectation
    loss.backward()
    assert pol.layers[0].attn.q_proj.weight.grad is not None
    assert ref.layers[0].attn.q_proj.weight.grad is None  # reference frozen-grad-wise


def test_grpo_loss_respects_completion_mask():
    pol, ref = _model(0), _model(1)
    ids = torch.randint(8, 48, (1, 6))
    am = torch.ones((1, 6), dtype=torch.bool)
    cm_full = torch.zeros((1, 6), dtype=torch.bool); cm_full[:] = True
    cm_none = torch.zeros((1, 6), dtype=torch.bool)
    adv = torch.tensor([1.0])
    loss_full, _ = grpo_loss(pol, ref, ids, am, cm_full, adv, beta=0.02)
    loss_none, _ = grpo_loss(pol, ref, ids, am, cm_none, adv, beta=0.02)
    # No supervised tokens -> loss is 0 (sum/1 where sum=0); full mask is nonzero.
    assert abs(float(loss_none)) < 1e-6
    assert float(loss_full) != 0.0
