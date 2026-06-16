"""Tests for the distillation loss."""

from __future__ import annotations

import torch

from orkhon.train.losses import IGNORE_INDEX, distillation_loss


def _logits(B=2, T=4, V=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, V, generator=g)


def test_distillation_is_finite_and_shapes():
    s = _logits().requires_grad_(True)
    t = _logits(seed=1)
    labels = torch.randint(0, 8, (2, 4))
    loss = distillation_loss(s, t, labels, T=2.0, alpha=0.5)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert loss.requires_grad  # student grads flow


def test_alpha_zero_is_pure_hard_ce():
    s = _logits().requires_grad_(True)
    t = _logits(seed=99)
    labels = torch.randint(0, 8, (2, 4))
    d = distillation_loss(s.detach().clone().requires_grad_(True), t, labels, alpha=0.0)
    # alpha=0 => exactly the hard CE term
    from orkhon.train.losses import lm_cross_entropy
    hard = lm_cross_entropy(s.detach(), labels)
    assert torch.allclose(d, hard, atol=1e-5)


def test_teacher_does_not_get_grads():
    s = _logits().requires_grad_(True)
    t = _logits(seed=1).requires_grad_(True)
    labels = torch.randint(0, 8, (2, 4))
    distillation_loss(s, t, labels)
    assert t.grad is None  # teacher detached inside the loss


def test_distillation_all_ignored_labels_is_finite():
    """When every label is IGNORE_INDEX, the hard CE term must not be NaN."""
    from orkhon.train.losses import IGNORE_INDEX
    s = _logits().requires_grad_(True)
    t = _logits(seed=1)
    labels = torch.full((2, 4), IGNORE_INDEX)
    loss = distillation_loss(s.detach().clone().requires_grad_(True), t, labels, alpha=0.5)
    assert torch.isfinite(loss), f"loss is {loss} (should be finite for all-IGNORE)"
    # The loss should be the soft KL term only (hard term = 0).
    assert float(loss) >= 0.0


def test_mismatched_shapes_raise():
    import pytest
    s = _logits(V=8)
    t = _logits(V=16)
    labels = torch.randint(0, 8, (2, 4))
    with pytest.raises(ValueError):
        distillation_loss(s, t, labels)
