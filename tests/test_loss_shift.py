"""Tests for lm_cross_entropy: IGNORE_INDEX masking, perfect-logit zero loss, shift."""

from __future__ import annotations

import torch

from orkhon.train.losses import IGNORE_INDEX, lm_cross_entropy


def test_perfect_logits_give_near_zero_loss():
    """When logits perfectly predict the shifted targets, loss -> ~0."""
    b, t, v = 2, 5, 7
    labels = torch.randint(0, v, (b, t))
    # Build logits so that argmax at position p equals labels[p+1] (the shift).
    logits = torch.zeros(b, t, v)
    for i in range(b):
        for p in range(t - 1):
            logits[i, p, labels[i, p + 1]] = 50.0  # huge logit on correct next token

    loss = lm_cross_entropy(logits, labels)
    assert loss.item() < 1e-3, f"expected ~0 loss, got {loss.item()}"


def test_ignore_index_positions_excluded():
    """Positions whose (shifted) label is IGNORE_INDEX must not contribute."""
    b, t, v = 1, 4, 5
    logits = torch.randn(b, t, v)
    labels = torch.randint(0, v, (b, t))

    base = lm_cross_entropy(logits, labels.clone())

    # Mask the LAST label (the only supervised position contributed by it after
    # shift is labels[:, 1:], i.e. positions 1..t-1). Mask all but one to compare.
    masked = labels.clone()
    masked[:, 2:] = IGNORE_INDEX  # only label at index 1 stays supervised
    loss_masked = lm_cross_entropy(logits, masked)

    # Compute the single-token reference loss for the one supervised position.
    # logits[:, 0] predicts labels[:, 1].
    import torch.nn.functional as F

    ref = F.cross_entropy(logits[:, 0, :].float(), labels[:, 1])
    assert torch.allclose(loss_masked, ref, atol=1e-5)
    # Sanity: masking changed the loss (different from the all-supervised case).
    assert not torch.allclose(loss_masked, base, atol=1e-6)


def test_all_ignored_returns_zero():
    """A fully-masked batch yields 0 loss (safe for empty micro-batches)."""
    b, t, v = 2, 4, 6
    logits = torch.randn(b, t, v)
    labels = torch.full((b, t), IGNORE_INDEX)
    loss = lm_cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0 or abs(loss.item()) < 1e-6


def test_shift_predicts_position_t_from_before_t():
    """logits[:, t] are scored against labels[:, t+1] (not labels[:, t])."""
    b, t, v = 1, 3, 4
    labels = torch.tensor([[0, 1, 2]])
    # Make position 0 confidently predict token 1 (the correct shifted label),
    # and a WRONG token for the unshifted label (token 0). If the shift were
    # wrong (predicting labels[:, t]), this would produce high loss.
    logits = torch.zeros(b, t, v)
    logits[0, 0, 1] = 50.0  # predict labels[0, 1] == 1  (correct under shift)
    logits[0, 1, 2] = 50.0  # predict labels[0, 2] == 2  (correct under shift)
    loss = lm_cross_entropy(logits, labels)
    assert loss.item() < 1e-3
