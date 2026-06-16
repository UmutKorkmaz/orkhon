"""Tests for dpo_loss, sequence_logprob masking, and frozen-reference behavior."""

from __future__ import annotations

import math

import pytest
import torch

from orkhon.train.losses import IGNORE_INDEX, dpo_loss, sequence_logprob


def test_policy_equals_reference_gives_log2():
    """When policy logps == reference logps, loss == -logsigmoid(0) == log(2)."""
    pc = torch.tensor([1.2, -0.5, 3.0])
    pr = torch.tensor([0.7, 0.1, -2.0])
    # Reference identical to policy -> inner term is exactly 0.
    loss, metrics = dpo_loss(pc, pr, pc.clone(), pr.clone(), beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6), loss.item()
    # Margin is policy(chosen-rejected) minus reference(chosen-rejected) == 0.
    assert abs(metrics["reward_margin"].item()) < 1e-6


def test_reference_is_frozen_no_grad_path():
    """dpo_loss must not require grad through reference logps."""
    pc = torch.tensor([1.0], requires_grad=True)
    pr = torch.tensor([0.5], requires_grad=True)
    rc = torch.tensor([0.8])  # no grad (frozen reference)
    rr = torch.tensor([0.4])
    loss, _ = dpo_loss(pc, pr, rc, rr, beta=0.2)
    loss.backward()
    assert pc.grad is not None and pr.grad is not None
    # Reference tensors have no grad attribute populated.
    assert rc.grad is None and rr.grad is None


def test_margin_increases_when_chosen_logp_raised():
    """Raising the policy's chosen logp increases the reward margin."""
    pr = torch.tensor([0.0])
    rc = torch.tensor([0.0])
    rr = torch.tensor([0.0])

    _, m_low = dpo_loss(torch.tensor([0.0]), pr, rc, rr, beta=0.1)
    _, m_high = dpo_loss(torch.tensor([2.0]), pr, rc, rr, beta=0.1)
    assert m_high["reward_margin"].item() > m_low["reward_margin"].item()
    # Loss decreases as chosen is preferred more strongly.
    loss_low, _ = dpo_loss(torch.tensor([0.0]), pr, rc, rr, beta=0.1)
    loss_high, _ = dpo_loss(torch.tensor([2.0]), pr, rc, rr, beta=0.1)
    assert loss_high.item() < loss_low.item()


def test_preference_accuracy_metric():
    """reward_accuracy is the fraction with chosen reward > rejected reward."""
    pc = torch.tensor([2.0, 0.0])
    pr = torch.tensor([0.0, 2.0])
    rc = torch.zeros(2)
    rr = torch.zeros(2)
    _, m = dpo_loss(pc, pr, rc, rr, beta=0.1)
    # First pair: chosen>rejected (correct). Second: chosen<rejected (wrong).
    assert m["reward_accuracy"].item() == 0.5


def test_sequence_logprob_masks_prompt():
    """sequence_logprob only sums logprobs over non-ignore (completion) positions."""
    b, t, v = 1, 5, 6
    logits = torch.randn(b, t, v)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 1, 2]])

    logp = sequence_logprob(logits, labels)

    # Manually sum the supervised positions under the same single shift:
    # labels[:, 1:] = [IGNORE, 3, 1, 2] scored against logits[:, :-1].
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)[0]
    shift_labels = labels[0, 1:]
    expected = 0.0
    for pos, lab in enumerate(shift_labels.tolist()):
        if lab != IGNORE_INDEX:
            expected += log_probs[pos, lab].item()
    assert abs(logp.item() - expected) < 1e-5


def test_sequence_logprob_ignores_when_all_masked():
    """A fully-masked sequence contributes 0 logprob."""
    b, t, v = 2, 4, 5
    logits = torch.randn(b, t, v)
    labels = torch.full((b, t), IGNORE_INDEX)
    logp = sequence_logprob(logits, labels)
    assert torch.allclose(logp, torch.zeros(b))


def test_sequence_logprob_shift_alignment():
    """logits[:, t] score labels[:, t+1] (the completion shift)."""
    b, t, v = 1, 3, 4
    labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
    # Confident, correct predictions under the shift -> logprob near 0 (log 1).
    logits = torch.zeros(b, t, v)
    logits[0, 0, 1] = 50.0  # predicts labels[0,1] == 1
    logits[0, 1, 2] = 50.0  # predicts labels[0,2] == 2
    logp = sequence_logprob(logits, labels)
    assert logp.item() > -1e-2  # ~0 (sum of two ~log(1) terms)
