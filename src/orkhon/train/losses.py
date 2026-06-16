"""Loss functions and log-probability helpers for the three training stages.

Three primitives live here:

* :func:`lm_cross_entropy` — the standard causal LM loss with the single
  next-token shift (``logits[:, :-1]`` predict ``labels[:, 1:]``). Mean over
  non-ignore positions only; pad / prompt spans flagged ``IGNORE_INDEX`` never
  contribute.
* :func:`sequence_logprob` — per-sequence sum of the log-probabilities the model
  assigns to its target tokens over the (non-ignore) completion positions. Used
  by DPO to score chosen/rejected completions. Same shift convention as the loss.
* :func:`dpo_loss` — the Direct Preference Optimization objective and its reward
  margin metric, computed from policy and (frozen) reference sequence logprobs.

The shift is the classic trap: a token at position ``t`` is predicted from the
hidden state at position ``t-1``, so logits are sliced ``[:, :-1]`` and labels
``[:, 1:]`` before any reduction. Keeping this in one place guarantees the loss,
the DPO logprobs, and the eval perplexity all agree.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def lm_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Mean cross-entropy over non-ignore next-token targets.

    Args:
        logits: ``[B, T, V]`` raw model outputs.
        labels: ``[B, T]`` target ids; ``ignore_index`` positions are skipped.
        ignore_index: label value marking masked positions (pad / non-assistant).

    Returns:
        Scalar tensor: mean negative log-likelihood over supervised tokens. If no
        token is supervised, returns ``0.0`` (so empty micro-batches are safe).

    The single shift is applied here: ``logits[:, :-1]`` predict ``labels[:, 1:]``.
    """
    if logits.dim() != 3:
        raise ValueError(f"expected logits [B, T, V], got shape {tuple(logits.shape)}")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab = shift_logits.size(-1)
    flat_logits = shift_logits.view(-1, vocab)
    flat_labels = shift_labels.view(-1)

    # Guard the all-ignored case: F.cross_entropy with reduction="mean" returns
    # NaN (0/0) when every target is ignore_index. Return a real 0.0 that still
    # carries grad so empty micro-batches are safe inside accumulation.
    if not torch.any(flat_labels != ignore_index):
        return (flat_logits.float().sum() * 0.0).reshape(())

    # Cross entropy in float32 for numerical stability even under autocast.
    return F.cross_entropy(
        flat_logits.float(),
        flat_labels,
        ignore_index=ignore_index,
        reduction="mean",
    )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    T: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Hinton-style knowledge distillation loss (same-tokenizer logit KD).

    ``L = (1 - alpha) * CE(student, hard labels) + alpha * T^2 * KL(teacher_T || student_T)``

    Teacher and student must share the vocabulary (same-tokenizer KD). Both are
    ``[B, T, V]``; the single next-token shift is applied internally, and the KL
    term is masked to supervised (non-``ignore_index``) positions so it matches
    the SFT masking. Teacher logits are detached (the teacher never trains).
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits must match; got {tuple(student_logits.shape)} "
            f"vs {tuple(teacher_logits.shape)} (same-tokenizer KD requires shared vocab)"
        )
    sl = student_logits[:, :-1, :].float()
    tl = teacher_logits[:, :-1, :].float().detach()
    lab = labels[:, 1:]

    # Use the guarded lm_cross_entropy (handles all-IGNORE batches → 0.0, not NaN).
    hard = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lab.reshape(-1),
                           ignore_index=ignore_index, reduction="mean")
    if not torch.isfinite(hard):
        hard = sl.sum() * 0.0  # all-IGNORE batch → zero hard term (safe gradient)

    s_logp = F.log_softmax(sl / T, dim=-1)
    t_p = F.softmax(tl / T, dim=-1)
    # Per-position KL, masked to supervised tokens.
    kl = t_p * (t_p.clamp_min(1e-12).log() - s_logp)
    kl = kl.sum(-1)  # [B, T']
    mask = (lab != ignore_index).float()
    soft = (kl * mask).sum() / mask.sum().clamp_min(1.0) * (T * T)

    return (1.0 - alpha) * hard + alpha * soft


def sequence_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Sum of target-token log-probabilities per sequence (completion-only).

    Args:
        logits: ``[B, T, V]`` raw model outputs.
        labels: ``[B, T]`` target ids; ``ignore_index`` positions (prompt / pad)
            are excluded from the sum.
        ignore_index: label value marking positions to skip.

    Returns:
        ``[B]`` tensor: for each sequence, the summed log p(target) over its
        supervised (completion) positions. Prompt positions contribute nothing.

    Same shift as :func:`lm_cross_entropy`: position ``t`` of ``labels[:, 1:]`` is
    scored against ``logits[:, :-1]`` at the same index.
    """
    if logits.dim() != 3:
        raise ValueError(f"expected logits [B, T, V], got shape {tuple(logits.shape)}")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)

    mask = shift_labels != ignore_index  # [B, T-1] True on supervised positions
    # Gather needs valid indices; replace ignore positions with 0 then zero them out.
    safe_labels = shift_labels.masked_fill(~mask, 0).unsqueeze(-1)  # [B, T-1, 1]
    token_logp = log_probs.gather(-1, safe_labels).squeeze(-1)  # [B, T-1]
    token_logp = token_logp * mask  # zero out non-completion positions

    return token_logp.sum(dim=-1)  # [B]


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Direct Preference Optimization loss with reward-margin metrics.

    Loss = ``-logsigmoid(beta * ((pc - pr) - (rc - rr)))`` averaged over the batch,
    where ``pc/pr`` are policy and ``rc/rr`` reference sequence logprobs for the
    chosen/rejected completions. When the policy equals the reference, the inner
    term is 0 and the loss is ``-logsigmoid(0) = log(2)``.

    Args:
        policy_chosen_logps: ``[B]`` policy logprob of the chosen completion.
        policy_rejected_logps: ``[B]`` policy logprob of the rejected completion.
        ref_chosen_logps: ``[B]`` frozen-reference logprob of the chosen completion.
        ref_rejected_logps: ``[B]`` frozen-reference logprob of the rejected completion.
        beta: DPO temperature; higher = stronger preference signal.

    Returns:
        ``(loss, metrics)`` where ``loss`` is a scalar and ``metrics`` carries the
        implicit rewards, the reward margin (chosen - rejected), and the preference
        accuracy (fraction with chosen reward > rejected reward).
    """
    # Implicit rewards: beta * (policy_logp - reference_logp).
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    logits = chosen_rewards - rejected_rewards  # beta * ((pc-rc) - (pr-rr))
    loss = -F.logsigmoid(logits).mean()

    metrics = {
        "reward_chosen": chosen_rewards.mean().detach(),
        "reward_rejected": rejected_rewards.mean().detach(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().detach(),
        "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().detach(),
    }
    return loss, metrics
