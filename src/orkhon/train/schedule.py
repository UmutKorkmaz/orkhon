"""Learning-rate schedule as a pure function of the step.

``lr_at(step, optim_cfg, max_steps)`` returns the LR for a given optimizer step:

1. Linear warmup from 0 to ``optim_cfg.lr`` over ``warmup_steps`` steps.
2. Then decay from ``optim_cfg.lr`` to ``optim_cfg.lr * min_lr_ratio`` according to
   ``optim_cfg.schedule`` (``cosine`` | ``linear`` | ``constant``) by ``max_steps``.

Keeping this a *pure function of step* (no internal counter) is what makes resume
exact: after restoring ``step`` from a checkpoint, the LR is recomputed, never
replayed. ``build_scheduler`` wires it onto an optimizer via ``LambdaLR`` for
callers that prefer the torch scheduler API, but the engine drives ``lr_at``
directly so the source of truth is the same in both paths.
"""

from __future__ import annotations

import math

import torch

from orkhon.config.schema import OptimConfig


def lr_at(step: int, optim_cfg: OptimConfig, max_steps: int) -> float:
    """Absolute learning rate for ``step`` (0-indexed optimizer step).

    Args:
        step: optimizer step index. Step 0 sits at the very start of warmup.
        optim_cfg: schedule hyperparameters (lr, min_lr_ratio, warmup_steps, schedule).
        max_steps: total optimizer steps (the decay horizon).

    Returns:
        The learning rate (a plain float) to set on every param group this step.
    """
    base_lr = optim_cfg.lr
    min_lr = base_lr * optim_cfg.min_lr_ratio
    warmup = optim_cfg.warmup_steps

    # --- Warmup: linear ramp 0 -> base_lr. ---
    if warmup > 0 and step < warmup:
        # step 0 -> lr ~ 0; step (warmup-1) approaches base_lr. We scale by
        # (step+1)/warmup so the last warmup step reaches base_lr exactly.
        return base_lr * (step + 1) / warmup

    # --- After warmup: decay base_lr -> min_lr by max_steps. ---
    schedule = optim_cfg.schedule
    if schedule == "constant":
        return base_lr

    # Progress through the decay phase in [0, 1].
    decay_steps = max(1, max_steps - warmup)
    progress = (step - warmup) / decay_steps
    progress = min(1.0, max(0.0, progress))

    if schedule == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return min_lr + coeff * (base_lr - min_lr)
    if schedule == "linear":
        return base_lr + progress * (min_lr - base_lr)

    raise ValueError(
        f"unknown schedule {schedule!r}; choose cosine | linear | constant"
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, optim_cfg: OptimConfig, max_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Wrap :func:`lr_at` in a ``LambdaLR`` for the torch scheduler API.

    The lambda returns a *multiplier* relative to the optimizer's base LR, so the
    effective LR matches :func:`lr_at` exactly. The engine sets LR directly via
    ``lr_at`` and does not require this, but it is provided for completeness and
    for callers that lean on ``scheduler.state_dict()`` for resume.
    """
    base_lr = optim_cfg.lr

    def lr_lambda(step: int) -> float:
        if base_lr == 0:
            return 0.0
        return lr_at(step, optim_cfg, max_steps) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
