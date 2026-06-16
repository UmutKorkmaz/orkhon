"""Optimizer construction with the standard decay / no-decay parameter split.

Weight decay should pull matmul (projection / embedding-output) weights toward
zero but must NOT decay norms, biases, or 1-D parameters — decaying those is a
known regularization bug that hurts training. The rule:

* parameters with ``ndim >= 2`` (the Linear / matmul weights) -> ``weight_decay``.
* everything else — RMSNorm weights, biases, any 1-D parameter -> ``0.0`` decay.

Embeddings are 2-D and would normally land in the decay group. Because the model
ties ``lm_head.weight`` to ``embed_tokens.weight`` (the same Parameter object),
we dedupe by parameter identity so the shared weight is added once. We also place
the (tied or untied) token embedding in the NO-decay group: token embeddings act
like a lookup table and are conventionally left un-decayed, matching the contract
("norms, biases, embeddings, 1D params" -> no decay).
"""

from __future__ import annotations

import torch
from torch import nn

from orkhon.config.schema import OptimConfig


def _is_embedding_param(model: nn.Module) -> set[int]:
    """Return ids of parameters owned by ``nn.Embedding`` modules.

    Used so token-embedding weights are routed to the no-decay group even though
    they are 2-D. Identity (``id``) is stable for the lifetime of the model and
    handles weight tying (the tied lm_head shares the same Parameter id).
    """
    embed_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                embed_ids.add(id(p))
    return embed_ids


def build_optimizer(
    model: nn.Module, optim_cfg: OptimConfig
) -> torch.optim.AdamW:
    """Build an AdamW optimizer with two parameter groups.

    Group 1 (decay): trainable parameters with ``ndim >= 2`` that are NOT
    embeddings (i.e. Linear / projection weights). Gets ``optim_cfg.weight_decay``.

    Group 2 (no-decay): norms, biases, all 1-D parameters, and embedding weights.
    Gets ``0.0`` weight decay.

    Args:
        model: the model whose ``requires_grad`` parameters are optimized.
        optim_cfg: hyperparameters (lr, betas, eps, weight_decay).

    Returns:
        Configured :class:`torch.optim.AdamW`. Parameters are deduped by identity
        so tied weights appear in exactly one group.
    """
    embed_ids = _is_embedding_param(model)

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    for param in model.parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:  # dedupe tied parameters (counted once)
            continue
        seen.add(pid)

        if param.ndim >= 2 and pid not in embed_ids:
            decay.append(param)
        else:
            no_decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": optim_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=optim_cfg.lr,
        betas=(optim_cfg.beta1, optim_cfg.beta2),
        eps=optim_cfg.eps,
    )
