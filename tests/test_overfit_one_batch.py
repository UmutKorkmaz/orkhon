"""A tiny model must be able to overfit a single fixed batch (loss drops sharply).

This is the canonical sanity check that the optimizer, loss, and backward path are
wired correctly: with a fixed batch and a reasonable LR, the loss should fall well
over 50% within a handful of steps. Kept tiny so it runs in well under a second.
"""

from __future__ import annotations

import pytest
import torch

from orkhon.config.schema import OptimConfig
from orkhon.model.config import ModelConfig
from orkhon.model.transformer import Transformer
from orkhon.train.losses import lm_cross_entropy
from orkhon.train.optim import build_optimizer
from orkhon.utils.seed import set_seed


def _tiny_model() -> Transformer:
    cfg = ModelConfig(
        vocab_size=64, block_size=16, n_layers=2, d_model=32,
        n_heads=4, n_kv_heads=2, intermediate_size=64,
        dropout=0.0, attn_impl="manual",
    )
    return Transformer(cfg)


@pytest.mark.slow
def test_overfits_single_batch():
    set_seed(0)
    torch.manual_seed(0)
    model = _tiny_model()
    model.train()

    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 64, (4, 8), generator=g)
    labels = x.clone()

    opt = build_optimizer(model, OptimConfig(lr=3e-3, weight_decay=0.0))

    def step() -> float:
        opt.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = lm_cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        return float(loss.detach().item())

    initial = step()
    final = initial
    for _ in range(60):
        final = step()

    assert final < 0.5 * initial, f"loss did not halve: {initial:.3f} -> {final:.3f}"
