"""Tests for build_optimizer's decay / no-decay parameter split."""

from __future__ import annotations

import torch

from orkhon.config.schema import OptimConfig
from orkhon.model.config import ModelConfig
from orkhon.model.transformer import Transformer
from orkhon.train.optim import build_optimizer


def _tiny_model(**overrides) -> Transformer:
    cfg = ModelConfig(
        vocab_size=64, block_size=16, n_layers=2, d_model=32,
        n_heads=4, n_kv_heads=2, intermediate_size=64,
        dropout=0.0, attn_impl="manual", use_bias=True, **overrides,
    )
    return Transformer(cfg)


def _groups(opt: torch.optim.Optimizer):
    decay = opt.param_groups[0]
    no_decay = opt.param_groups[1]
    return decay, no_decay


def _ids(group) -> set[int]:
    return {id(p) for p in group["params"]}


def test_two_groups_with_expected_weight_decay():
    model = _tiny_model()
    opt = build_optimizer(model, OptimConfig(weight_decay=0.1))
    decay, no_decay = _groups(opt)
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0


def test_norms_biases_embeddings_in_no_decay():
    model = _tiny_model()
    opt = build_optimizer(model, OptimConfig(weight_decay=0.1))
    _, no_decay = _groups(opt)
    no_decay_ids = _ids(no_decay)

    # Embedding weight -> no-decay (even though it is 2-D).
    assert id(model.embed_tokens.weight) in no_decay_ids
    # RMSNorm weights (1-D) -> no-decay.
    assert id(model.final_norm.weight) in no_decay_ids
    # Every bias and every 1-D parameter -> no-decay.
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name:
            assert id(p) in no_decay_ids, f"{name} should be in no-decay"


def test_matmul_weights_in_decay():
    model = _tiny_model()
    opt = build_optimizer(model, OptimConfig(weight_decay=0.1))
    decay, _ = _groups(opt)
    decay_ids = _ids(decay)

    embed_id = id(model.embed_tokens.weight)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matmul = p.ndim >= 2 and id(p) != embed_id
        if is_matmul:
            assert id(p) in decay_ids, f"{name} (matmul weight) should be in decay"


def test_all_params_covered_once_and_tied_deduped():
    model = _tiny_model(tie_word_embeddings=True)
    opt = build_optimizer(model, OptimConfig())
    decay, no_decay = _groups(opt)

    all_ids = _ids(decay) | _ids(no_decay)
    overlap = _ids(decay) & _ids(no_decay)
    assert not overlap, "a parameter landed in both groups"

    # Unique trainable parameters (deduped by identity, handling tied weights).
    unique_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert all_ids == unique_trainable

    # Tied lm_head shares the embedding Parameter -> counted exactly once.
    total_listed = len(decay["params"]) + len(no_decay["params"])
    assert total_listed == len(unique_trainable)
