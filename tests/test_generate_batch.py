"""Tests for batched generation + repetition penalty."""

from __future__ import annotations

import torch

from orkhon.model import Transformer, generate, generate_batch
from orkhon.model.config import ModelConfig
from orkhon.model.generation import _apply_repetition_penalty


def _model(seed: int = 0, init_std: float = 0.4) -> Transformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=64, block_size=64, n_layers=3, d_model=64, n_heads=4,
                      n_kv_heads=2, intermediate_size=128, init_std=init_std,
                      tie_word_embeddings=False, attn_impl="manual")
    return Transformer(cfg).eval()


def test_generate_batch_matches_single_greedy():
    """Batched greedy must equal single-sequence greedy for every prompt, even at
    different prompt lengths (left-pad + key-mask + RoPE relative invariance)."""
    m = _model()
    prompts = [[5, 6, 7, 8, 9], [10, 11], [20, 21, 22, 23]]
    batched = generate_batch(m, prompts, 10, pad_id=0, temperature=0.0, eos_ids=())
    single = [generate(m, p, 10, temperature=0.0, eos_ids=()) for p in prompts]
    assert batched == single


def test_generate_batch_matches_single_with_repetition_penalty():
    """Batched generation with rep_penalty must match single — pad_id in the
    left-pad region must NOT be unfairly penalized for shorter prompts."""
    m = _model(seed=2)
    # Variable-length prompts; pad_id=0 is NOT in any real prompt.
    prompts = [[10, 11, 12], [20, 21, 22, 23, 24, 25]]
    RP = 2.0
    batched = generate_batch(m, prompts, 8, pad_id=0, temperature=0.0,
                             repetition_penalty=RP, eos_ids=())
    single = [generate(m, p, 8, temperature=0.0, repetition_penalty=RP, eos_ids=())
              for p in prompts]
    assert batched == single, f"batch {batched} != single {single}"


def test_generate_batch_per_row_eos():
    """Each row stops at its own eos; finished rows don't keep appending."""
    m = _model()
    # force eos to be whatever the first sampled token is for row 0
    prompts = [[1, 2, 3], [4, 5, 6, 7]]
    first = generate(m, prompts[0], 1, temperature=0.0)[0]
    out = generate_batch(m, prompts, 8, pad_id=0, temperature=0.0, eos_ids=(first,))
    assert out[0] == [first]  # stopped immediately at eos
    assert len(out[1]) >= 1


def test_repetition_penalty_changes_output_and_is_noop_at_one():
    m = _model(seed=1)
    base = generate(m, [5, 5, 5, 5], 12, temperature=0.0, repetition_penalty=1.0)
    pen = generate(m, [5, 5, 5, 5], 12, temperature=0.0, repetition_penalty=1.5)
    assert base != pen  # penalty alters greedy choices
    # penalty == 1.0 is identical to not passing it
    assert base == generate(m, [5, 5, 5, 5], 12, temperature=0.0)


def test_repetition_penalty_math():
    logits = torch.tensor([2.0, -2.0, 0.5, 1.0])
    prev = torch.tensor([0, 1])  # seen tokens 0 and 1
    out = _apply_repetition_penalty(logits, prev, 2.0)
    assert torch.allclose(out, torch.tensor([1.0, -4.0, 0.5, 1.0]))  # 2/2, -2*2, unchanged
    # no-op at 1.0
    assert torch.equal(_apply_repetition_penalty(logits, prev, 1.0), logits)
