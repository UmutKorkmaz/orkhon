"""Tests for the loglikelihood / multiple-choice evaluator."""

from __future__ import annotations

import math

import torch

from orkhon.eval.loglikelihood import (
    choice_loglikelihoods,
    loglikelihood,
    multiple_choice,
    run_multiple_choice,
)
from orkhon.model import Transformer
from orkhon.model.config import ModelConfig


class _Tok:
    """Minimal tokenizer stub: char-level encode for deterministic tests."""

    def encode(self, text: str) -> list[int]:
        # Map chars to ids >= 10 so they never collide with special 0..9.
        return [10 + (ord(c) % 50) for c in text]


def _model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=64, block_size=32, n_layers=2, d_model=32, n_heads=4,
                      n_kv_heads=2, intermediate_size=64, init_std=0.3,
                      tie_word_embeddings=False)
    return Transformer(cfg).eval()


def test_loglikelihood_matches_manual_logprob_sum():
    model = _model()
    ctx = [11, 12]
    cont = [13, 14]
    # Recompute by hand.
    x = torch.tensor([ctx + cont])
    import torch.nn.functional as F
    with torch.no_grad():
        lp = F.log_softmax(model(x)[0][0].float(), dim=-1)
        expected = float(lp[1, 13] + lp[2, 14])  # cont token 0 from pos len(ctx)-1=1
    assert math.isclose(loglikelihood(model, ctx, cont, device="cpu"), expected, rel_tol=1e-5)


def test_multiple_choice_picks_highest_loglikelihood():
    model = _model()
    tok = _Tok()
    ctx = "ab"
    choices = ["cd", "ef", "gh"]
    scores, _ = choice_loglikelihoods(model, tok, ctx, choices, device="cpu")
    best = int(max(range(len(scores)), key=lambda i: scores[i]))
    assert multiple_choice(model, tok, ctx, choices, device="cpu") == best


def test_run_multiple_choice_accuracy():
    model = _model()
    tok = _Tok()
    examples = []
    for ctx in ["ab", "xy", "pq"]:
        choices = ["c", "d", "e"]
        best = multiple_choice(model, tok, ctx, choices, device="cpu")
        examples.append({"context": ctx, "choices": choices, "label": best})
    res = run_multiple_choice(examples, model, tok, device="cpu")
    assert res["n"] == 3 and res["acc"] == 1.0


def test_loglikelihood_caps_to_context_window():
    """Long context+continuation must not crash (right-truncated to block_size)."""
    model = _model()  # block_size=32
    ctx = list(range(10, 40)) * 2   # 60 tokens — exceeds block_size 32
    cont = [41, 42, 43]
    val = loglikelihood(model, ctx, cont, device="cpu")
    assert isinstance(val, float) and val == val  # finite, not NaN
