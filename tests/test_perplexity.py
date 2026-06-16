"""Tests for orkhon.eval.perplexity.evaluate."""

from __future__ import annotations

import math

import torch
from torch import nn

from orkhon.eval.perplexity import IGNORE_INDEX, evaluate
from orkhon.model.config import ModelConfig
from orkhon.model.transformer import Transformer


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        block_size=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        dropout=0.0,
        attn_impl="manual",
    )


def _tiny_model() -> Transformer:
    torch.manual_seed(0)
    return Transformer(_tiny_cfg()).eval()


class _FakePacked:
    """Minimal PackedDataset stand-in returning fixed (x, y) batches."""

    def __init__(self, vocab_size: int, seq_len: int, batches: int) -> None:
        self.seq_len = seq_len
        self._batches = batches
        self._calls = 0
        g = torch.Generator().manual_seed(1)
        self._x = torch.randint(0, vocab_size, (batches, 4, seq_len), generator=g)

    def get_batch(self, batch_size, device="cpu"):
        i = self._calls % self._batches
        self._calls += 1
        x = self._x[i, :batch_size]
        # y is x shifted left by one (next-token targets); reuse a roll for shape.
        y = torch.roll(x, shifts=-1, dims=1)
        return x, y


def test_evaluate_returns_finite_loss_and_ppl():
    model = _tiny_model()
    ds = _FakePacked(model.cfg.vocab_size, seq_len=8, batches=3)
    out = evaluate(model, ds, max_batches=3, batch_size=4, seq_len=8, device="cpu")

    assert math.isfinite(out["loss"])
    assert math.isfinite(out["ppl"])
    assert out["tokens"] > 0
    # ppl is exp(loss) by construction.
    assert math.isclose(out["ppl"], math.exp(out["loss"]), rel_tol=1e-6)


def test_ppl_equals_exp_loss_for_uniform_logits():
    """A model emitting uniform logits over V tokens has ppl == V exactly."""

    vocab = 7

    class _Uniform(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.training = False

        def forward(self, input_ids, attention_mask=None, past=None, use_cache=False):
            b, t = input_ids.shape
            logits = torch.zeros(b, t, vocab)  # uniform after softmax
            return logits, None

        def eval(self):
            self.training = False
            return self

    class _DS:
        def get_batch(self, batch_size, device="cpu"):
            x = torch.zeros(batch_size, 5, dtype=torch.long)
            y = torch.zeros(batch_size, 5, dtype=torch.long)
            return x, y

    out = evaluate(_Uniform(), _DS(), max_batches=2, batch_size=3, seq_len=5)
    # Uniform over `vocab` => loss = ln(vocab), ppl = vocab.
    assert math.isclose(out["loss"], math.log(vocab), rel_tol=1e-5)
    assert math.isclose(out["ppl"], vocab, rel_tol=1e-5)


def test_ignore_and_pad_excluded_from_denominator():
    """IGNORE_INDEX labels must not count toward tokens or NLL."""

    vocab = 5

    class _Uniform(nn.Module):
        def forward(self, input_ids, attention_mask=None, past=None, use_cache=False):
            b, t = input_ids.shape
            return torch.zeros(b, t, vocab), None

    class _CollatedDS:
        """One batch: only the last position is supervised; rest IGNORE_INDEX."""

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {}

        def collate(self, batch):
            # input length 4 => after shift, 3 target positions.
            input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
            labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4]])
            attn = torch.ones(1, 4, dtype=torch.bool)
            return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    out = evaluate(_Uniform(), _CollatedDS(), max_batches=1, batch_size=1)
    # Exactly one supervised token (the last label after the shift).
    assert out["tokens"] == 1
    # Uniform => per-token loss is ln(vocab); single token => loss == ln(vocab).
    assert math.isclose(out["loss"], math.log(vocab), rel_tol=1e-5)


def test_no_supervised_tokens_returns_inf():
    vocab = 5

    class _Uniform(nn.Module):
        def forward(self, input_ids, attention_mask=None, past=None, use_cache=False):
            b, t = input_ids.shape
            return torch.zeros(b, t, vocab), None

    class _AllIgnored:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {}

        def collate(self, batch):
            input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
            labels = torch.full((1, 3), IGNORE_INDEX)
            attn = torch.ones(1, 3, dtype=torch.bool)
            return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    out = evaluate(_Uniform(), _AllIgnored(), max_batches=1, batch_size=1)
    assert out["tokens"] == 0
    assert out["loss"] == float("inf")
    assert out["ppl"] == float("inf")
