"""Perplexity evaluation over packed or supervised datasets.

Perplexity is ``exp(total_nll / total_tokens)`` where the sums run over real
target tokens only — ``IGNORE_INDEX`` (masked / non-assistant) and pad positions
never count toward either the numerator or the denominator. This mirrors the
training loss shift (``logits[:, :-1]`` predict ``labels[:, 1:]``) so eval and
train numbers are directly comparable.

The evaluator is dataset-shape agnostic. It supports two batch sources:

* :class:`orkhon.data.PackedDataset` via ``get_batch(batch_size, device)`` ->
  ``(x, y)``; ``y`` is already next-token aligned to ``x`` (every token
  supervised, no further shift needed).
* :class:`orkhon.data.SFTDataset` (map-style with ``collate``) -> batches of
  ``{input_ids, labels, attention_mask}`` where labels are aligned to inputs and
  carry ``IGNORE_INDEX`` on prompt/pad spans (one internal next-token shift).

Both paths reduce to the same token-weighted NLL accumulation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _nll_sum_and_count(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[float, int]:
    """Summed NLL and supervised-token count for already-aligned tensors.

    ``logits`` (``[B, T, V]``) and ``targets`` (``[B, T]``) must be position-for-
    position aligned — the caller is responsible for any next-token shift. Returns
    ``(nll_sum, n_tokens)`` over non-ignore targets.
    """
    vocab = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab).float()
    flat_targets = targets.reshape(-1)

    n_tokens = int((flat_targets != ignore_index).sum().item())
    if n_tokens == 0:
        return 0.0, 0

    nll_sum = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=ignore_index,
        reduction="sum",
    )
    return float(nll_sum.item()), n_tokens


def _eval_packed(model, dataset, max_batches, batch_size, device):
    """Accumulate NLL over PackedDataset batches (no extra shift: y aligns to x)."""
    total_nll = 0.0
    total_tokens = 0
    for _ in range(max_batches):
        x, y = dataset.get_batch(batch_size, device)
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        nll, n = _nll_sum_and_count(logits, y)
        total_nll += nll
        total_tokens += n
    return total_nll, total_tokens


def _iter_collated_batches(dataset: Any, max_batches: int, batch_size: int):
    """Yield collated batches from a map-style dataset that exposes ``collate``."""
    n = len(dataset)
    produced = 0
    idx = 0
    while produced < max_batches and idx < n:
        items = [dataset[j] for j in range(idx, min(idx + batch_size, n))]
        yield dataset.collate(items)
        idx += batch_size
        produced += 1


def _eval_collated(model, dataset, max_batches, batch_size, device):
    """Accumulate NLL over collated batches with the single next-token shift."""
    total_nll = 0.0
    total_tokens = 0
    for batch in _iter_collated_batches(dataset, max_batches, batch_size):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        logits, _ = model(input_ids, attention_mask=attention_mask)
        # Single shift: logits at position t predict labels at position t+1.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        nll, n = _nll_sum_and_count(shift_logits, shift_labels)
        total_nll += nll
        total_tokens += n
    return total_nll, total_tokens


@torch.no_grad()
def evaluate(
    model,
    dataset,
    *,
    max_batches: int = 50,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str | torch.device = "cpu",
) -> dict:
    """Compute token-weighted loss and perplexity over ``dataset``.

    Args:
        model: a :class:`orkhon.model.Transformer` (or any module returning
            ``(logits, _)`` from ``forward``).
        dataset: a :class:`~orkhon.data.PackedDataset` (``get_batch``) or a
            map-style dataset exposing ``__getitem__``/``__len__``/``collate``
            (e.g. :class:`~orkhon.data.SFTDataset`).
        max_batches: cap on number of batches drawn from the dataset.
        batch_size: rows per batch.
        seq_len: informational; PackedDataset already fixes its own window.
        device: device to run the model on.

    Returns:
        ``{"loss": float, "ppl": float, "tokens": int}`` where ``ppl == exp(loss)``
        and ``loss`` is the mean NLL over supervised tokens only. If no supervised
        tokens are seen, ``loss``/``ppl`` are ``inf`` and ``tokens`` is 0.
    """
    was_training = model.training
    model.eval()

    if hasattr(dataset, "get_batch"):
        total_nll, total_tokens = _eval_packed(
            model, dataset, max_batches, batch_size, device
        )
    else:
        total_nll, total_tokens = _eval_collated(
            model, dataset, max_batches, batch_size, device
        )

    if was_training:
        model.train()

    if total_tokens == 0:
        return {"loss": float("inf"), "ppl": float("inf"), "tokens": 0}

    loss = total_nll / total_tokens
    ppl = math.exp(loss)
    return {"loss": loss, "ppl": ppl, "tokens": total_tokens}
