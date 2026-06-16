"""Loglikelihood evaluation — the engine behind multiple-choice benchmarks.

HellaSwag, ARC, MMLU, TR-MMLU etc. are all scored the same way: for each candidate
continuation, compute ``log P(continuation | context)`` under the model, and pick
the highest. This module implements that directly on Orkhon's hand-written model
(no ``lm-eval`` dependency, no ``AutoModel``) — one forward pass over
``context + continuation``, summing the log-probabilities of the continuation
tokens.

Two standard metrics:
- ``acc``      — argmax of the raw loglikelihood.
- ``acc_norm`` — argmax of the loglikelihood normalized by continuation length
  (in bytes), which corrects for longer answers being penalized.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def loglikelihood(model, context_ids: list[int], continuation_ids: list[int], *, device) -> float:
    """Return ``sum log P(continuation | context)`` (a single forward pass).

    The token at continuation position ``i`` is predicted by the logits at sequence
    position ``len(context) - 1 + i`` (the standard next-token shift). If the full
    sequence exceeds the model's context window, the context is right-truncated
    (keeping the most recent tokens) so the forward pass stays within bounds.
    """
    if not continuation_ids:
        return 0.0
    max_ctx = int(getattr(model, "_rope_max_seq", model.cfg.block_size))
    ids = context_ids + continuation_ids
    if len(ids) > max_ctx:
        ids = ids[-max_ctx:]
        c0 = len(ids) - len(continuation_ids)
    else:
        c0 = len(context_ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    logprobs = F.log_softmax(logits[0].float(), dim=-1)  # [T, V]
    total = 0.0
    for i, tok in enumerate(continuation_ids):
        total += float(logprobs[c0 - 1 + i, tok])
    return total


def choice_loglikelihoods(
    model,
    tokenizer,
    context: str,
    choices: Sequence[str],
    *,
    device,
) -> tuple[list[float], list[int]]:
    """Loglikelihood of each choice, using string-difference tokenization.

    We tokenize the FULL ``context + choice`` and the ``context`` alone, then the
    continuation is the difference — this is robust to byte-level BPE merge
    boundaries (e.g. a leading space changing the first token) without special
    handling. Returns ``(loglikes, cont_token_counts)``.
    """
    ctx_ids = tokenizer.encode(context)
    scores: list[float] = []
    counts: list[int] = []
    for choice in choices:
        full = tokenizer.encode(context + choice)
        # The continuation is full minus the context prefix. (With byte-level BPE
        # on a plain string concatenation this is exact; we guard defensively.)
        k = len(ctx_ids)
        cont = full[k:] if len(full) >= k else full
        scores.append(loglikelihood(model, ctx_ids, cont, device=device))
        counts.append(max(len(cont), 1))
    return scores, counts


def multiple_choice(
    model, tokenizer, context: str, choices: Sequence[str], *, device, normalize: bool = False
) -> int:
    """Return the index of the best choice.

    ``normalize=True`` uses byte-length-normalized loglikelihood (``acc_norm``):
    each score divided by the UTF-8 byte length of the choice, which corrects for
    longer continuations being inherently less probable.
    """
    scores, counts = choice_loglikelihoods(model, tokenizer, context, choices, device=device)
    if normalize:
        lens = [max(len(c.encode("utf-8")), 1) for c in choices]
        scores = [s / n for s, n in zip(scores, lens)]
    return int(max(range(len(scores)), key=lambda i: scores[i]))


def run_multiple_choice(
    examples: Sequence[dict],
    model,
    tokenizer,
    *,
    device,
    limit: int | None = None,
) -> dict:
    """Score a list of ``{context, choices, label}`` examples; return metrics.

    Returns ``{"acc": float, "acc_norm": float, "n": int}``. ``label`` is the
    0-indexed correct choice (or an int if the task uses letters).
    """
    n = 0
    correct = 0
    correct_norm = 0
    for ex in examples[:limit] if limit else examples:
        label = ex["label"]
        if isinstance(label, str):
            label = "abcdefghijklmnopqrstuvwxyz".index(label.lower())
        choices = ex["choices"]
        pred = multiple_choice(model, tokenizer, ex["context"], choices, device=device)
        pred_norm = multiple_choice(
            model, tokenizer, ex["context"], choices, device=device, normalize=True
        )
        correct += int(pred == label)
        correct_norm += int(pred_norm == label)
        n += 1
    return {
        "acc": correct / n if n else float("nan"),
        "acc_norm": correct_norm / n if n else float("nan"),
        "n": n,
    }
