"""Autoregressive generation with a KV-cache (single-sequence and batched).

``generate`` prefills the prompt in one forward pass (populating the cache), then
decodes one token at a time. Cached decode reuses stored keys/values, so each
step is a single-token forward whose logits match a full no-cache re-forward over
the growing sequence (parity is verified in the tests).

``generate_batch`` decodes many prompts at once. Prompts are **left-padded** to a
common length and a key-padding mask hides the pad positions. This is correct
without any per-sequence position handling because RoPE is *relative*: a uniform
per-sequence absolute shift leaves every real-token distance unchanged (the same
invariance behind the KV-cache parity). Batched generation is the shared primitive
behind fast serving, in-loop eval, and rejection-sampling / GRPO rollouts.

Sampling: ``temperature == 0`` is greedy (argmax). Otherwise a **repetition
penalty** (CTRL-style; ``> 1`` discourages repeats) is applied to already-seen
tokens, then temperature, then optional ``top_k`` / ``top_p`` filtering, then a
multinomial draw. Both functions return ONLY the newly generated ids.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from orkhon.model.kv_cache import KVCache


def _apply_repetition_penalty(
    logits: torch.Tensor, prev_ids: torch.Tensor | None, penalty: float
) -> torch.Tensor:
    """CTRL-style repetition penalty on a ``[..., vocab]`` logits row/batch.

    ``prev_ids`` are the already-seen token ids (1-D for a single row, or a
    ``[B, L]`` long tensor for a batch). Positive logits of seen tokens are
    divided by ``penalty``, negative ones multiplied — both reduce their
    probability. ``penalty == 1.0`` is a no-op.
    """
    if penalty == 1.0 or prev_ids is None or prev_ids.numel() == 0:
        return logits
    out = logits.clone()
    if out.dim() == 1:
        seen = torch.unique(prev_ids)
        vals = out[seen]
        out[seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
        return out
    # Batched: [B, V] logits, [B, L] prev ids.
    gathered = out.gather(1, prev_ids)
    gathered = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
    out.scatter_(1, prev_ids, gathered)
    return out


def _filter_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Apply temperature, then top-k and top-p filtering. ``logits``: [vocab]."""
    logits = logits / max(temperature, 1e-8)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth_val = torch.topk(logits, k).values[-1]
        logits = torch.where(
            logits < kth_val, torch.full_like(logits, float("-inf")), logits
        )

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # Keep tokens up to and including the one crossing the top_p threshold.
        remove = cumprobs - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            0, sorted_idx, sorted_logits
        )

    return logits


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    *,
    prev_ids: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
) -> int:
    """Pick the next token id from final-position logits ``[vocab]``."""
    logits = _apply_repetition_penalty(logits, prev_ids, repetition_penalty)
    if temperature == 0:
        return int(torch.argmax(logits).item())
    filtered = _filter_logits(logits, temperature, top_k, top_p)
    probs = F.softmax(filtered, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model,
    prompt_ids: list[int] | torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    eos_ids: tuple[int, ...] = (),
    device=None,
) -> list[int]:
    """Generate up to ``max_new_tokens`` ids; return ONLY the new ids."""
    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    if isinstance(prompt_ids, torch.Tensor):
        ids = prompt_ids.flatten().tolist()
    else:
        ids = list(prompt_ids)

    eos_set = set(eos_ids)
    block_size = getattr(model, "_rope_max_seq", model.cfg.block_size)

    # Prefill the prompt and seed the cache.
    cache = KVCache(model.cfg.n_layers)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits, cache = model(input_ids, past=cache, use_cache=True)
    next_logits = logits[0, -1, :]

    generated: list[int] = []
    for _ in range(max_new_tokens):
        prev = torch.tensor(ids + generated, dtype=torch.long, device=device)
        next_id = _sample_token(
            next_logits, temperature, top_k, top_p,
            prev_ids=prev, repetition_penalty=repetition_penalty,
        )
        generated.append(next_id)
        if next_id in eos_set:
            break

        # Guard against exceeding the trained context window.
        if cache.length >= block_size:
            break

        step_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, cache = model(step_input, past=cache, use_cache=True)
        next_logits = logits[0, -1, :]

    if was_training:
        model.train()
    return generated


@torch.no_grad()
def generate_batch(
    model,
    prompts: list[list[int]],
    max_new_tokens: int,
    *,
    pad_id: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    eos_ids: tuple[int, ...] = (),
    device=None,
) -> list[list[int]]:
    """Generate for a batch of prompts at once; return ONLY new ids per prompt.

    Prompts are left-padded to a common length; a key-padding mask hides the pads.
    Each sequence stops at its own ``eos`` (finished rows just stop appending).
    """
    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    bsz = len(prompts)
    max_len = max(len(p) for p in prompts)
    eos_set = set(eos_ids)
    block_size = getattr(model, "_rope_max_seq", model.cfg.block_size)

    # Left-pad: real tokens end at the last column so logits[:, -1] is next-token.
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    key_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
    for i, p in enumerate(prompts):
        input_ids[i, max_len - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
        key_mask[i, max_len - len(p):] = True

    cache = KVCache(model.cfg.n_layers)
    logits, cache = model(input_ids, attention_mask=key_mask, past=cache, use_cache=True)
    next_logits = logits[:, -1, :]  # [B, V]

    seqs: list[list[int]] = [p[:] for p in prompts]  # running context per row
    out: list[list[int]] = [[] for _ in range(bsz)]
    finished = [False] * bsz

    for _ in range(max_new_tokens):
        # Repetition penalty per row using each row's ACTUAL context (not the padded
        # tensor — pad_id in the left-pad region would be unfairly penalized for
        # shorter prompts, breaking batch-vs-single parity).
        penal = next_logits.clone()
        for i in range(bsz):
            row_prev = torch.tensor(seqs[i], dtype=torch.long, device=device)
            if row_prev.numel() > 0:
                penal[i] = _apply_repetition_penalty(next_logits[i], row_prev, repetition_penalty)

        step_ids: list[int] = []
        for i in range(bsz):
            row = penal[i]
            if temperature == 0:
                tok = int(torch.argmax(row).item())
            else:
                filt = _filter_logits(row, temperature, top_k, top_p)
                tok = int(torch.multinomial(F.softmax(filt, dim=-1), 1).item())
            step_ids.append(tok)
            if not finished[i]:
                out[i].append(tok)
                seqs[i].append(tok)
                if tok in eos_set:
                    finished[i] = True

        if all(finished) or cache.length >= block_size:
            break

        step_input = torch.tensor([[t] for t in step_ids], dtype=torch.long, device=device)
        key_mask = torch.cat(
            [key_mask, torch.ones((bsz, 1), dtype=torch.bool, device=device)], dim=1
        )
        logits, cache = model(step_input, attention_mask=key_mask, past=cache, use_cache=True)
        next_logits = logits[:, -1, :]

    if was_training:
        model.train()
    return out
