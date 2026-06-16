"""Plain text completion for a *base* (pretrained) model.

Unlike ``orkhon.serve.chat_cli`` (which renders the chat template and is meant for
SFT/instruct checkpoints), this continues raw text — the right interface for a
pretraining-only model such as the TinyStories base model.

Pretraining packed each document as ``encode(doc) + <eos>``, so the model learned
that a fresh document begins right after ``<eos>``. We therefore prefix the prompt
with ``<eos>`` (``fresh_doc=True``) so generation starts a clean document, and stop
when the model emits the next ``<eos>`` (end of document).
"""

from __future__ import annotations

from pathlib import Path

import torch

from orkhon.model import generate
from orkhon.tokenizer import load_tokenizer
from orkhon.train.checkpoint import load_model_from_checkpoint
from orkhon.utils import resolve_device


def complete(
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    prompt: str,
    *,
    device: str = "auto",
    tag: str = "last",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
    repetition_penalty: float = 1.3,
    fresh_doc: bool = True,
) -> str:
    """Continue ``prompt`` with the base model; returns prompt + completion."""
    dev = resolve_device(device)
    model, _ = load_model_from_checkpoint(checkpoint_dir, device=dev, tag=tag)
    model.eval()
    tok = load_tokenizer(tokenizer_dir)

    prefix = [tok.special.eos] if fresh_doc else []
    ids = prefix + tok.encode(prompt)
    with torch.no_grad():
        new_ids = generate(
            model,
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_ids=(tok.special.eos,),
            device=dev,
        )
    return prompt + tok.decode(new_ids, skip_special=True)
