"""Smoke evaluation: a fast regression gate after training.

``run_smoke_eval`` loads a trained checkpoint + tokenizer, generates a few
completions for a fixed set of chat prompts, and (optionally) measures validation
loss/perplexity on a held-out dataset. It returns a compact summary that CI can
diff between runs to catch regressions (NaNs, broken generation, loss blow-ups).

This module never trains — it only loads, generates, and evaluates. Defaults are
conservative (greedy, short) so the gate is deterministic and quick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from orkhon.eval.perplexity import evaluate
from orkhon.model.generation import generate
from orkhon.tokenizer.render import encode_for_inference
from orkhon.tokenizer.tokenizer import load_tokenizer
from orkhon.train.checkpoint import load_model_from_checkpoint
from orkhon.utils.device import resolve_device

# A small, fixed set of single-turn prompts used as the generation smoke set.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "Hello",
    "What is two plus two?",
    "Say something nice.",
)


def _generate_reply(
    model,
    tokenizer,
    user_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device,
) -> str:
    """Render a single-turn prompt, generate, and decode the assistant reply."""
    messages = [{"role": "user", "content": user_text}]
    prompt_ids = encode_for_inference(messages, tokenizer.encode, tokenizer.special)
    new_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_ids=(tokenizer.special.end,),
        device=device,
    )
    return tokenizer.decode(new_ids, skip_special=True)


def run_smoke_eval(
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    val_dataset=None,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    max_batches: int = 10,
    batch_size: int = 4,
    seq_len: int = 64,
    device: str = "auto",
    tag: str = "last",
) -> dict:
    """Generate sample completions and (optionally) measure validation loss.

    Args:
        checkpoint_dir: directory holding ``ckpt_<tag>.pt`` + ``model_config.json``.
        tokenizer_dir: directory with the trained tokenizer.
        prompts: user prompts to generate single-turn replies for.
        val_dataset: optional dataset (PackedDataset / SFTDataset) for loss/ppl.
        max_new_tokens / temperature / top_k / top_p: generation controls.
        max_batches / batch_size / seq_len: validation-eval controls.
        device: device preference string ("auto", "cpu", "mps", "cuda").
        tag: checkpoint tag ("last" or "best").

    Returns:
        ``{"completions": [{"prompt", "reply"}...], "val_loss", "val_ppl",
        "all_finite"}``. ``val_loss``/``val_ppl`` are ``None`` when no dataset is
        given. ``all_finite`` is False if any reply errored or loss is non-finite.
    """
    dev = resolve_device(device)
    model, _cfg = load_model_from_checkpoint(checkpoint_dir, device=dev, tag=tag)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    completions: list[dict] = []
    all_finite = True
    for user_text in prompts:
        reply = _generate_reply(
            model,
            tokenizer,
            user_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=dev,
        )
        completions.append({"prompt": user_text, "reply": reply})

    val_loss: float | None = None
    val_ppl: float | None = None
    if val_dataset is not None:
        metrics = evaluate(
            model,
            val_dataset,
            max_batches=max_batches,
            batch_size=batch_size,
            seq_len=seq_len,
            device=dev,
        )
        val_loss = metrics["loss"]
        val_ppl = metrics["ppl"]
        if not (isinstance(val_loss, float) and val_loss == val_loss and val_loss != float("inf")):
            all_finite = False

    return {
        "completions": completions,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "all_finite": all_finite,
        "num_prompts": len(completions),
    }
