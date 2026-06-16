"""Supervised fine-tuning stage: assistant-only cross-entropy.

``run`` loads pretrained weights from ``cfg.init_from`` (via
:func:`load_model_from_checkpoint`), reads ``{"messages": [...]}`` examples through
:class:`orkhon.data.SFTDataset`, and trains with the same
:func:`lm_cross_entropy` — but the dataset's ``labels`` already mask everything
except assistant content + closing ``<|end|>`` (``IGNORE_INDEX`` elsewhere), so the
loss is assistant-only by construction.

Padding is handled by :meth:`SFTDataset.collate`; the model receives the
``attention_mask`` so it ignores pad keys.
"""

from __future__ import annotations

import random

import torch

from orkhon.config.schema import SFTConfig
from orkhon.data.dataset import SFTDataset
from orkhon.tokenizer.render import IGNORE_INDEX
from orkhon.tokenizer.tokenizer import load_tokenizer
from orkhon.train.checkpoint import load_model_from_checkpoint
from orkhon.train.engine import prepare_run, run_training_loop
from orkhon.train.losses import lm_cross_entropy
from orkhon.train.optim import build_optimizer
from orkhon.utils.logging import get_logger

_logger = get_logger("orkhon.train.sft")


def _iter_batches(dataset: SFTDataset, batch_size: int, rng: random.Random):
    """Yield collated batches forever (random sampling with replacement)."""
    n = len(dataset)
    while True:
        idx = [rng.randrange(n) for _ in range(min(batch_size, n))]
        yield dataset.collate([dataset[i] for i in idx])


def run(cfg: SFTConfig) -> dict:
    """Fine-tune from ``cfg.init_from`` and return the training summary."""
    if cfg.data.sft_path is None:
        raise ValueError("sft requires cfg.data.sft_path (messages jsonl)")

    device, dtype = prepare_run(cfg.train)

    # Load pretrained weights + architecture from the checkpoint.
    model, model_cfg = load_model_from_checkpoint(cfg.init_from, device=device)
    tokenizer = load_tokenizer(cfg.tokenizer.dir)
    # Tool-use SFT: if the tokenizer was retrofitted (+tool/+image tokens) its vocab
    # may exceed the model's. Grow the embeddings (preserving old rows) so the new
    # tokens are learnable. If model_vocab >= tokenizer_vocab, no resize is needed
    # (the extra embedding rows are simply unused) — common when a model was trained
    # with a larger target vocab than the tokenizer ended up producing.
    if tokenizer.vocab_size > model_cfg.vocab_size:
        from orkhon.model.resize import resize_token_embeddings
        model, model_cfg = resize_token_embeddings(model, tokenizer.vocab_size)
        model.to(device)
    model.train()
    optimizer = build_optimizer(model, cfg.optim)
    train_ds = SFTDataset(cfg.data.sft_path, tokenizer)
    val_ds = (
        SFTDataset(cfg.data.val_sft_path, tokenizer)
        if cfg.data.val_sft_path
        else None
    )

    rng = random.Random(cfg.train.seed)
    train_batches = _iter_batches(train_ds, cfg.train.batch_size, rng)

    # The model can only attend within its block_size (RoPE table length). Cap the
    # context so an over-long example never indexes past the position table.
    max_ctx = min(cfg.train.seq_len, model_cfg.block_size)

    def _supervised_tokens(labels: torch.Tensor) -> int:
        return int((labels[:, 1:] != IGNORE_INDEX).sum().item())

    def micro_batch_loss(step: int) -> tuple[torch.Tensor, int]:
        batch = next(train_batches)
        input_ids = batch["input_ids"][:, :max_ctx].to(device)
        labels = batch["labels"][:, :max_ctx].to(device)
        attn = batch["attention_mask"][:, :max_ctx].to(device)
        logits, _ = model(input_ids, attention_mask=attn)
        loss = lm_cross_entropy(logits, labels)
        return loss, _supervised_tokens(labels)

    @torch.no_grad()
    def eval_fn() -> dict | None:
        if val_ds is None:
            return None
        was_training = model.training
        model.eval()
        val_rng = random.Random(cfg.train.seed + 1)
        batches = _iter_batches(val_ds, cfg.train.batch_size, val_rng)
        losses = []
        for _ in range(max(1, cfg.train.eval_iters)):
            batch = next(batches)
            logits, _ = model(
                batch["input_ids"][:, :max_ctx].to(device),
                attention_mask=batch["attention_mask"][:, :max_ctx].to(device),
            )
            losses.append(
                lm_cross_entropy(logits, batch["labels"][:, :max_ctx].to(device)).item()
            )
        if was_training:
            model.train()
        return {"val_loss": sum(losses) / len(losses)}

    _logger.info("sft: %d examples from %s", len(train_ds), cfg.data.sft_path)

    result = run_training_loop(
        model=model,
        optimizer=optimizer,
        micro_batch_loss=micro_batch_loss,
        optim_cfg=cfg.optim,
        train_cfg=cfg.train,
        model_cfg=model_cfg,
        device=device,
        dtype=dtype,
        eval_fn=eval_fn,
    )
    return result.as_dict()
