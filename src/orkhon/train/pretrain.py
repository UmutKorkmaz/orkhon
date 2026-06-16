"""Pretraining stage: next-token prediction over packed token windows.

``run`` builds the model from ``cfg.arch_path``, opens the packed train/val bins
under ``cfg.data.prepared_dir``, and trains with :func:`lm_cross_entropy` over
``(x -> y)`` windows sampled by :class:`orkhon.data.PackedDataset`. All the loop
mechanics (accumulation, clipping, scheduling, checkpointing, resume) live in
:mod:`orkhon.train.engine`; this module only wires the data and the loss.
"""

from __future__ import annotations

from pathlib import Path

import torch

from orkhon.config.load import load_model_config
from orkhon.config.schema import PretrainConfig
from orkhon.data.pack import PackedDataset
from orkhon.data.shard import ShardedPackedDataset
from orkhon.model.transformer import Transformer
from orkhon.train.distributed import rank
from orkhon.train.engine import prepare_run, run_training_loop
from orkhon.train.losses import lm_cross_entropy
from orkhon.train.optim import build_optimizer
from orkhon.utils.logging import get_logger

_logger = get_logger("orkhon.train.pretrain")


def run(cfg: PretrainConfig) -> dict:
    """Pretrain a model and return ``{final_train_loss, best_val_loss, steps}``."""
    if cfg.data.prepared_dir is None:
        raise ValueError("pretrain requires cfg.data.prepared_dir (packed bins)")

    device, dtype = prepare_run(cfg.train)

    model_cfg = load_model_config(cfg.arch_path)
    model = Transformer(model_cfg).to(device)
    model.train()
    optimizer = build_optimizer(model, cfg.optim)

    prepared = Path(cfg.data.prepared_dir)
    seq_len = cfg.train.seq_len
    # ShardedPackedDataset auto-detects the format: sharded (manifest.json +
    # train/shard_*.bin) or the legacy single train.bin. Either way, same get_batch.
    train_ds = ShardedPackedDataset(prepared, seq_len)

    val_path = prepared / "val.bin"
    val_ds = (
        PackedDataset(val_path, seq_len)
        if val_path.exists() and val_path.stat().st_size > (seq_len + 1) * 2
        else None
    )

    batch_size = cfg.train.batch_size
    micro_offsets: dict[int, int] = {}

    def micro_batch_loss(step: int) -> tuple[torch.Tensor, int]:
        micro_idx = micro_offsets.get(step, 0)
        micro_offsets[step] = micro_idx + 1
        sample_step = step * max(1, cfg.train.grad_accum_steps) + micro_idx
        x, y = train_ds.get_batch(
            batch_size,
            device,
            seed=cfg.train.seed,
            step=sample_step,
            rank=rank(),
        )
        logits, _ = model(x)
        loss = lm_cross_entropy(logits, _labels_from_targets(x, y))
        # Supervised tokens per micro-batch = the shifted target count.
        n_tokens = batch_size * (seq_len - 1)
        return loss, n_tokens

    @torch.no_grad()
    def eval_fn() -> dict | None:
        if val_ds is None:
            return None
        was_training = model.training
        model.eval()
        losses = []
        for _ in range(max(1, cfg.train.eval_iters)):
            x, y = val_ds.get_batch(batch_size, device)
            logits, _ = model(x)
            losses.append(lm_cross_entropy(logits, _labels_from_targets(x, y)).item())
        if was_training:
            model.train()
        return {"val_loss": sum(losses) / len(losses)}

    _logger.info(
        "pretrain: %d params, %d train tokens",
        sum(p.numel() for p in model.parameters()),
        len(train_ds),
    )

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


def _labels_from_targets(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Build a label tensor aligned to ``x`` for the engine's single-shift loss.

    ``PackedDataset`` returns ``y`` already shifted by one relative to ``x``. The
    shared :func:`lm_cross_entropy` applies its own single shift
    (``logits[:, :-1]`` vs ``labels[:, 1:]``), so we construct ``labels`` such that
    ``labels[:, t+1] == y[:, t]`` — i.e. ``labels = [x[:, 0], y[:, 0], y[:, 1], ...]``.
    Position 0's label is never scored (no logit precedes it after the shift), so
    its value is irrelevant; we use ``x[:, 0]`` for a clean, full-length tensor.
    """
    # labels[:, 0] = x[:, 0] (unused after shift); labels[:, 1:] = y[:, :-1].
    labels = x.clone()
    labels[:, 1:] = y[:, :-1]
    return labels
