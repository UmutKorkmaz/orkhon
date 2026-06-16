"""Knowledge distillation: train a student from a frozen teacher's logits.

Same-tokenizer logit KD (Hinton): the student minimizes a blend of hard-label
cross-entropy and soft KL to the teacher's temperature-scaled distribution
(:func:`orkhon.train.losses.distillation_loss`). It reuses the shared training
engine, so resume / checkpointing / mixed-precision / the NaN guard all apply.

Realistic use at Orkhon's scale: the teacher is a *larger* Orkhon checkpoint (same
tokenizer/vocab) and the student a smaller one — distillation reaches comparable
quality at far lower FLOPs than training the student from scratch. (Cross-tokenizer
KD / on-policy GKD are later additions; this is the offline same-tokenizer stage.)
"""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel

from orkhon.config.load import load_model_config
from orkhon.data.shard import ShardedPackedDataset
from orkhon.model.transformer import Transformer
from orkhon.train.engine import prepare_run, run_training_loop
from orkhon.train.losses import distillation_loss
from orkhon.train.optim import build_optimizer
from orkhon.utils.logging import get_logger

_logger = get_logger("orkhon.train.distill")


class DistillConfig(BaseModel):
    model_config = {"extra": "forbid"}

    arch_path: str            # student architecture
    teacher: str              # teacher checkpoint dir (frozen)
    prepared_dir: str         # sharded/flat pretrain data (shared vocab)
    out_dir: str = "runs/distill"
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    schedule: str = "cosine"
    temperature: float = 2.0
    alpha: float = 0.5        # 1.0 = pure distillation, 0.0 = pure hard labels
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 1337
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 2000
    seq_len: int = 512
    log_interval: int = 50
    eval_interval: int = 500
    eval_iters: int = 50
    ckpt_interval: int = 500


def run(cfg: DistillConfig) -> dict:
    """Distill the teacher into a fresh student built from ``cfg.arch_path``."""
    from orkhon.train.checkpoint import load_model_from_checkpoint

    device, dtype = prepare_run(_shim(cfg))

    model_cfg = load_model_config(cfg.arch_path)
    student = Transformer(model_cfg).to(device)
    student.train()
    optimizer = build_optimizer(student, _optim(cfg))

    teacher, tcfg = load_model_from_checkpoint(cfg.teacher, device=device)
    if tcfg.vocab_size != model_cfg.vocab_size:
        raise ValueError(
            f"teacher vocab {tcfg.vocab_size} != student vocab "
            f"{model_cfg.vocab_size}; same-tokenizer KD requires a shared vocab"
        )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ds = ShardedPackedDataset(cfg.prepared_dir, cfg.seq_len)

    def micro_batch_loss(step: int):
        x, y = ds.get_batch(cfg.batch_size, device)
        s_logits, _ = student(x)
        with torch.no_grad():
            t_logits, _ = teacher(x)
        loss = distillation_loss(s_logits, t_logits, y, T=cfg.temperature,
                                 alpha=cfg.alpha)
        return loss, x.numel()

    def eval_fn():
        import numpy as np
        total, n = 0.0, 0
        for _ in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, "cpu")
            with torch.no_grad():
                sl, _ = student(x)
            # eval on hard-label CE (the student's actual language modeling loss)
            from orkhon.train.losses import lm_cross_entropy
            total += float(lm_cross_entropy(sl, y).item()) * x.numel()
            n += x.numel()
        return {"val_loss": total / max(n, 1)}

    res = run_training_loop(
        model=student, optimizer=optimizer, micro_batch_loss=micro_batch_loss,
        optim_cfg=_optim(cfg), train_cfg=_shim(cfg), model_cfg=model_cfg,
        device=device, dtype=dtype, eval_fn=eval_fn, resume=False,
    )
    _logger.info("distillation done: %s", res.final_train_loss)
    return res.as_dict()


# --- Thin shims so run_training_loop's existing config types are satisfied. ---
from orkhon.config.schema import OptimConfig, TrainConfig


def _optim(cfg: DistillConfig) -> OptimConfig:
    return OptimConfig(lr=cfg.lr, weight_decay=cfg.weight_decay, grad_clip=cfg.grad_clip,
                       warmup_steps=cfg.warmup_steps, min_lr_ratio=cfg.min_lr_ratio,
                       schedule=cfg.schedule)


def _shim(cfg: DistillConfig) -> TrainConfig:
    return TrainConfig(device=cfg.device, dtype=cfg.dtype, seed=cfg.seed,
                       batch_size=cfg.batch_size, grad_accum_steps=cfg.grad_accum_steps,
                       max_steps=cfg.max_steps, seq_len=cfg.seq_len,
                       log_interval=cfg.log_interval, eval_interval=cfg.eval_interval,
                       eval_iters=cfg.eval_iters, ckpt_interval=cfg.ckpt_interval,
                       out_dir=cfg.out_dir)
