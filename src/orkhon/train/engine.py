"""The shared training loop reused by pretrain, SFT, and DPO.

Each stage supplies two closures:

* ``micro_batch_loss(step) -> (loss, n_tokens)`` — runs one micro-batch forward and
  returns the *mean* loss for that micro-batch plus the number of supervised tokens
  it covered (for throughput accounting).
* ``eval_loss() -> dict`` — returns ``{"val_loss": float, ...}`` for periodic
  validation, or ``None`` if the stage has no validation set.

The engine owns the cross-cutting concerns:

* device / dtype resolution and autocast (via :mod:`orkhon.utils`).
* gradient accumulation. The micro-batch loss is divided by ``grad_accum_steps``
  before ``backward`` so the accumulated gradient equals the gradient of the mean
  over the full effective batch — the classic accumulation trap.
* gradient clipping at ``optim.grad_clip``.
* LR scheduling per optimizer step via :func:`orkhon.train.schedule.lr_at` (pure
  function of step -> exact resume).
* periodic eval, JSONL metrics logging, and checkpointing (last + best-by-val).
* resume from ``out_dir/ckpt_last.pt`` (model + optimizer + step + rng + lr).

Keeping all of this in one place means the three stages differ only in how a
micro-batch loss is computed, not in how training is driven.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Skip an optimizer step if the loss exceeds this multiple of its running EMA.
_SPIKE_MULT = 8.0

import torch
from torch import nn

from orkhon.config.schema import OptimConfig, TrainConfig
from orkhon.train.checkpoint import load_checkpoint, save_checkpoint
from orkhon.train.distributed import (
    all_reduce_mean,
    barrier,
    cleanup_distributed,
    init_distributed,
    is_distributed,
    is_main_process,
    rank,
    resolve_wrap_mode,
    world_size,
    wrap_model,
)
from orkhon.train.metrics import grad_global_norm, tokens_per_second
from orkhon.train.schedule import lr_at
from orkhon.utils.dtype import autocast_ctx, resolve_dtype
from orkhon.utils.device import resolve_device
from orkhon.utils.logging import JsonlMetrics, get_logger
from orkhon.utils.seed import get_rng_state, set_rng_state, set_seed

# Closure types.
MicroBatchLoss = Callable[[int], tuple[torch.Tensor, int]]
EvalFn = Callable[[], dict | None]

_logger = get_logger("orkhon.train")


@dataclass
class TrainResult:
    """Summary returned by :func:`run_training_loop`."""

    final_train_loss: float
    best_val_loss: float
    steps: int
    n_skipped: int = 0

    def as_dict(self) -> dict:
        return {
            "final_train_loss": self.final_train_loss,
            "best_val_loss": self.best_val_loss,
            "steps": self.steps,
            "n_skipped": self.n_skipped,
        }


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module behind a DDP/FSDP wrapper (else ``model``).

    Checkpoints must persist the *unwrapped* state dict so they load identically
    in single-process inference/export paths. Both DDP and the classic FSDP
    wrapper expose the original model as ``.module``; ``fully_shard`` mutates the
    model in place and has no wrapper, so ``model`` is already correct there.
    """
    return getattr(model, "module", model)


def maybe_resume(
    out_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device,
) -> int:
    """Restore model + optimizer + RNG from ``out_dir/ckpt_last.pt`` if present.

    Returns the step to *continue from* (0 when no checkpoint exists). LR is not
    stored explicitly; it is recomputed from the step by the scheduler, so resume
    is exact as long as the step is restored.
    """
    ckpt_path = Path(out_dir) / "ckpt_last.pt"
    if not ckpt_path.exists():
        return 0

    ckpt = load_checkpoint(out_dir, tag="last", map_location=map_location)
    unwrap_model(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("rng_state") is not None:
        set_rng_state(ckpt["rng_state"])
    start_step = int(ckpt["step"])
    _logger.info("resumed from %s at step %d", ckpt_path, start_step)
    return start_step


def run_training_loop(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    micro_batch_loss: MicroBatchLoss,
    optim_cfg: OptimConfig,
    train_cfg: TrainConfig,
    model_cfg,
    device: torch.device,
    dtype: torch.dtype,
    eval_fn: EvalFn | None = None,
    resume: bool = True,
) -> TrainResult:
    """Drive the full training loop and return a summary.

    Args:
        model: the (already device-placed) model to train.
        optimizer: its optimizer (two param groups from :func:`build_optimizer`).
        micro_batch_loss: closure returning ``(loss, n_tokens)`` for one micro-batch.
        optim_cfg: LR schedule + clipping hyperparameters.
        train_cfg: loop control (max_steps, intervals, grad_accum, out_dir, seed).
        model_cfg: architecture config saved into checkpoints.
        device / dtype: resolved compute device and autocast dtype.
        eval_fn: optional validation closure returning ``{"val_loss": ...}``.
        resume: when True, continue from ``out_dir/ckpt_last.pt`` if it exists.

    Returns:
        :class:`TrainResult` with final train loss, best val loss, and steps run.
    """
    # --- Distributed setup (no-op when WORLD_SIZE <= 1; single-process is identical). ---
    init_distributed(device)
    n_ranks = world_size()
    distributed = is_distributed()
    main = is_main_process()
    if distributed:
        # Data sharding via a rank-disjoint sampler stream: each rank advances its
        # own NumPy RNG so PackedDataset.get_batch draws different windows. Combined
        # with the same global batch this gives an effective batch of
        # batch_size * grad_accum * world_size with no overlapping samples per step.
        set_seed(train_cfg.seed + rank())
        model = wrap_model(model, resolve_wrap_mode(), device=device)
        _logger.info("distributed: %d ranks, effective batch x%d", n_ranks, n_ranks)

    out_dir = Path(train_cfg.out_dir)
    if main:
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier()  # ensure out_dir exists before non-main ranks proceed

    start_step = (
        maybe_resume(out_dir, model, optimizer, map_location=device)
        if resume
        else 0
    )

    # Only the main process writes the metrics log; other ranks discard their copy.
    grad_accum = max(1, train_cfg.grad_accum_steps)
    max_steps = train_cfg.max_steps
    from orkhon.train.monitor import build_metrics_sink
    metrics = build_metrics_sink(
        out_dir, enabled=main, backend=train_cfg.monitor,
        project="orkhon", run_name=out_dir.name, config={"max_steps": max_steps},
    )

    best_val_loss = float("inf")
    last_train_loss = float("nan")
    loss_ema: float | None = None
    n_skipped = 0

    try:
        for step in range(start_step, max_steps):
            # LR is a pure function of step (exact resume).
            lr = lr_at(step, optim_cfg, max_steps)
            _set_lr(optimizer, lr)

            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            step_loss_sum = 0.0
            step_tokens = 0
            for _ in range(grad_accum):
                with autocast_ctx(device, dtype):
                    loss, n_tokens = micro_batch_loss(step)
                # DIVIDE by grad_accum so the accumulated grad is the mean
                # gradient over the effective batch (the known trap).
                (loss / grad_accum).backward()
                step_loss_sum += float(loss.detach().item())
                step_tokens += int(n_tokens)

            grad_norm = grad_global_norm(model)
            avg_loss = step_loss_sum / grad_accum

            # NaN / loss-spike guard: a single bad step (fp overflow, data glitch)
            # can wreck a multi-day run. If the loss or grad norm is non-finite, or
            # the loss spikes far above its running average, SKIP the update (drop
            # the gradients) instead of corrupting the weights.
            is_bad = (
                not math.isfinite(grad_norm)
                or not math.isfinite(avg_loss)
                or (loss_ema is not None and avg_loss > _SPIKE_MULT * loss_ema)
            )
            if is_bad:
                optimizer.zero_grad(set_to_none=True)
                n_skipped += 1
                if main:
                    _logger.warning(
                        "step %d SKIPPED (loss=%.3f grad_norm=%.3f ema=%.3f): "
                        "non-finite or spike; %d skipped total",
                        step, avg_loss, grad_norm, loss_ema or float("nan"), n_skipped,
                    )
                continue

            if optim_cfg.grad_clip and optim_cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.grad_clip)
            optimizer.step()
            loss_ema = avg_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * avg_loss

            elapsed = time.perf_counter() - t0
            # The logged loss is the global mean across ranks (matches what one
            # large batch would yield); a no-op identity in single-process runs.
            if distributed:
                avg_loss = all_reduce_mean(avg_loss, device)
            last_train_loss = avg_loss
            # Throughput is aggregate: sum each rank's local tok/s into a cluster total.
            tps = tokens_per_second(step_tokens, elapsed) * n_ranks

            if main and step % train_cfg.log_interval == 0:
                metrics.log(
                    step,
                    train_loss=avg_loss,
                    lr=lr,
                    grad_norm=grad_norm,
                    tokens_per_sec=tps,
                )
                _logger.info(
                    "step %d/%d loss %.4f lr %.2e grad_norm %.3f %.0f tok/s",
                    step, max_steps, avg_loss, lr, grad_norm, tps,
                )

            # --- Periodic evaluation + best checkpoint. ---
            is_eval_step = (
                eval_fn is not None
                and train_cfg.eval_interval > 0
                and (step + 1) % train_cfg.eval_interval == 0
            )
            if is_eval_step:
                val_metrics = eval_fn()
                if val_metrics and "val_loss" in val_metrics:
                    val_loss = float(val_metrics["val_loss"])
                    # All ranks must agree on the best-checkpoint decision, so the
                    # val loss is reduced to the global mean on every rank.
                    if distributed:
                        val_loss = all_reduce_mean(val_loss, device)
                        val_metrics = {**val_metrics, "val_loss": val_loss}
                    if main:
                        metrics.log(step, **val_metrics)
                        _logger.info("step %d val_loss %.4f", step, val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if main:
                            _save(
                                out_dir, model, optimizer, step + 1, model_cfg,
                                train_cfg, tag="best",
                            )

            # --- Periodic last checkpoint (main process only). ---
            if (
                main
                and train_cfg.ckpt_interval > 0
                and (step + 1) % train_cfg.ckpt_interval == 0
            ):
                _save(
                    out_dir, model, optimizer, step + 1, model_cfg,
                    train_cfg, tag="last",
                )

        # Always write a final 'last' checkpoint at the end of the run (main only).
        if main:
            _save(
                out_dir, model, optimizer, max_steps, model_cfg, train_cfg, tag="last"
            )
    finally:
        if metrics is not None:
            metrics.close()
        # Drain pending collectives and tear down the group before returning.
        barrier()
        cleanup_distributed()

    return TrainResult(
        final_train_loss=last_train_loss,
        best_val_loss=best_val_loss,
        steps=max_steps,
        n_skipped=n_skipped,
    )


def _save(out_dir, model, optimizer, step, model_cfg, train_cfg, tag):
    """Persist a checkpoint, capturing the current RNG state for exact resume.

    Always saves the *unwrapped* module so checkpoints load in single-process
    inference/export paths regardless of the training parallelism strategy.
    """
    save_checkpoint(
        out_dir=out_dir,
        model=unwrap_model(model),
        optimizer=optimizer,
        scheduler_state=None,
        step=step,
        rng_state=get_rng_state(),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        tag=tag,
    )


def prepare_run(train_cfg: TrainConfig) -> tuple[torch.device, torch.dtype]:
    """Seed, then resolve the compute device and autocast dtype for a stage."""
    set_seed(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    dtype = resolve_dtype(train_cfg.dtype, device)
    return device, dtype
