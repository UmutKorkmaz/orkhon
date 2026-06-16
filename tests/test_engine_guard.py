"""The training engine's NaN/loss-spike guard skips bad steps, not the run."""

from __future__ import annotations

import torch

from orkhon.config.schema import OptimConfig, TrainConfig
from orkhon.model import Transformer
from orkhon.model.config import ModelConfig
from orkhon.train.engine import run_training_loop
from orkhon.train.optim import build_optimizer


def _tiny():
    cfg = ModelConfig(vocab_size=48, block_size=16, n_layers=2, d_model=32,
                      n_heads=4, n_kv_heads=2, intermediate_size=64)
    return Transformer(cfg), cfg


def test_nan_step_is_skipped_weights_stay_finite(tmp_path):
    model, cfg = _tiny()
    opt = build_optimizer(model, OptimConfig(lr=1e-3))
    train_cfg = TrainConfig(max_steps=6, batch_size=2, seq_len=8, log_interval=1,
                            eval_interval=100, ckpt_interval=100, out_dir=str(tmp_path))

    def micro_batch_loss(step):
        x = torch.randint(0, 48, (2, 8))
        logits, _ = model(x)
        loss = logits.float().mean()
        if step == 2:
            loss = loss * float("nan")  # poison step 2
        return loss, 16

    run_training_loop(
        model=model, optimizer=opt, micro_batch_loss=micro_batch_loss,
        optim_cfg=OptimConfig(lr=1e-3, grad_clip=1.0), train_cfg=train_cfg,
        model_cfg=cfg, device=torch.device("cpu"), dtype=torch.float32, resume=False,
    )

    # The NaN step must have been skipped — no parameter became NaN/Inf.
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_skip_then_checkpoint_then_resume_no_corruption(tmp_path):
    """A NaN skip + checkpoint + resume must not crash or corrupt weights.

    The design: the saved step is the LR-schedule position (not the update count).
    A NaN skip loses one update at that position — the schedule continues correctly
    on resume. This test proves the path doesn't crash and weights stay finite.
    """
    model, cfg = _tiny()
    opt = build_optimizer(model, OptimConfig(lr=1e-3))
    train_cfg = TrainConfig(max_steps=6, batch_size=2, seq_len=8, log_interval=1,
                            eval_interval=100, ckpt_interval=3, out_dir=str(tmp_path))

    call_count = [0]

    def micro_batch_loss(step):
        x = torch.randint(0, 48, (2, 8))
        logits, _ = model(x)
        loss = logits.float().mean()
        call_count[0] += 1
        if step == 1:
            loss = loss * float("nan")  # poison step 1
        return loss, 16

    res = run_training_loop(
        model=model, optimizer=opt, micro_batch_loss=micro_batch_loss,
        optim_cfg=OptimConfig(lr=1e-3, grad_clip=1.0), train_cfg=train_cfg,
        model_cfg=cfg, device=torch.device("cpu"), dtype=torch.float32, resume=False,
    )
    assert res.n_skipped >= 1  # at least the NaN step was skipped
    assert all(torch.isfinite(p).all() for p in model.parameters())

    # Resume from the saved checkpoint — must not crash.
    model2, cfg2 = _tiny()
    opt2 = build_optimizer(model2, OptimConfig(lr=1e-3))
    res2 = run_training_loop(
        model=model2, optimizer=opt2, micro_batch_loss=micro_batch_loss,
        optim_cfg=OptimConfig(lr=1e-3, grad_clip=1.0),
        train_cfg=TrainConfig(max_steps=6, batch_size=2, seq_len=8, log_interval=1,
                              eval_interval=100, ckpt_interval=100, out_dir=str(tmp_path)),
        model_cfg=cfg, device=torch.device("cpu"), dtype=torch.float32, resume=True,
    )
    assert all(torch.isfinite(p).all() for p in model2.parameters())
