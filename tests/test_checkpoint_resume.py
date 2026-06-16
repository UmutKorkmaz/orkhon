"""Exact-resume test: save mid-training, reload, and continue identically.

Resume is exact when, after restoring model + optimizer + step (LR is a pure
function of step), the next optimizer step produces the SAME loss as an
uninterrupted run. We drive the shared engine with a deterministic data source
(fixed batches indexed by step) so any divergence is attributable to resume logic.
"""

from __future__ import annotations

import torch

from orkhon.config.schema import OptimConfig, TrainConfig
from orkhon.model.config import ModelConfig
from orkhon.model.transformer import Transformer
from orkhon.train.checkpoint import load_checkpoint
from orkhon.train.engine import run_training_loop
from orkhon.train.losses import lm_cross_entropy
from orkhon.train.optim import build_optimizer
from orkhon.utils.seed import set_seed


def _cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=64, block_size=16, n_layers=2, d_model=32,
        n_heads=4, n_kv_heads=2, intermediate_size=64,
        dropout=0.0, attn_impl="manual",
    )


def _fixed_batches(n_steps: int, b: int = 4, t: int = 8, v: int = 64):
    """Deterministic per-step (x, labels) so two runs see identical data."""
    g = torch.Generator().manual_seed(123)
    batches = []
    for _ in range(n_steps + 2):
        x = torch.randint(0, v, (b, t), generator=g)
        batches.append((x, x.clone()))
    return batches


def _train_cfg(out_dir, max_steps) -> TrainConfig:
    return TrainConfig(
        device="cpu", dtype="float32", seed=7, batch_size=4,
        grad_accum_steps=1, max_steps=max_steps, seq_len=8,
        log_interval=100, eval_interval=0, eval_iters=0,
        ckpt_interval=100, out_dir=str(out_dir),
    )


def _build(out_dir, max_steps):
    set_seed(7)
    torch.manual_seed(7)
    model = Transformer(_cfg())
    model.train()
    opt = build_optimizer(model, OptimConfig(lr=1e-2, warmup_steps=2, weight_decay=0.0))
    return model, opt


def _run(model, opt, batches, out_dir, max_steps, resume):
    def micro_batch_loss(step: int):
        x, labels = batches[step]
        logits, _ = model(x)
        return lm_cross_entropy(logits, labels), x.numel()

    return run_training_loop(
        model=model, optimizer=opt, micro_batch_loss=micro_batch_loss,
        optim_cfg=OptimConfig(lr=1e-2, warmup_steps=2, weight_decay=0.0),
        train_cfg=_train_cfg(out_dir, max_steps), model_cfg=_cfg(),
        device=torch.device("cpu"), dtype=torch.float32,
        eval_fn=None, resume=resume,
    )


def test_reload_restores_identical_params(tmp_path):
    out_dir = tmp_path / "run"
    batches = _fixed_batches(5)
    model, opt = _build(out_dir, 5)
    _run(model, opt, batches, out_dir, max_steps=5, resume=False)

    ckpt = load_checkpoint(out_dir, tag="last")
    assert ckpt["step"] == 5

    fresh = Transformer(_cfg())
    fresh.load_state_dict(ckpt["model"])
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), fresh.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2), f"param {n1} differs after reload"


def test_resume_continues_with_same_next_loss(tmp_path):
    """Train 3 then resume to 5 == train 5 uninterrupted, step by step."""
    batches = _fixed_batches(5)

    # Reference: uninterrupted 5-step run; capture loss of step index 3 and 4.
    ref_dir = tmp_path / "ref"
    ref_losses: list[float] = []

    model_ref, opt_ref = _build(ref_dir, 5)

    def ref_loss(step: int):
        x, labels = batches[step]
        logits, _ = model_ref(x)
        loss = lm_cross_entropy(logits, labels)
        ref_losses.append(float(loss.detach().item()))
        return loss, x.numel()

    run_training_loop(
        model=model_ref, optimizer=opt_ref, micro_batch_loss=ref_loss,
        optim_cfg=OptimConfig(lr=1e-2, warmup_steps=2, weight_decay=0.0),
        train_cfg=_train_cfg(ref_dir, 5), model_cfg=_cfg(),
        device=torch.device("cpu"), dtype=torch.float32, eval_fn=None, resume=False,
    )

    # Interrupted: run 3 steps, save 'last', then a FRESH model+opt resumes to 5.
    int_dir = tmp_path / "int"
    model_a, opt_a = _build(int_dir, 3)
    _run(model_a, opt_a, batches, int_dir, max_steps=3, resume=False)

    # Fresh objects, resume from int_dir/ckpt_last.pt (step 3) and continue to 5.
    model_b, opt_b = _build(int_dir, 5)
    resumed_losses: list[float] = []

    def resumed_loss(step: int):
        x, labels = batches[step]
        logits, _ = model_b(x)
        loss = lm_cross_entropy(logits, labels)
        resumed_losses.append((step, float(loss.detach().item())))
        return loss, x.numel()

    run_training_loop(
        model=model_b, optimizer=opt_b, micro_batch_loss=resumed_loss,
        optim_cfg=OptimConfig(lr=1e-2, warmup_steps=2, weight_decay=0.0),
        train_cfg=_train_cfg(int_dir, 5), model_cfg=_cfg(),
        device=torch.device("cpu"), dtype=torch.float32, eval_fn=None, resume=True,
    )

    # The resumed run should execute steps 3 and 4 with the SAME losses as the
    # uninterrupted reference at those steps.
    by_step = dict(resumed_losses)
    assert 3 in by_step and 4 in by_step
    assert by_step[3] == ref_losses[3], (by_step[3], ref_losses[3])
    assert by_step[4] == ref_losses[4], (by_step[4], ref_losses[4])
