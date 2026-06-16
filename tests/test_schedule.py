"""Tests for the pure-function LR schedule (warmup + cosine/linear/constant)."""

from __future__ import annotations

import math

import pytest

from orkhon.config.schema import OptimConfig
from orkhon.train.schedule import lr_at


def _cfg(**kw) -> OptimConfig:
    base = dict(lr=1.0, min_lr_ratio=0.1, warmup_steps=10, schedule="cosine")
    base.update(kw)
    return OptimConfig(**base)


def test_lr_near_zero_at_step_zero():
    cfg = _cfg()
    # Step 0 is the first warmup step: lr = lr * 1/warmup (small, near 0).
    lr0 = lr_at(0, cfg, max_steps=100)
    assert lr0 == pytest.approx(cfg.lr / cfg.warmup_steps)
    assert lr0 < cfg.lr * 0.2


def test_lr_peaks_at_end_of_warmup():
    cfg = _cfg(warmup_steps=10)
    peak = lr_at(cfg.warmup_steps - 1, cfg, max_steps=100)
    assert peak == pytest.approx(cfg.lr)
    # Just before the peak, LR is strictly lower.
    assert lr_at(cfg.warmup_steps - 2, cfg, max_steps=100) < peak


def test_cosine_decays_to_min_lr_at_max_steps():
    cfg = _cfg(schedule="cosine", min_lr_ratio=0.1, warmup_steps=10)
    max_steps = 100
    final = lr_at(max_steps, cfg, max_steps=max_steps)
    assert final == pytest.approx(cfg.lr * cfg.min_lr_ratio, abs=1e-9)

    # Halfway through decay, cosine sits between min and peak.
    mid = lr_at(55, cfg, max_steps=max_steps)
    assert cfg.lr * cfg.min_lr_ratio < mid < cfg.lr


def test_cosine_midpoint_value():
    """At the decay midpoint, cosine coefficient is 0.5."""
    cfg = _cfg(schedule="cosine", min_lr_ratio=0.0, warmup_steps=0)
    max_steps = 100
    mid = lr_at(50, cfg, max_steps=max_steps)
    expected = 0.5 * (1 + math.cos(math.pi * 0.5)) * cfg.lr  # = 0.5 * lr
    assert mid == pytest.approx(expected, abs=1e-9)


def test_linear_schedule_reaches_min_lr():
    cfg = _cfg(schedule="linear", min_lr_ratio=0.2, warmup_steps=5)
    max_steps = 50
    final = lr_at(max_steps, cfg, max_steps=max_steps)
    assert final == pytest.approx(cfg.lr * cfg.min_lr_ratio, abs=1e-9)


def test_constant_schedule_holds_peak_after_warmup():
    cfg = _cfg(schedule="constant", warmup_steps=5)
    for step in (5, 20, 99):
        assert lr_at(step, cfg, max_steps=100) == pytest.approx(cfg.lr)


def test_schedule_is_pure_function_of_step():
    """Same step -> same LR regardless of call order (exact resume)."""
    cfg = _cfg()
    a = [lr_at(s, cfg, 100) for s in range(0, 100, 7)]
    b = [lr_at(s, cfg, 100) for s in reversed(range(0, 100, 7))][::-1]
    assert a == b
