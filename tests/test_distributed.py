"""Unit tests for the multi-GPU helpers in :mod:`orkhon.train.distributed`.

Two layers:

* Fast, offline, CPU-only tests of the single-process degradation: every helper
  must behave as a no-op / identity when ``WORLD_SIZE <= 1`` so the non-distributed
  training path is byte-for-byte unchanged.
* A ``@pytest.mark.slow`` gloo smoke that spawns two processes, inits a real
  process group, and all-reduces a tensor — proving the group plumbing works
  without needing CUDA or a network.

No network access is required; gloo runs over localhost.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from orkhon.train.distributed import (
    DistInfo,
    WRAP_ENV,
    all_reduce_mean,
    barrier,
    cleanup_distributed,
    dist_info,
    init_distributed,
    is_distributed,
    is_main_process,
    local_rank,
    rank,
    resolve_wrap_mode,
    world_size,
    wrap_model,
)


# --------------------------------------------------------------------------- #
# Single-process degradation (fast, offline, CPU).                            #
# --------------------------------------------------------------------------- #


def test_world_size_is_one_without_torchrun():
    assert world_size() == 1


def test_rank_and_local_rank_default_to_zero():
    assert rank() == 0
    assert local_rank() == 0


def test_is_main_process_true_in_single_process():
    assert is_main_process() is True


def test_is_distributed_false_in_single_process():
    assert is_distributed() is False


def test_dist_info_snapshot_describes_single_process():
    info = dist_info()
    assert isinstance(info, DistInfo)
    assert info.world_size == 1
    assert info.rank == 0
    assert info.is_main is True
    assert info.is_distributed is False


def test_init_distributed_is_noop_without_world_size():
    # Returns the single-process info and never touches the process group.
    info = init_distributed(torch.device("cpu"))
    assert info.world_size == 1
    assert not (torch.distributed.is_available() and torch.distributed.is_initialized())


def test_barrier_and_cleanup_are_safe_noops():
    # Neither should raise when no group is initialized.
    barrier()
    cleanup_distributed()


def test_all_reduce_mean_is_identity_without_group():
    assert all_reduce_mean(3.5, torch.device("cpu")) == 3.5


def test_wrap_model_none_returns_same_object():
    model = nn.Linear(4, 4)
    wrapped = wrap_model(model, "none")
    assert wrapped is model  # identity, not just equal


def test_wrap_model_rejects_unknown_mode():
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        wrap_model(model, "tensorparallel")


def test_wrap_model_ddp_without_group_returns_model_unchanged():
    # No process group -> wrapping is meaningless; helper degrades to identity
    # instead of raising, so a misconfigured single-process run still works.
    model = nn.Linear(4, 4)
    assert wrap_model(model, "ddp") is model
    assert wrap_model(model, "fsdp") is model


# --------------------------------------------------------------------------- #
# resolve_wrap_mode.                                                          #
# --------------------------------------------------------------------------- #


def test_resolve_wrap_mode_defaults_to_none_in_single_process():
    # No env var, not distributed -> "none".
    assert resolve_wrap_mode() == "none"


def test_resolve_wrap_mode_honors_explicit_argument():
    assert resolve_wrap_mode("fsdp") == "fsdp"
    assert resolve_wrap_mode("DDP") == "ddp"  # case-insensitive


def test_resolve_wrap_mode_reads_env_var(monkeypatch):
    monkeypatch.setenv(WRAP_ENV, "fsdp")
    assert resolve_wrap_mode() == "fsdp"


def test_resolve_wrap_mode_rejects_garbage(monkeypatch):
    monkeypatch.setenv(WRAP_ENV, "nonsense")
    with pytest.raises(ValueError):
        resolve_wrap_mode()


def test_resolve_wrap_mode_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv(WRAP_ENV, "fsdp")
    assert resolve_wrap_mode("ddp") == "ddp"


# --------------------------------------------------------------------------- #
# Env-var parsing of dist_info (simulate torchrun without launching it).      #
# --------------------------------------------------------------------------- #


def test_dist_info_reads_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    info = dist_info()
    assert (info.world_size, info.rank, info.local_rank) == (4, 2, 1)
    assert info.is_distributed is True
    assert info.is_main is False
    # is_distributed() reflects the env too (no group initialized).
    assert is_distributed() is True


def test_dist_info_ignores_garbage_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "not-a-number")
    # Falls back to the single-process default rather than crashing.
    assert dist_info().world_size == 1


# --------------------------------------------------------------------------- #
# gloo 2-process smoke (slow; spawns real processes, no CUDA / no network).   #
# --------------------------------------------------------------------------- #


def _all_reduce_worker(rank_: int, world: int, port: int, out_path: str) -> None:
    """Child process: init a gloo group, all-reduce a per-rank tensor, write result.

    Each rank contributes ``rank + 1``; the SUM all-reduce must yield
    ``1 + 2 + ... + world`` on every rank. We also exercise the package helpers
    (``world_size``/``rank``/``is_main_process``/``all_reduce_mean``) under a live
    group to prove they read the group, not just the env.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank_)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = "0"

    torch.distributed.init_process_group(backend="gloo", init_method="env://")
    try:
        assert world_size() == world
        assert rank() == rank_
        assert is_main_process() == (rank_ == 0)

        t = torch.tensor([float(rank_ + 1)])
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

        # all_reduce_mean must average the per-rank values: mean(1..world).
        mean_val = all_reduce_mean(float(rank_ + 1), torch.device("cpu"))

        if rank_ == 0:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(f"{t.item()},{mean_val}")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.slow
def test_gloo_two_process_all_reduce(tmp_path):
    """Spawn two gloo processes; verify SUM and mean all-reduce across ranks."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")

    world = 2
    port = 29555
    out_path = str(tmp_path / "reduced.txt")

    mp = torch.multiprocessing.get_context("spawn")
    procs = [
        mp.Process(target=_all_reduce_worker, args=(r, world, port, out_path))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    for p in procs:
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    summed, mean_val = open(out_path, encoding="utf-8").read().split(",")
    # SUM of (rank+1) over ranks 0,1 = 1 + 2 = 3.
    assert float(summed) == pytest.approx(3.0)
    # mean of (1, 2) = 1.5.
    assert float(mean_val) == pytest.approx(1.5)
