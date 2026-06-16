"""Multi-GPU helpers for data-parallel and sharded training.

This module turns the single-process training loop into an (optionally)
distributed one with the smallest possible surface area. Everything is driven by
the environment variables ``torchrun`` sets — ``RANK``, ``WORLD_SIZE``,
``LOCAL_RANK`` — so launching ``torchrun --nproc_per_node=N -m orkhon ...`` is the
only thing that changes between single- and multi-GPU runs.

Design rules (so single-process behavior stays byte-for-byte identical):

* :func:`is_distributed` is ``False`` whenever ``WORLD_SIZE`` is unset or ``1``.
  Every other helper degrades to a trivial answer in that case
  (``world_size() == 1``, ``rank() == 0``, ``is_main_process() is True``).
* :func:`init_distributed` is a no-op unless ``WORLD_SIZE > 1``; it never touches
  the process group otherwise, so importing this module has no side effects.
* :func:`wrap_model` with ``mode="none"`` returns the model object unchanged
  (identity), so the non-distributed code path never sees a wrapper.

Backends: NCCL on CUDA (the only sane choice for GPU all-reduce), GLOO elsewhere
(CPU smoke tests, MPS boxes that cannot use NCCL).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn

from orkhon.utils.logging import get_logger

_logger = get_logger("orkhon.train.distributed")

# Supported model-wrapping strategies.
WRAP_MODES = ("none", "ddp", "fsdp")

# Environment variable that selects the wrap strategy when running under torchrun.
# Read here (not in the config schema) so the YAML contract is untouched.
WRAP_ENV = "ORKHON_DISTRIBUTED"


@dataclass(frozen=True)
class DistInfo:
    """Immutable snapshot of the current process's place in the world."""

    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` on absence or garbage."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def dist_info() -> DistInfo:
    """Read RANK / WORLD_SIZE / LOCAL_RANK from the environment.

    Defaults describe a single-process run (rank 0, world size 1), so this is safe
    to call even when not launched under ``torchrun``.
    """
    world = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local = _env_int("LOCAL_RANK", 0)
    return DistInfo(rank=rank, world_size=world, local_rank=local)


def is_distributed() -> bool:
    """True only when launched as a multi-process group (``WORLD_SIZE > 1``)."""
    return dist_info().world_size > 1


def _group_ready() -> bool:
    """True when ``torch.distributed`` is available and a group is initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def world_size() -> int:
    """Number of processes in the group (``1`` when not distributed)."""
    if _group_ready():
        return torch.distributed.get_world_size()
    return dist_info().world_size


def rank() -> int:
    """This process's global rank (``0`` when not distributed)."""
    if _group_ready():
        return torch.distributed.get_rank()
    return dist_info().rank


def local_rank() -> int:
    """This process's node-local rank (selects the local CUDA device)."""
    return dist_info().local_rank


def is_main_process() -> bool:
    """True on global rank 0 (the only process that logs and checkpoints)."""
    return rank() == 0


def _backend_for(device: torch.device) -> str:
    """Pick the collective backend: NCCL on CUDA, GLOO everywhere else."""
    return "nccl" if device.type == "cuda" else "gloo"


def init_distributed(device: torch.device | None = None) -> DistInfo:
    """Initialize the process group when running under ``torchrun``.

    No-op (returns the single-process :class:`DistInfo`) when ``WORLD_SIZE <= 1``
    or a group is already initialized. On CUDA it also pins this process to
    ``cuda:LOCAL_RANK`` so every rank owns a distinct GPU.

    Args:
        device: the resolved compute device; selects the backend. When ``None``,
            CUDA is assumed if available (the realistic multi-GPU case).

    Returns:
        The :class:`DistInfo` for this process.
    """
    info = dist_info()
    if not info.is_distributed:
        return info
    if _group_ready():
        return info

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backend = _backend_for(device)
    # torchrun exports MASTER_ADDR / MASTER_PORT; env:// reads them.
    torch.distributed.init_process_group(backend=backend, init_method="env://")

    if device.type == "cuda":
        torch.cuda.set_device(info.local_rank)

    _logger.info(
        "initialized %s group: rank %d/%d (local_rank %d)",
        backend, info.rank, info.world_size, info.local_rank,
    )
    return info


def cleanup_distributed() -> None:
    """Destroy the process group if one is active (safe to always call)."""
    if _group_ready():
        torch.distributed.destroy_process_group()


def barrier() -> None:
    """Block until all ranks reach this point (no-op when not distributed)."""
    if _group_ready():
        torch.distributed.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Average a scalar across all ranks (identity when not distributed).

    Used to turn each rank's local loss into the global mean for logging, so the
    reported curve matches what a single large batch would have produced.
    """
    if not _group_ready():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return float(t.item()) / world_size()


def resolve_wrap_mode(explicit: str | None = None) -> str:
    """Resolve the wrap strategy from an explicit value or the env var.

    Order: explicit argument > ``ORKHON_DISTRIBUTED`` env var > ``"ddp"`` when
    distributed (the safe default for multi-GPU) > ``"none"``.
    """
    candidate = explicit or os.environ.get(WRAP_ENV)
    if candidate:
        candidate = candidate.strip().lower()
        if candidate not in WRAP_MODES:
            raise ValueError(
                f"distributed mode must be one of {WRAP_MODES}, got {candidate!r}"
            )
        return candidate
    return "ddp" if is_distributed() else "none"


def wrap_model(model: nn.Module, mode: str, device: torch.device | None = None) -> nn.Module:
    """Wrap ``model`` for the requested parallelism strategy.

    Args:
        model: the (already device-placed) model.
        mode: one of ``"none"`` (return unchanged), ``"ddp"`` (replicate +
            all-reduce gradients), or ``"fsdp"`` (fully sharded data parallel —
            shard params/grads/optimizer state across ranks for large models).
        device: compute device (used to pick ``device_ids`` for DDP on CUDA).

    Returns:
        The wrapped module. For ``"none"`` this is the original object (identity),
        so non-distributed runs are unaffected.
    """
    if mode not in WRAP_MODES:
        raise ValueError(f"mode must be one of {WRAP_MODES}, got {mode!r}")
    if mode == "none":
        return model
    if not _group_ready():
        # No group -> wrapping would be meaningless (and DDP/FSDP would error).
        _logger.warning("wrap_model(mode=%r) requested without a process group; "
                        "returning the model unchanged", mode)
        return model

    if device is None:
        device = next(model.parameters()).device

    if mode == "ddp":
        return _wrap_ddp(model, device)
    return _wrap_fsdp(model, device)


def _wrap_ddp(model: nn.Module, device: torch.device) -> nn.Module:
    """DistributedDataParallel: replicate the model, all-reduce grads each step."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    if device.type == "cuda":
        return DDP(model, device_ids=[local_rank()], output_device=local_rank())
    # CPU/GLOO (and the gloo smoke test) take no device_ids.
    return DDP(model)


def _wrap_fsdp(model: nn.Module, device: torch.device) -> nn.Module:
    """FullyShardedDataParallel: shard params/grads/opt-state per transformer block.

    Prefers the modern ``fully_shard`` (per-module sharding, PyTorch 2.4+); falls
    back to the classic :class:`FullyShardedDataParallel` wrapper with an
    auto-wrap policy keyed on the model's ``Block`` type. Each transformer block
    becomes its own communication/sharding unit so memory scales ~1/world_size.
    """
    block_cls = _transformer_block_cls(model)

    # Modern composable API (torch.distributed.fsdp.fully_shard): wrap each block,
    # then the root. Returns the same module object (sharding is applied in place).
    try:
        from torch.distributed.fsdp import fully_shard

        if block_cls is not None:
            for module in model.modules():
                if isinstance(module, block_cls):
                    fully_shard(module)
        fully_shard(model)
        return model
    except Exception as exc:  # pragma: no cover - exercised only on real GPUs
        _logger.warning("fully_shard unavailable (%s); using FSDP wrapper", exc)

    # Classic wrapper fallback.
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if block_cls is None:
        return FSDP(model)

    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    return FSDP(model, auto_wrap_policy=ModuleWrapPolicy({block_cls}))


def _transformer_block_cls(model: nn.Module) -> type | None:
    """Best-effort lookup of the per-layer ``Block`` class for FSDP wrapping.

    Imports lazily so this module never hard-depends on the model package layout;
    returns ``None`` if the class cannot be located (FSDP then shards the root).
    """
    try:
        from orkhon.model.block import TransformerBlock  # type: ignore

        return TransformerBlock
    except Exception:
        return None
