"""Shared utilities: seeding, device/dtype resolution, paths, hashing, logging."""

from orkhon.utils.device import resolve_device
from orkhon.utils.dtype import autocast_ctx, resolve_dtype
from orkhon.utils.seed import get_rng_state, set_rng_state, set_seed

__all__ = [
    "resolve_device",
    "resolve_dtype",
    "autocast_ctx",
    "set_seed",
    "get_rng_state",
    "set_rng_state",
]
