"""Checkpoint save/load with exact-resume guarantees.

A checkpoint is a single ``.pt`` file holding everything needed to resume training
bit-for-bit:

    {model, optimizer, scheduler, step, rng_state, model_config(dict), train_config(dict)}

Two tags are written under ``out_dir``:

* ``ckpt_last.pt`` — the most recent state (resume target).
* ``ckpt_best.pt`` — the lowest-validation-loss state seen so far.

A human-readable ``model_config.json`` is also written so the exporter / loaders
can reconstruct the architecture without unpickling a checkpoint.

``load_model_from_checkpoint`` rebuilds a :class:`~orkhon.model.Transformer` from the
stored ``model_config`` and loads its weights — the path used by SFT (to init from
a pretrained model) and DPO (to build the frozen reference).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from orkhon.model.config import ModelConfig
from orkhon.model.transformer import Transformer

_TAG_FILES = {"last": "ckpt_last.pt", "best": "ckpt_best.pt"}
_MODEL_CONFIG_JSON = "model_config.json"


def _ckpt_path(out_dir: str | Path, tag: str) -> Path:
    if tag not in _TAG_FILES:
        raise ValueError(f"unknown checkpoint tag {tag!r}; use 'last' or 'best'")
    return Path(out_dir) / _TAG_FILES[tag]


def _as_config_dict(cfg: Any) -> dict:
    """Coerce a model/train config (dataclass or pydantic) to a plain dict."""
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()  # ModelConfig
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()  # pydantic v2
    raise TypeError(f"cannot serialize config of type {type(cfg).__name__}")


def save_checkpoint(
    out_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict | None,
    step: int,
    rng_state: dict,
    model_cfg: ModelConfig | dict,
    train_cfg: Any,
    tag: str = "last",
) -> Path:
    """Write a checkpoint ``.pt`` (and ``model_config.json``) under ``out_dir``.

    Args:
        out_dir: destination directory (created if missing).
        model: model whose ``state_dict`` is saved.
        optimizer: optimizer whose ``state_dict`` is saved (for exact resume).
        scheduler_state: optional scheduler ``state_dict`` (LR is also a pure
            function of step, so this is informational/optional).
        step: the optimizer step count completed so far.
        rng_state: snapshot from :func:`orkhon.utils.get_rng_state`.
        model_cfg: the architecture config (stored as a dict).
        train_cfg: the stage config (stored as a dict).
        tag: ``'last'`` or ``'best'``.

    Returns:
        The path of the written checkpoint file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg_dict = _as_config_dict(model_cfg)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "step": step,
        "rng_state": rng_state,
        "model_config": model_cfg_dict,
        "train_config": _as_config_dict(train_cfg),
    }

    path = _ckpt_path(out_dir, tag)
    torch.save(payload, path)

    # Always (re)write the readable model config alongside the checkpoint.
    (out_dir / _MODEL_CONFIG_JSON).write_text(
        json.dumps(model_cfg_dict, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_checkpoint(
    dir: str | Path,
    tag: str = "last",
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint dict written by :func:`save_checkpoint`.

    Args:
        dir: directory containing ``ckpt_<tag>.pt``.
        tag: ``'last'`` or ``'best'``.
        map_location: device to map tensors onto.

    Returns:
        The full checkpoint dict (model/optimizer/scheduler/step/rng/configs).
    """
    path = _ckpt_path(dir, tag)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    # weights_only=False: the payload holds RNG/config objects, not just tensors.
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(
    dir: str | Path,
    device: str | torch.device = "cpu",
    tag: str = "last",
) -> tuple[Transformer, ModelConfig]:
    """Rebuild a model from a checkpoint and load its weights.

    Reads ``model_config`` from the checkpoint, constructs a
    :class:`~orkhon.model.Transformer`, loads the saved weights, and moves it to
    ``device``. Returns ``(model, model_config)``.
    """
    ckpt = load_checkpoint(dir, tag=tag, map_location=device)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, cfg
