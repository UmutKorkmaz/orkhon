"""YAML loading, dotlist CLI overrides, and typed config construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Type, TypeVar

import yaml
from pydantic import BaseModel

from orkhon.model.config import ModelConfig

T = TypeVar("T", bound=BaseModel)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(data).__name__}")
    return data


def _coerce_scalar(value: str) -> Any:
    """Parse an override value with YAML rules (so 3e-4, true, [1,2] work)."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(cfg: dict[str, Any], overrides: Sequence[str] | None) -> dict[str, Any]:
    """Apply ``a.b.c=value`` dotlist overrides onto a nested dict (in place copy)."""
    if not overrides:
        return cfg
    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        node = out
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = _coerce_scalar(raw)
    return out


def load_stage_config(
    cls: Type[T], path: str | Path, overrides: Sequence[str] | None = None
) -> T:
    """Load YAML, apply overrides, and validate into a pydantic stage config."""
    data = apply_overrides(load_yaml(path), overrides)
    return cls.model_validate(data)


def load_model_config(path: str | Path, overrides: Sequence[str] | None = None) -> ModelConfig:
    """Load a model/arch YAML into a :class:`ModelConfig`."""
    data = apply_overrides(load_yaml(path), overrides)
    return ModelConfig.from_dict(data)
