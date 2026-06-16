"""Decoder-only Transformer implemented by hand: RoPE, GQA, RMSNorm, SwiGLU, KV-cache."""

from orkhon.model.config import ModelConfig
from orkhon.model.hf import (
    from_pretrained,
    hf_config_to_model_config,
    save_as_orkhon_checkpoint,
)
from orkhon.model.generation import generate, generate_batch
from orkhon.model.kv_cache import KVCache
from orkhon.model.transformer import Transformer, build_model

__all__ = [
    "ModelConfig",
    "Transformer",
    "build_model",
    "KVCache",
    "generate",
    "generate_batch",
    "from_pretrained",
    "save_as_orkhon_checkpoint",
    "hf_config_to_model_config",
]
