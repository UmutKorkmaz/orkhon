"""Load a HuggingFace Llama-architecture checkpoint into Orkhon's Transformer.

This bridges the ecosystem: a clean Llama-style model on the Hub (Llama 2/3,
TinyLlama, many SmolLM/OpenELM-free derivatives, etc.) can be pulled down,
parsed into an :class:`~orkhon.model.config.ModelConfig`, and have its weights
mapped onto Orkhon's hand-written modules. The result can then be SFT'd, DPO'd,
or served through the rest of the stack.

Public surface:

* :func:`from_pretrained` — download + build + load weights, returns
  ``(Transformer, ModelConfig)``.
* :func:`save_as_orkhon_checkpoint` — persist a loaded model as an Orkhon
  checkpoint (reusing :func:`orkhon.train.checkpoint.save_checkpoint`) so it
  drops straight into the SFT / serving paths.

Scope is deliberately narrow: only architectures whose math matches Orkhon's
(RoPE + GQA + RMSNorm + SwiGLU, no projection bias, no QK-norm, dense FFN). Any
HF config field that signals a feature this Transformer cannot represent raises
:class:`NotImplementedError` naming the feature — we never silently mismap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from orkhon.model.config import ModelConfig
from orkhon.model.transformer import Transformer

_CONFIG_JSON = "config.json"
_WEIGHTS_SINGLE = "model.safetensors"
_WEIGHTS_INDEX = "model.safetensors.index.json"

# Architectures whose tensor layout / math equals Orkhon's. Other ``architectures``
# entries (e.g. Qwen2/Qwen3 with QK-norm, MoE variants) are rejected up front.
_SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "LlamaModel",
}


# --------------------------------------------------------------------------- #
# Config parsing                                                              #
# --------------------------------------------------------------------------- #
def _require_clean_llama(hf: dict[str, Any]) -> None:
    """Reject HF configs that use features Orkhon's architecture lacks.

    Raises a clear :class:`NotImplementedError` naming the offending feature so
    callers never get a silently wrong weight mapping.
    """
    # Projection bias: our Linear layers are bias-free unless use_bias, and the
    # HF weights would carry *_proj.bias tensors we have nowhere to put.
    if hf.get("attention_bias", False):
        raise NotImplementedError(
            "unsupported feature: attention_bias=True (Orkhon attention "
            "projections are bias-free)"
        )
    if hf.get("mlp_bias", False):
        raise NotImplementedError(
            "unsupported feature: mlp_bias=True (Orkhon SwiGLU is bias-free)"
        )

    # Qwen3-style QK normalization adds q_norm/k_norm weights with no Orkhon home.
    if hf.get("use_qk_norm", False) or hf.get("qk_norm", False):
        raise NotImplementedError(
            "unsupported feature: QK-norm (q_norm/k_norm) — not implemented in "
            "Orkhon attention"
        )

    # Sliding-window / local attention changes the mask Orkhon builds.
    sliding = hf.get("sliding_window")
    if sliding is not None and sliding is not False:
        # Some configs ship sliding_window with use_sliding_window=False; honor
        # the explicit disable flag when present.
        if hf.get("use_sliding_window", True):
            raise NotImplementedError(
                f"unsupported feature: sliding_window={sliding} — Orkhon uses "
                "full causal attention"
            )

    # Mixture-of-Experts: dense SwiGLU only.
    if hf.get("num_experts") or hf.get("num_local_experts"):
        raise NotImplementedError(
            "unsupported feature: MoE (num_experts) — Orkhon has a dense FFN"
        )

    # head_dim that does not equal hidden_size / num_attention_heads. Our
    # ModelConfig supports an explicit head_dim, but the q/o projection shapes in
    # Orkhon assume q_out == n_heads * head_dim AND the residual stream is
    # d_model; HF decoupled-head_dim models keep o_proj: (n_heads*head_dim)->
    # hidden which we *do* support, but only when head_dim divides cleanly. Guard
    # the genuinely incompatible case where head_dim is given yet inconsistent.
    hidden = hf["hidden_size"]
    n_heads = hf["num_attention_heads"]
    head_dim = hf.get("head_dim")
    if head_dim is not None and head_dim != hidden // n_heads:
        raise NotImplementedError(
            f"unsupported feature: head_dim={head_dim} != hidden_size//"
            f"num_attention_heads ({hidden}//{n_heads}={hidden // n_heads}); "
            "decoupled head_dim is not supported"
        )

    # Non-RoPE position handling (e.g. learned/alibi) would need other weights.
    pos_emb = hf.get("position_embedding_type")
    if pos_emb is not None and pos_emb not in ("rope", "rotary"):
        raise NotImplementedError(
            f"unsupported feature: position_embedding_type={pos_emb!r} "
            "(Orkhon is RoPE-only)"
        )


def hf_config_to_model_config(hf: dict[str, Any]) -> ModelConfig:
    """Translate a parsed HF Llama ``config.json`` dict into a :class:`ModelConfig`.

    Validates that the architecture is a clean Llama variant first (see
    :func:`_require_clean_llama`).
    """
    archs = hf.get("architectures")
    if archs:
        if not any(a in _SUPPORTED_ARCHITECTURES for a in archs):
            raise NotImplementedError(
                f"unsupported architectures={archs}; only Llama-style "
                f"{sorted(_SUPPORTED_ARCHITECTURES)} are supported"
            )

    _require_clean_llama(hf)

    n_heads = int(hf["num_attention_heads"])
    n_kv_heads = int(hf.get("num_key_value_heads", n_heads))
    hidden = int(hf["hidden_size"])

    return ModelConfig(
        vocab_size=int(hf["vocab_size"]),
        block_size=int(hf.get("max_position_embeddings", 2048)),
        n_layers=int(hf["num_hidden_layers"]),
        d_model=hidden,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=hf.get("head_dim"),  # None -> derived as hidden // n_heads
        intermediate_size=int(hf["intermediate_size"]),
        norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
        rope_theta=float(hf.get("rope_theta", 10000.0)),
        dropout=0.0,
        use_bias=False,
        tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
    )


# --------------------------------------------------------------------------- #
# Weight name mapping                                                         #
# --------------------------------------------------------------------------- #
def _hf_to_orkhon_key_map(n_layers: int) -> dict[str, str]:
    """Build the full HF-name -> Orkhon-name mapping for every target param.

    ``lm_head.weight`` is included; callers handle the tied-embedding fallback
    when the HF state dict omits it.
    """
    mapping: dict[str, str] = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "final_norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for i in range(n_layers):
        h = f"model.layers.{i}"
        o = f"layers.{i}"
        mapping[f"{h}.self_attn.q_proj.weight"] = f"{o}.attn.q_proj.weight"
        mapping[f"{h}.self_attn.k_proj.weight"] = f"{o}.attn.k_proj.weight"
        mapping[f"{h}.self_attn.v_proj.weight"] = f"{o}.attn.v_proj.weight"
        mapping[f"{h}.self_attn.o_proj.weight"] = f"{o}.attn.o_proj.weight"
        mapping[f"{h}.mlp.gate_proj.weight"] = f"{o}.mlp.gate_proj.weight"
        mapping[f"{h}.mlp.up_proj.weight"] = f"{o}.mlp.up_proj.weight"
        mapping[f"{h}.mlp.down_proj.weight"] = f"{o}.mlp.down_proj.weight"
        mapping[f"{h}.input_layernorm.weight"] = f"{o}.attn_norm.weight"
        mapping[f"{h}.post_attention_layernorm.weight"] = f"{o}.mlp_norm.weight"
    return mapping


def _build_orkhon_state_dict(
    hf_state: dict[str, torch.Tensor], cfg: ModelConfig
) -> dict[str, torch.Tensor]:
    """Remap an HF Llama state dict onto Orkhon parameter names.

    Handles tied embeddings (a missing ``lm_head.weight`` reuses
    ``embed_tokens.weight``). Raises :class:`KeyError` if any required non-tied
    target weight is absent, and :class:`NotImplementedError` if HF ships bias
    tensors we have no slot for.
    """
    key_map = _hf_to_orkhon_key_map(cfg.n_layers)

    # Defensive: reject stray bias tensors that slipped past the config guard.
    for name in hf_state:
        if name.endswith(".bias") and (".self_attn." in name or ".mlp." in name):
            raise NotImplementedError(
                f"unsupported feature: projection bias tensor {name!r} present "
                "in weights"
            )

    out: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for hf_name, ork_name in key_map.items():
        if hf_name in hf_state:
            out[ork_name] = hf_state[hf_name]
        elif hf_name == "lm_head.weight":
            # Tied embeddings: lm_head shares the token-embedding weight.
            embed = hf_state.get("model.embed_tokens.weight")
            if embed is None:
                missing.append(hf_name)
            else:
                out["lm_head.weight"] = embed
        else:
            missing.append(hf_name)

    if missing:
        raise KeyError(
            "HF checkpoint is missing required weights: " + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    return out


# --------------------------------------------------------------------------- #
# Download helpers                                                            #
# --------------------------------------------------------------------------- #
def _resolve_file(
    repo_or_path: str, filename: str, revision: str | None, required: bool
) -> Path | None:
    """Return a local path for ``filename`` from a local dir or the Hub.

    Local directories short-circuit the network. For a Hub repo id, downloads
    via ``hf_hub_download``. Returns ``None`` when ``required`` is False and the
    entry is absent.
    """
    local = Path(repo_or_path)
    if local.is_dir():
        candidate = local / filename
        if candidate.exists():
            return candidate
        if required:
            raise FileNotFoundError(f"{filename} not found in {repo_or_path}")
        return None

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_or_path, filename=filename, revision=revision
        )
        return Path(path)
    except EntryNotFoundError:
        if required:
            raise
        return None


def _load_weight_files(
    repo_or_path: str, revision: str | None
) -> dict[str, torch.Tensor]:
    """Download (or read locally) the safetensors shards and merge them.

    Supports both single-file ``model.safetensors`` and sharded layouts driven
    by ``model.safetensors.index.json``.
    """
    from safetensors.torch import load_file

    index_path = _resolve_file(
        repo_or_path, _WEIGHTS_INDEX, revision, required=False
    )
    state: dict[str, torch.Tensor] = {}

    if index_path is not None:
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
        for shard in shards:
            shard_path = _resolve_file(
                repo_or_path, shard, revision, required=True
            )
            assert shard_path is not None
            state.update(load_file(str(shard_path)))
        return state

    single = _resolve_file(
        repo_or_path, _WEIGHTS_SINGLE, revision, required=True
    )
    assert single is not None
    state.update(load_file(str(single)))
    return state


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def from_pretrained(
    repo_or_path: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    revision: str | None = None,
) -> tuple[Transformer, ModelConfig]:
    """Load an HF Llama-architecture checkpoint into an Orkhon ``Transformer``.

    Args:
        repo_or_path: a Hugging Face repo id (``"org/model"``) or a local
            directory containing ``config.json`` + safetensors weights.
        device: device to place the model on.
        dtype: optional dtype to cast the model to after loading. If ``None``,
            keeps the checkpoint's native dtype.
        revision: optional git revision (branch / tag / commit) for Hub repos.

    Returns:
        ``(model, cfg)`` — a built and weight-loaded :class:`Transformer` and the
        derived :class:`ModelConfig`.

    Raises:
        NotImplementedError: if the HF config uses a feature Orkhon's
            architecture cannot represent (bias, QK-norm, sliding window, MoE,
            decoupled head_dim, non-RoPE positions, non-Llama architecture).
        KeyError: if required weight tensors are missing.
    """
    config_path = _resolve_file(
        repo_or_path, _CONFIG_JSON, revision, required=True
    )
    assert config_path is not None
    hf_config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    cfg = hf_config_to_model_config(hf_config)

    hf_state = _load_weight_files(repo_or_path, revision)
    orkhon_state = _build_orkhon_state_dict(hf_state, cfg)

    model = Transformer(cfg)
    # strict=True would reject the tied lm_head (it shares embed_tokens' storage
    # and may not be a distinct key after tying). Tying is reapplied by the
    # constructor when cfg.tie_word_embeddings, so loading the explicit key is a
    # harmless no-op; we keep strict to catch genuine name/shape mismatches.
    missing, unexpected = model.load_state_dict(orkhon_state, strict=False)
    _verify_load(model, orkhon_state, missing, unexpected)

    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, cfg


def _verify_load(
    model: Transformer,
    orkhon_state: dict[str, torch.Tensor],
    missing: list[str],
    unexpected: list[str],
) -> None:
    """Assert every model parameter was sourced from the HF checkpoint.

    With tied embeddings, ``lm_head.weight`` aliases ``embed_tokens.weight``, so
    PyTorch may report it 'missing' even though it is correctly populated. We
    treat that single alias as benign and fail on anything else.
    """
    if unexpected:
        raise KeyError(f"unexpected keys when loading HF weights: {unexpected}")

    benign = set()
    if model.cfg.tie_word_embeddings:
        benign.add("lm_head.weight")
    real_missing = [m for m in missing if m not in benign]
    if real_missing:
        raise KeyError(
            f"model parameters left uninitialized after load: {real_missing}"
        )


def save_as_orkhon_checkpoint(
    model: Transformer,
    cfg: ModelConfig,
    out_dir: str | Path,
) -> Path:
    """Persist a loaded model as an Orkhon checkpoint under ``out_dir``.

    Reuses :func:`orkhon.train.checkpoint.save_checkpoint` so the result is a
    fully-formed resume target (model + optimizer + step + RNG + configs) that
    SFT / DPO / serving load through the normal path. A readable
    ``model_config.json`` is written alongside.

    Because there is no real training state for an imported model, a fresh
    optimizer over the model's parameters is created (``step=0``) and an empty
    train config is recorded.

    Args:
        model: the loaded :class:`Transformer`.
        cfg: its :class:`ModelConfig`.
        out_dir: destination directory (created if missing).

    Returns:
        The path of the written ``ckpt_last.pt``.
    """
    from orkhon.train.checkpoint import save_checkpoint
    from orkhon.utils import get_rng_state

    # A minimal optimizer satisfies save_checkpoint's contract and yields a
    # valid (if untrained) optimizer state. We avoid build_optimizer to keep this
    # free of an OptimConfig dependency.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.0
    )

    return save_checkpoint(
        out_dir=out_dir,
        model=model,
        optimizer=optimizer,
        scheduler_state=None,
        step=0,
        rng_state=get_rng_state(),
        model_cfg=cfg,
        train_cfg={"source": "from_pretrained"},
        tag="last",
    )
