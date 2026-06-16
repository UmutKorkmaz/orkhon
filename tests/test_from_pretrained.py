"""Offline tests for loading HF Llama-architecture checkpoints into Orkhon.

No network: we build a synthetic tiny-Llama ``config.json`` + safetensors
state dict in a temp dir, and exercise BOTH code paths:

* the local-directory short-circuit (pass the temp dir directly), and
* the Hub path with ``huggingface_hub.hf_hub_download`` monkeypatched to return
  the temp files.

Coverage:
* every Orkhon target parameter is mapped + shape-loads, and a forward runs;
* tied embeddings (no ``lm_head.weight``) are handled;
* sharded ``model.safetensors.index.json`` layout loads;
* unsupported features (attention_bias, q_norm, sliding_window, MoE,
  decoupled head_dim, non-Llama arch) raise ``NotImplementedError``;
* ``save_as_orkhon_checkpoint`` writes a checkpoint reloadable by the trainer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import orkhon.model.hf as fp
from orkhon.model.hf import (
    from_pretrained,
    hf_config_to_model_config,
    save_as_orkhon_checkpoint,
)

# --------------------------------------------------------------------------- #
# Synthetic tiny-Llama fixtures                                               #
# --------------------------------------------------------------------------- #
N_LAYERS = 2
D_MODEL = 32
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = D_MODEL // N_HEADS  # 8
INTER = 64
VOCAB = 50


def _base_config(**overrides) -> dict:
    """A clean tiny-Llama HF config dict."""
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": D_MODEL,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV_HEADS,
        "intermediate_size": INTER,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }
    cfg.update(overrides)
    return cfg


def _hf_state_dict(tie: bool = False) -> dict[str, torch.Tensor]:
    """Build a full HF Llama state dict with correct shapes (random values)."""
    g = torch.Generator().manual_seed(0)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, dtype=torch.float32) * 0.02

    q_out = N_HEADS * HEAD_DIM
    kv_out = N_KV_HEADS * HEAD_DIM

    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": rnd(VOCAB, D_MODEL),
        "model.norm.weight": torch.ones(D_MODEL),
    }
    for i in range(N_LAYERS):
        h = f"model.layers.{i}"
        sd[f"{h}.self_attn.q_proj.weight"] = rnd(q_out, D_MODEL)
        sd[f"{h}.self_attn.k_proj.weight"] = rnd(kv_out, D_MODEL)
        sd[f"{h}.self_attn.v_proj.weight"] = rnd(kv_out, D_MODEL)
        sd[f"{h}.self_attn.o_proj.weight"] = rnd(D_MODEL, q_out)
        sd[f"{h}.mlp.gate_proj.weight"] = rnd(INTER, D_MODEL)
        sd[f"{h}.mlp.up_proj.weight"] = rnd(INTER, D_MODEL)
        sd[f"{h}.mlp.down_proj.weight"] = rnd(D_MODEL, INTER)
        sd[f"{h}.input_layernorm.weight"] = torch.ones(D_MODEL)
        sd[f"{h}.post_attention_layernorm.weight"] = torch.ones(D_MODEL)
    if not tie:
        sd["lm_head.weight"] = rnd(VOCAB, D_MODEL)
    return sd


def _write_repo(
    dir_path: Path,
    config: dict | None = None,
    state: dict[str, torch.Tensor] | None = None,
    *,
    sharded: bool = False,
) -> Path:
    """Write config.json + safetensors (single or sharded) into ``dir_path``."""
    dir_path.mkdir(parents=True, exist_ok=True)
    if config is None:
        config = _base_config()
    if state is None:
        state = _hf_state_dict()
    (dir_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    if not sharded:
        save_file(state, str(dir_path / "model.safetensors"))
        return dir_path

    # Split keys across two shards and write an index.
    keys = list(state.keys())
    half = len(keys) // 2
    groups = {"model-00001-of-00002.safetensors": keys[:half],
              "model-00002-of-00002.safetensors": keys[half:]}
    weight_map = {}
    for shard, shard_keys in groups.items():
        save_file({k: state[k] for k in shard_keys}, str(dir_path / shard))
        for k in shard_keys:
            weight_map[k] = shard
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (dir_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    return dir_path


def _patch_hub(monkeypatch, repo_dir: Path) -> None:
    """Monkeypatch hf_hub_download to serve files from a local temp repo dir."""
    from huggingface_hub.utils import EntryNotFoundError

    def fake_download(repo_id, filename, revision=None, **kwargs):
        candidate = repo_dir / filename
        if not candidate.exists():
            raise EntryNotFoundError(f"{filename} not found")
        return str(candidate)

    # The implementation imports hf_hub_download inside the function from the
    # huggingface_hub package, so patch it on the package module.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)


# --------------------------------------------------------------------------- #
# Config translation                                                          #
# --------------------------------------------------------------------------- #
def test_config_fields_map_to_model_config():
    cfg = hf_config_to_model_config(_base_config())
    assert cfg.d_model == D_MODEL
    assert cfg.n_layers == N_LAYERS
    assert cfg.n_heads == N_HEADS
    assert cfg.n_kv_heads == N_KV_HEADS
    assert cfg.intermediate_size == INTER
    assert cfg.vocab_size == VOCAB
    assert cfg.norm_eps == 1e-5
    assert cfg.rope_theta == 10000.0
    assert cfg.block_size == 128
    assert cfg.tie_word_embeddings is False
    assert cfg.use_bias is False
    assert cfg.hd() == HEAD_DIM


def test_tie_word_embeddings_flag_reflected():
    cfg = hf_config_to_model_config(_base_config(tie_word_embeddings=True))
    assert cfg.tie_word_embeddings is True


# --------------------------------------------------------------------------- #
# Name mapping covers every target parameter                                  #
# --------------------------------------------------------------------------- #
def test_name_mapping_covers_every_target_param():
    cfg = hf_config_to_model_config(_base_config())
    state = _hf_state_dict()
    orkhon_state = fp._build_orkhon_state_dict(state, cfg)

    from orkhon.model.transformer import Transformer

    model = Transformer(cfg)
    target_names = set(model.state_dict().keys())
    # buffers (rope_cos/rope_sin) are non-persistent -> not in state_dict.
    mapped_names = set(orkhon_state.keys())

    # Every learnable target param (sans tied lm_head alias) must be supplied.
    missing = target_names - mapped_names
    assert missing == set(), f"unmapped target params: {missing}"
    # And shapes must match exactly.
    for name, tensor in model.state_dict().items():
        assert orkhon_state[name].shape == tensor.shape, name


# --------------------------------------------------------------------------- #
# Local-dir load path                                                         #
# --------------------------------------------------------------------------- #
def test_from_pretrained_local_dir_loads_and_forwards(tmp_path):
    repo = _write_repo(tmp_path / "repo")
    model, cfg = from_pretrained(str(repo), device="cpu")

    assert cfg.n_layers == N_LAYERS
    input_ids = torch.randint(0, VOCAB, (2, 7))
    logits, _ = model(input_ids)
    assert logits.shape == (2, 7, VOCAB)
    assert torch.isfinite(logits).all()


def test_from_pretrained_dtype_cast(tmp_path):
    repo = _write_repo(tmp_path / "repo")
    model, _ = from_pretrained(str(repo), device="cpu", dtype=torch.float16)
    assert model.embed_tokens.weight.dtype == torch.float16


def test_from_pretrained_weights_match_source(tmp_path):
    state = _hf_state_dict()
    repo = _write_repo(tmp_path / "repo", state=state)
    model, _ = from_pretrained(str(repo), device="cpu")
    # Spot-check that a mapped weight equals the source tensor.
    torch.testing.assert_close(
        model.layers[0].attn.q_proj.weight,
        state["model.layers.0.self_attn.q_proj.weight"],
    )
    torch.testing.assert_close(
        model.final_norm.weight, state["model.norm.weight"]
    )


# --------------------------------------------------------------------------- #
# Tied embeddings (lm_head absent)                                            #
# --------------------------------------------------------------------------- #
def test_tied_embeddings_no_lm_head(tmp_path):
    config = _base_config(tie_word_embeddings=True)
    state = _hf_state_dict(tie=True)
    assert "lm_head.weight" not in state
    repo = _write_repo(tmp_path / "repo", config=config, state=state)

    model, cfg = from_pretrained(str(repo), device="cpu")
    assert cfg.tie_word_embeddings is True
    # lm_head shares storage with embed_tokens.
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    logits, _ = model(torch.randint(0, VOCAB, (1, 5)))
    assert logits.shape == (1, 5, VOCAB)


# --------------------------------------------------------------------------- #
# Hub path (monkeypatched download) — single + sharded                        #
# --------------------------------------------------------------------------- #
def test_from_pretrained_hub_path_monkeypatched(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "hub_repo")
    _patch_hub(monkeypatch, repo)

    model, cfg = from_pretrained("fake-org/tiny-llama", device="cpu")
    logits, _ = model(torch.randint(0, VOCAB, (1, 4)))
    assert logits.shape == (1, 4, VOCAB)


def test_from_pretrained_hub_sharded(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path / "hub_sharded", sharded=True)
    _patch_hub(monkeypatch, repo)

    model, _ = from_pretrained("fake-org/tiny-llama-sharded", device="cpu")
    logits, _ = model(torch.randint(0, VOCAB, (1, 6)))
    assert logits.shape == (1, 6, VOCAB)


# --------------------------------------------------------------------------- #
# Unsupported-feature guards                                                  #
# --------------------------------------------------------------------------- #
def test_attention_bias_raises():
    with pytest.raises(NotImplementedError, match="attention_bias"):
        hf_config_to_model_config(_base_config(attention_bias=True))


def test_qk_norm_raises():
    with pytest.raises(NotImplementedError, match="QK-norm"):
        hf_config_to_model_config(_base_config(use_qk_norm=True))


def test_sliding_window_raises():
    with pytest.raises(NotImplementedError, match="sliding_window"):
        hf_config_to_model_config(_base_config(sliding_window=256))


def test_moe_raises():
    with pytest.raises(NotImplementedError, match="MoE"):
        hf_config_to_model_config(_base_config(num_experts=8))


def test_decoupled_head_dim_raises():
    # head_dim that does not equal hidden_size // num_attention_heads.
    with pytest.raises(NotImplementedError, match="head_dim"):
        hf_config_to_model_config(_base_config(head_dim=HEAD_DIM + 4))


def test_unsupported_architecture_raises():
    with pytest.raises(NotImplementedError, match="architectures"):
        hf_config_to_model_config(
            _base_config(architectures=["Qwen3ForCausalLM"])
        )


def test_mlp_bias_raises():
    with pytest.raises(NotImplementedError, match="mlp_bias"):
        hf_config_to_model_config(_base_config(mlp_bias=True))


def test_bias_tensor_in_weights_raises():
    cfg = hf_config_to_model_config(_base_config())
    state = _hf_state_dict()
    state["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(N_HEADS * HEAD_DIM)
    with pytest.raises(NotImplementedError, match="bias"):
        fp._build_orkhon_state_dict(state, cfg)


def test_missing_required_weight_raises():
    cfg = hf_config_to_model_config(_base_config())
    state = _hf_state_dict()
    del state["model.layers.1.mlp.down_proj.weight"]
    with pytest.raises(KeyError, match="missing required weights"):
        fp._build_orkhon_state_dict(state, cfg)


# --------------------------------------------------------------------------- #
# Checkpoint round-trip                                                       #
# --------------------------------------------------------------------------- #
def test_save_as_orkhon_checkpoint_roundtrip(tmp_path):
    repo = _write_repo(tmp_path / "repo")
    model, cfg = from_pretrained(str(repo), device="cpu")

    out_dir = tmp_path / "ckpt"
    path = save_as_orkhon_checkpoint(model, cfg, out_dir)
    assert path.exists()
    assert (out_dir / "model_config.json").exists()

    from orkhon.train.checkpoint import load_model_from_checkpoint

    reloaded, rcfg = load_model_from_checkpoint(out_dir, device="cpu")
    assert rcfg.to_dict() == cfg.to_dict()
    # Reloaded weights equal the imported ones.
    torch.testing.assert_close(
        reloaded.layers[0].attn.q_proj.weight,
        model.layers[0].attn.q_proj.weight,
    )
    logits, _ = reloaded(torch.randint(0, VOCAB, (1, 5)))
    assert logits.shape == (1, 5, VOCAB)
