"""Tests for HF export + reload parity (orkhon.export.to_hf)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from orkhon.export.model_card import generate_model_card, write_model_card_from_dir
from orkhon.export.to_hf import export, load_exported_model, reload_and_check
from orkhon.model.config import ModelConfig


def test_export_writes_expected_files(checkpoint_dir, tokenizer_dir, tmp_path):
    out_dir = tmp_path / "hf"
    export(checkpoint_dir, out_dir, tokenizer_dir)

    assert (out_dir / "config.json").exists()
    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "tokenizer_config.json").exists()
    # HF model card (README.md with YAML front-matter) is written by default.
    card = (out_dir / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---") and "license: apache-2.0" in card

    config = json.loads((out_dir / "config.json").read_text())
    assert config["model_type"] == "orkhon"
    assert "architectures" in config
    # All ModelConfig fields are present.
    for field in ModelConfig.__dataclass_fields__:
        assert field in config


def test_reload_and_check_logits_match(checkpoint_dir, tokenizer_dir, tmp_path):
    out_dir = tmp_path / "hf"
    export(checkpoint_dir, out_dir, tokenizer_dir)

    result = reload_and_check(out_dir, checkpoint_dir, device="cpu", atol=1e-3)
    assert result["ok"] is True
    assert result["max_abs_diff"] <= 1e-3


def test_exported_model_forward_matches_original(checkpoint_dir, tokenizer_dir, tmp_path):
    out_dir = tmp_path / "hf"
    export(checkpoint_dir, out_dir, tokenizer_dir)

    from orkhon.train.checkpoint import load_model_from_checkpoint

    original, cfg = load_model_from_checkpoint(checkpoint_dir, device="cpu")
    exported, _ = load_exported_model(out_dir, device="cpu")

    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        a, _ = original(x)
        b, _ = exported(x)
    assert torch.allclose(a, b, atol=1e-3)

    # Tied weight survives the round trip.
    if cfg.tie_word_embeddings:
        assert exported.lm_head.weight is exported.embed_tokens.weight


def test_model_card_generation(checkpoint_dir, tokenizer_dir, tmp_path):
    out_dir = tmp_path / "hf"
    export(checkpoint_dir, out_dir, tokenizer_dir)

    config = json.loads((out_dir / "config.json").read_text())
    cfg = ModelConfig.from_dict(config)
    card = generate_model_card(
        cfg, model_name="orkhon-tiny", eval_results={"loss": 1.23, "ppl": 3.42}
    )
    assert "orkhon-tiny" in card
    assert "Parameters" in card
    assert "1.2300" in card  # eval loss rendered

    readme = write_model_card_from_dir(out_dir, model_name="orkhon-tiny")
    assert Path(readme).exists()
    assert "orkhon-tiny" in Path(readme).read_text()
