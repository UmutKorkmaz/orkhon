"""Tests for the model zoo registry (offline, tiny model + tokenizer)."""

from __future__ import annotations

import json

import torch

from orkhon.model import ModelConfig, Transformer
from orkhon.registry import build_index, next_name, register_imported, register_model
from orkhon.tokenizer.train import train_tokenizer


def _tiny_checkpoint(tmp_path, vocab_size):
    cfg = ModelConfig(vocab_size=vocab_size, block_size=32, n_layers=2, d_model=32,
                      n_heads=4, n_kv_heads=2, intermediate_size=64)
    model = Transformer(cfg).eval()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    torch.save({"model": model.state_dict(), "model_config": cfg.to_dict()},
               ckpt_dir / "ckpt_last.pt")
    return ckpt_dir


def _tiny_tokenizer(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["the cat sat on the mat"] * 50 + ["a dog ran fast"] * 50),
                      encoding="utf-8")
    tok_dir = tmp_path / "tok"
    train_tokenizer(corpus_paths=str(corpus), out_dir=str(tok_dir), vocab_size=300)
    return tok_dir


def test_register_model_writes_full_archive(tmp_path):
    tok_dir = _tiny_tokenizer(tmp_path)
    from orkhon.tokenizer import load_tokenizer
    vocab = load_tokenizer(tok_dir).vocab_size
    ckpt = _tiny_checkpoint(tmp_path, vocab)
    root = tmp_path / "models"

    dest = register_model(
        "bumin-mini", ckpt, tok_dir, kind="base", lineage="a tiny test model",
        sample_prompts=["the cat"], generate_mode="complete",
        device="cpu", dest_root=str(root), date="20260614",
    )

    assert dest.name == "bumin-mini-20260614"
    for f in ["manifest.json", "model_card.md", "samples.txt", "run.sh",
              "code_snapshot.tgz", "checkpoint/ckpt_last.pt", "checkpoint/model_config.json",
              "tokenizer/tokenizer.json"]:
        assert (dest / f).exists(), f"missing {f}"

    man = json.loads((dest / "manifest.json").read_text())
    assert man["name"] == "bumin-mini" and man["kind"] == "base"
    assert man["params_m"] > 0 and man["n_layers"] == 2

    # The archived checkpoint reloads through the normal path.
    from orkhon.train.checkpoint import load_model_from_checkpoint
    m2, c2 = load_model_from_checkpoint(dest / "checkpoint", device="cpu")
    assert c2.n_layers == 2

    index = build_index(root)
    assert "bumin-mini" in index.read_text()


def test_register_imported_records_repro_without_weights(tmp_path):
    root = tmp_path / "models"
    dest = register_imported(
        "kashgar", repo="HuggingFaceTB/SmolLM2-135M", params_m=135.0,
        lineage="imported open base", sample={"prompt": "The capital of France is",
        "output": " the capital of the country."}, dest_root=str(root), date="20260614",
    )
    man = json.loads((dest / "manifest.json").read_text())
    assert man["kind"] == "imported" and man["source_repo"].endswith("SmolLM2-135M")
    assert not (dest / "checkpoint").exists()  # weights NOT re-stored
    assert "import-hf" in (dest / "run.sh").read_text()


def test_next_name_skips_used(tmp_path):
    root = tmp_path / "models"
    (root / "bumin-mini-20260101").mkdir(parents=True)
    assert next_name(root) == "tonyuk"  # bumin-mini taken -> next in pool
