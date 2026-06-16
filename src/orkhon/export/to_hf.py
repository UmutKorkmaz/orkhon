"""Export a Orkhon checkpoint to a Hugging-Face-style directory.

``export`` writes a self-contained directory:

* ``config.json``        — all :class:`~orkhon.model.ModelConfig` fields plus a
  ``model_type`` and ``architectures`` entry, so the architecture is fully
  reconstructable without the original checkpoint.
* ``model.safetensors``  — the model ``state_dict`` (tensors only). When the
  embedding and LM head are weight-tied, the duplicate ``lm_head.weight`` is
  dropped (safetensors forbids shared storage) and re-tied on reload; a flag in
  ``config.json`` records this.
* tokenizer files        — ``tokenizer.json``, ``special_tokens_map.json``,
  ``tokenizer_config.json`` (which embeds the chat template) copied verbatim.

``reload_and_check`` rebuilds a :class:`~orkhon.model.Transformer` purely from the
exported files and asserts its logits match the original checkpoint within a
tolerance on a fixed prompt — the parity guarantee the exporter must uphold.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from orkhon.model.config import ModelConfig
from orkhon.model.transformer import Transformer
from orkhon.train.checkpoint import load_checkpoint

MODEL_TYPE = "orkhon"
_TOKENIZER_FILES = (
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
)
_FIXED_PROMPT = [1, 2, 3, 4, 5]  # arbitrary but fixed ids for parity checking


def _strip_tied_lm_head(
    state_dict: dict[str, torch.Tensor], tied: bool
) -> dict[str, torch.Tensor]:
    """Return a safetensors-safe state dict (no shared storage).

    When weights are tied, ``lm_head.weight`` shares storage with
    ``embed_tokens.weight``; safetensors refuses duplicate storage, so we drop the
    LM head copy. Otherwise we clone any tensors that still share storage to be safe.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if tied and k == "lm_head.weight":
            continue
        # Clone to guarantee contiguous, non-aliased tensors for safetensors.
        out[k] = v.detach().cpu().contiguous().clone()
    return out


def export(
    checkpoint_dir: str | Path,
    out_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    tag: str = "last",
    model_name: str | None = None,
    eval_json: str | Path | None = None,
    write_card: bool = True,
) -> Path:
    """Export ``checkpoint_dir`` to a HF-style directory at ``out_dir``.

    Args:
        checkpoint_dir: directory holding ``ckpt_<tag>.pt`` + ``model_config.json``.
        out_dir: destination directory (created if missing).
        tokenizer_dir: directory with the trained tokenizer files to copy.
        tag: checkpoint tag ("last" or "best").
        model_name: display name for the HF model card.
        eval_json: optional path to an eval-results JSON (folded into the card).
        write_card: write an HF-ready ``README.md`` model card into ``out_dir``.

    Returns:
        The ``out_dir`` path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(checkpoint_dir, tag=tag, map_location="cpu")
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    state_dict = ckpt["model"]

    # --- config.json (architecture metadata) ---
    config = dict(cfg.to_dict())
    config["model_type"] = MODEL_TYPE
    config["architectures"] = ["OrkhonForCausalLM"]
    config["tie_word_embeddings"] = cfg.tie_word_embeddings
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    # --- model.safetensors (state dict, tie-safe) ---
    safe_state = _strip_tied_lm_head(state_dict, cfg.tie_word_embeddings)
    save_file(
        safe_state,
        str(out_dir / "model.safetensors"),
        metadata={"format": "pt", "model_type": MODEL_TYPE},
    )

    # --- tokenizer files (verbatim copy) ---
    tokenizer_dir = Path(tokenizer_dir)
    for name in _TOKENIZER_FILES:
        src = tokenizer_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # --- HF model card (README.md) ---
    if write_card:
        from orkhon.export.model_card import generate_hf_model_card

        eval_results = None
        if eval_json and Path(eval_json).exists():
            eval_results = json.loads(Path(eval_json).read_text(encoding="utf-8"))
        card = generate_hf_model_card(
            cfg, model_name=model_name or out_dir.name, eval_results=eval_results,
            tool_token_trained=None,  # current checkpoints predate <|tool|>
        )
        (out_dir / "README.md").write_text(card, encoding="utf-8")

    return out_dir


def load_exported_model(
    out_dir: str | Path, device: str | torch.device = "cpu"
) -> tuple[Transformer, ModelConfig]:
    """Rebuild a :class:`Transformer` from an exported HF directory.

    Reads ``config.json`` + ``model.safetensors``, reconstructs the config,
    builds the model, loads weights (re-tying the LM head when configured), and
    moves it to ``device``.
    """
    out_dir = Path(out_dir)
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    cfg = ModelConfig.from_dict(config)

    model = Transformer(cfg)
    state_dict = load_file(str(out_dir / "model.safetensors"))

    # If tied, the LM head was dropped on export; re-tie before/after load. The
    # Transformer constructor already tied lm_head.weight to the embedding, so
    # loading just the embedding row is sufficient and load_state_dict(strict=False)
    # tolerates the absent lm_head.weight key.
    strict = not cfg.tie_word_embeddings
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if unexpected:
        raise RuntimeError(f"unexpected keys in exported state dict: {unexpected}")
    if cfg.tie_word_embeddings:
        # Only lm_head.weight may be "missing"; it is tied to the embedding.
        leftover = [m for m in missing if m != "lm_head.weight"]
        if leftover:
            raise RuntimeError(f"missing keys in exported state dict: {leftover}")
        model.lm_head.weight = model.embed_tokens.weight

    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def reload_and_check(
    out_dir: str | Path,
    checkpoint_dir: str | Path,
    device: str | torch.device = "cpu",
    *,
    tag: str = "last",
    atol: float = 1e-3,
    prompt_ids: list[int] | None = None,
) -> dict:
    """Assert exported model logits match the original within ``atol``.

    Rebuilds the model from ``out_dir`` and the original from ``checkpoint_dir``,
    runs both on a fixed prompt, and asserts the logits agree elementwise.

    Returns:
        ``{"max_abs_diff": float, "atol": float, "ok": True}`` on success.

    Raises:
        AssertionError: if any logit differs by more than ``atol``.
    """
    from orkhon.train.checkpoint import load_model_from_checkpoint

    ids = list(prompt_ids) if prompt_ids is not None else list(_FIXED_PROMPT)

    original, cfg = load_model_from_checkpoint(checkpoint_dir, device=device, tag=tag)
    original.eval()
    # Clamp prompt ids into the vocab range so the fixed prompt is always valid.
    ids = [i % cfg.vocab_size for i in ids][: cfg.block_size]
    if not ids:
        ids = [0]

    exported, _ = load_exported_model(out_dir, device=device)

    x = torch.tensor([ids], dtype=torch.long, device=device)
    orig_logits, _ = original(x)
    new_logits, _ = exported(x)

    max_abs_diff = float((orig_logits - new_logits).abs().max().item())
    assert max_abs_diff <= atol, (
        f"exported logits diverge from original: max_abs_diff={max_abs_diff:.3e} "
        f"> atol={atol:.3e}"
    )
    return {"max_abs_diff": max_abs_diff, "atol": atol, "ok": True}
