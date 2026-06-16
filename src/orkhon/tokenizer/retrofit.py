"""Append <|tool|> / <image> to an EXISTING tokenizer without shifting any ids.

Retraining a tokenizer would change every token id and invalidate all checkpoints.
Instead we copy the tokenizer artifacts and append the missing special tokens at the
END of the vocab, so every existing id is preserved and only new ids are added. A
model trained on the old tokenizer is then made compatible by resizing its embedding
matrix (see :mod:`orkhon.model.resize`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from orkhon.tokenizer.tokenizer import load_tokenizer


def ensure_tool_tokens(src_dir: str | Path, out_dir: str | Path) -> dict:
    """Copy ``src_dir`` tokenizer -> ``out_dir``, appending <|tool|>/<image> if absent.

    Returns ``{"added": [...], "vocab_before": N, "vocab_after": M, "tool_id": int|None}``.
    Existing token ids are NEVER changed.
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        s = src_dir / name
        if s.exists():
            shutil.copy2(s, out_dir / name)

    tk = load_tokenizer(out_dir)
    before = tk.vocab_size
    added: list[str] = []
    for tok in ("<|tool|>", "<image>"):
        if tk.token_to_id(tok) is None:
            added.append(tok)
    if added:
        tk._tk.add_special_tokens(added)  # append at the end; preserves all existing ids
        tk._tk.save(str(out_dir / "tokenizer.json"))
    after = load_tokenizer(out_dir).vocab_size

    # Refresh the special_tokens_map so SpecialIds.tool/image resolve.
    smap_path = out_dir / "special_tokens_map.json"
    smap = json.loads(smap_path.read_text()) if smap_path.exists() else {}
    for tok in ("<|tool|>", "<image>"):
        smap.setdefault(tok, tok) if tok in added or tk.token_to_id(tok) is not None else None
    # Always ensure the map lists them (idempotent).
    for tok in ("<|tool|>", "<image>"):
        if load_tokenizer(out_dir).token_to_id(tok) is not None:
            smap[tok] = tok
    smap_path.write_text(json.dumps(smap, indent=2), encoding="utf-8")

    fresh = load_tokenizer(out_dir)
    return {"added": added, "vocab_before": before, "vocab_after": fresh.vocab_size,
            "tool_id": fresh.special.tool, "image_id": fresh.special.image}
