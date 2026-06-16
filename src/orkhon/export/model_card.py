"""Generate a Markdown model card from an architecture config + eval results.

``generate_model_card`` produces a README.md-style document summarizing the model
shape (from :class:`~orkhon.model.ModelConfig`) and any evaluation metrics, with a
short usage snippet. It is intentionally template-light: a few well-structured
sections that render cleanly on the Hub and in a repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orkhon.model.config import ModelConfig


def _fmt_int(n: int) -> str:
    """Human-friendly integer (e.g. 12,345,678)."""
    return f"{n:,}"


def _params_human(n: int) -> str:
    """Compact parameter count (e.g. 1.2M, 350K)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _config_table(cfg: ModelConfig) -> str:
    rows = [
        ("vocab_size", cfg.vocab_size),
        ("block_size", cfg.block_size),
        ("n_layers", cfg.n_layers),
        ("d_model", cfg.d_model),
        ("n_heads", cfg.n_heads),
        ("n_kv_heads", cfg.n_kv_heads),
        ("head_dim", cfg.hd()),
        ("intermediate_size", cfg.intermediate()),
        ("rope_theta", cfg.rope_theta),
        ("norm_eps", cfg.norm_eps),
        ("tie_word_embeddings", cfg.tie_word_embeddings),
    ]
    lines = ["| Field | Value |", "| --- | --- |"]
    lines += [f"| `{k}` | {v} |" for k, v in rows]
    return "\n".join(lines)


def _eval_section(eval_results: dict[str, Any] | None) -> str:
    if not eval_results:
        return "_No evaluation results provided._"

    lines = ["| Metric | Value |", "| --- | --- |"]
    for key in ("loss", "ppl", "val_loss", "val_ppl", "accuracy", "tokens"):
        if key in eval_results and eval_results[key] is not None:
            val = eval_results[key]
            if isinstance(val, float):
                val = f"{val:.4f}"
            lines.append(f"| `{key}` | {val} |")
    # Include any remaining scalar metrics not covered above.
    if len(lines) == 2:
        for k, v in eval_results.items():
            if isinstance(v, (int, float, str)):
                lines.append(f"| `{k}` | {v} |")
    return "\n".join(lines)


def generate_model_card(
    cfg: ModelConfig,
    *,
    model_name: str = "orkhon",
    eval_results: dict[str, Any] | None = None,
    description: str | None = None,
) -> str:
    """Render a Markdown model card string.

    Args:
        cfg: the architecture config.
        model_name: display name / repo id.
        eval_results: optional dict of metrics (loss/ppl/accuracy/...).
        description: optional free-text description paragraph.

    Returns:
        The model card as a Markdown string.
    """
    params = cfg.estimate_params()
    desc = description or (
        "A from-scratch decoder-only Transformer (RoPE, grouped-query attention, "
        "RMSNorm, SwiGLU) trained with the Orkhon stack."
    )

    card = f"""---
license: apache-2.0
library_name: orkhon
tags:
- text-generation
- orkhon
---

# {model_name}

{desc}

- **Architecture:** decoder-only Transformer (GQA + RoPE + RMSNorm + SwiGLU)
- **Parameters:** ~{_params_human(params)} ({_fmt_int(params)})
- **Context length:** {cfg.block_size} tokens
- **Vocabulary:** {_fmt_int(cfg.vocab_size)}

## Configuration

{_config_table(cfg)}

## Evaluation

{_eval_section(eval_results)}

## Usage

```python
from orkhon.export.to_hf import load_exported_model
from orkhon.tokenizer import load_tokenizer
from orkhon.model import generate
from orkhon.tokenizer.render import encode_for_inference

model, cfg = load_exported_model("{model_name}")
tok = load_tokenizer("{model_name}")

messages = [{{"role": "user", "content": "Hello"}}]
prompt_ids = encode_for_inference(messages, tok.encode, tok.special)
new_ids = generate(model, prompt_ids, max_new_tokens=64, temperature=0.0,
                   eos_ids=(tok.special.end,))
print(tok.decode(new_ids))
```
"""
    return card


def write_model_card(
    cfg: ModelConfig,
    out_path: str | Path,
    *,
    model_name: str = "orkhon",
    eval_results: dict[str, Any] | None = None,
    description: str | None = None,
) -> Path:
    """Render and write a model card to ``out_path`` (typically ``README.md``)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    card = generate_model_card(
        cfg,
        model_name=model_name,
        eval_results=eval_results,
        description=description,
    )
    out_path.write_text(card, encoding="utf-8")
    return out_path


def generate_hf_model_card(
    cfg: ModelConfig,
    *,
    model_name: str = "orkhon",
    eval_results: dict[str, Any] | None = None,
    description: str | None = None,
    training_data: str | None = None,
    tool_token_trained: bool | None = None,
    license_id: str = "apache-2.0",
) -> str:
    """An HF-Hub-ready model card: YAML front-matter + the standard card + limitations.

    Includes a ``tool_token_trained`` flag (None = predates the <|tool|> token) so
    consumers know whether native tool-calling is available.
    """
    body = generate_model_card(cfg, model_name=model_name, eval_results=eval_results,
                               description=description)
    # The base card opens with its own YAML front-matter; strip it so the HF card
    # has exactly ONE front-matter block (the one we emit below).
    if body.startswith("---"):
        end = body.find("\n---", 3)
        if end >= 0:
            body = body[end + 4 :].lstrip("\n")
    flag_text = {True: "yes", False: "no", None: "unknown (predates the <|tool|> token)"}[
        tool_token_trained
    ]
    front = [
        "---",
        "language: ['en', 'tr']",
        f"license: {license_id}",
        "library_name: orkhon",
        "tags: ['orkhon', 'from-scratch', 'turkic']",
        f"parameters: {cfg.estimate_params()}",
        "---",
        "",
    ]
    tool_note = ("native `<|tool|>` calls available." if tool_token_trained
                 else "native tool-calling not available; tool use is loop/prompt-based.")
    extra = [
        "## Training data", "", training_data or "_see the Orkhon model card / lineage._", "",
        "## Limitations", "",
        "- A small from-scratch model: fluent in surface form, **not a reliable knowledge source**.",
        "- Facts may be confabulated; no repetition penalty unless enabled at inference.",
        f"- **Tool token trained:** {flag_text} — {tool_note}",
        "- Not safety-aligned; do not deploy without an alignment layer.",
        "",
    ]
    return "\n".join(front) + body + "\n\n" + "\n".join(extra)



def write_model_card_from_dir(
    export_dir: str | Path,
    *,
    model_name: str = "orkhon",
    eval_json: str | Path | None = None,
) -> Path:
    """Generate ``README.md`` inside an exported dir using its ``config.json``."""
    export_dir = Path(export_dir)
    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    cfg = ModelConfig.from_dict(config)
    eval_results = None
    if eval_json is not None and Path(eval_json).exists():
        eval_results = json.loads(Path(eval_json).read_text(encoding="utf-8"))
    return write_model_card(
        cfg,
        export_dir / "README.md",
        model_name=model_name,
        eval_results=eval_results,
    )
