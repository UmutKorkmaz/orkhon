"""Golden-set chat scoring: exact / contains matching of assistant replies.

A lightweight behavioral check: for a tiny set of ``(prompt, expected)`` pairs,
generate the assistant reply and score it against the expected answer with one of
two matchers:

* ``exact``   — normalized (stripped, case-folded) reply equals expected.
* ``contains``— expected appears as a normalized substring of the reply.

This is intentionally simple — it is a smoke/regression aid, not a benchmark. It
returns per-example results and an aggregate accuracy so CI can gate on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from orkhon.model.generation import generate
from orkhon.tokenizer.render import encode_for_inference
from orkhon.tokenizer.tokenizer import OrkhonTokenizer

# A GoldenItem: {"prompt": str | [messages], "expected": str, "match": "exact"|"contains"}
GoldenItem = dict


def _normalize(text: str) -> str:
    """Lowercase + strip surrounding whitespace for tolerant comparison."""
    return text.strip().lower()


def _score(reply: str, expected: str, match: str) -> bool:
    """Return True if ``reply`` satisfies the ``match`` rule against ``expected``."""
    r = _normalize(reply)
    e = _normalize(expected)
    if match == "exact":
        return r == e
    if match == "contains":
        return e in r
    raise ValueError(f"unknown match mode {match!r}; use 'exact' or 'contains'")


def _prompt_messages(prompt) -> list[dict]:
    """Normalize a prompt field (string or messages list) to a messages list."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


def score_golden_set(
    model,
    tokenizer: OrkhonTokenizer,
    golden: Sequence[GoldenItem],
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device=None,
) -> dict:
    """Generate and score replies for a golden set of prompt/expected pairs.

    Args:
        model: a trained :class:`orkhon.model.Transformer`.
        tokenizer: the matching :class:`OrkhonTokenizer`.
        golden: items ``{"prompt", "expected", "match"}`` (``match`` defaults to
            ``"contains"`` when omitted).
        max_new_tokens / temperature / top_k / top_p: generation controls
            (greedy by default for determinism).
        device: device override (defaults to the model's device).

    Returns:
        ``{"results": [{"prompt", "expected", "reply", "match", "passed"}...],
        "accuracy": float, "passed": int, "total": int}``.
    """
    results: list[dict] = []
    passed = 0
    for item in golden:
        match = item.get("match", "contains")
        messages = _prompt_messages(item["prompt"])
        prompt_ids = encode_for_inference(messages, tokenizer.encode, tokenizer.special)
        new_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_ids=(tokenizer.special.end,),
            device=device,
        )
        reply = tokenizer.decode(new_ids, skip_special=True)
        ok = _score(reply, item["expected"], match)
        passed += int(ok)
        results.append(
            {
                "prompt": item["prompt"],
                "expected": item["expected"],
                "reply": reply,
                "match": match,
                "passed": ok,
            }
        )

    total = len(golden)
    accuracy = passed / total if total else 0.0
    return {
        "results": results,
        "accuracy": accuracy,
        "passed": passed,
        "total": total,
    }


def load_golden_set(path: str | Path) -> list[GoldenItem]:
    """Load a golden set from a JSONL file (one item per line)."""
    import json

    items: list[GoldenItem] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
