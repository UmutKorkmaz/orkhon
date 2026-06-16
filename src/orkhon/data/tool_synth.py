"""Synthesize tool-call SFT traces so a model learns to NATIVELY emit <|tool|> calls.

Three trace shapes, all deterministic + verified against ``parse_tool_call``:
  1. tool call  — user asks an arithmetic question; assistant emits <|tool|>{calculator...};
     a tool turn gives the answer; assistant states it.
  2. retrieve   — user asks a doc question; assistant emits <|tool|>{retrieve...}; tool
     returns a passage; assistant answers.
  3. direct     — a question that needs NO tool; assistant answers plainly (a negative so
     the model doesn't over-call tools).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

_TOOL = "<|tool|>"


def _calc_trace(rng: random.Random) -> dict:
    a, b = rng.randint(1, 99), rng.randint(1, 99)
    op = rng.choice(["+", "-", "*"])
    expr = f"{a}{op}{b}"
    ans = {"+": a + b, "-": a - b, "*": a * b}[op]  # explicit dispatch (no eval)
    return {"messages": [
        {"role": "user", "content": f"What is {a} {op} {b}?"},
        {"role": "assistant", "content": f"Let me compute that. {_TOOL}" +
            json.dumps({"name": "calculator", "arguments": {"expression": expr}})},
        {"role": "tool", "content": str(ans)},
        {"role": "assistant", "content": f"The answer is {ans}."},
    ]}


def _retrieve_trace(rng: random.Random) -> dict:
    topic = rng.choice(["the sky", "the ocean", "a forest", "the city", "a river"])
    return {"messages": [
        {"role": "user", "content": f"Tell me about {topic}."},
        {"role": "assistant", "content": f"Let me look that up. {_TOOL}" +
            json.dumps({"name": "retrieve", "arguments": {"query": topic}})},
        {"role": "tool", "content": f"Excerpt about {topic}: it is a part of the natural world."},
        {"role": "assistant", "content": f"Here is what I found about {topic}: it is a part of the natural world."},
    ]}


def _direct_trace(rng: random.Random) -> dict:
    return {"messages": [
        {"role": "user", "content": rng.choice(["Hello!", "How are you?", "What is your name?"])},
        {"role": "assistant", "content": rng.choice([
            "Hello! How can I help?", "I'm doing well, thanks!", "I'm Orkhon, an AI assistant."])},
    ]}


def make_tool_sft(
    out_path: str | Path,
    *,
    out_val: str | Path | None = None,
    n: int = 5000,
    val_frac: float = 0.05,
    seed: int = 1337,
) -> dict:
    """Write ``n`` deterministic tool-call SFT traces (jsonl); optional held-out val split."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r = rng.random()
        rows.append(_calc_trace(rng) if r < 0.45
                     else _retrieve_trace(rng) if r < 0.80
                     else _direct_trace(rng))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_val = int(n * val_frac)
    train, val = rows[n_val:], rows[:n_val]
    with open(out_path, "w", encoding="utf-8") as w:
        for r in train:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    if out_val:
        with open(out_val, "w", encoding="utf-8") as w:
            for r in val:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"train": len(train), "val": len(val), "out": str(out_path)}
