"""Deterministic synthetic smoke datasets.

These tiny, fully-deterministic datasets exist so the whole training stack
(pretrain -> SFT -> DPO) can be exercised end-to-end and a ~6M-param model can
*measurably* reduce loss. The text is simple, highly structured English with lots
of repeated algorithmic patterns (counting, arithmetic, fixed Q&A formats) — easy
to learn, so loss curves move within a handful of steps.

Everything is seeded; no network access. Re-running produces byte-identical files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# --- generation knobs -------------------------------------------------------
_SEED = 1337
_PRETRAIN_LINES = 4000
_SFT_EXAMPLES = 80
_DPO_EXAMPLES = 40

_NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve",
]
_ANIMALS = ["cat", "dog", "fox", "owl", "bee", "ant", "cow", "hen"]
_COLORS = ["red", "blue", "green", "gray", "black", "white", "brown", "amber"]


def _pretrain_lines(rng: random.Random) -> list[str]:
    """Build a list of simple, learnable templated English sentences."""
    lines: list[str] = []
    for _ in range(_PRETRAIN_LINES):
        kind = rng.randint(0, 4)
        if kind == 0:
            # Counting sequences: strongly predictable next token.
            start = rng.randint(0, 6)
            seq = " ".join(str(start + i) for i in range(6))
            lines.append(f"count from {start}: {seq} .")
        elif kind == 1:
            # Number-word association.
            n = rng.randint(0, 12)
            lines.append(f"the number {n} is written as {_NUMBER_WORDS[n]} .")
        elif kind == 2:
            # Simple arithmetic facts.
            a = rng.randint(0, 9)
            b = rng.randint(0, 9)
            lines.append(f"if you add {a} and {b} you get {a + b} .")
        elif kind == 3:
            # Fixed-structure descriptive sentence.
            animal = rng.choice(_ANIMALS)
            color = rng.choice(_COLORS)
            lines.append(f"the {color} {animal} sat quietly on the soft warm mat .")
        else:
            # Repeated Q&A pattern (teaches the colon/answer structure).
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
            lines.append(f"question: what is {a} plus {b} ? answer: {a + b} .")
    return lines


def make_smoke_corpus(out: str | Path = "data/smoke/pretrain.txt") -> Path:
    """Write the deterministic pretraining corpus (one document per line)."""
    rng = random.Random(_SEED)
    lines = _pretrain_lines(rng)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _sft_example(rng: random.Random) -> dict:
    """One SFT example teaching a crisp 'Answer: <n>.' arithmetic format."""
    a = rng.randint(0, 9)
    b = rng.randint(0, 9)
    user = f"What is {a} plus {b}?"
    assistant = f"Answer: {a + b}."
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def make_smoke_sft(out: str | Path = "data/smoke/sft.jsonl") -> Path:
    """Write deterministic SFT examples teaching a fixed answer format."""
    rng = random.Random(_SEED + 1)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    # Deduplicate so the format is taught on varied inputs.
    while len(rows) < _SFT_EXAMPLES:
        ex = _sft_example(rng)
        key = (ex["messages"][0]["content"], ex["messages"][1]["content"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(ex)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def _dpo_example(rng: random.Random) -> dict:
    """One preference pair: chosen uses the crisp format, rejected is sloppy."""
    a = rng.randint(0, 9)
    b = rng.randint(0, 9)
    total = a + b
    prompt = [{"role": "user", "content": f"What is {a} plus {b}?"}]
    chosen = f"Answer: {total}."
    # Rejected is verbose/unformatted and sometimes wrong-looking — clearly worse.
    rejected = f"hmm i think maybe it could be around {total} or so, not totally sure"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def make_smoke_dpo(out: str | Path = "data/smoke/dpo.jsonl") -> Path:
    """Write deterministic DPO preference pairs (chosen clearly better)."""
    rng = random.Random(_SEED + 2)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict] = []
    while len(rows) < _DPO_EXAMPLES:
        ex = _dpo_example(rng)
        key = ex["prompt"][0]["content"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(ex)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def make_all(base: str | Path = "data/smoke") -> dict[str, Path]:
    """Generate all three smoke datasets under ``base``."""
    base = Path(base)
    return {
        "pretrain": make_smoke_corpus(base / "pretrain.txt"),
        "sft": make_smoke_sft(base / "sft.jsonl"),
        "dpo": make_smoke_dpo(base / "dpo.jsonl"),
    }


if __name__ == "__main__":
    paths = make_all()
    for name, path in paths.items():
        print(f"wrote {name}: {path}")
