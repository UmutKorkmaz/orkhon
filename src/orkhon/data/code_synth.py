"""Fill-in-the-middle (FIM) code-completion data synthesis.

FIM is how code models (StarCoder, Code Llama) learn to *complete* code, not just
continue it: split a code document into ``prefix | middle | suffix`` at a random
point and teach the model to produce ``middle`` given ``prefix`` and ``suffix``.

We emit standard SFT ``{messages}`` rows so the existing SFT trainer consumes them
unchanged (no new special tokens, no tokenizer change). Delimiters are plain text
(``<fim_prefix>`` etc. as ordinary BPE tokens) — good enough for a small model and
avoids the irreversible tokenizer freeze.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

# Plain-text delimiters (ordinary tokens; no tokenizer change).
PRE, SUF, MID = "<fim_prefix>", "<fim_suffix>", "<fim_mid>"

_MIN_CHARS = 40


def fim_example(code: str, rng: random.Random) -> dict | None:
    """Turn one code document into a FIM SFT example, or None if too short."""
    code = code.rstrip()
    if len(code) < _MIN_CHARS:
        return None
    # Split on a line boundary near the middle for natural prefix/suffixes.
    lines = code.split("\n")
    if len(lines) < 3:
        return None
    cut = rng.randint(1, len(lines) - 2)
    prefix = "\n".join(lines[:cut])
    middle = lines[cut]
    suffix = "\n".join(lines[cut + 1 :])
    if not middle.strip():
        return None
    prompt = f"Complete the code (fill in the middle).\n{PRE} {prefix}\n{SUF} {suffix}\n{MID} "
    return {"messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": middle}]}


def make_fim_sft(
    code_docs: Iterable[str],
    out_path: str | Path,
    *,
    max_examples: int = 2000,
    seed: int = 1337,
) -> int:
    """Write ``max_examples`` FIM SFT rows (jsonl) from an iterable of code docs."""
    rng = random.Random(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as w:
        for code in code_docs:
            if n >= max_examples:
                break
            ex = fim_example(code, rng)
            if ex is not None:
                w.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
    return n
