"""Synthetic Old Turkic transliteration data for SFT.

Transliteration (rune → Latin) is a deterministic mapping (see ``translit.py``),
so we can manufacture an arbitrarily large, *correct* training set: sample rune
sequences and pair them with their exact Latin transliteration. This teaches a
model the script ⇄ value correspondence without any scholarly corpus.

This does NOT manufacture *translation* data (→ Turkish/English) — that requires
sourced attestations (Tekin/Erdal/Uppsala) and is deliberately left out so the
model is never trained on invented meanings.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from orkhon.data.old_turkic.translit import OLD_TURKIC_RANGE, rune_to_latin

# Use the Orkhon-variant runes (the script of the Orkhon inscriptions) for realism.
def _orkhon_runes() -> list[str]:
    import unicodedata

    out = []
    for cp in range(OLD_TURKIC_RANGE[0], OLD_TURKIC_RANGE[1] + 1):
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "ORKHON" in name or "COMMON" in name:
            out.append(ch)
    return out


_INSTRUCTIONS = [
    "Transliterate this Old Turkic (Orkhon) text into Latin:",
    "Bu Göktürk yazısını Latin harflerine çevir:",
    "Read this runic Old Turkic and give the transliteration:",
]


def make_translit_sft(
    out_path: str | Path,
    n: int = 2000,
    *,
    min_words: int = 1,
    max_words: int = 6,
    min_len: int = 2,
    max_len: int = 8,
    seed: int = 1337,
) -> int:
    """Write ``n`` deterministic rune→Latin transliteration SFT examples (jsonl)."""
    rng = random.Random(seed)
    runes = _orkhon_runes()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "w", encoding="utf-8") as w:
        for _ in range(n):
            words = []
            for _w in range(rng.randint(min_words, max_words)):
                wlen = rng.randint(min_len, max_len)
                words.append("".join(rng.choice(runes) for _ in range(wlen)))
            runic = " ".join(words)
            latin = rune_to_latin(runic)
            instr = rng.choice(_INSTRUCTIONS)
            row = {
                "messages": [
                    {"role": "user", "content": f"{instr} {runic}"},
                    {"role": "assistant", "content": latin},
                ]
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written
