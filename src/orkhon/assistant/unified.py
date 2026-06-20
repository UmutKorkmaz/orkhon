"""Deterministic specialist routing for the unified Orkhon assistant.

Small LLMs are the wrong place to perform exact symbolic transforms. The unified
assistant therefore keeps deterministic tasks in code and leaves open-ended
language behavior to the model.
"""

from __future__ import annotations

import re

from orkhon.data.old_turkic import OLD_TURKIC_RANGE, rune_to_latin


_ADD_RE = re.compile(
    r"(?:what\s+is\s+)?(?P<a>\d{1,4})\s*\+\s*(?P<b>\d{1,4})(?:\s*(?:\?|kactir|kaçtır))?",
    re.IGNORECASE,
)


def _is_old_turkic(ch: str) -> bool:
    cp = ord(ch)
    return OLD_TURKIC_RANGE[0] <= cp <= OLD_TURKIC_RANGE[1]


def _extract_old_turkic(text: str) -> str:
    parts: list[str] = []
    in_gap = False
    for ch in text:
        if _is_old_turkic(ch):
            if in_gap and parts and parts[-1] != " ":
                parts.append(" ")
            parts.append(ch)
            in_gap = False
        elif ch.isspace() and parts:
            in_gap = True
        elif parts:
            in_gap = True
    return "".join(parts).strip()


def _looks_like_translation_request(text: str) -> bool:
    lower = text.lower()
    translation_terms = (
        "translate",
        "translation",
        "modern turkish",
        "modern english",
        "ceviri",
        "çeviri",
        "tercüme",
        "tercume",
        "turkceye",
        "türkçeye",
        "ingilizceye",
    )
    transliteration_terms = (
        "transliterate",
        "transliteration",
        "latin",
        "read",
        "oku",
        "harflerine",
        "translitere",
    )
    return any(t in lower for t in translation_terms) and not any(
        t in lower for t in transliteration_terms
    )


def _is_transliteration_request(text: str) -> bool:
    lower = text.lower()
    terms = (
        "old turkic",
        "kokturk",
        "gokturk",
        "göktürk",
        "eski turkce",
        "eski türkçe",
        "rune",
        "runic",
        "transliterate",
        "transliteration",
        "latin",
        "harflerine",
        "cevir",
        "çevir",
        "oku",
        "read",
    )
    return any(t in lower for t in terms)


def _math_reply(text: str) -> str | None:
    match = _ADD_RE.search(text)
    if not match:
        return None
    a = int(match.group("a"))
    b = int(match.group("b"))
    lower = text.lower()
    if "kactir" in lower or "kaçtır" in lower:
        return f"Cevap {a + b}."
    return f"The answer is {a + b}."


def deterministic_reply(message: str) -> str | None:
    """Return an exact tool-style reply when the message matches a known task.

    Supported tasks:
    - Old Turkic/Kokturk rune to Latin transliteration.
    - Cautious response for translation requests involving runes.
    - Simple integer addition.
    """
    text = (message or "").strip()
    if not text:
        return None

    runic = _extract_old_turkic(text)
    if runic and _looks_like_translation_request(text):
        transliteration = rune_to_latin(runic)
        return (
            "I can transliterate these Old Turkic/Kokturk runes into Latin, but "
            "reliable translation to modern Turkish or English needs sourced "
            f"inscription data. Latin transliteration: {transliteration}"
        )
    if runic and _is_transliteration_request(text):
        return rune_to_latin(runic)

    return _math_reply(text)
