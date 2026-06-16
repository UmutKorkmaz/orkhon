"""Old Turkic rune ⇄ Latin transliteration, derived from the Unicode standard.

Every character in the Old Turkic Unicode block (U+10C00–U+10C4F) carries an
authoritative name of the form ``OLD TURKIC LETTER <VARIANT> <VALUE>`` where
``VARIANT`` is ``ORKHON`` / ``YENISEI`` / ``COMMON`` and ``VALUE`` is the scholarly
transliteration value (e.g. ``A``, ``AE``, ``B1``, ``B2``). We build the
rune → Latin map straight from those names, so this layer is **correct by
construction** — no training, no guessing.

Notes on the script (why this is transliteration, not perfect rendering):
- Old Turkic is written right-to-left; Unicode stores it in logical order, so a
  rune string transliterates left-to-right value-by-value.
- Many consonants have **back/back-vowel (¹)** vs **front/front-vowel (²)** forms
  (the ``B1``/``B2`` distinction). Rune → Latin is unambiguous (each rune has a
  value). The reverse (Latin/Turkish → runes) needs vowel-harmony rules and the
  conventional omission of some vowels, which is genuinely hard and is left to a
  learned model + sourced data — not done here.
"""

from __future__ import annotations

import unicodedata

# The Unicode Old Turkic block.
OLD_TURKIC_RANGE = (0x10C00, 0x10C4F)
_PREFIX = "OLD TURKIC LETTER "


def _build_map() -> dict[str, tuple[str, str]]:
    """rune char -> (variant, value), parsed from Unicode names."""
    out: dict[str, tuple[str, str]] = {}
    for cp in range(OLD_TURKIC_RANGE[0], OLD_TURKIC_RANGE[1] + 1):
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if not name.startswith(_PREFIX):
            continue
        rest = name[len(_PREFIX):]
        parts = rest.split(" ", 1)
        if len(parts) == 2:
            variant, value = parts[0], parts[1]
        else:  # no explicit variant
            variant, value = "COMMON", parts[0]
        out[ch] = (variant, value)
    return out


_RUNE_MAP = _build_map()


def rune_inventory() -> dict[str, str]:
    """Return ``{rune: 'VARIANT VALUE'}`` for every Old Turkic rune (for display)."""
    return {ch: f"{v[0].title()} {v[1]}" for ch, v in _RUNE_MAP.items()}


def contains_old_turkic(text: str) -> bool:
    """True if ``text`` contains any character in the Old Turkic block."""
    lo, hi = OLD_TURKIC_RANGE
    return any(lo <= ord(ch) <= hi for ch in text)


def rune_to_latin(text: str, *, sep: str = "") -> str:
    """Transliterate Old Turkic runes in ``text`` to their Latin values.

    Each rune becomes its scholarly value (lowercased; the back/front index is
    kept, e.g. ``b1``/``b2``). Non-rune characters (spaces, Latin, punctuation)
    pass through unchanged. ``sep`` is inserted between consecutive rune values
    (useful for legibility, e.g. ``sep='·'``).
    """
    pieces: list[str] = []
    prev_was_rune = False
    for ch in text:
        if ch in _RUNE_MAP:
            value = _RUNE_MAP[ch][1].lower()
            if prev_was_rune and sep:
                pieces.append(sep)
            pieces.append(value)
            prev_was_rune = True
        else:
            pieces.append(ch)
            prev_was_rune = False
    return "".join(pieces)
