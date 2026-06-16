"""Old Turkic (Göktürk) script support.

The Orkhon inscriptions are the project's namesake. There is **no** corpus large
enough to *pretrain* Old Turkic fluency (the entire attested corpus is on the order
of a few hundred lines), so Orkhon's realistic Göktürk capability is a
**transliteration / translation assistant**, not a fluent speaker.

This package provides the deterministic, authoritative foundation: the rune ⇄ Latin
transliteration map (built from the Unicode Old Turkic block U+10C00–U+10C4F), which
is correct by construction and needs no training. Translation (→ Turkish/English)
requires real scholarly data (Tekin, Erdal, the Uppsala runiform database) and is a
separate, data-gated SFT step on top of the ``bengü`` Turkish base.
"""

from orkhon.data.old_turkic.translit import (
    OLD_TURKIC_RANGE,
    contains_old_turkic,
    rune_inventory,
    rune_to_latin,
)

__all__ = [
    "OLD_TURKIC_RANGE",
    "rune_to_latin",
    "contains_old_turkic",
    "rune_inventory",
]
