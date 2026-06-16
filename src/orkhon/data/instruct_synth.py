"""Synthesize a story-instruction dataset from a plain story corpus.

Turns a TinyStories-style base model (a free-running story continuer) into an
*instruction follower* by pairing each story with a short, derived instruction
like ``"Write a short story about Lily."``. The instruction is derived
heuristically from the story itself:

1. The first capitalized character name(s) found via a simple regex, else
2. a salient noun (first lowercase content word past a small stoplist), else
3. a generic fallback (``"a little adventure"``).

From each story we emit:

* an **SFT** line ``{"messages": [{user: instruction}, {assistant: story}]}``;
* a **DPO** preference line ``{"prompt": [{user: instruction}], "chosen": story,
  "rejected": bad}`` where ``bad`` is a deterministically-degraded response
  (a truncated story, a different random story, or a sentence-shuffled story).

Everything is seeded and offline: re-running with the same inputs/seed produces
byte-identical files. A small val split is held out and written to ``*_val.jsonl``.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

# --- heuristics knobs -------------------------------------------------------
# Words we never treat as a "name" even if capitalized (sentence starters etc).
_STOP_CAPS = frozenset(
    {
        "The", "A", "An", "Once", "One", "There", "It", "He", "She", "They",
        "We", "I", "You", "His", "Her", "Their", "This", "That", "When",
        "Then", "But", "And", "So", "If", "As", "At", "In", "On", "Of",
        "To", "For", "With", "Day", "Today", "Tomorrow", "Yesterday",
    }
)
# Words skipped when falling back to "a salient noun".
_STOP_WORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "and", "in", "on", "it", "is", "was",
        "were", "be", "this", "that", "with", "for", "as", "at", "by", "but",
        "or", "so", "if", "then", "when", "there", "once", "one", "day",
        "his", "her", "their", "they", "he", "she", "we", "you", "i",
    }
)
_GENERIC_SUBJECT = "a little adventure"
_NAME_RE = re.compile(r"\b([A-Z][a-z]+)\b")
_WORD_RE = re.compile(r"[A-Za-z]+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def _pick_subject(story: str) -> str:
    """Derive a short instruction subject from a story (deterministic).

    Tries a capitalized character name first, then a salient lowercase noun,
    then a generic fallback. Never returns an empty string.
    """
    for match in _NAME_RE.finditer(story):
        word = match.group(1)
        if word not in _STOP_CAPS:
            return word

    for match in _WORD_RE.finditer(story):
        word = match.group(0)
        if word.lower() not in _STOP_WORDS and len(word) > 2:
            return word.lower()

    return _GENERIC_SUBJECT


def _instruction_for(story: str) -> str:
    """Build a short instruction string for a story."""
    subject = _pick_subject(story)
    return f"Write a short story about {subject}."


def _truncate(story: str) -> str:
    """A clearly-worse response: the story cut off partway (incomplete)."""
    words = story.split()
    if len(words) <= 3:
        return words[0] if words else story
    keep = max(2, len(words) // 3)
    return " ".join(words[:keep])


def _shuffle_sentences(story: str, rng: random.Random) -> str:
    """A clearly-worse response: same sentences, scrambled order (incoherent)."""
    sentences = [s.strip() for s in _SENTENCE_RE.findall(story) if s.strip()]
    if len(sentences) < 2:
        return _truncate(story)
    order = list(range(len(sentences)))
    rng.shuffle(order)
    # If the shuffle is a no-op, rotate so the result is actually different.
    if order == sorted(order):
        order = order[1:] + order[:1]
    return " ".join(sentences[i] for i in order)


def _bad_response(
    story: str, all_stories: list[str], idx: int, rng: random.Random
) -> str:
    """Deterministically build a clearly-worse 'rejected' response.

    Rotates among three degradation strategies so the preference signal is
    varied: truncation, a different random story, or sentence shuffling.
    """
    strategy = rng.randint(0, 2)
    if strategy == 0:
        bad = _truncate(story)
    elif strategy == 1 and len(all_stories) > 1:
        # Pick a different story deterministically.
        other = rng.randrange(len(all_stories))
        if other == idx:
            other = (other + 1) % len(all_stories)
        bad = all_stories[other]
    else:
        bad = _shuffle_sentences(story, rng)

    # Guarantee rejected != chosen (degenerate single-word stories etc).
    if bad == story:
        bad = _truncate(story) if _truncate(story) != story else f"{story} ..."
    if bad == story:
        bad = "I do not want to tell a story right now."
    return bad


def _read_stories(corpus_txt: str | Path) -> list[str]:
    """Read non-empty, stripped stories (one per line) from a corpus file."""
    path = Path(corpus_txt)
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    stories: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                stories.append(text)
    if not stories:
        raise ValueError(f"no stories found in {path}")
    return stories


def _write_jsonl(rows: list[dict], path: str | Path) -> None:
    """Write rows as one JSON object per line (UTF-8, stable formatting)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _val_count(n: int, *, fraction: float = 0.05, cap: int = 200) -> int:
    """Size of the held-out val split: ~fraction of n, at least 1, capped."""
    if n <= 1:
        return 0
    return max(1, min(cap, int(round(n * fraction))))


def make_story_instructions(
    corpus_txt: str | Path,
    out_sft_jsonl: str | Path,
    out_dpo_jsonl: str | Path,
    *,
    max_examples: int = 4000,
    seed: int = 1337,
) -> dict[str, int]:
    """Synthesize SFT + DPO instruction datasets from a story corpus.

    Reads stories (one per line) from ``corpus_txt`` and, for each, derives a
    short instruction and emits a matched SFT message pair and a DPO preference
    pair. A small validation split is held out and written alongside the train
    files as ``*_val.jsonl`` (derived from ``out_sft_jsonl`` / ``out_dpo_jsonl``
    by inserting ``_val`` before the suffix).

    Args:
        corpus_txt: Path to a UTF-8 corpus, one story per line.
        out_sft_jsonl: Output path for the SFT train split.
        out_dpo_jsonl: Output path for the DPO train split.
        max_examples: Cap on stories consumed (after which we stop).
        seed: Seed controlling all degradation/shuffling — fully deterministic.

    Returns:
        Counts dict with keys ``stories``, ``sft_train``, ``sft_val``,
        ``dpo_train``, ``dpo_val``.
    """
    stories = _read_stories(corpus_txt)
    if max_examples is not None and len(stories) > max_examples:
        stories = stories[:max_examples]
    n = len(stories)

    # One RNG drives all degradation choices; seeded => deterministic output.
    rng = random.Random(seed)

    sft_rows: list[dict] = []
    dpo_rows: list[dict] = []
    for idx, story in enumerate(stories):
        instruction = _instruction_for(story)
        sft_rows.append(
            {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": story},
                ]
            }
        )
        bad = _bad_response(story, stories, idx, rng)
        dpo_rows.append(
            {
                "prompt": [{"role": "user", "content": instruction}],
                "chosen": story,
                "rejected": bad,
            }
        )

    n_val = _val_count(n)
    n_train = n - n_val

    sft_train, sft_val = sft_rows[:n_train], sft_rows[n_train:]
    dpo_train, dpo_val = dpo_rows[:n_train], dpo_rows[n_train:]

    sft_val_path = _val_path(out_sft_jsonl)
    dpo_val_path = _val_path(out_dpo_jsonl)

    _write_jsonl(sft_train, out_sft_jsonl)
    _write_jsonl(dpo_train, out_dpo_jsonl)
    if sft_val:
        _write_jsonl(sft_val, sft_val_path)
    if dpo_val:
        _write_jsonl(dpo_val, dpo_val_path)

    return {
        "stories": n,
        "sft_train": len(sft_train),
        "sft_val": len(sft_val),
        "dpo_train": len(dpo_train),
        "dpo_val": len(dpo_val),
    }


def _val_path(train_path: str | Path) -> Path:
    """Derive the val-split path by inserting ``_val`` before the suffix."""
    p = Path(train_path)
    return p.with_name(f"{p.stem}_val{p.suffix}")


if __name__ == "__main__":  # pragma: no cover - manual convenience entrypoint
    counts = make_story_instructions(
        "data/tinystories/corpus.txt",
        "data/instruct/tinystories_sft.jsonl",
        "data/instruct/tinystories_dpo.jsonl",
    )
    print(json.dumps(counts, indent=2))
