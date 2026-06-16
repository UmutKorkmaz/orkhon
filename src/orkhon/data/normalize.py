"""Text normalization and document iteration for corpus preparation.

Normalization is intentionally conservative: NFC unicode form + control-char
stripping (keeping ``\\n`` and ``\\t``) + whitespace trim. We avoid lowercasing or
punctuation munging so the model sees natural text. Documents shorter than a small
threshold are dropped because they add noise without learnable signal.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterator

# Minimum normalized length (chars) for a document to be kept.
MIN_DOC_CHARS = 1

# Control chars to preserve (newline, tab); everything else < 0x20 or DEL is stripped.
_KEEP_CONTROL = {"\n", "\t"}


def normalize_text(text: str) -> str:
    """Normalize a single document: NFC, strip control chars, trim edges."""
    text = unicodedata.normalize("NFC", text)
    cleaned = []
    for ch in text:
        if ch in _KEEP_CONTROL:
            cleaned.append(ch)
            continue
        category = unicodedata.category(ch)
        # Drop control (Cc) and format (Cf) chars; keep everything else.
        if category in ("Cc", "Cf"):
            continue
        cleaned.append(ch)
    return "".join(cleaned).strip()


def _iter_txt_docs(path: Path) -> Iterator[str]:
    """Yield one document per non-empty line of a .txt file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            yield line


def _iter_jsonl_docs(path: Path) -> Iterator[str]:
    """Yield text from a .jsonl file.

    Each line is a JSON object; we extract a ``text`` field if present, otherwise
    fall back to a ``content`` field. Lines without text are skipped.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content")
            if isinstance(text, str):
                yield text


def iter_documents(corpus_path: str | Path, min_chars: int = MIN_DOC_CHARS) -> Iterator[str]:
    """Iterate normalized documents from a corpus file (.txt or .jsonl).

    Short documents (after normalization) are filtered out. The dispatch is by file
    suffix; unknown suffixes are treated as plain text.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"corpus path not found: {path}")

    raw_iter = _iter_jsonl_docs(path) if path.suffix == ".jsonl" else _iter_txt_docs(path)
    for raw in raw_iter:
        doc = normalize_text(raw)
        if len(doc) >= min_chars:
            yield doc
