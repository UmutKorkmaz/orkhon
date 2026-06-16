"""Document loaders — turn files on disk into :class:`Document` objects.

MVP text formats: ``.md .txt .py .yaml .yml .json .rst``. Binary/unknown files are
skipped. Paths are stored relative to a root so citations read ``docs/foo.md`` not
absolute paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from orkhon.rag.types import Document

_TEXT_SUFFIXES = {".md", ".txt", ".py", ".yaml", ".yml", ".json", ".rst", ".toml", ".cfg", ".ini"}


def iter_files(inputs: Iterable[str | Path], *, include=None, exclude=None) -> Iterator[Path]:
    """Yield real files under the given inputs (dirs are walked)."""
    inc = set(include) if include else _TEXT_SUFFIXES
    exc = set(exclude) if exclude else set()
    seen: set[Path] = set()
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            if p.suffix in inc and p.suffix not in exc and p not in seen:
                seen.add(p)
                yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in inc and f.suffix not in exc and f not in seen:
                    seen.add(f)
                    yield f


def load_text(path: str | Path, root: str | Path | None = None) -> Document | None:
    """Read a text file into a Document; ``path`` is stored relative to ``root``."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return None
    rel = str(p.relative_to(root)) if root else str(p)
    return Document(path=rel, text=text, metadata={"bytes": len(text.encode("utf-8"))})
