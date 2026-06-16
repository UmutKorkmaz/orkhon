"""Project path helpers and run-directory creation."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repo root = three levels up from this file (src/orkhon/utils/paths.py)."""
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(name: str, base: str | Path = "runs") -> Path:
    """Create and return runs/<name>/ (caller supplies a unique name)."""
    return ensure_dir(Path(base) / name)
