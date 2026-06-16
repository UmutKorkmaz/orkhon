"""Serialize evaluation results to JSON.

A tiny, dependency-light helper so every eval entrypoint (perplexity, smoke,
chat scoring) writes its results in a consistent, machine-readable shape that the
model-card generator and CI gates can read back.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_report(results: dict[str, Any], out_path: str | Path) -> Path:
    """Write ``results`` (plus a UTC timestamp) to ``out_path`` as JSON.

    Args:
        results: a JSON-serializable dict of metrics/outputs.
        out_path: destination ``.json`` file (parent dirs are created).

    Returns:
        The path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(results)
    payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())

    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return out_path


def read_report(path: str | Path) -> dict[str, Any]:
    """Load an eval report JSON written by :func:`write_report`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
