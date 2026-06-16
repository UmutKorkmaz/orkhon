"""Stream real, diverse text from the Hugging Face Hub into Orkhon's corpus format.

Output format: one document per line in a ``.txt`` file, with internal whitespace
flattened to single spaces. This is exactly what the tokenizer trainer
(``_iter_corpus_lines``) and the data packer (``_iter_txt_docs``) already consume,
so a streamed corpus drops straight into the existing pipeline (same convention as
``data/download.py``).

Unlike ``download_tinystories`` (which streams a single raw ``.txt`` over HTTP),
this module uses ``datasets.load_dataset(..., streaming=True)`` to iterate arbitrary
Hub text datasets document-by-document without materializing them on disk. The
flagship target is ``HuggingFaceFW/fineweb-edu`` (sample-10BT), a large, high-quality
web corpus that serves as the backbone for the 50M-local and 125M-cloud runs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

# Collapse all runs of whitespace (newlines, tabs, repeated spaces) to one space.
_WS = re.compile(r"\s+")

# Emit a progress callback every N written documents.
_PROGRESS_EVERY = 10000

# Convenience descriptor for the flagship FineWeb-Edu sample shard.
FINEWEB_EDU = dict(
    dataset="HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    text_field="text",
)

# Turkish Wikipedia — clean, non-gated; the entry point for Turkish data (a
# bilingual tokenizer + a Turkish-capable model). For scale, FineWeb-2 `tur_Latn`
# (HuggingFaceFW/fineweb-2) and CulturaX `tr` are the big web pools.
WIKIPEDIA_TR = dict(
    dataset="wikimedia/wikipedia",
    name="20231101.tr",
    text_field="text",
)


def _flatten(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and trim.

    Mirrors ``data/download._flatten`` so every corpus writer produces identical
    one-line-per-document output.
    """
    return _WS.sub(" ", text).strip()


def stream_hf_text(
    dataset: str,
    out_path: str | Path,
    *,
    name: Optional[str] = None,
    split: str = "train",
    text_field: str = "text",
    max_docs: Optional[int] = None,
    min_chars: int = 200,
    on_progress: Optional[Callable[[int], None]] = None,
) -> int:
    """Stream a Hub text dataset and write one document per line to ``out_path``.

    Loads ``dataset`` (optionally a config ``name``) in streaming mode so nothing
    is materialized on disk, iterates rows, extracts ``obj[text_field]``, flattens
    whitespace to single spaces, skips documents shorter than ``min_chars`` (measured
    after flattening), and writes one document per line to ``out_path`` (a ``.txt``
    file). Stops once ``max_docs`` documents have been written.

    Args:
        dataset: Hub dataset id, e.g. ``"HuggingFaceFW/fineweb-edu"``.
        out_path: Destination ``.txt`` path; parent dirs are created.
        name: Optional dataset config name (e.g. ``"sample-10BT"``).
        split: Dataset split to stream (default ``"train"``).
        text_field: Key in each row holding the document text (default ``"text"``).
        max_docs: Maximum documents to write; ``None`` means no cap.
        min_chars: Skip documents whose flattened length is below this threshold.
        on_progress: Optional callback invoked with the running written count.

    Returns:
        The number of documents written to ``out_path``.
    """
    # Imported lazily so importing this module (and the package) never requires
    # ``datasets`` to be present, and so tests can monkeypatch the symbol.
    from datasets import load_dataset

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(dataset, name, split=split, streaming=True)

    written = 0
    with open(out, "w", encoding="utf-8") as w:
        for obj in stream:
            if max_docs is not None and written >= max_docs:
                break
            raw = obj.get(text_field) if isinstance(obj, dict) else None
            if not raw:
                continue
            line = _flatten(raw)
            if len(line) < min_chars:
                continue
            w.write(line + "\n")
            written += 1
            if on_progress is not None and written % _PROGRESS_EVERY == 0:
                on_progress(written)
            if max_docs is not None and written >= max_docs:
                break

    return written


def download_wikipedia_tr(
    out_path: str | Path,
    max_docs: Optional[int] = None,
    on_progress: Optional[Callable[[int], None]] = None,
) -> int:
    """Stream Turkish Wikipedia into Orkhon's corpus format."""
    d = WIKIPEDIA_TR
    return stream_hf_text(
        d["dataset"], out_path, name=d["name"], split="train",
        text_field=d["text_field"], max_docs=max_docs, min_chars=200,
        on_progress=on_progress,
    )


def download_fineweb_edu(
    out_path: str | Path,
    max_docs: Optional[int] = None,
    on_progress: Optional[Callable[[int], None]] = None,
) -> int:
    """Stream the FineWeb-Edu sample shard into Orkhon's corpus format.

    Thin convenience wrapper over :func:`stream_hf_text` pinned to the
    :data:`FINEWEB_EDU` descriptor (``HuggingFaceFW/fineweb-edu``, config
    ``sample-10BT``, text field ``text``).

    Args:
        out_path: Destination ``.txt`` path; parent dirs are created.
        max_docs: Maximum documents to write; ``None`` means no cap.
        on_progress: Optional callback invoked with the running written count.

    Returns:
        The number of documents written to ``out_path``.
    """
    return stream_hf_text(
        FINEWEB_EDU["dataset"],
        out_path,
        name=FINEWEB_EDU["name"],
        text_field=FINEWEB_EDU["text_field"],
        max_docs=max_docs,
        on_progress=on_progress,
    )
