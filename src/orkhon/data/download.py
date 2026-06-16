"""Download real public corpora into Orkhon's corpus format.

Output format: one document per line in a ``.txt`` file (internal whitespace
flattened to single spaces). This is exactly what both the tokenizer trainer
(``_iter_corpus_lines``) and the data packer (``_iter_txt_docs``) already consume,
so a downloaded corpus drops straight into the existing pipeline.

TinyStories (Eldan & Li, 2023) is a synthetic corpus of short, simple stories
designed so that even very small models learn fluent, coherent English. Stories
in the raw file are separated by an ``<|endoftext|>`` marker.
"""

from __future__ import annotations

import codecs
import re
import ssl
import urllib.request
from pathlib import Path


def _ssl_context() -> ssl.SSLContext:
    """SSL context with a real CA bundle.

    The macOS python.org interpreter ships without system CA certs, so a plain
    urlopen of an https URL fails verification. Prefer certifi's bundle when
    available, falling back to the platform default.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

TINYSTORIES_URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}
_STORY_SEP = "<|endoftext|>"
_WS = re.compile(r"\s+")

# Read 1 MiB per network chunk.
_CHUNK = 1 << 20


def _flatten(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and trim."""
    return _WS.sub(" ", text).strip()


def download_tinystories(
    out_path: str | Path,
    split: str = "train",
    max_stories: int | None = None,
    on_progress=None,
) -> int:
    """Stream TinyStories and write one story per line to ``out_path``.

    Streams the (large) raw file, splits on ``<|endoftext|>`` incrementally so
    memory stays flat, and stops early once ``max_stories`` have been written.
    Returns the number of stories written.
    """
    if split not in TINYSTORIES_URLS:
        raise ValueError(f"split must be one of {sorted(TINYSTORIES_URLS)}, got {split!r}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    written = 0

    req = urllib.request.Request(
        TINYSTORIES_URLS[split], headers={"User-Agent": "orkhon-data/0.1"}
    )
    ctx = _ssl_context()
    with urllib.request.urlopen(req, context=ctx) as resp, open(out, "w", encoding="utf-8") as w:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            buffer += decoder.decode(chunk)
            parts = buffer.split(_STORY_SEP)
            buffer = parts.pop()  # last piece may be an incomplete story
            for story in parts:
                line = _flatten(story)
                if line:
                    w.write(line + "\n")
                    written += 1
                    if on_progress is not None and written % 10000 == 0:
                        on_progress(written)
                    if max_stories is not None and written >= max_stories:
                        return written
        # Flush a final complete story in the tail.
        buffer += decoder.decode(b"", final=True)
        line = _flatten(buffer)
        if line and (max_stories is None or written < max_stories):
            w.write(line + "\n")
            written += 1

    return written
