"""Runtime tokenizer wrapper around a trained HF ``tokenizers`` model.

:class:`OrkhonTokenizer` is the thin, dependency-light object the rest of the stack
uses. It deliberately does NOT add ``<bos>``/``<eos>`` in :meth:`encode` — callers
that need chat scaffolding go through :mod:`orkhon.tokenizer.render`, and the
pretraining packer adds ``<eos>`` document separators explicitly. This keeps a
single source of truth for special-token placement.

CRITICAL: :meth:`encode` strips special-token matching from the content path via a
content-only tokenizer clone, so literal ``<|end|>`` in document text is treated as
ordinary bytes, NOT as a control token.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

from orkhon.tokenizer.special_tokens import SpecialIds, special_ids


class OrkhonTokenizer:
    """BPE tokenizer with resolved special ids and chat-aware helpers.

    Attributes:
        special: resolved :class:`SpecialIds`.
        vocab_size: total vocabulary size (including special tokens).
    """

    def __init__(self, tk: Tokenizer) -> None:
        self._tk = tk
        self.special: SpecialIds = special_ids(self._tk.token_to_id)
        self.vocab_size: int = self._tk.get_vocab_size()
        # Content-only clone: same BPE merges, but NO added (special) tokens, so
        # encode("hello<|end|>world") treats <|end|> as ordinary bytes, not id 7.
        spec = json.loads(tk.to_str())
        spec["added_tokens"] = []
        self._content_tk = Tokenizer.from_str(json.dumps(spec))

    # --- core BPE codec ---------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """BPE-encode ``text`` into token ids WITHOUT adding bos/eos.

        Special-token scaffolding is the caller's responsibility (see
        :mod:`orkhon.tokenizer.render`), so this returns only content ids.
        Literal special-token strings in ``text`` (e.g. ``<|end|>``) are encoded
        as ordinary bytes — they do NOT become control-token ids.
        """
        return self._content_tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids back to text.

        Args:
            ids: token ids to decode.
            skip_special: when True, drop special tokens (pad/bos/eos/role markers).
        """
        return self._tk.decode(list(ids), skip_special_tokens=skip_special)

    # --- vocabulary lookups ----------------------------------------------

    def token_to_id(self, tok: str) -> int | None:
        return self._tk.token_to_id(tok)

    def id_to_token(self, i: int) -> str:
        return self._tk.id_to_token(i)


def load_tokenizer(dir: str | Path) -> OrkhonTokenizer:
    """Load a :class:`OrkhonTokenizer` from a directory containing tokenizer.json."""
    path = Path(dir) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {dir}")
    tk = Tokenizer.from_file(str(path))
    return OrkhonTokenizer(tk)
