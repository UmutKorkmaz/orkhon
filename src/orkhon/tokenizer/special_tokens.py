"""Canonical special tokens with STABLE ids.

The order of :data:`SPECIAL_TOKENS` is load-bearing: these tokens are passed to the
BPE trainer first, so they receive ids ``0..7`` deterministically, before the
byte alphabet and merges. Never reorder this list — ids would drift and break every
existing checkpoint/tokenizer. New special tokens must be appended at the end.

Role tokens open a chat message; ``<|end|>`` closes it and is the assistant stop
token. ``<eos>`` separates pretraining documents (distinct from ``<|end|>``).
``<pad>`` is a real id (never aliased to eos) so loss masking is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"

# Tool / image tokens — appended (id 8, 9) for tokenizers trained AFTER this
# change. Older tokenizers (8 specials, ids 0..7) still load: their ids are None.
TOOL = "<|tool|>"
IMAGE = "<image>"

# Order defines ids 0..7 (the original, load-bearing specials — never reorder).
# TOOL/IMAGE are appended so NEW tokenizers get ids 8/9; old ones stay at 0..7.
SPECIAL_TOKENS: list[str] = [PAD, BOS, EOS, UNK, SYSTEM, USER, ASSISTANT, END, TOOL, IMAGE]

ROLE_TOKENS: dict[str, str] = {
    "system": SYSTEM,
    "user": USER,
    "assistant": ASSISTANT,
    "tool": TOOL,  # present only on tokenizers trained with the tool token
}


@dataclass(frozen=True)
class SpecialIds:
    """Resolved integer ids for the special tokens, read from a trained tokenizer."""

    pad: int
    bos: int
    eos: int
    unk: int
    system: int
    user: int
    assistant: int
    end: int
    # Tool/image are OPTIONAL: tokenizers trained before the tool change have only
    # ids 0..7, so these resolve to None and tool use is simply unavailable on them.
    tool: int | None = None
    image: int | None = None

    @property
    def all(self) -> tuple[int, ...]:
        return (self.pad, self.bos, self.eos, self.unk,
                self.system, self.user, self.assistant, self.end)

    def role_id(self, role: str) -> int:
        table = {"system": self.system, "user": self.user, "assistant": self.assistant}
        if role == "tool":
            if self.tool is None:
                raise KeyError("this tokenizer has no <|tool|> token (trained before the tool change)")
            return self.tool
        return table[role]


def special_ids(token_to_id) -> SpecialIds:
    """Build :class:`SpecialIds` from a callable mapping token string -> id.

    ``token_to_id`` is typically ``tokenizers.Tokenizer.token_to_id`` or a dict's
    ``__getitem__``. The eight core specials (ids 0..7) must be present; the tool
    and image specials (ids 8/9) are optional (None on older tokenizers).
    """
    def lookup(tok: str) -> int:
        tid = token_to_id(tok)
        if tid is None:
            raise KeyError(f"special token {tok!r} not present in tokenizer")
        return tid

    def maybe(tok: str) -> int | None:
        tid = token_to_id(tok)
        return None if tid is None else int(tid)

    return SpecialIds(
        pad=lookup(PAD), bos=lookup(BOS), eos=lookup(EOS), unk=lookup(UNK),
        system=lookup(SYSTEM), user=lookup(USER), assistant=lookup(ASSISTANT), end=lookup(END),
        tool=maybe(TOOL), image=maybe(IMAGE),
    )
