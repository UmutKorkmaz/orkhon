"""Orkhon family chat Space — every live model behind one ChatInterface.

A single gradio.ChatInterface with a model selector. Each family member is
loaded on first use (lru_cache) via ``orkhon.export.to_hf.load_exported_model``
from its HF repo. Chat members (``chat_cli.reply``) hold a real conversation;
base members continue the prompt (they "finish the thought").

API contract for orkhon.umutkorkmaz.net's /api/chat proxy:
    respond(message: str, history: list, member_key: str) -> assistant text
The site calls  predict("respond", [message, history, member_key]).

Local test (loads the already-exported folders, no HF download):
    ORKHON_FAMILY_LOCAL_ROOT=exports/huggingface \
      uv run --extra demo python spaces/orkhon-demo/app.py
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("orkhon.space")

HF_OWNER = os.environ.get("ORKHON_HF_OWNER", "korkmazumut")
LOCAL_ROOT = os.environ.get("ORKHON_FAMILY_LOCAL_ROOT")  # point at exports/huggingface to test locally
DEVICE = os.environ.get("ORKHON_DEVICE", "cpu")

# Family registry. Normal public members are chat assistants; exact Kokturk
# transliteration and simple arithmetic are routed deterministically before the
# model is loaded.
FAMILY = {
    "tangri": {
        "repo": f"{HF_OWNER}/orkhon-tangri",
        "local": "orkhon-tangri",
        "interface": "chat",
        "note": "100M unified EN/TR/Kokturk assistant.",
    },
    "bunghu": {
        "repo": f"{HF_OWNER}/orkhon-bunghu",
        "local": "orkhon-bunghu",
        "interface": "chat",
        "note": "57M unified assistant from the bilingual branch.",
    },
    "tegin": {
        "repo": f"{HF_OWNER}/orkhon-tegin",
        "local": "orkhon-tegin",
        "interface": "chat",
        "note": "22M unified assistant from the former story-instruct branch.",
    },
    "tonyuk": {
        "repo": f"{HF_OWNER}/orkhon-tonyuk",
        "local": "orkhon-tonyuk",
        "interface": "chat",
        "note": "22M unified assistant from the story base.",
    },
    "istem": {
        "repo": f"{HF_OWNER}/orkhon-istem",
        "local": "orkhon-istem",
        "interface": "chat",
        "note": "51M unified assistant from the FineWeb-Edu base.",
    },
    "bumin-mini": {
        "repo": f"{HF_OWNER}/orkhon-bumin-mini",
        "local": "orkhon-bumin-mini",
        "interface": "chat",
        "note": "4M compact unified assistant smoke model.",
    },
    "kashgar": {
        "repo": f"{HF_OWNER}/orkhon-kashgar",
        "local": "orkhon-kashgar",
        "interface": "chat",
        "note": "Imported/open-base assistant slot once weights are archived.",
    },
}

DEFAULT_MEMBER = "tangri"


_ADD_RE = re.compile(
    r"(?:what\s+is\s+)?(?P<a>\d{1,4})\s*\+\s*(?P<b>\d{1,4})(?:\s*(?:\?|kactir|kaçtır))?",
    re.IGNORECASE,
)


def _deterministic_reply(message: str) -> str | None:
    try:
        from orkhon.assistant import deterministic_reply

        return deterministic_reply(message)
    except Exception:
        pass

    from orkhon.data.old_turkic import OLD_TURKIC_RANGE, rune_to_latin

    text = (message or "").strip()
    runes = "".join(
        ch if OLD_TURKIC_RANGE[0] <= ord(ch) <= OLD_TURKIC_RANGE[1] else " "
        for ch in text
    )
    runes = " ".join(runes.split())
    lower = text.lower()
    if runes and any(s in lower for s in ("transliterate", "latin", "rune", "kokturk", "gokturk", "old turkic", "eski turkce", "harflerine", "cevir", "oku")):
        return rune_to_latin(runes)
    if runes and any(s in lower for s in ("translate", "translation", "modern turkish", "ceviri", "çeviri", "tercume", "tercüme")):
        return (
            "I can transliterate these Old Turkic/Kokturk runes into Latin, but "
            "reliable translation to modern Turkish or English needs sourced "
            f"inscription data. Latin transliteration: {rune_to_latin(runes)}"
        )

    match = _ADD_RE.search(text)
    if match:
        a = int(match.group("a"))
        b = int(match.group("b"))
        if "kactir" in lower or "kaçtır" in lower:
            return f"Cevap {a + b}."
        return f"The answer is {a + b}."
    return None


def _model_path(member_key: str) -> str:
    member = FAMILY[member_key]
    if LOCAL_ROOT:
        candidate = Path(LOCAL_ROOT) / member["local"]
        if candidate.exists():
            return str(candidate)
    from huggingface_hub import snapshot_download

    return snapshot_download(member["repo"])


@lru_cache(maxsize=8)
def _load(member_key: str):
    """Load + cache (model, tokenizer) for a family member."""
    from orkhon.export.to_hf import load_exported_model
    from orkhon.tokenizer import load_tokenizer

    path = _model_path(member_key)
    model, _cfg = load_exported_model(path, device=DEVICE)
    model.eval()
    tokenizer = load_tokenizer(path)
    log.info("loaded %s from %s", member_key, path)
    return model, tokenizer


def _history_to_msgs(history) -> list[dict[str, str]]:
    """Normalize gradio history (dict or tuple form) into chat messages."""
    msgs: list[dict[str, str]] = []
    for h in history or []:
        if isinstance(h, dict):
            role = h.get("role") or "user"
            content = h.get("content") or ""
            if content:
                msgs.append({"role": role, "content": content})
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            user, assistant = h[0], h[1]
            if user:
                msgs.append({"role": "user", "content": str(user)})
            if assistant:
                msgs.append({"role": "assistant", "content": str(assistant)})
    return msgs


def _complete(member_key: str, prompt: str) -> str:
    from orkhon.model.generation import generate

    model, tok = _load(member_key)
    ids = [tok.special.eos] + tok.encode(prompt)
    # Decoding for a small high-perplexity base model: moderate temperature
    # (low temp locks a weak model onto wrong tokens / repetition death-spiral),
    # top_k 40 (diffuse small-model distributions need hard truncation, not pure
    # top-p), and repetition_penalty to kill the degenerate loops seen in the
    # training samples. This is the research-aligned band (Chip Huyen / arXiv
    # 2509.23234); it CANNOT make a 57M/ppl-47 model factual — that ceiling is
    # the model, not the sampler.
    new_ids = generate(
        model,
        ids,
        max_new_tokens=160,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.3,
        eos_ids=(tok.special.eos,),
        device=DEVICE,
    )
    return prompt + tok.decode(new_ids, skip_special=True)


def _chat(member_key: str, message: str, history) -> str:
    from orkhon.serve.chat_cli import reply

    model, tok = _load(member_key)
    msgs = _history_to_msgs(history)
    msgs.append({"role": "user", "content": message})
    return reply(
        msgs,
        model,
        tok,
        max_new_tokens=200,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.3,
        device=DEVICE,
    )


def respond(message: str, history, member_key: str = DEFAULT_MEMBER):
    """ChatInterface entry point. Routes to chat or completion per member."""
    message = (message or "").strip()
    if not message:
        return ""
    routed = _deterministic_reply(message)
    if routed is not None:
        return routed
    if member_key not in FAMILY:
        member_key = DEFAULT_MEMBER
    member = FAMILY[member_key]
    try:
        if member["interface"] == "chat":
            return _chat(member_key, message, history)
        return _complete(member_key, message)
    except Exception as exc:  # never crash the chat; surface a readable error
        log.exception("respond failed for %s", member_key)
        return f"[{member_key} could not respond: {exc}]"


def _build_demo():
    import gradio as gr

    with gr.Blocks(title="Orkhon model family") as demo:
        gr.Markdown(
            "# Orkhon model family\n"
            "An auditable from-scratch LLM stack. Pick a voice and write to it — "
            "chat members answer; base members continue your text."
        )
        chat = gr.ChatInterface(
            respond,
            additional_inputs=[
                gr.Dropdown(
                    choices=list(FAMILY.keys()),
                    value=DEFAULT_MEMBER,
                    label="Model",
                    info="Chat members answer; base (completion) members continue your text.",
                )
            ],
            examples=[
                ["What can you help me with?", "tangri"],
                ["Transliterate this Old Turkic text into Latin: 𐰃𐰡𐰞𐰜", "tangri"],
            ],
        )
    return demo


if __name__ == "__main__":
    _build_demo().launch()
