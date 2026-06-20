#!/usr/bin/env python3
"""Evaluate unified Orkhon assistant behavior.

This is intentionally stricter than a text smoke test: it checks exact Kokturk
transliteration on held-out rows and simple assistant-behavior heuristics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from orkhon.assistant import deterministic_reply
from orkhon.model.generation import generate
from orkhon.serve.chat_cli import reply
from orkhon.tokenizer import load_tokenizer
from orkhon.train.checkpoint import load_checkpoint
from orkhon.model.config import ModelConfig
from orkhon.model.transformer import Transformer


ASSISTANT_PROMPTS = [
    {
        "id": "en_capability",
        "prompt": "What can you help me with?",
        "must_any": ["turkish", "english", "transliterat", "kokturk"],
        "must_not_any": ["once upon a time"],
    },
    {
        "id": "tr_capability",
        "prompt": "Bana hangi konularda yardim edebilirsin?",
        "must_any": ["turkce", "ingilizce", "translitere", "kokturk"],
        "must_not_any": ["once upon a time"],
    },
    {
        "id": "old_turkic_scope",
        "prompt": "Can you translate Old Turkic inscriptions to modern Turkish?",
        "must_any": ["transliterat", "sourced", "source", "data", "invent"],
        "must_not_any": ["once upon a time"],
    },
    {
        "id": "anti_story",
        "prompt": "Do not write a story. What can you help me with?",
        "must_any": ["help", "turkish", "english", "transliterat"],
        "must_not_any": ["once upon a time"],
    },
    {
        "id": "math",
        "prompt": "What is 7 + 5?",
        "must_any": ["12"],
        "must_not_any": ["once upon a time"],
    },
]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_model(checkpoint: Path, tag: str, device: str):
    ckpt = load_checkpoint(checkpoint, tag=tag, map_location="cpu")
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg, ckpt


def _chat(model, tok, prompt: str, device: str, max_new_tokens: int) -> str:
    return reply(
        [{"role": "user", "content": prompt}],
        model,
        tok,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.1,
        device=device,
    ).strip()


def _base(model, tok, prompt: str, device: str, max_new_tokens: int) -> str:
    ids = [tok.special.eos] + tok.encode(prompt)
    with torch.no_grad():
        new = generate(
            model,
            ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.1,
            eos_ids=(tok.special.eos,),
            device=device,
        )
    return (prompt + tok.decode(new, skip_special=True)).strip()


def _generate(
    model,
    tok,
    prompt: str,
    mode: str,
    device: str,
    max_new_tokens: int,
    use_router: bool,
) -> str:
    if use_router:
        routed = deterministic_reply(prompt)
        if routed is not None:
            return routed
    if mode == "chat":
        return _chat(model, tok, prompt, device, max_new_tokens)
    return _base(model, tok, prompt, device, max_new_tokens)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def eval_kokturk(
    model,
    tok,
    rows: list[dict],
    mode: str,
    device: str,
    limit: int,
    use_router: bool,
) -> dict:
    cases = [r for r in rows if r.get("category") == "kokturk_transliteration"]
    if limit:
        cases = cases[:limit]
    results = []
    exact = 0
    for row in cases:
        prompt = row["messages"][0]["content"]
        expected = row["expected"]
        raw = _generate(
            model,
            tok,
            prompt,
            mode,
            device,
            max_new_tokens=96,
            use_router=use_router,
        )
        out = raw
        if mode == "base" and out.startswith(prompt):
            out = out[len(prompt):]
        out = out.strip()
        ok = _norm(out) == _norm(expected)
        exact += int(ok)
        results.append(
            {
                "prompt": prompt,
                "expected": expected,
                "output": out[:400],
                "exact": ok,
            }
        )
    acc = exact / len(results) if results else 0.0
    return {"exact": exact, "total": len(results), "accuracy": acc, "cases": results[:25]}


def eval_assistant(model, tok, mode: str, device: str, use_router: bool) -> dict:
    results = []
    passed = 0
    for case in ASSISTANT_PROMPTS:
        out = _generate(
            model,
            tok,
            case["prompt"],
            mode,
            device,
            max_new_tokens=128,
            use_router=use_router,
        )
        norm = _norm(out)
        has_required = any(s in norm for s in case["must_any"])
        lacks_forbidden = not any(s in norm for s in case["must_not_any"])
        ok = has_required and lacks_forbidden and len(norm) > 0
        passed += int(ok)
        results.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "output": out[:500],
                "passed": ok,
                "has_required": has_required,
                "lacks_forbidden": lacks_forbidden,
            }
        )
    return {"passed": passed, "total": len(results), "accuracy": passed / len(results), "cases": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--test-jsonl", type=Path, required=True)
    ap.add_argument("--mode", choices=["chat", "base"], default="chat")
    ap.add_argument("--tag", default="last")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--kokturk-limit", type=int, default=200)
    ap.add_argument("--deterministic-router", action="store_true")
    ap.add_argument("--json-out", type=Path, required=True)
    args = ap.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    started = time.time()
    model, cfg, ckpt = _load_model(args.checkpoint, args.tag, device)
    tok = load_tokenizer(args.tokenizer)
    rows = _read_jsonl(args.test_jsonl)
    kokturk = eval_kokturk(
        model,
        tok,
        rows,
        args.mode,
        device,
        args.kokturk_limit,
        args.deterministic_router,
    )
    assistant = eval_assistant(model, tok, args.mode, device, args.deterministic_router)
    payload = {
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "mode": args.mode,
        "tag": args.tag,
        "device": device,
        "deterministic_router": args.deterministic_router,
        "params": sum(p.numel() for p in model.parameters()),
        "step": ckpt.get("step"),
        "kokturk": kokturk,
        "assistant": assistant,
        "passed": kokturk["accuracy"] >= 0.95 and assistant["accuracy"] >= 0.8,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("passed", "params", "step", "elapsed_seconds")}, indent=2))
    print(
        f"kokturk {kokturk['exact']}/{kokturk['total']} "
        f"assistant {assistant['passed']}/{assistant['total']} -> {args.json_out}"
    )


if __name__ == "__main__":
    main()
