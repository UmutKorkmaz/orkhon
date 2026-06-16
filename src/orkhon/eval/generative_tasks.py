"""Generative task adapters: GSM8K (math) and MBPP (code), + tiny offline fixtures."""

from __future__ import annotations

from typing import Sequence

# Tiny offline fixtures so tests + smoke run without network. Real runs stream
# the public datasets via the `hub` extra (see _hf_examples).
GSM8K_FIXTURE: list[dict] = [
    {"id": 0, "prompt": "Janet has 2 apples and buys 3 more. How many apples? End with \\boxed{}.",
     "gold": "5"},
    {"id": 1, "prompt": "What is 7 * 6? End with \\boxed{}.", "gold": "42"},
    {"id": 2, "prompt": "A train goes 60 km in 1 hour. How far in 2 hours? \\boxed{}.",
     "gold": "120"},
]
MBPP_FIXTURE: list[dict] = [
    {"id": 0, "prompt": "Write a function `add(a, b)` returning a+b.",
     "tests": ["add(2,3) == 5"]},
    {"id": 1, "prompt": "Write `sq(x)` returning x squared.",
     "tests": ["sq(4) == 16"]},
]


def _hf_examples(dataset: str, *, split: str, limit: int | None, mapper) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    out = []
    for i, row in enumerate(ds):
        ex = mapper(row, i)
        if ex is not None:
            out.append(ex)
        if limit and len(out) >= limit:
            break
    return out


def gsm8k_examples(limit: int | None = None, *, split: str = "test",
                   fixture: bool = False) -> list[dict]:
    if fixture:
        return [dict(e) for e in GSM8K_FIXTURE]
    def m(row, i):
        q = row.get("question")
        a = row.get("answer", "")
        # GSM8K answers end with "#### <number>".
        import re
        mm = re.search(r"####\s*(-?[\d,]+)", a)
        if not (q and mm):
            return None
        gold = mm.group(1).replace(",", "")
        return {"id": i, "prompt": q + "\n\nEnd your answer with \\boxed{}.", "gold": gold}
    return _hf_examples("openai/gsm8k", split=split, limit=limit, mapper=m)


def mbpp_examples(limit: int | None = None, *, split: str = "test",
                  fixture: bool = False) -> list[dict]:
    if fixture:
        return [dict(e) for e in MBPP_FIXTURE]
    def m(row, i):
        code = row.get("code") or ""
        prompts = row.get("test_list") or []
        text = row.get("prompt") or row.get("text") or ""
        if not (text and prompts):
            return None
        # tests are python assert strings already
        tests = [t.replace("assert ", "").strip() if t.startswith("assert ") else t
                 for t in prompts]
        return {"id": i, "prompt": text, "tests": tests}
    return _hf_examples("mbpp", split=split, limit=limit, mapper=m)


GEN_TASKS = {"gsm8k": gsm8k_examples, "mbpp": mbpp_examples}


def load_generative_task(name: str, limit: int | None = None, *, fixture: bool = False) -> list[dict]:
    if name not in GEN_TASKS:
        raise ValueError(f"unknown generative task {name!r}; choose from {sorted(GEN_TASKS)}")
    return GEN_TASKS[name](limit, fixture=fixture)
