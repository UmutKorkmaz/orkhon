"""Benchmark task sets for `orkhon bench`.

Two layers:
- A tiny **built-in** multiple-choice set (English + Turkish) so the harness works
  offline and is unit-testable. It is a *plumbing* signal, not a real benchmark.
- **Adapters** for real public tasks (HellaSwag, ARC-easy, MMLU, TR-MMLU) loaded
  from the HuggingFace Hub via the existing `data/hf_text` path when available.
  Each adapter yields ``{context, choices, label}`` rows consumed by
  :func:`orkhon.eval.loglikelihood.run_multiple_choice`.
"""

from __future__ import annotations

from typing import Iterator, Sequence

# A tiny offline set. The Turkish rows double as a sanity signal for bengü.
BUILTIN: list[dict] = [
    # --- English common sense ---
    {"context": "The capital of France is",
     "choices": [" Paris", " Tokyo", " A banana", " The moon"], "label": 0},
    {"context": "Water boils at",
     "choices": [" 100 degrees Celsius", " -40 degrees", " midnight", " blue"],
     "label": 0},
    {"context": "Birds typically travel through the air by",
     "choices": [" flying", " swimming underground", " singing loudly at",
                 " planting seeds in"],
     "label": 0},
    {"context": "After the sun sets, the sky usually becomes",
     "choices": [" dark", " made of glass", " extremely loud", " frozen solid"],
     "label": 0},
    # --- Turkish common sense ---
    {"context": "Türkiye'nin başkenti",
     "choices": [" Ankara'dır.", " Tokyo'dur.", " Muzdur.", " Ay'dır."], "label": 0},
    {"context": "İnsanlar suyu",
     "choices": [" içer.", " dinlerler.", " boyarlar.", " uçururlar."], "label": 0},
    {"context": "Geceleri gökyüzünde genellikle",
     "choices": [" yıldızlar görünür.", " arabalar yağar.", " müzik çalar.",
                 " deniz donar."],
     "label": 0},
    {"context": "Kediler genellikle",
     "choices": [" miyavlar.", " havlar.", " şarkı söyler.", " uçar."], "label": 0},
]


def builtin_examples(limit: int | None = None) -> list[dict]:
    out = [dict(e) for e in BUILTIN]
    return out[:limit] if limit else out


def _hf_examples(dataset: str, *, name=None, split: str, limit: int | None,
                 mapper) -> list[dict]:
    """Lazy HF-dataset adapter: stream `dataset`, map each row via `mapper`."""
    from datasets import load_dataset

    ds = load_dataset(dataset, name=name, split=split, streaming=True)
    out: list[dict] = []
    for row in ds:
        ex = mapper(row)
        if ex is not None:
            out.append(ex)
        if limit and len(out) >= limit:
            break
    return out


def hellaswag_examples(limit: int | None = None) -> list[dict]:
    """HellaSwag (commonsense NLI). label is the gold ending index."""
    def m(row):
        ctx = row.get("ctx")
        endings = row.get("endings")
        label = row.get("label")
        if not (ctx and endings and label is not None):
            return None
        return {"context": ctx, "choices": list(endings), "label": int(label)}

    return _hf_examples("Rowan/hellaswag", split="validation", limit=limit, mapper=m)


def arc_easy_examples(limit: int | None = None) -> list[dict]:
    """ARC-Easy (grade-school science)."""
    def m(row):
        q = row.get("question")
        choices = row.get("choices")
        ak = row.get("answerKey")
        if not (q and choices and ak):
            return None
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if ak not in labels:
            return None
        return {"context": q + " Answer:", "choices": [" " + t for t in texts],
                "label": labels.index(ak)}

    return _hf_examples("allenai/ai2_arc", name="ARC-Easy", split="validation",
                        limit=limit, mapper=m)


TASKS = {
    "builtin": builtin_examples,
    "hellaswag": hellaswag_examples,
    "arc_easy": arc_easy_examples,
}


def load_task(name: str, limit: int | None = None) -> list[dict]:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name](limit)
