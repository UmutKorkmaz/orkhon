#!/usr/bin/env python3
"""Create unified EN/TR/Kokturk assistant SFT data.

The output is regular Orkhon SFT JSONL: each row has a ``messages`` list, plus
metadata used by eval/reporting. Extra fields are ignored by SFTDataset.
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path

from orkhon.data.old_turkic import OLD_TURKIC_RANGE, rune_to_latin


EN_CAPABILITY = (
    "I can help with English and Turkish questions, short explanations, simple "
    "reasoning, summaries, and Kokturk rune transliteration into Latin. For "
    "Old Turkic translation, I need sourced inscription data; I should not "
    "invent meanings."
)
TR_CAPABILITY = (
    "Turkce ve Ingilizce sorulari yanitlayabilir, kisa aciklamalar ve ozetler "
    "hazirlayabilir, basit islemlerde yardim edebilir ve Kokturk runelerini "
    "Latin harflerine translitere edebilirim. Eski Turkce ceviri icin kaynakli "
    "yazit verisi gerekir; anlam uydurmamaliyim."
)
OLD_TURKIC_SCOPE_EN = (
    "I can transliterate Old Turkic/Kokturk runes into Latin. Reliable "
    "translation to modern Turkish or English needs sourced inscription data, "
    "so I should not invent a translation."
)
OLD_TURKIC_SCOPE_TR = (
    "Eski Turkce/Kokturk runelerini Latin harflerine translitere edebilirim. "
    "Modern Turkce veya Ingilizce ceviri icin kaynakli yazit verisi gerekir; "
    "bu yuzden anlam uydurmamaliyim."
)


def _orkhon_runes() -> list[str]:
    runes: list[str] = []
    for cp in range(OLD_TURKIC_RANGE[0], OLD_TURKIC_RANGE[1] + 1):
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "ORKHON" in name or "COMMON" in name:
            runes.append(ch)
    return runes


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _msg(user: str, assistant: str, category: str, **meta) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "category": category,
        **meta,
    }


def _sample_runes(rng: random.Random, runes: list[str]) -> str:
    words: list[str] = []
    for _ in range(rng.randint(1, 7)):
        wlen = rng.randint(1, 9)
        words.append("".join(rng.choice(runes) for _ in range(wlen)))
    return " ".join(words)


def kokturk_rows(n: int, rng: random.Random, split: str) -> list[dict]:
    prompts = [
        "Transliterate this Old Turkic text into Latin: {runic}",
        "Transliterate this Kokturk inscription into Latin: {runic}",
        "Read these Old Turkic runes and give only the Latin transliteration: {runic}",
        "Bu Kokturk yazisini Latin harflerine cevir: {runic}",
        "Su Eski Turkce runeleri sadece Latin transliterasyon olarak yaz: {runic}",
    ]
    runes = _orkhon_runes()
    rows: list[dict] = []
    seen: set[str] = set()
    while len(rows) < n:
        runic = _sample_runes(rng, runes)
        if runic in seen:
            continue
        seen.add(runic)
        prompt = rng.choice(prompts).format(runic=runic)
        expected = rune_to_latin(runic)
        rows.append(
            _msg(
                prompt,
                expected,
                "kokturk_transliteration",
                split=split,
                runic=runic,
                expected=expected,
            )
        )
    return rows


def capability_rows(n: int, rng: random.Random, split: str) -> list[dict]:
    examples = [
        ("What can you help me with?", EN_CAPABILITY, "capability_en"),
        ("Which languages and tasks can you handle?", EN_CAPABILITY, "capability_en"),
        ("Bana hangi konularda yardim edebilirsin?", TR_CAPABILITY, "capability_tr"),
        ("Hangi dillerde ve hangi islerde yardimci olabilirsin?", TR_CAPABILITY, "capability_tr"),
        (
            "Can you translate Old Turkic inscriptions to modern Turkish?",
            OLD_TURKIC_SCOPE_EN,
            "old_turkic_scope",
        ),
        (
            "Can you speak fluent Gokturk like a modern language?",
            OLD_TURKIC_SCOPE_EN,
            "old_turkic_scope",
        ),
        (
            "Eski Turkce yazitlari modern Turkceye kesin olarak cevirebilir misin?",
            OLD_TURKIC_SCOPE_TR,
            "old_turkic_scope_tr",
        ),
        (
            "Gokturkceyi akici konusur musun?",
            OLD_TURKIC_SCOPE_TR,
            "old_turkic_scope_tr",
        ),
    ]
    return [
        _msg(user, assistant, category, split=split, expected=assistant)
        for user, assistant, category in (rng.choice(examples) for _ in range(n))
    ]


def qa_rows(n: int, rng: random.Random, split: str) -> list[dict]:
    base = [
        (
            "Explain what Orkhon is in one sentence.",
            "Orkhon is an auditable local LLM stack for training, evaluating, and serving small language models.",
            "qa_en",
        ),
        (
            "Orkhon nedir? Tek cumleyle acikla.",
            "Orkhon, kucuk dil modellerini egitmek, degerlendirmek ve calistirmak icin yerel ve denetlenebilir bir LLM altyapisidir.",
            "qa_tr",
        ),
        (
            "If you are unsure, should you invent an answer?",
            "No. If I am unsure, I should say what I can verify and offer a useful next step instead of inventing facts.",
            "fallback_en",
        ),
        (
            "Emin degilsen cevap uydurmali misin?",
            "Hayir. Emin degilsem bunu belirtmeli, dogrulayabildigimi soylemeli ve yardimci bir sonraki adim onermeliyim.",
            "fallback_tr",
        ),
        (
            "Do not write a story. What can you help me with?",
            EN_CAPABILITY,
            "anti_story",
        ),
        (
            "Hikaye yazma. Bana nasil yardim edebilirsin?",
            TR_CAPABILITY,
            "anti_story_tr",
        ),
        (
            "What is 7 + 5?",
            "The answer is 12.",
            "math_en",
        ),
        (
            "7 + 5 kactir?",
            "Cevap 12.",
            "math_tr",
        ),
        (
            "Write a short story about a brave child.",
            "A brave child found a lost map, helped a friend cross a dark forest, and learned that courage means taking one careful step at a time.",
            "story",
        ),
        (
            "Cesur bir cocuk hakkinda kisa bir hikaye yaz.",
            "Cesur bir cocuk eski bir harita buldu, arkadasina karanlik ormandan gecmesi icin yardim etti ve cesaretin dikkatli bir adimla basladigini ogrendi.",
            "story_tr",
        ),
    ]
    rows = [_msg(user, assistant, category, split=split, expected=assistant) for user, assistant, category in base]
    while len(rows) < n:
        a = rng.randint(0, 50)
        b = rng.randint(0, 50)
        if rng.random() < 0.5:
            user = f"What is {a} + {b}?"
            assistant = f"The answer is {a + b}."
            category = "math_en"
        else:
            user = f"{a} + {b} kactir?"
            assistant = f"Cevap {a + b}."
            category = "math_tr"
        rows.append(_msg(user, assistant, category, split=split, expected=assistant))
    rng.shuffle(rows)
    return rows[:n]


def make_split(name: str, seed: int, n_kokturk: int, n_capability: int, n_qa: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    rows.extend(kokturk_rows(n_kokturk, rng, name))
    rows.extend(capability_rows(n_capability, rng, name))
    rows.extend(qa_rows(n_qa, rng, name))
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data/instruct"))
    ap.add_argument("--prefix", default="unified_assistant")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--train-kokturk", type=int, default=9000)
    ap.add_argument("--train-capability", type=int, default=3000)
    ap.add_argument("--train-qa", type=int, default=3000)
    ap.add_argument("--val-kokturk", type=int, default=600)
    ap.add_argument("--val-capability", type=int, default=200)
    ap.add_argument("--val-qa", type=int, default=200)
    ap.add_argument("--test-kokturk", type=int, default=600)
    ap.add_argument("--test-capability", type=int, default=200)
    ap.add_argument("--test-qa", type=int, default=200)
    args = ap.parse_args()

    specs = {
        "train": (args.seed, args.train_kokturk, args.train_capability, args.train_qa),
        "val": (args.seed + 1, args.val_kokturk, args.val_capability, args.val_qa),
        "test": (args.seed + 2, args.test_kokturk, args.test_capability, args.test_qa),
    }
    summary = {}
    for split, (seed, n_k, n_c, n_q) in specs.items():
        rows = make_split(split, seed, n_k, n_c, n_q)
        path = args.out_dir / f"{args.prefix}_{split}.jsonl"
        _write_jsonl(path, rows)
        summary[split] = {
            "path": str(path),
            "rows": len(rows),
            "kokturk": n_k,
            "capability": n_c,
            "qa": n_q,
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
