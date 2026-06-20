# tonyuk  ·  20260620

**instruct** model · **21.9M** params · vocab 8192
· 6L · d512 · heads 8/4 · ctx 512

> 22M unified EN/TR/Kokturk assistant SFT from the Tonyukuk TinyStories base. Exact rune transliteration is served through the deterministic assistant router.

## Metrics

- (none recorded)

## Samples

**'What can you help me with?'**

> I can help with English and Turkish questions, short explanations, simple reasoning, summaries, and Kokturk rune transliteration into Latin. For Old Turkic translation, I need sourced inscription data; I should not invent meanings

**'Bana hangi konularda yardim edebilirsin?'**

> Turkce ve Ingilizce sorulari yanitlayabilir, kisa aciklamalar ve ozetler hazirlayabilir, basit islemlerde yardim edebilir ve Kokturk runelerini Latin harflerine translitere ed

**'Do not write a story. What can you help me with?'**

> I can help with English and Turkish questions, short explanations, simple reasoning, summaries, and Kokturk rune transliteration into Latin. For Old Turkic translation, I need sourced inscription data; I should not invent meanings

## Run it

```bash
uv run orkhon chat --checkpoint models/tonyuk-20260620/checkpoint --tokenizer models/tonyuk-20260620/tokenizer
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_