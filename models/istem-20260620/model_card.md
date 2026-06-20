# istem  ·  20260620

**instruct** model · **51.4M** params · vocab 16384
· 9L · d640 · heads 10/5 · ctx 512

> 51M unified EN/TR/Kokturk assistant SFT from the FineWeb-Edu base. Exact rune transliteration is served through the deterministic assistant router.

## Metrics

- (none recorded)

## Samples

**'What can you help me with?'**

> I can help with English and Turkish questions, short explanations, simple reasoning, summaries, and Kokturk rune transliteration into Latin. For Old Turkic translation, I need sourced inscription data; I should not invent meanings.

**'Bana hangi konularda yardim edebilirsin?'**

> Turkce ve Ingilizce sorulari yanitlayabilir, kisa aciklamalar ve ozetler hazirlayabilir, basit islemlerde yardim edebilir ve Kokturk runelerini Latin harflerine translitere edebilirim. Eski Turkce

**'Do not write a story. What can you help me with?'**

> I can help with English and Turkish questions, short explanations, simple reasoning, summaries, and Kokturk rune transliteration into Latin. For Old Turkic translation, I need sourced inscription data; I should not invent meanings.

## Run it

```bash
uv run orkhon chat --checkpoint models/istem-20260620/checkpoint --tokenizer models/istem-20260620/tokenizer
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_