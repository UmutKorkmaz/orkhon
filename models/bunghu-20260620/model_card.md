# bunghu  ·  20260620

**instruct** model · **57.4M** params · vocab 32768
· 8L · d640 · heads 10/5 · ctx 512

> 57M unified EN/TR/Kokturk assistant SFT from the bilingual branch. Exact rune transliteration is served through the deterministic assistant router.

## Metrics

- (none recorded)

## Samples

**'What can you help me with?'**

> I can help with English and Turkish questions, short explanations, simple reasoning, summaries, and Kokturk rune transliteration into Latin. For Old Turkic translation, I need sourced inscription data; I should not invent meanings.

**'Bana hangi konularda yardim edebilirsin?'**

> Turkce ve Ingilizce sorulari yanitlayabilir, kisa aciklamalar ve ozetler hazirlayabilir, basit islemlerde yardim edebilir ve Kokturk runelerini Latin harflerine translitere edebilirim. Eski Turkce ceviri icin kaynakli yazit verisi gerekir; anlam uydurmamaliyim.

**'Do not write a story. What can you help me with?'**

> I can help with English and Turkish questions, short explanations, simple reasoning, summaries, and Kokturk rune transliteration into Latin. For Old Turkic translation, I need sourced inscription data; I should not invent meanings.

## Run it

```bash
uv run orkhon chat --checkpoint models/bunghu-20260620/checkpoint --tokenizer models/bunghu-20260620/tokenizer
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_