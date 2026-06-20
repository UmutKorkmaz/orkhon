# tangri  ·  20260620

**instruct** model · **100.7M** params · vocab 32768
· 12L · d768 · heads 12/4 · ctx 1024

> 100M unified EN/TR/Kokturk assistant SFT from the mixed Tangri base; exact rune transliteration is served through the deterministic assistant router.

## Metrics

- (none recorded)

## Samples

**'What can you help me with?'**

> I can help with English and Turkish questions, short explanations, simple reasoning, summaries, and Kokturk rune transliteration into Latin. For Old Turkic translation, I need sourced inscription data; I should not invent meanings.

**'Can you translate Old Turkic inscriptions to modern Turkish?'**

> I can transliterate Old Turkic/Kokturk runes into Latin. Reliable translation to modern Turkish or English needs sourced inscription data, so I should not invent a translation.

**'Do not write a story. What can you help me with?'**

> I can help with English and Turkish questions, short explanations, simple reasoning, summaries, and Kokturk rune transliteration into Latin. For Old Turkic translation, I need sourced inscription data; I should not invent meanings.

## Run it

```bash
uv run orkhon chat --checkpoint models/tangri-20260620/checkpoint --tokenizer models/tangri-20260620/tokenizer
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_