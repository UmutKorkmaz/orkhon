# bumin-mini  ·  20260620

**instruct** model · **4.2M** params · vocab 4096
· 4L · d256 · heads 4/2 · ctx 256

> 4M compact unified EN/TR/Kokturk assistant SFT from the original Bumin smoke checkpoint. Exact rune transliteration is served through the deterministic assistant router.

## Metrics

- (none recorded)

## Samples

**'What can you help me with?'**

> I can help with English and Turkish questions, short explanations, simple reasoning, summaries, and Kokturk run

**'Bana hangi konularda yardim edebilirsin?'**

> Turkce ve Ingilizce sorulari yanitlayabilir, kisa aciklamalar ve ozetler hazirlayabilir, basit

**'Do not write a story. What can you help me with?'**

> I can help with English and Turkish questions, short explanations, simple reasoning, summaries, and Kokturk run

## Run it

```bash
uv run orkhon chat --checkpoint models/bumin-mini-20260620/checkpoint --tokenizer models/bumin-mini-20260620/tokenizer
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_