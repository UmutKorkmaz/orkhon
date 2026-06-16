# bumin · 20260614

[English](model_card.md) | [Türkçe](model_card.tr.md)

**instruct** model · **4.2M** parametre · vocab 4096 · 4L · d256 · heads 4/2 · ctx 256

> İlk Orkhon modeli. Sentetik aritmetik Q&A üzerinde SFT edilmiş 4M parametreli sıfırdan Transformer.
> `Answer: N.` chat formatını uçtan uca öğrenir.

## Metrikler

- **val_loss**: 1.6348
- **perplexity**: 5.128
- **eval_tokens**: 20480

## Örnekler

**`What is 2 plus 2?`**

> Answer: 4.

**`What is 7 plus 5?`**

> Answer: 12.

## Çalıştırma

```bash
uv run orkhon chat --checkpoint models/bumin-20260614/checkpoint --tokenizer models/bumin-20260614/tokenizer
```

## İçerik

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_
