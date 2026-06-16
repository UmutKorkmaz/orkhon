# tonyukuk · 20260614

[English](model_card.md) | [Türkçe](model_card.tr.md)

**base** model · **21.9M** parametre · vocab 8192 · 6L · d512 · heads 8/4 · ctx 512

> 47.4M TinyStories token üzerinde sıfırdan pretrain edilmiş 22M model. Akıcı ve tutarlı kısa hikayeler yazar.
> Val loss 1.55, perplexity yaklaşık 5.3.

## Metrikler

- **val_loss**: 1.5622
- **perplexity**: 4.769
- **eval_tokens**: 163840

## Örnekler

**`Once upon a time, there was a little robot named Bo.`**

> Bo was a very happy robot. He liked to play with his friends in the park...

**`The dragon was not scary at all.`**

> It was a big, friendly dragon. He liked to play with kids and make them laugh...

## Çalıştırma

```bash
uv run orkhon generate --checkpoint models/tonyukuk-20260614/checkpoint --tokenizer models/tonyukuk-20260614/tokenizer -p "Once upon a time"
```

## İçerik

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_
