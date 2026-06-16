# kultigin · 20260614

[English](model_card.md) | [Türkçe](model_card.tr.md)

**instruct** model · **21.9M** parametre · vocab 8192 · 6L · d512 · heads 8/4 · ctx 512

> `tonyukuk` modelinin sentetik hikaye komutlarıyla SFT edilmiş hali. `Write a short story about ...` gibi
> komutlara hikaye ile cevap verir. 22M ölçek ve TinyStories vocab'ı nedeniyle içerik grounding'i sınırlıdır.

## Metrikler

- Kayıtlı metrik yok.

## Örnekler

**`Write a short story about a brave knight.`**

> Once upon a time, there was a brave knight...

**`Write a short story about a magic garden.`**

> Once upon a time, there was a little girl named Daisy...

## Çalıştırma

```bash
uv run orkhon chat --checkpoint models/kultigin-20260614/checkpoint --tokenizer models/kultigin-20260614/tokenizer
```

## İçerik

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_
