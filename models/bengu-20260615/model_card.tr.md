# bengü · 20260615

[English](model_card.md) | [Türkçe](model_card.tr.md)

**base** model · **57.4M** parametre · vocab 32768 · 8L · d640 · heads 10/5 · ctx 512

> Türkçe konuşan Orkhon modeli. 136M token üzerinde sıfırdan eğitilmiş 57M EN+TR bilingual base:
> yaklaşık %75 Türkçe Wikipedia + %25 İngilizce. 32k bilingual tokenizer Türkçe token sayısını ciddi biçimde
> düşürür. 14k step, val loss 3.88. Türkçe morphology/agglutination daha iyi; facts confabulated ve repetition
> görülebilir. Gerçek `bengü` hattı `tengri`-350M + Old-Turkic üstünde büyüyecektir.
> Ayrıntı: [`docs/turkic-languages.tr.md`](../../docs/turkic-languages.tr.md).

## Metrikler

- **val_loss**: 3.8523
- **perplexity**: 47.102
- **eval_tokens**: 163840

## Örnekler

**`Türkiye'nin başkenti Ankara`**

> 'da bulunan bir futbol takımı...

**`Göktürkler, Orta Asya'da`**

> yaşayan Türk halklarından biridir...

**`The Roman Empire was`**

> the first Roman Empire to be founded in the Roman Empire...

## Çalıştırma

```bash
uv run orkhon generate --checkpoint models/bengu-20260615/checkpoint --tokenizer models/bengu-20260615/tokenizer -p "Once upon a time"
```

## İçerik

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_
