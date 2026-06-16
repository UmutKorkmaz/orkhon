# bengü-göktürk · 20260615

[English](model_card.md) | [Türkçe](model_card.tr.md)

**instruct** model · **57.4M** parametre · vocab 32768 · 8L · d640 · heads 10/5 · ctx 512

> Göktürk / Eski Türkçe transliteratörü. `bengü` bilingual base, Unicode Old Turkic block U+10C00-U+10C4F'ten
> üretilmiş 6000 deterministik rune→Latin çiftiyle SFT edildi. Model runeleri okuyup Latin transliterasyon
> döndürür. SFT loss 0.01.
>
> **Not:** bu transliterasyondur, çeviri değildir. Rune→Latin deterministiktir; modern Türkçe çeviri için
> kaynaklı akademik veri gerekir.

## Metrikler

- Kayıtlı metrik yok.

## Örnek

**`Transliterate this Old Turkic (Orkhon) text into Latin: 𐰃𐰡𐰞𐰜`**

> ieltaloek

## Çalıştırma

```bash
uv run orkhon chat --checkpoint models/bengu-gokturk-20260615/checkpoint --tokenizer models/bengu-gokturk-20260615/tokenizer
```

## İçerik

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_
