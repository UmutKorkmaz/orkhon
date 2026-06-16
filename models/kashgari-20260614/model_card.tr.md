# kashgari · 20260614

[English](model_card.md) | [Türkçe](model_card.tr.md)

**imported** model · **135.0M** parametre

> 135M Llama-architecture açık base modelinin Orkhon'un elle yazılmış Transformer'ına exact logit parity ile
> yüklenmiş hali. Gerektiğinde yeniden import edilir; weights bu arşivde yeniden saklanmaz.

## Metrikler

- Kayıtlı metrik yok.

## Örnek

**`The capital of France is`**

> the capital of the country.

## Yeniden import

```bash
uv run orkhon import-hf --repo HuggingFaceTB/SmolLM2-135M --out runs/kashgari
```

## İçerik

- `checkpoint/` — import sonrası üretilecek inference weights + `model_config.json`
- kaynak repo tokenizer'ı kullanılır
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_
