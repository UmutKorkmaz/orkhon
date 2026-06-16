# istemi · 20260614

[English](model_card.md) | [Türkçe](model_card.tr.md)

**base** model · **51.4M** parametre · vocab 16384 · 9L · d640 · heads 10/5 · ctx 512

> 162.8M FineWeb-Edu gerçek web token'ı üzerinde sıfırdan pretrain edilmiş 51M model. Orkhon'un gerçek dünya
> metni üzerindeki ilk modeli. 20k-step run: val loss 3.84, ppl ~46. 200× hedefe göre undertrained; R1 aynı
> boyuta 10B token okutacaktır.

## Metrikler

- **val_loss**: 3.8396
- **perplexity**: 46.506
- **eval_tokens**: 163840

## Örnekler

**`The history of the Roman Empire`**

> is a mystery. The Roman Empire was ...

**`Photosynthesis is the process by which`**

> the plant is grown. The plant is then used ...

## Çalıştırma

```bash
uv run orkhon generate --checkpoint models/istemi-20260614/checkpoint --tokenizer models/istemi-20260614/tokenizer -p "Once upon a time"
```

## İçerik

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_
