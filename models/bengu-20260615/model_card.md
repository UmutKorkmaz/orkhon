# bengu  ·  20260615

**base** model · **57.4M** params · vocab 32768
· 8L · d640 · heads 10/5 · ctx 512

> The Turkish-speaking Orkhon model — a 57M bilingual (EN+TR) base trained from scratch on 136M tokens (~75%% Turkish Wikipedia + 25%% English) with a 32k bilingual tokenizer that cuts Turkish token count ~64%. 14000 steps, val loss 3.88. Fluent, grammatical Wikipedia-register Turkish (correct morphology/agglutination); facts confabulated and some repetition (no repetition penalty yet) — expected at this scale. First step of the 'bengu' (Bengu Tas, eternal stone) Turkic branch; the real one is tengri-350M + Old-Turkic. See docs/turkic-languages.md.

## Metrics

- **val_loss**: 3.8523
- **perplexity**: 47.102
- **eval_tokens**: 163840

## Samples

**"Türkiye'nin başkenti Ankara"**

> 'da bulunan bir futbol takımı. Tarihçe Ankara Futbol Federasyonu'nun kuruluş tarihi olan 1950'li yıllarda Ankara Futbol Federasyonu'nun kuruluş tarihi olan 1950'li yıllarda Ankara Futbol Federasyonu'nun kuruluş tarihi olan 1950'li yıllarda Ankara Futbol Federasyonu'nun kuruluş tarihi olan 1950'li yıllarda Ankara Futbol Federasyonu'nun kuruluş tarihi olan 1950'li yıllarda Ankara Futbol Federasyonu'nun kuruluş tarihi olan 1950'li yıllarda Ankara Futbol Federasyonu'nun kuruluş tarihi olan 1950'li yıllarda Ankara Futbol Federasyonu'nun kuruluş tarihi olan

**"Göktürkler, Orta Asya'da"**

>  yaşayan Türk halklarından biridir. Göktürkler, Orta Asya'da yaşayan Türk halklarından biridir. Göktürkler, Orta Asya'da yaşayan Türk halklarından biridir. Göktürkler, Orta Asya'da yaşayan Türk halklarından biridir. Göktürkler, Orta Asya'da yaşayan Türk halklarından biridir. Göktürkler, Orta Asya'da yaşayan Türk halklarından biridir. Göktürkler, Orta Asya'da yaşayan Türk halklarından biridir. Göktürkler, Orta Asya'da yaşayan Türk

**'Türk mutfağının en ünlü yemeği'**

>  olan "Türk mutfağının en önemli yemeklerinden biri" olarak kabul edilir. Türk mutfağının en önemli yemeklerinden biri olan "Türk mutfağının en önemli yemeklerinden biri" olarak kabul edilir. Türk mutfağının en önemli yemeklerinden biri olan "Türk mutfağının en önemli yemeklerinden biri" olarak kabul edilir. Türk mutfağının en önemli yemeklerinden biri olan "Türk mutfağının en önemli yemeklerinden biri" olarak kabul edilir. Türk mutfağının en önemli yemeklerinden biri olan "Türk mutfağının en önemli yemek

**'The Roman Empire was'**

>  the first Roman Empire to be founded in the Roman Empire. The Roman Empire was the capital of the Roman Empire. The Roman Empire was the capital of the Roman Empire. The Roman Empire was the capital of the Roman Empire. The Roman Empire was the capital of the Roman Empire. The Roman Empire was the capital of the Roman Empire. The Roman Empire was the capital of the Roman Empire. The Roman Empire was the capital of the Roman Empire. The Roman Empire was the capital of the Roman Empire.

## Run it

```bash
uv run orkhon generate --checkpoint models/bengu-20260615/checkpoint --tokenizer models/bengu-20260615/tokenizer -p "Once upon a time"
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_