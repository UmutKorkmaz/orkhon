# istemi  ·  20260614

**base** model · **51.4M** params · vocab 16384
· 9L · d640 · heads 10/5 · ctx 512

> A 51M model pretrained from scratch on 162.8M FineWeb-Edu (real web) tokens — the first Orkhon model on real-world knowledge. Full 20k-step run: val loss 3.84 (ppl ~46). Undertrained vs the 200x target (only ~3 tok/param); R1 would feed it 10B tokens.

## Metrics

- **val_loss**: 3.8396
- **perplexity**: 46.506
- **eval_tokens**: 163840

## Samples

**'The history of the Roman Empire'**

>  is a mystery. The Roman Empire was aric society, and the Roman Empire was a part of the Roman Empire. The Roman Empire was a part of the Roman Empire, and the Roman Empire was a part of the Roman Empire. The Roman Empire was a part of the Roman Empire, and the Roman Empire was a part of the Roman Empire. The Roman Empire was a part of the Roman Empire, and the Roman Empire was a part of the Roman Empire. The Roman Empire was a part

**'Photosynthesis is the process by which'**

>  the plant is grown. The plant is then used to produce the plant. The plant is then used to produce the plant. The plant is then used to produce the plant. The plant is then used to produce the plant. The plant is then used to produce the plant. The plant is then used to produce the plant. The plant is then used to produce the plant. The plant is then used to produce the plant. The plant is then used to produce the plant. The plant is then used

**'To make bread, you first'**

>  need to make bread, then you need to make bread. The first thing you need to make bread is to make bread. The second thing you need to make bread is to make bread. The first thing you need to make bread is to make bread. The second thing you need to make bread is to make bread. The first thing you need to make bread is to make bread. The first thing you need to make bread is to make bread. The first thing you need to make bread is to

## Run it

```bash
uv run orkhon generate --checkpoint models/istemi-20260614/checkpoint --tokenizer models/istemi-20260614/tokenizer -p "Once upon a time"
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_