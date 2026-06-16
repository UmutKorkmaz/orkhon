# bumin  ·  20260614

**instruct** model · **4.2M** params · vocab 4096
· 4L · d256 · heads 4/2 · ctx 256

> The first Orkhon model — a 4M from-scratch transformer SFT'd on synthetic arithmetic Q&A. Learns the 'Answer: N.' chat format end-to-end.

## Metrics

- **val_loss**: 1.6348
- **perplexity**: 5.128
- **eval_tokens**: 20480

## Samples

**'What is 2 plus 2?'**

> Answer: 4.

**'What is 7 plus 5?'**

> Answer: 12.

## Run it

```bash
uv run orkhon chat --checkpoint models/bumin-20260614/checkpoint --tokenizer models/bumin-20260614/tokenizer
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_