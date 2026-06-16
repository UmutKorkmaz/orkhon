# tonyukuk  ·  20260614

**base** model · **21.9M** params · vocab 8192
· 6L · d512 · heads 8/4 · ctx 512

> A 22M model pretrained from scratch on 47.4M TinyStories tokens. Writes fluent, coherent short stories. Val loss 1.55, perplexity ~5.3.

## Metrics

- **val_loss**: 1.5622
- **perplexity**: 4.769
- **eval_tokens**: 163840

## Samples

**'Once upon a time, there was a little robot named Bo.'**

>  Bo was a very happy robot. He liked to play with his friends in the park. One day, Bo and his friends were playing a game. They had to find a big ball to play with. Bo was very good at finding things. He found a ball, a toy car, and a toy car. He was very happy. But then, something unexpected happened. A big wind came and blew Bo and his toy car far away. The wind was so strong that it blew Bo and his

**'The dragon was not scary at all.'**

>  It was a big, friendly dragon. He liked to play with kids and make them laugh. He had a big, red nose and big ears. He was very friendly and liked to play with kids. One day, the dragon saw a little girl crying. She was crying and crying. The dragon wanted to help her. He said, "Why are you sad?" The little girl said, "I lost my toy. I can't find it." The dragon wanted to help the little girl.

## Run it

```bash
uv run orkhon generate --checkpoint models/tonyukuk-20260614/checkpoint --tokenizer models/tonyukuk-20260614/tokenizer -p "Once upon a time"
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_