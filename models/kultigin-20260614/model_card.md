# kultigin  ·  20260614

**instruct** model · **21.9M** params · vocab 8192
· 6L · d512 · heads 8/4 · ctx 512

> tonyukuk SFT'd on synthetic story instructions — responds to 'Write a short story about ...' with a story. Learns the instruction format (content grounding limited by 22M scale + TinyStories vocabulary).

## Metrics

- (none recorded)

## Samples

**'Write a short story about a brave knight.'**

> Once upon a time, there was a brave knight. He was always looking for ways to help people. One day, he decided to go on an adventure. He packed his bag and went to the forest. He walked and walked until he came to a big tree. He looked up to the top of the tree and saw a little bird. The dragon said, "Hello, little bird. Do

**'Write a short story about a magic garden.'**

> Once upon a time, there was a little girl named Daisy. Daisy was only three years old and she loved to explore the garden. One day, Daisy decided to go for a walk in the garden. As she walked, she saw a big, red apple on the ground. She picked it up and took a big bite. It was so yummy! Daisy wanted to show her friends the apple,

## Run it

```bash
uv run orkhon chat --checkpoint models/kultigin-20260614/checkpoint --tokenizer models/kultigin-20260614/tokenizer
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_