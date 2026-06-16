# bengu-gokturk  ·  20260615

**instruct** model · **57.4M** params · vocab 32768
· 8L · d640 · heads 10/5 · ctx 512

> A Göktürk (Old Turkic) transliterator — the bengu bilingual base SFT'd on 6000 deterministic rune->Latin pairs (correct-by-construction, from the Unicode Old Turkic block U+10C00-U+10C4F). The model named for the Orkhon inscriptions, reading them: paste runes, get the Latin transliteration. SFT loss 0.01. NOTE: this is transliteration, NOT translation; rune->Latin is deterministic, translation to modern Turkish needs sourced scholarly data.

## Metrics

- (none recorded)

## Samples

**'Transliterate this Old Turkic (Orkhon) text into Latin: 𐰃𐰡𐰞𐰜'**

> ieltaloek

## Run it

```bash
uv run orkhon chat --checkpoint models/bengu-gokturk-20260615/checkpoint --tokenizer models/bengu-gokturk-20260615/tokenizer
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`
- `tokenizer/` — the tokenizer
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_