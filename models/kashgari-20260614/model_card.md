# kashgari  ·  20260614

**imported** model · **135.0M** params · vocab None
· NoneL · dNone · heads None/None · ctx None

> A real 135M Llama-architecture open base loaded into Orkhon's hand-written Transformer with EXACT logit parity vs transformers. Re-importable on demand; weights not re-stored.

## Metrics

- (none recorded)

## Samples

**'The capital of France is'**

>  the capital of the country.

The capital of France

## Run it

```bash
uv run orkhon import-hf --repo HuggingFaceTB/SmolLM2-135M --out runs/kashgari
```

## Contents

- `checkpoint/` — inference weights + `model_config.json`  (imported: re-download via the command above)
- (uses the source repo's tokenizer)
- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`

_Orkhon v0.1.0_