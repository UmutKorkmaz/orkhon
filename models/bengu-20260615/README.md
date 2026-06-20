---
license: apache-2.0
library_name: orkhon
pipeline_tag: text-generation
language:
- tr
- en
tags:
- orkhon
- base-model
- turkish
- bilingual
- from-scratch
---

# Orkhon Bengu

Bengu is a 57M parameter bilingual base model trained from scratch for the
Orkhon project. It is the first Turkish-focused checkpoint in the Bengu Tas
branch: a compact EN+TR language model trained on roughly 136M tokens, mostly
Turkish Wikipedia with an English slice for bilingual coverage.

This is a research checkpoint, not a production assistant. It produces fluent
Wikipedia-register Turkish and shows correct morphology/agglutination, but it
also confabulates facts and repeats phrases. Use it to inspect the model family,
tokenizer, and generation behavior, not as a reliable factual source.

## Model Details

| Field | Value |
| --- | --- |
| Date | 2026-06-15 |
| Type | Base language model |
| Parameters | 57.35M |
| Vocabulary | 32,768 |
| Layers | 8 |
| Hidden size | 640 |
| Attention heads | 10 query / 5 key-value |
| Context length | 512 tokens |
| Training source | `runs/bengu` |
| Generation mode | Completion |
| Orkhon version | 0.1.0 |

## Evaluation

| Metric | Value |
| --- | ---: |
| Validation loss | 3.8523 |
| Perplexity | 47.102 |
| Eval tokens | 163,840 |

## Run Locally

Run from the repository root:

```bash
uv run orkhon generate \
  --checkpoint models/bengu-20260615/checkpoint \
  --tokenizer models/bengu-20260615/tokenizer \
  -p "Once upon a time"
```

Or use the bundled script:

```bash
./models/bengu-20260615/run.sh
```

## Example Behavior

Prompt:

```text
Gokturkler, Orta Asya'da
```

Observed behavior: continues in Turkish with grammatical prose, but may repeat
phrases and invent details. See `samples.txt` for captured generations.

## Files

- `checkpoint/` - Orkhon checkpoint weights and `model_config.json`
- `tokenizer/` - bilingual tokenizer files
- `manifest.json` - machine-readable model metadata
- `eval.json` - evaluation metrics
- `samples.txt` - captured sample generations
- `run.sh` - local smoke-test command
- `code_snapshot.tgz` - code snapshot used for the export
- `model_card.md` and `model_card.tr.md` - short English/Turkish cards

## Limitations

- Small base model trained for exploration, not instruction following.
- Can repeat phrases, especially on open-ended prompts.
- Can generate plausible but false facts.
- Not safety aligned and not suitable for medical, legal, financial, or other
  high-stakes use.

## License

Released under Apache-2.0. See the repository `LICENSE` file for the full terms.
