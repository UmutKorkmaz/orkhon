---
license: apache-2.0
library_name: orkhon
pipeline_tag: text-generation
language:
- en
tags:
- orkhon
- base-model
- tinystories
- story-generation
- from-scratch
---

# Orkhon Tonyukuk

Tonyukuk is a 22M parameter base model pretrained from scratch on 47.4M
TinyStories tokens. It is a compact Orkhon checkpoint for fluent short-story
completion and for comparing base versus instruction-tuned behavior with
Kultigin.

The model writes simple, coherent children's-story style text. It is not a
general-purpose language model and has limited world knowledge.

## Model Details

| Field | Value |
| --- | --- |
| Date | 2026-06-14 |
| Type | Base story model |
| Parameters | 21.90M |
| Vocabulary | 8,192 |
| Layers | 6 |
| Hidden size | 512 |
| Attention heads | 8 query / 4 key-value |
| Context length | 512 tokens |
| Training source | `runs/tinystories` |
| Generation mode | Completion |
| Orkhon version | 0.1.0 |

## Evaluation

| Metric | Value |
| --- | ---: |
| Validation loss | 1.5622 |
| Perplexity | 4.769 |
| Eval tokens | 163,840 |

## Run Locally

Run from the repository root:

```bash
uv run orkhon generate \
  --checkpoint models/tonyukuk-20260614/checkpoint \
  --tokenizer models/tonyukuk-20260614/tokenizer \
  -p "Once upon a time"
```

Or use the bundled script:

```bash
./models/tonyukuk-20260614/run.sh
```

## Example Behavior

Prompt:

```text
Once upon a time, there was a little robot named Bo.
```

Observed behavior: continues in a simple TinyStories-like style. See
`samples.txt` for captured generations.

## Files

- `checkpoint/` - Orkhon checkpoint weights and `model_config.json`
- `tokenizer/` - tokenizer files
- `manifest.json` - machine-readable model metadata
- `eval.json` - evaluation metrics
- `samples.txt` - captured sample generations
- `run.sh` - local generation command
- `code_snapshot.tgz` - code snapshot used for the export
- `model_card.md` and `model_card.tr.md` - short English/Turkish cards

## Limitations

- TinyStories-scale base model with limited domain coverage.
- Can repeat or produce simplistic details.
- No instruction tuning in this checkpoint.
- Not safety aligned and not suitable for high-stakes use.

## License

Released under Apache-2.0. See the repository `LICENSE` file for the full terms.
