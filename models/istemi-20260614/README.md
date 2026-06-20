---
license: apache-2.0
library_name: orkhon
pipeline_tag: text-generation
language:
- en
tags:
- orkhon
- base-model
- fineweb-edu
- from-scratch
- research
---

# Orkhon Istemi

Istemi is a 51M parameter base model pretrained from scratch on 162.8M
FineWeb-Edu tokens. It is the first Orkhon checkpoint trained on real-world web
text rather than only synthetic or toy corpora.

The checkpoint is intentionally undertrained relative to common scaling targets:
it saw about 3 tokens per parameter, while a stronger run would use orders of
magnitude more data. Treat it as a real-data research milestone, not a reliable
knowledge model.

## Model Details

| Field | Value |
| --- | --- |
| Date | 2026-06-14 |
| Type | Base language model |
| Parameters | 51.42M |
| Vocabulary | 16,384 |
| Layers | 9 |
| Hidden size | 640 |
| Attention heads | 10 query / 5 key-value |
| Context length | 512 tokens |
| Training source | `runs/fineweb_50m` |
| Generation mode | Completion |
| Orkhon version | 0.1.0 |

## Evaluation

| Metric | Value |
| --- | ---: |
| Validation loss | 3.8396 |
| Perplexity | 46.506 |
| Eval tokens | 163,840 |

## Run Locally

Run from the repository root:

```bash
uv run orkhon generate \
  --checkpoint models/istemi-20260614/checkpoint \
  --tokenizer models/istemi-20260614/tokenizer \
  -p "Once upon a time"
```

Or use the bundled script:

```bash
./models/istemi-20260614/run.sh
```

## Example Behavior

Prompt:

```text
The history of the Roman Empire
```

Observed behavior: continues in English with plausible-looking web prose, but
can repeat and invent facts. See `samples.txt` for captured generations.

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

- Undertrained 51M base model.
- Can repeat, derail, and hallucinate.
- No instruction tuning or preference tuning.
- Not safety aligned and not suitable for medical, legal, financial, or other
  high-stakes use.

## License

Released under Apache-2.0. See the repository `LICENSE` file for the full terms.
