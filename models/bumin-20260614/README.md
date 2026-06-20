---
license: apache-2.0
library_name: orkhon
pipeline_tag: text-generation
language:
- en
tags:
- orkhon
- instruct
- arithmetic
- synthetic-data
- from-scratch
---

# Orkhon Bumin

Bumin is the first Orkhon checkpoint: a 4.2M parameter from-scratch instruct
model trained on synthetic arithmetic Q&A. It is intentionally tiny so it runs
quickly on CPU and is useful as the fastest checkpoint for demos, smoke tests,
and Hugging Face Space bring-up.

The model learns the simple chat format `Answer: N.` for small arithmetic
prompts. It is not a general-purpose assistant.

## Model Details

| Field | Value |
| --- | --- |
| Date | 2026-06-14 |
| Type | Instruct model |
| Parameters | 4.20M |
| Vocabulary | 4,096 |
| Layers | 4 |
| Hidden size | 256 |
| Attention heads | 4 query / 2 key-value |
| Context length | 256 tokens |
| Training source | `runs/sft_smoke` |
| Generation mode | Chat |
| Orkhon version | 0.1.0 |

## Evaluation

| Metric | Value |
| --- | ---: |
| Validation loss | 1.6348 |
| Perplexity | 5.128 |
| Eval tokens | 20,480 |

## Run Locally

Run from the repository root:

```bash
uv run orkhon chat \
  --checkpoint models/bumin-20260614/checkpoint \
  --tokenizer models/bumin-20260614/tokenizer
```

Or use the bundled script:

```bash
./models/bumin-20260614/run.sh
```

## Example Behavior

Prompt:

```text
What is 2 plus 2?
```

Output:

```text
Answer: 4.
```

More captured examples are in `samples.txt`.

## Files

- `checkpoint/` - Orkhon checkpoint weights and `model_config.json`
- `tokenizer/` - tokenizer files
- `manifest.json` - machine-readable model metadata
- `eval.json` - evaluation metrics
- `samples.txt` - captured sample generations
- `run.sh` - local chat command
- `code_snapshot.tgz` - code snapshot used for the export
- `model_card.md` and `model_card.tr.md` - short English/Turkish cards

## Limitations

- Extremely small model trained for arithmetic-format smoke tests.
- Does not contain broad world knowledge.
- Expected to fail outside its narrow synthetic task.
- Not safety aligned and not suitable for high-stakes use.

## License

Released under Apache-2.0. See the repository `LICENSE` file for the full terms.
