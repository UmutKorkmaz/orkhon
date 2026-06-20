---
license: apache-2.0
library_name: orkhon
pipeline_tag: text-generation
language:
- en
tags:
- orkhon
- instruct
- tinystories
- story-generation
- sft
---

# Orkhon Kultigin

Kultigin is a 22M parameter instruct checkpoint built by supervised
fine-tuning Tonyukuk on synthetic story instructions. It learns the instruction
shape for prompts such as "Write a short story about ...", while retaining the
small TinyStories-scale vocabulary and knowledge limits.

This is a compact story-format demonstration checkpoint, not a general writing
assistant.

## Model Details

| Field | Value |
| --- | --- |
| Date | 2026-06-14 |
| Type | Instruct story model |
| Parameters | 21.90M |
| Vocabulary | 8,192 |
| Layers | 6 |
| Hidden size | 512 |
| Attention heads | 8 query / 4 key-value |
| Context length | 512 tokens |
| Training source | `runs/tinystories_sft` |
| Generation mode | Chat |
| Orkhon version | 0.1.0 |

## Run Locally

Run from the repository root:

```bash
uv run orkhon chat \
  --checkpoint models/kultigin-20260614/checkpoint \
  --tokenizer models/kultigin-20260614/tokenizer
```

Or use the bundled script:

```bash
./models/kultigin-20260614/run.sh
```

## Example Behavior

Prompt:

```text
Write a short story about a brave knight.
```

Observed behavior: produces a short children's-story style continuation, but
may drift or introduce odd details. See `samples.txt` for captured generations.

## Files

- `checkpoint/` - Orkhon checkpoint weights and `model_config.json`
- `tokenizer/` - tokenizer files
- `manifest.json` - machine-readable model metadata
- `samples.txt` - captured sample generations
- `run.sh` - local chat command
- `code_snapshot.tgz` - code snapshot used for the export
- `model_card.md` and `model_card.tr.md` - short English/Turkish cards

## Limitations

- Narrow story-instruction checkpoint.
- Content grounding is limited by the 22M scale and TinyStories vocabulary.
- No recorded evaluation metrics in this export.
- Not safety aligned and not suitable for high-stakes use.

## License

Released under Apache-2.0. See the repository `LICENSE` file for the full terms.
