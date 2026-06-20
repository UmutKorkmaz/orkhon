---
license: apache-2.0
library_name: orkhon
pipeline_tag: text-generation
language:
- otk
- tr
tags:
- orkhon
- instruct
- old-turkic
- gokturk
- transliteration
- sft
---

# Orkhon Bengu-Gokturk

Bengu-Gokturk is a 57M parameter instruct checkpoint for Old Turkic rune to
Latin transliteration. It starts from the Bengu bilingual base and is SFT'd on
6,000 deterministic rune-to-Latin pairs generated from the Unicode Old Turkic
block.

This model performs transliteration, not translation. It maps Old Turkic runes
to Latin characters. It does not translate the text into modern Turkish or
English, and it should not be treated as a scholarly translation model.

## Model Details

| Field | Value |
| --- | --- |
| Date | 2026-06-15 |
| Type | Instruct transliteration model |
| Parameters | 57.35M |
| Vocabulary | 32,768 |
| Layers | 8 |
| Hidden size | 640 |
| Attention heads | 10 query / 5 key-value |
| Context length | 512 tokens |
| Training source | `runs/gokturk` |
| Generation mode | Chat |
| Orkhon version | 0.1.0 |

## Training Signal

The supervised data is deterministic and correct-by-construction for the
transliteration task. No translation supervision is included.

## Run Locally

Run from the repository root:

```bash
uv run orkhon chat \
  --checkpoint models/bengu-gokturk-20260615/checkpoint \
  --tokenizer models/bengu-gokturk-20260615/tokenizer
```

Or use the bundled script:

```bash
./models/bengu-gokturk-20260615/run.sh
```

## Example Behavior

Prompt:

```text
Transliterate this Old Turkic (Orkhon) text into Latin: [Old Turkic runes]
```

Observed behavior: returns a Latin transliteration. See `samples.txt` for the
captured rune sample and model output.

## Files

- `checkpoint/` - Orkhon checkpoint weights and `model_config.json`
- `tokenizer/` - bilingual tokenizer files
- `manifest.json` - machine-readable model metadata
- `samples.txt` - captured sample generation
- `run.sh` - local chat command
- `code_snapshot.tgz` - code snapshot used for the export
- `model_card.md` and `model_card.tr.md` - short English/Turkish cards

## Limitations

- Transliteration only; no semantic translation.
- The task is narrow and deterministic.
- Outputs should be checked before use in historical or linguistic work.
- Not safety aligned and not suitable for high-stakes use.

## License

Released under Apache-2.0. See the repository `LICENSE` file for the full terms.
