---
license: apache-2.0
library_name: orkhon
pipeline_tag: text-generation
language:
- en
base_model: HuggingFaceTB/SmolLM2-135M
tags:
- orkhon
- imported
- smollm2
- logit-parity
- transformer-import
---

# Orkhon Kashgari

Kashgari is an import/parity checkpoint entry for
`HuggingFaceTB/SmolLM2-135M`. It demonstrates that a real 135M
Llama-architecture open base model can be loaded into Orkhon's hand-written
Transformer with exact logit parity against `transformers`.

This folder does not store the imported weights. It stores the manifest, sample,
run command, and code snapshot needed to reproduce the import.

## Model Details

| Field | Value |
| --- | --- |
| Date | 2026-06-14 |
| Type | Imported base model |
| Source model | `HuggingFaceTB/SmolLM2-135M` |
| Parameters | 135M |
| Stored weights | No |
| Training source | Upstream source repo |
| Generation mode | Re-import before use |
| Orkhon version | 0.1.0 |

## Reproduce the Import

Run from the repository root:

```bash
uv run orkhon import-hf \
  --repo HuggingFaceTB/SmolLM2-135M \
  --out runs/kashgari
```

Or use the bundled script:

```bash
./models/kashgari-20260614/run.sh
```

## Example Behavior

Prompt:

```text
The capital of France is
```

Observed behavior is captured in `samples.txt`. This folder is primarily about
import correctness and parity, not about evaluating the upstream model.

## Files

- `manifest.json` - machine-readable import metadata
- `samples.txt` - captured sample generation
- `run.sh` - re-import command
- `code_snapshot.tgz` - code snapshot used for the export
- `model_card.md` and `model_card.tr.md` - short English/Turkish cards

No `checkpoint/` or `tokenizer/` directory is included because the model is
re-importable from the upstream Hugging Face repository.

## Limitations

- No weights are stored in this folder.
- Requires access to `HuggingFaceTB/SmolLM2-135M` to reproduce the import.
- This card documents Orkhon import/parity behavior, not a new trained model.
- Verify the upstream model card before using the imported weights in downstream
  applications.

## License

The Orkhon import metadata and local code are released under Apache-2.0. The
upstream `HuggingFaceTB/SmolLM2-135M` model is also listed on Hugging Face as
Apache-2.0; keep upstream attribution when redistributing imported weights.
