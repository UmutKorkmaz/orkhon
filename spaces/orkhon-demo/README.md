---
title: Orkhon model family
emoji: 🪨
colorFrom: gray
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# Orkhon model family demo

Gradio Space for the Orkhon model family. The app loads the published HF model
repos prepared by `scripts/prepare_hf_family.py`:

- `orkhon-tangri`
- `orkhon-bunghu`
- `orkhon-tegin`
- `orkhon-tonyuk`
- `orkhon-istem`
- `orkhon-bumin-mini`

`tangri` is the default demo. All normal public members are unified assistants:
English, Turkish, and Kokturk/Old Turkic rune-to-Latin transliteration are part
of the same line, not a separate `-gokturk` branch.

## Space settings

| Setting | Value |
| --- | --- |
| SDK | Gradio |
| Hardware | CPU basic is enough for first demo; upgrade if latency is poor |
| App file | `app.py` |

## Environment

| Var | Required | Meaning |
| --- | --- | --- |
| `ORKHON_HF_OWNER` | no | HF user/org for the model repos; default `korkmazumut` |
| `ORKHON_DEVICE` | no | `cpu` default; use `cuda` only on GPU hardware |
| `ORKHON_FAMILY_LOCAL_ROOT` | local only | local prepared export root, e.g. `exports/huggingface` |

## Local test

Prepare the family exports first:

```bash
uv run python scripts/prepare_hf_family.py --owner korkmazumut
```

Then run the Space locally against those exports:

```bash
ORKHON_FAMILY_LOCAL_ROOT=exports/huggingface \
  uv run --extra demo python spaces/orkhon-demo/app.py
```

## Public safety note

Do not enable code execution tools on the public Space. This demo only runs text
generation plus deterministic transliteration through exported Orkhon weights.
