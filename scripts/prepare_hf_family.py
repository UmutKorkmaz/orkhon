#!/usr/bin/env python3
"""Prepare Hugging Face upload folders for the Orkhon model family.

The script exports each runnable local model-zoo member into a HF-style folder
(`config.json`, `model.safetensors`, tokenizer files, README) and writes a
family upload plan. It does not upload or publish anything by itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orkhon.export.to_hf import export


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "exports" / "huggingface"
GITHUB_URL = "https://github.com/UmutKorkmaz/orkhon"


@dataclass(frozen=True)
class FamilyMember:
    name: str
    repo_suffix: str
    model_dir: str
    kind: str
    interface: str
    summary: str
    prompt: str
    extra_tags: tuple[str, ...] = ()

    @property
    def local_path(self) -> Path:
        return ROOT / "models" / self.model_dir


FAMILY: tuple[FamilyMember, ...] = (
    FamilyMember(
        name="bumin-mini",
        repo_suffix="orkhon-bumin-mini",
        model_dir="bumin-mini-20260620",
        kind="instruct",
        interface="chat",
        summary="4M unified assistant smoke model for EN/TR/Kokturk behavior.",
        prompt="What is 2 plus 2?",
        extra_tags=("smoke-test", "chat", "turkish", "old-turkic"),
    ),
    FamilyMember(
        name="tonyuk",
        repo_suffix="orkhon-tonyuk",
        model_dir="tonyuk-20260620",
        kind="instruct",
        interface="chat",
        summary="22M unified assistant tuned from the Tonyukuk story base.",
        prompt="What can you help me with?",
        extra_tags=("tinystories", "instruction-tuned", "chat", "old-turkic"),
    ),
    FamilyMember(
        name="tegin",
        repo_suffix="orkhon-tegin",
        model_dir="tegin-20260620",
        kind="instruct",
        interface="chat",
        summary="22M unified assistant tuned from the former Kultegin instruct model.",
        prompt="Do not write a story. What can you help me with?",
        extra_tags=("instruction-tuned", "chat", "old-turkic"),
    ),
    FamilyMember(
        name="istem",
        repo_suffix="orkhon-istem",
        model_dir="istem-20260620",
        kind="instruct",
        interface="chat",
        summary="51M unified assistant tuned from the FineWeb-Edu base.",
        prompt="Bana hangi konularda yardim edebilirsin?",
        extra_tags=("fineweb-edu", "instruction-tuned", "chat", "old-turkic"),
    ),
    FamilyMember(
        name="kashgar",
        repo_suffix="orkhon-kashgar",
        model_dir="kashgar-20260620",
        kind="imported",
        interface="chat",
        summary="135M imported assistant slot once local weights are archived.",
        prompt="Explain what Orkhon is in one sentence.",
        extra_tags=("imported", "chat", "old-turkic"),
    ),
    FamilyMember(
        name="bunghu",
        repo_suffix="orkhon-bunghu",
        model_dir="bunghu-20260620",
        kind="instruct",
        interface="chat",
        summary="57M unified EN/TR/Kokturk assistant tuned from the bilingual branch.",
        prompt="Transliterate this Old Turkic (Orkhon) text into Latin: 𐰃𐰡𐰞𐰜",
        extra_tags=("turkish", "bilingual", "old-turkic", "transliteration", "chat"),
    ),
    FamilyMember(
        name="tangri",
        repo_suffix="orkhon-tangri",
        model_dir="tangri-20260620",
        kind="instruct",
        interface="chat",
        summary="100M unified EN/TR/Kokturk assistant trained from the mixed Tangri base.",
        prompt="Can you translate Old Turkic inscriptions to modern Turkish?",
        extra_tags=("turkish", "bilingual", "old-turkic", "transliteration", "chat"),
    ),
    FamilyMember(
        name="qaghan",
        repo_suffix="orkhon-qaghan",
        model_dir="qaghan-20260620",
        kind="instruct",
        interface="chat",
        summary="Future larger unified assistant slot.",
        prompt="What can you help me with?",
        extra_tags=("planned", "chat"),
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_lines(eval_results: dict[str, Any]) -> str:
    if not eval_results:
        return "_No headline evaluation JSON is recorded for this member yet._"

    rows = ["| Metric | Value |", "| --- | ---: |"]
    for key, value in eval_results.items():
        if isinstance(value, (int, float, str)) and value != "":
            rows.append(f"| `{key}` | {value} |")
    if len(rows) == 2:
        return "_Evaluation JSON exists, but it does not expose scalar headline metrics._"
    return "\n".join(rows)


def _family_table(owner: str, current: str) -> str:
    rows = ["| Member | Role | HF repo |", "| --- | --- | --- |"]
    for member in FAMILY:
        repo_id = f"{owner}/{member.repo_suffix}"
        marker = "current" if member.name == current else ""
        rows.append(
            f"| `{member.name}` {marker} | {member.summary} | "
            f"[{repo_id}](https://huggingface.co/{repo_id}) |"
        )
    return "\n".join(rows)


def _readme(member: FamilyMember, owner: str, manifest: dict[str, Any], eval_results: dict[str, Any]) -> str:
    repo_id = f"{owner}/{member.repo_suffix}"
    params = manifest.get("params_m")
    params_text = f"{params:.1f}M" if isinstance(params, (int, float)) else "unknown"
    tags = [
        "orkhon",
        "from-scratch",
        "pytorch",
        "safetensors",
        "text-generation",
        "turkic",
        *member.extra_tags,
    ]
    tag_lines = "\n".join(f"- {tag}" for tag in dict.fromkeys(tags))
    language_lines = "- en\n- tr\n- otk"

    return f"""---
license: apache-2.0
library_name: orkhon
pipeline_tag: text-generation
language:
{language_lines}
tags:
{tag_lines}
---

# Orkhon / {member.name}

{member.summary}

This repository is one member of the **Orkhon model family**, an auditable
from-scratch LLM stack covering tokenizer, pretraining, post-training,
evaluation, serving, and Hugging Face export.

- **Family:** Orkhon
- **Member:** `{member.name}`
- **Kind:** `{member.kind}`
- **Interface:** `{member.interface}`
- **Parameters:** ~{params_text}
- **Source code:** [{GITHUB_URL}]({GITHUB_URL})
- **Local model-zoo folder:** `models/{member.model_dir}`

## Family Context

{_family_table(owner, member.name)}

Members without archived local weights are skipped by the preparation script.

## Intended Use

- Inspecting and reproducing the Orkhon training/export path.
- Running small local demos on CPU/MPS/CUDA.
- Comparing Orkhon family members by training stage and data mix.
- Rune-to-Latin Old Turkic transliteration demos.

## Not Intended For

- Reliable factual QA.
- Safety-critical decisions.
- Claims of state-of-the-art Turkish or general LLM performance.
- Treating Old Turkic transliteration as modern Turkish translation.

## Evaluation

{_metric_lines(eval_results)}

Current benchmark reports in the source repo are smoke baselines unless marked
otherwise. Do not treat limit-20 benchmark runs as headline capability claims.

## Example Prompt

```text
{member.prompt}
```

## Usage

Install Orkhon from the source repo, then load this exported folder:

```bash
pip install git+{GITHUB_URL}
```

```python
from huggingface_hub import snapshot_download
from orkhon.export.to_hf import load_exported_model
from orkhon.tokenizer import load_tokenizer

path = snapshot_download("{repo_id}")
model, cfg = load_exported_model(path, device="cpu")
tok = load_tokenizer(path)
```

Use `orkhon.serve.chat_cli.reply` for the unified assistant members.

## Files

- `model.safetensors` - exported inference weights
- `config.json` - Orkhon architecture config
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
- `manifest.json` - model-zoo metadata from the source repo
- `samples.txt` - saved local samples when available
- `orkhon_model_card.md` - original local model card

## Limitations

This is a small research/education model family. The models can repeat,
confabulate, and fail outside their narrow data scale. Public demos should keep
the claim narrow: Orkhon is an auditable from-scratch stack with a Turkic branch,
not a frontier assistant.
"""


def _copy_sidecars(src_dir: Path, out_dir: Path) -> None:
    copies = {
        "manifest.json": "manifest.json",
        "samples.txt": "samples.txt",
        "eval.json": "eval.json",
        "model_card.md": "orkhon_model_card.md",
        "model_card.tr.md": "orkhon_model_card.tr.md",
    }
    for src_name, dst_name in copies.items():
        src = src_dir / src_name
        if src.exists():
            shutil.copy2(src, out_dir / dst_name)


def prepare(owner: str, out_root: Path) -> list[tuple[FamilyMember, Path]]:
    out_root.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[FamilyMember, Path]] = []

    for member in FAMILY:
        src = member.local_path
        checkpoint = src / "checkpoint"
        tokenizer = src / "tokenizer"
        if not (checkpoint / "ckpt_last.pt").exists() or not (tokenizer / "tokenizer.json").exists():
            continue

        out_dir = out_root / member.repo_suffix
        eval_json = src / "eval.json"
        export(
            checkpoint_dir=checkpoint,
            out_dir=out_dir,
            tokenizer_dir=tokenizer,
            model_name=f"{owner}/{member.repo_suffix}",
            eval_json=eval_json if eval_json.exists() else None,
        )
        _copy_sidecars(src, out_dir)

        manifest = _read_json(src / "manifest.json")
        eval_results = _read_json(eval_json)
        (out_dir / "README.md").write_text(
            _readme(member, owner, manifest, eval_results),
            encoding="utf-8",
        )
        prepared.append((member, out_dir))

    _write_family_docs(owner, out_root, prepared)
    return prepared


def _write_family_docs(
    owner: str, out_root: Path, prepared: list[tuple[FamilyMember, Path]]
) -> None:
    repo_rows = "\n".join(
        f"- `{owner}/{member.repo_suffix}` from `{path.relative_to(ROOT)}`"
        for member, path in prepared
    )
    upload_cmds = "\n".join(
        "uv run hf upload "
        f"{owner}/{member.repo_suffix} "
        f"{path.relative_to(ROOT)} . "
        "--type model --no-private "
        f"--commit-message \"Publish Orkhon {member.name}\""
        for member, path in prepared
    )
    collection_cmds = "\n".join(
        f"uv run hf collections add-item {owner}/orkhon-model-family "
        f"{owner}/{member.repo_suffix} model --exists-ok"
        for member, _path in prepared
    )

    (out_root / "FAMILY.md").write_text(
        f"""# Orkhon Model Family

Prepared Hugging Face repos:

{repo_rows}

Positioning:

> Orkhon is an auditable from-scratch LLM stack with unified English, Turkish,
> and Old Turkic/Kokturk transliteration behavior across the normal model line.

Suggested collection title:

```text
Orkhon Model Family
```

Suggested collection description:

```text
Auditable from-scratch LLM stack and model zoo with unified EN/TR/Kokturk
assistant checkpoints.
```
""",
        encoding="utf-8",
    )

    (out_root / "UPLOAD_PLAN.md").write_text(
        f"""# Hugging Face Upload Plan

This plan is prepared for Hugging Face user/org `{owner}`.

## Upload model repos

Run only after confirming that these files should be published publicly:

```bash
{upload_cmds}
```

## Create family collection

After the model repos exist:

```bash
uv run hf collections create "Orkhon Model Family" --namespace {owner} \\
  --description "Auditable from-scratch LLM stack and model zoo." --exists-ok
{collection_cmds}
```

If the CLI returns a generated collection slug, replace
`{owner}/orkhon-model-family` in the `add-item` commands with that slug.

## Publish family Space

Package `spaces/orkhon-demo` as a Gradio Space after the repos above exist:

```bash
uv run hf upload {owner}/orkhon-demo spaces/orkhon-demo . \\
  --type space --no-private --commit-message "Publish Orkhon family demo"
```
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="korkmazumut", help="HF username or org.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for prepared HF repos.",
    )
    args = parser.parse_args()

    prepared = prepare(args.owner, args.out)
    for member, path in prepared:
        print(f"prepared {args.owner}/{member.repo_suffix}: {path}")
    print(f"wrote {args.out / 'UPLOAD_PLAN.md'}")


if __name__ == "__main__":
    main()
