"""Orkhon model zoo — named, dated, self-contained model archives.

Every model Orkhon trains can be *registered* into ``models/<name>-<YYYYMMDD>/``,
a self-contained folder that stores everything needed to reproduce, run, and
understand that model:

    models/tonyukuk-20260614/
      model_card.md        human-readable card (lineage, arch, metrics, samples)
      manifest.json        machine-readable metadata
      checkpoint/          the weights (slim, inference-only) + model_config.json
      tokenizer/           the tokenizer it was trained with
      samples.txt          generated outputs
      eval.json            perplexity / metrics (when applicable)
      code_snapshot.tgz    the exact src/ + configs that produced it
      run.sh               one command to talk to this model

Models are codenamed after figures from Turkic history (the project is named for
the Orkhon inscriptions). :func:`next_name` hands out the next unused codename so
future models slot in automatically.

Two registration paths:
  * :func:`register_model`    — Orkhon-trained models (stores the weights).
  * :func:`register_imported` — models imported from the HF Hub (stores the repo
    id + reproduction command instead of re-storing re-downloadable weights).
"""

from __future__ import annotations

import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

from orkhon import __version__
from orkhon.data import PackedDataset
from orkhon.eval.perplexity import evaluate
from orkhon.tokenizer import load_tokenizer
from orkhon.train.checkpoint import load_model_from_checkpoint

# Clean ASCII model names, handed out in pool order to new models.
NAME_POOL: list[str] = [
    "bumin-mini",
    "tonyuk",
    "tegin",
    "istem",
    "kashgar",
    "bunghu",
    "tangri",
    "qaghan",
    "otuken",
    "balasagun",
    "kutadgu",
    "oguz",
    "manas",
    "tarkan",
    "ergenekon",
    "atilla",
    "timur",
    "korkut",
    "umay",
    "yenisei",
    "alp",
    "asena",
    "ulugbeg",
    "selcuk",
    "bozok",
    "altan",
    "koroglu",
    "alpamis",
    "yabgu",
]

DEFAULT_ROOT = "models"


def today_stamp() -> str:
    """Date stamp ``YYYYMMDD`` for folder naming."""
    return datetime.now().strftime("%Y%m%d")


def _existing_names(dest_root: str | Path) -> set[str]:
    root = Path(dest_root)
    if not root.exists():
        return set()
    names = set()
    for p in root.iterdir():
        if p.is_dir() and "-" in p.name:
            names.add(p.name.rsplit("-", 1)[0])
    return names


def next_name(dest_root: str | Path = DEFAULT_ROOT) -> str:
    """Return the next unused codename from :data:`NAME_POOL`."""
    used = _existing_names(dest_root)
    for name in NAME_POOL:
        if name not in used:
            return name
    # Pool exhausted — fall back to an indexed name.
    return f"orkhon-{len(used) + 1}"


def _snapshot_code(dest: Path, config_files: Sequence[str | Path]) -> str:
    """Tar src/orkhon + pyproject + the given configs into code_snapshot.tgz."""
    repo = Path(__file__).resolve().parents[2]
    out = dest / "code_snapshot.tgz"
    with tarfile.open(out, "w:gz") as tar:
        src = repo / "src" / "orkhon"
        for py in sorted(src.rglob("*.py")):
            tar.add(py, arcname=str(py.relative_to(repo)))
        tmpl = src / "tokenizer" / "chat_template.jinja"
        if tmpl.exists():
            tar.add(tmpl, arcname=str(tmpl.relative_to(repo)))
        for extra in ("pyproject.toml",):
            p = repo / extra
            if p.exists():
                tar.add(p, arcname=extra)
        for cfg in config_files:
            p = repo / cfg if not Path(cfg).is_absolute() else Path(cfg)
            if p.exists():
                tar.add(p, arcname=str(Path(cfg)))
    return out.name


def _save_slim_checkpoint(dest: Path, model, cfg) -> None:
    """Write an inference-only checkpoint (weights + config, no optimizer)."""
    ckpt_dir = dest / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.to("cpu") for k, v in model.state_dict().items()}
    torch.save({"model": state, "model_config": cfg.to_dict()}, ckpt_dir / "ckpt_last.pt")
    (ckpt_dir / "model_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2), encoding="utf-8"
    )


def _generate_samples(model, tok, prompts: Sequence[str], mode: str, device) -> list[dict]:
    """Generate a completion/reply per prompt. mode is 'complete' or 'chat'."""
    from orkhon.model import generate
    from orkhon.serve.chat_cli import reply
    from orkhon.tokenizer.render import encode_for_inference  # noqa: F401 (chat path)

    out = []
    model.eval()
    for p in prompts:
        if mode == "chat":
            text = reply([{"role": "user", "content": p}], model, tok,
                         temperature=0.0, max_new_tokens=80, device=device)
        else:  # complete (base model): signal a fresh doc with <eos>, stop at <eos>
            ids = [tok.special.eos] + tok.encode(p)
            new = generate(model, ids, max_new_tokens=100, temperature=0.0,
                           eos_ids=(tok.special.eos,), device=device)
            text = tok.decode(new, skip_special=True)
        out.append({"prompt": p, "output": text})
    return out


def _write_card(dest: Path, manifest: dict, samples: list[dict]) -> None:
    m = manifest
    lines = [
        f"# {m['name']}  ·  {m['date']}",
        "",
        f"**{m['kind']}** model · **{m['params_m']:.1f}M** params · vocab {m['vocab_size']}",
        f"· {m['n_layers']}L · d{m['d_model']} · heads {m['n_heads']}/{m['n_kv_heads']}"
        f" · ctx {m['block_size']}",
        "",
        f"> {m['lineage']}",
        "",
        "## Metrics",
        "",
    ]
    if m.get("metrics"):
        for k, v in m["metrics"].items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- (none recorded)")
    lines += ["", "## Samples", ""]
    for s in samples:
        lines.append(f"**{s['prompt']!r}**")
        lines.append("")
        lines.append(f"> {s['output']}")
        lines.append("")
    lines += [
        "## Run it",
        "",
        "```bash",
        m["run_hint"],
        "```",
        "",
        "## Contents",
        "",
        "- `checkpoint/` — inference weights + `model_config.json`"
        + ("  (imported: re-download via the command above)" if m["kind"] == "imported" else ""),
        "- `tokenizer/` — the tokenizer" if m["kind"] != "imported" else "- (uses the source repo's tokenizer)",
        "- `samples.txt`, `eval.json`, `code_snapshot.tgz`, `manifest.json`",
        "",
        f"_Orkhon v{m['orkhon_version']}_",
    ]
    (dest / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


def register_model(
    name: str,
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    kind: str,
    lineage: str,
    sample_prompts: Sequence[str],
    generate_mode: str = "complete",
    eval_prepared: str | Path | None = None,
    eval_seq_len: int = 512,
    config_files: Sequence[str] = (),
    device: str = "cpu",
    dest_root: str | Path = DEFAULT_ROOT,
    date: str | None = None,
    tag: str = "last",
) -> Path:
    """Archive an Orkhon-trained model into ``<dest_root>/<name>-<date>/``."""
    date = date or today_stamp()
    dest = Path(dest_root) / f"{name}-{date}"
    dest.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model_from_checkpoint(checkpoint_dir, device=device, tag=tag)
    tok = load_tokenizer(tokenizer_dir)

    samples = _generate_samples(model, tok, sample_prompts, generate_mode, device)
    (dest / "samples.txt").write_text(
        "\n\n".join(f"### {s['prompt']}\n{s['output']}" for s in samples), encoding="utf-8"
    )

    metrics: dict = {}
    if eval_prepared is not None:
        val_bin = Path(eval_prepared) / "val.bin"
        if val_bin.exists():
            ds = PackedDataset(str(val_bin), eval_seq_len)
            res = evaluate(model, ds, max_batches=40, batch_size=8,
                           seq_len=eval_seq_len, device=device)
            metrics = {"val_loss": round(res["loss"], 4), "perplexity": round(res["ppl"], 3),
                       "eval_tokens": res["tokens"]}
            (dest / "eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    _save_slim_checkpoint(dest, model, cfg)
    shutil.copytree(tokenizer_dir, dest / "tokenizer", dirs_exist_ok=True)
    snapshot = _snapshot_code(dest, config_files)

    rel = f"{Path(dest_root).name}/{name}-{date}"
    verb = "chat" if generate_mode == "chat" else "generate"
    parg = "" if generate_mode == "chat" else ' -p "Once upon a time"'
    run_hint = (f"uv run orkhon {verb} --checkpoint {rel}/checkpoint "
                f"--tokenizer {rel}/tokenizer{parg}")

    manifest = {
        "name": name, "date": date, "kind": kind, "lineage": lineage,
        "params_m": cfg.estimate_params() / 1e6, "vocab_size": cfg.vocab_size,
        "n_layers": cfg.n_layers, "d_model": cfg.d_model, "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads, "block_size": cfg.block_size,
        "arch": cfg.to_dict(), "metrics": metrics, "n_samples": len(samples),
        "source_checkpoint": str(checkpoint_dir), "generate_mode": generate_mode,
        "code_snapshot": snapshot, "run_hint": run_hint,
        "orkhon_version": __version__,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_card(dest, manifest, samples)
    (dest / "run.sh").write_text("#!/usr/bin/env bash\nset -e\n" + run_hint + "\n", encoding="utf-8")
    (dest / "run.sh").chmod(0o755)
    return dest


def register_imported(
    name: str,
    *,
    repo: str,
    params_m: float,
    lineage: str,
    sample: dict | None = None,
    config_files: Sequence[str] = (),
    dest_root: str | Path = DEFAULT_ROOT,
    date: str | None = None,
) -> Path:
    """Archive a Hub-imported model as a manifest + reproduction command.

    Weights are NOT re-stored (they re-download from ``repo``); the archive records
    exactly how to recreate the Orkhon checkpoint.
    """
    date = date or today_stamp()
    dest = Path(dest_root) / f"{name}-{date}"
    dest.mkdir(parents=True, exist_ok=True)

    repro = f"uv run orkhon import-hf --repo {repo} --out runs/{name}"
    samples = [sample] if sample else []
    if samples:
        (dest / "samples.txt").write_text(
            f"### {samples[0]['prompt']}\n{samples[0]['output']}", encoding="utf-8"
        )
    snapshot = _snapshot_code(dest, config_files)
    manifest = {
        "name": name, "date": date, "kind": "imported", "lineage": lineage,
        "params_m": params_m, "vocab_size": None, "n_layers": None, "d_model": None,
        "n_heads": None, "n_kv_heads": None, "block_size": None, "arch": None,
        "metrics": {}, "n_samples": len(samples), "source_repo": repo,
        "code_snapshot": snapshot, "run_hint": repro, "orkhon_version": __version__,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_card(dest, manifest, samples)
    (dest / "run.sh").write_text("#!/usr/bin/env bash\nset -e\n" + repro + "\n", encoding="utf-8")
    (dest / "run.sh").chmod(0o755)
    return dest


def build_index(dest_root: str | Path = DEFAULT_ROOT) -> Path:
    """Scan all model folders and (re)write ``<dest_root>/registry.md``."""
    root = Path(dest_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for man in sorted(root.glob("*/manifest.json")):
        m = json.loads(man.read_text(encoding="utf-8"))
        metric = ""
        if m.get("metrics"):
            mm = m["metrics"]
            if "perplexity" in mm:
                metric = f"ppl {mm['perplexity']}"
            elif mm:
                metric = ", ".join(f"{k} {v}" for k, v in list(mm.items())[:1])
        rows.append((m["date"], m["name"], m["kind"], m.get("params_m") or 0,
                     metric, man.parent.name))

    lines = [
        "# Orkhon model zoo", "",
        "Named, dated, self-contained model archives. Each folder has weights,"
        " tokenizer, samples, eval, a code snapshot, and `run.sh`. New models:"
        " `orkhon register ...`.", "",
        "| Date | Name | Kind | Params | Metric | Folder |",
        "|------|------|------|--------|--------|--------|",
    ]
    for date, name, kind, params, metric, folder in rows:
        lines.append(f"| {date} | **{name}** | {kind} | {params:.0f}M | {metric} | `{folder}/` |")
    lines.append("")
    out = root / "registry.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
