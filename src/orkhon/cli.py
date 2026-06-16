"""Orkhon command-line interface.

A single ``typer`` app (exported as :data:`app`) that wires the whole pipeline:

    tokenizer train -> data prepare/synth -> train pretrain/sft/dpo
        -> eval -> chat -> serve -> export hf

The CLI is a thin orchestration layer: every command resolves a config (via the
shared ``orkhon.config`` loaders, with ``--set k=v`` dotlist overrides) and then
delegates to the already-built package functions. Progress is reported through
``rich`` so a long pipeline reads clearly in a terminal.

Run ``orkhon --help`` (or ``python -m orkhon --help``) to see the command tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from orkhon.config.load import load_model_config, load_stage_config
from orkhon.config.schema import DPOConfig, PretrainConfig, SFTConfig

console = Console()

app = typer.Typer(
    name="orkhon",
    help="Orkhon — a from-scratch LLM stack: tokenizer -> data -> pretrain "
    "-> SFT -> DPO -> eval -> serve -> export.",
    no_args_is_help=True,
    add_completion=False,
)

# Sub-apps: grouped commands that read naturally as `orkhon <group> <verb>`.
tokenizer_app = typer.Typer(help="Tokenizer training.", no_args_is_help=True)
data_app = typer.Typer(help="Dataset synthesis and packing.", no_args_is_help=True)
train_app = typer.Typer(help="Training stages (pretrain, SFT, DPO).", no_args_is_help=True)
export_app = typer.Typer(help="Export trained checkpoints.", no_args_is_help=True)

app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(export_app, name="export")

rag_app = typer.Typer(help="Retrieval-Augmented Generation (ingest/search).", no_args_is_help=True)
app.add_typer(rag_app, name="rag")


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


def _step(title: str) -> None:
    """Print a prominent section banner for a pipeline step."""
    console.print(Panel(title, style="bold cyan", expand=False))


def _ok(message: str) -> None:
    console.print(f"[bold green]✓[/bold green] {message}")


def _summary_table(title: str, payload: dict) -> None:
    """Render a flat dict as a small two-column rich table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    for key, value in payload.items():
        if isinstance(value, float):
            table.add_row(str(key), f"{value:.4f}")
        else:
            table.add_row(str(key), str(value))
    console.print(table)


def _fail(message: str) -> "typer.Exit":
    """Print an error and return an Exit(1) the caller should ``raise``."""
    console.print(f"[bold red]error:[/bold red] {message}")
    return typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# tokenizer                                                                   #
# --------------------------------------------------------------------------- #


@tokenizer_app.command("train")
def tokenizer_train(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML with corpus/out/vocab_size."
    ),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus", help="Corpus .txt path (overrides config)."
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Output tokenizer directory (overrides config)."
    ),
    vocab_size: Optional[int] = typer.Option(
        None, "--vocab-size", help="Target vocab size including 8 specials."
    ),
    min_frequency: Optional[int] = typer.Option(
        None, "--min-frequency", help="Minimum BPE merge frequency."
    ),
) -> None:
    """Train a byte-level BPE tokenizer (writes tokenizer.json + maps)."""
    from orkhon.tokenizer.train import train_tokenizer

    cfg = _load_yaml_or_empty(config)
    corpus_path = corpus or cfg.get("corpus")
    out_dir = out or cfg.get("out")
    vsize = vocab_size if vocab_size is not None else cfg.get("vocab_size")
    min_freq = (
        min_frequency
        if min_frequency is not None
        else int(cfg.get("min_frequency", 2))
    )

    if corpus_path is None or out_dir is None or vsize is None:
        raise _fail(
            "need corpus, out, and vocab_size (via --config or the flags)."
        )

    _step(f"Train tokenizer (vocab_size={vsize})")
    console.print(f"corpus: [yellow]{corpus_path}[/yellow]  ->  out: [yellow]{out_dir}[/yellow]")
    train_tokenizer(
        corpus_paths=str(corpus_path),
        out_dir=str(out_dir),
        vocab_size=int(vsize),
        min_frequency=min_freq,
    )
    _ok(f"tokenizer written to {out_dir}")


@tokenizer_app.command("retrofit-tools")
def tokenizer_retrofit(
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Existing tokenizer dir."),
    out: Path = typer.Option(..., "--out", help="Output tokenizer dir (with <|tool|>/<image>)."),
) -> None:
    """Append <|tool|>/<image> to an existing tokenizer (preserves all old ids)."""
    from orkhon.tokenizer.retrofit import ensure_tool_tokens

    _step("Retrofit tokenizer with tool/image tokens")
    res = ensure_tool_tokens(tokenizer, out)
    _ok(f"vocab {res['vocab_before']} -> {res['vocab_after']} (added {res['added']}); tool_id={res['tool_id']}")


@tokenizer_app.command("fertility")
def tokenizer_fertility(
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Candidate tokenizer dir."),
    baseline: Optional[Path] = typer.Option(None, "--baseline", help="Optional baseline tokenizer dir."),
    english: Optional[Path] = typer.Option(None, "--english", help="Optional English sample file."),
    turkish: Optional[Path] = typer.Option(None, "--turkish", help="Optional Turkish sample file."),
    old_turkic: Optional[Path] = typer.Option(None, "--old-turkic", help="Optional Old Turkic sample file."),
    min_turkish_bpt: float = typer.Option(3.5, "--min-turkish-bpt"),
    max_old_turkic_tpr: float = typer.Option(1.5, "--max-old-turkic-tpr"),
    max_english_regression: float = typer.Option(0.0, "--max-english-regression"),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if the gate fails."),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="Write gate report JSON."),
) -> None:
    """Measure tokenizer fertility for the pre-R3 multilingual freeze gate."""
    from orkhon.tokenizer.fertility import (
        DEFAULT_ENGLISH,
        DEFAULT_OLD_TURKIC,
        DEFAULT_TURKISH,
        evaluate_fertility_gate,
        read_sample_file,
    )

    _step("Tokenizer fertility gate")
    english_samples = read_sample_file(english) if english else DEFAULT_ENGLISH
    turkish_samples = read_sample_file(turkish) if turkish else DEFAULT_TURKISH
    old_turkic_samples = read_sample_file(old_turkic) if old_turkic else DEFAULT_OLD_TURKIC
    gate = evaluate_fertility_gate(
        tokenizer,
        baseline_dir=baseline,
        english=english_samples,
        turkish=turkish_samples,
        old_turkic=old_turkic_samples,
        min_turkish_bytes_per_token=min_turkish_bpt,
        max_old_turkic_tokens_per_rune=max_old_turkic_tpr,
        max_english_regression=max_english_regression,
    )

    rows = {
        "passed": gate.passed,
        "english_bytes_per_token": gate.candidate.english_bytes_per_token,
        "turkish_bytes_per_token": gate.candidate.turkish_bytes_per_token,
        "old_turkic_tokens_per_rune": gate.candidate.old_turkic_tokens_per_rune,
    }
    _summary_table("fertility", rows)
    for name, ok in gate.checks.items():
        console.print(f"{'✓' if ok else '✗'} {name}")

    if json_out:
        import json

        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(gate.to_dict(), indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        _ok(f"fertility report written to {json_out}")

    if strict and not gate.passed:
        raise _fail("tokenizer fertility gate failed")


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #


@data_app.command("tool-synth")
def data_tool_synth(
    out: Path = typer.Option(Path("data/tool/sft.jsonl"), "--out"),
    val_out: Optional[Path] = typer.Option(None, "--val-out"),
    n: int = typer.Option(5000, "--n"),
    seed: int = typer.Option(1337, "--seed"),
) -> None:
    """Synthesize tool-call SFT traces (calculator/retrieve/direct-answer)."""
    from orkhon.data.tool_synth import make_tool_sft

    _step("Synthesize tool-use SFT traces")
    res = make_tool_sft(out, out_val=val_out, n=n, seed=seed)
    _ok(f"train {res['train']}  val {res['val']}  -> {res['out']}")


@data_app.command("synth")
def data_synth(
    base: Path = typer.Option(
        Path("data/smoke"), "--base", help="Directory for the smoke datasets."
    ),
) -> None:
    """Regenerate the deterministic smoke datasets (pretrain/sft/dpo)."""
    from orkhon.data.synth import make_all

    _step("Synthesize smoke datasets")
    paths = make_all(base)
    for name, path in paths.items():
        _ok(f"{name}: {path}")


@data_app.command("download")
def data_download(
    dataset: str = typer.Option("tinystories", "--dataset", help="tinystories | fineweb-edu | <hf-repo-id>."),
    split: str = typer.Option("train", "--split", help="Split (tinystories: train|valid)."),
    out: Path = typer.Option(..., "--out", help="Output .txt (one doc per line)."),
    max_stories: Optional[int] = typer.Option(
        None, "--max-stories", help="Cap docs (tinystories)."
    ),
    max_docs: Optional[int] = typer.Option(
        None, "--max-docs", help="Cap docs (fineweb-edu / hf datasets)."
    ),
    name: Optional[str] = typer.Option(None, "--name", help="HF dataset config name."),
    text_field: str = typer.Option("text", "--text-field", help="Text field for hf datasets."),
    min_chars: int = typer.Option(200, "--min-chars", help="Skip docs shorter than this (hf datasets)."),
) -> None:
    """Download a real public corpus into the one-doc-per-line .txt format."""
    _step(f"Download {dataset} ({split})")
    console.print(f"-> [yellow]{out}[/yellow]")
    prog = lambda c: console.print(f"  {c:,} docs…")

    if dataset == "tinystories":
        from orkhon.data.download import download_tinystories
        n = download_tinystories(out, split=split, max_stories=max_stories, on_progress=prog)
    elif dataset == "fineweb-edu":
        from orkhon.data.hf_text import download_fineweb_edu
        n = download_fineweb_edu(out, max_docs=max_docs, on_progress=prog)
    elif dataset == "wikipedia-tr":
        from orkhon.data.hf_text import download_wikipedia_tr
        n = download_wikipedia_tr(out, max_docs=max_docs, on_progress=prog)
    else:  # treat as a generic HF dataset repo id
        from orkhon.data.hf_text import stream_hf_text
        n = stream_hf_text(
            dataset, out, name=name, split=split, text_field=text_field,
            max_docs=max_docs, min_chars=min_chars, on_progress=prog,
        )
    _ok(f"wrote {n:,} docs to {out}")


@data_app.command("instruct")
def data_instruct(
    corpus: Path = typer.Option(..., "--corpus", help="Story corpus .txt (one per line)."),
    out_sft: Path = typer.Option(
        Path("data/instruct/tinystories_sft.jsonl"), "--out-sft", help="SFT jsonl output."
    ),
    out_dpo: Path = typer.Option(
        Path("data/instruct/tinystories_dpo.jsonl"), "--out-dpo", help="DPO jsonl output."
    ),
    max_examples: int = typer.Option(4000, "--max-examples"),
    seed: int = typer.Option(1337, "--seed"),
) -> None:
    """Synthesize instruction (SFT) + preference (DPO) data from a story corpus."""
    from orkhon.data.instruct_synth import make_story_instructions

    _step("Synthesize instruction datasets")
    counts = make_story_instructions(
        corpus, out_sft, out_dpo, max_examples=max_examples, seed=seed
    )
    for k, v in counts.items():
        _ok(f"{k}: {v}")


@data_app.command("curate")
def data_curate(
    corpus: Path = typer.Option(..., "--corpus", help="Input corpus .txt (one doc/line)."),
    out: Path = typer.Option(..., "--out", help="Cleaned corpus .txt."),
    dedupe: str = typer.Option("exact", "--dedupe", help="none | exact | minhash."),
    ngram: int = typer.Option(13, "--ngram", help="Decontam n-gram length."),
) -> None:
    """Dedup + decontaminate a corpus (streaming; writes one cleaned doc per line)."""
    from orkhon.data.decontam import build_ngram_index, decontaminate
    from orkhon.data.dedupe import dedupe_exact, dedupe_minhash
    from orkhon.data.normalize import iter_documents
    from orkhon.eval.benchmarks import builtin_examples

    _step("Curate corpus (dedup + decontam)")
    docs = iter_documents(corpus)
    if dedupe == "exact":
        docs = dedupe_exact(docs)
    elif dedupe == "minhash":
        docs = dedupe_minhash(docs)
    elif dedupe != "none":
        raise _fail(f"--dedupe must be none|exact|minhash, got {dedupe!r}")
    # Decontaminate against the built-in eval set (add real benchmarks later).
    bench = [e["context"] + " " + " ".join(e["choices"]) for e in builtin_examples()]
    idx = build_ngram_index(bench, n=ngram)
    n_in = n_out = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as w:
        for doc in decontaminate(docs, idx, n=ngram):
            w.write(doc + "\n")
            n_out += 1
    _ok(f"kept {n_out} docs (dedupe={dedupe}, decontam ngram={ngram}) -> {out}")


@data_app.command("prepare")
def data_prepare(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML with corpus/tokenizer/out."
    ),
    corpus: Optional[Path] = typer.Option(None, "--corpus", help="Corpus path."),
    tokenizer: Optional[Path] = typer.Option(
        None, "--tokenizer", help="Tokenizer directory."
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Prepared output dir."),
    val_fraction: Optional[float] = typer.Option(
        None, "--val-fraction", help="Fraction of docs routed to validation."
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Split hash seed."),
    sharded: bool = typer.Option(False, "--sharded", help="Stream to sharded bins (no RAM ceiling)."),
) -> None:
    """Tokenize and pack a corpus into train.bin / val.bin / meta.json."""
    from orkhon.data.tokenize import prepare_pretrain

    cfg = _load_yaml_or_empty(config)
    corpus_path = corpus or cfg.get("corpus")
    tok_dir = tokenizer or cfg.get("tokenizer_dir") or cfg.get("tokenizer")
    out_dir = out or cfg.get("out")
    vfrac = val_fraction if val_fraction is not None else float(cfg.get("val_fraction", 0.1))
    sd = seed if seed is not None else int(cfg.get("seed", 1337))

    if corpus_path is None or tok_dir is None or out_dir is None:
        raise _fail(
            "need corpus, tokenizer, and out (via --config or the flags)."
        )

    _step("Prepare pretraining shards")
    console.print(
        f"corpus: [yellow]{corpus_path}[/yellow]  tokenizer: [yellow]{tok_dir}[/yellow]"
        f"  ->  out: [yellow]{out_dir}[/yellow]"
    )
    if sharded:
        from orkhon.data.shard import prepare_pretrain_sharded
        meta = prepare_pretrain_sharded(str(corpus_path), str(tok_dir), str(out_dir), val_fraction=vfrac, seed=sd)
    else:
        meta = prepare_pretrain(
        corpus_path=str(corpus_path),
        tokenizer_dir=str(tok_dir),
        out_dir=str(out_dir),
        val_fraction=vfrac,
        seed=sd,
    )
    _summary_table("prepared shards", meta)
    _ok(f"shards written to {out_dir}")


# --------------------------------------------------------------------------- #
# train                                                                       #
# --------------------------------------------------------------------------- #


@train_app.command("pretrain")
def train_pretrain(
    config: Path = typer.Option(..., "--config", "-c", help="Pretrain YAML."),
    set_: list[str] = typer.Option(
        [], "--set", help="Dotlist override, e.g. --set train.max_steps=10.",
    ),
) -> None:
    """Run the pretraining stage and checkpoint to its out_dir."""
    from orkhon.train import pretrain

    cfg = load_stage_config(PretrainConfig, config, set_)
    _step("Pretrain")
    console.print(f"out_dir: [yellow]{cfg.train.out_dir}[/yellow]  steps: {cfg.train.max_steps}")
    summary = pretrain.run(cfg)
    _summary_table("pretrain summary", summary)
    _ok(f"checkpoints in {cfg.train.out_dir}")


@train_app.command("sft")
def train_sft(
    config: Path = typer.Option(..., "--config", "-c", help="SFT YAML."),
    set_: list[str] = typer.Option(
        [], "--set", help="Dotlist override, e.g. --set train.max_steps=10.",
    ),
) -> None:
    """Run supervised fine-tuning (assistant-only loss) from a pretrained ckpt."""
    from orkhon.train import sft

    cfg = load_stage_config(SFTConfig, config, set_)
    _step("Supervised fine-tuning")
    console.print(
        f"init_from: [yellow]{cfg.init_from}[/yellow]  ->  out_dir: "
        f"[yellow]{cfg.train.out_dir}[/yellow]"
    )
    summary = sft.run(cfg)
    _summary_table("sft summary", summary)
    _ok(f"checkpoints in {cfg.train.out_dir}")


@train_app.command("dpo")
def train_dpo(
    config: Path = typer.Option(..., "--config", "-c", help="DPO YAML."),
    set_: list[str] = typer.Option(
        [], "--set", help="Dotlist override, e.g. --set train.max_steps=10.",
    ),
) -> None:
    """Run Direct Preference Optimization against a frozen reference model."""
    from orkhon.train import dpo

    cfg = load_stage_config(DPOConfig, config, set_)
    _step("Direct preference optimization")
    console.print(
        f"init_from: [yellow]{cfg.init_from}[/yellow]  ref_from: "
        f"[yellow]{cfg.ref_from or cfg.init_from}[/yellow]  ->  "
        f"[yellow]{cfg.train.out_dir}[/yellow]"
    )
    summary = dpo.run(cfg)
    _summary_table("dpo summary", summary)
    _ok(f"checkpoints in {cfg.train.out_dir}")


@train_app.command("grpo")
def train_grpo(
    config: Path = typer.Option(..., "--config", "-c", help="GRPO YAML."),
    set_: list[str] = typer.Option(
        [], "--set", help="Dotlist override, e.g. --set train.max_steps=10.",
    ),
) -> None:
    """Run GRPO/RLVR: verifiable-reward RL with group-relative advantages."""
    from orkhon.config.schema import GRPOConfig
    from orkhon.train import grpo

    cfg = load_stage_config(GRPOConfig, config, set_)
    _step("GRPO / RLVR (verifiable rewards)")
    console.print(
        f"init_from: [yellow]{cfg.init_from}[/yellow]  beta {cfg.beta} "
        f"group {cfg.group_size}  reward={cfg.reward_kind}"
    )
    summary = grpo.run(cfg)
    _summary_table("grpo summary", summary)
    _ok(f"checkpoints in {cfg.train.out_dir}")


# --------------------------------------------------------------------------- #
# rag                                                                         #
# --------------------------------------------------------------------------- #


@rag_app.command("ingest")
def rag_ingest(
    paths: list[Path] = typer.Argument(..., help="Files/dirs to ingest."),
    out: Path = typer.Option(Path("data/rag/index"), "--out", help="Index output dir."),
    embed_model: str = typer.Option("hashing", "--embed-model",
                                    help="'hashing' (offline) or a sentence-transformers model name."),
    chunk_chars: int = typer.Option(1200, "--chunk-chars"),
    overlap: int = typer.Option(160, "--overlap"),
) -> None:
    """Ingest files into a persisted RAG index (chunks + embeddings)."""
    from orkhon.rag import ingest as _ingest

    _step("RAG ingest")
    summary = _ingest([str(p) for p in paths], out, embed_model=embed_model,
                      chunk_chars=chunk_chars, overlap=overlap)
    _summary_table("rag ingest", summary)


@rag_app.command("search")
def rag_search(
    query: str = typer.Argument(..., help="Query string."),
    index: Path = typer.Option(Path("data/rag/index"), "--index", help="Index dir."),
    top_k: int = typer.Option(5, "--top-k"),
) -> None:
    """Search a RAG index and print cited passages."""
    from orkhon.rag import format_hits, retrieve
    from orkhon.rag.embed import build_embedder
    from orkhon.rag.store import VectorStore

    store = VectorStore.load(index)
    emb = build_embedder(store.embed_model or "hashing")
    hits = retrieve(query, store, emb, top_k=top_k)
    if not hits:
        console.print("[no results]")
        return
    console.print(format_hits(hits))


# --------------------------------------------------------------------------- #
# eval                                                                        #
# --------------------------------------------------------------------------- #


@app.command("eval")
def eval_cmd(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint dir."),
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Tokenizer dir."),
    prepared: Optional[Path] = typer.Option(
        None,
        "--prepared",
        help="Prepared shards dir (uses its val.bin). Defaults to data/prepared/smoke.",
    ),
    split: str = typer.Option("val", "--split", help="val | train shard."),
    tag: str = typer.Option("last", "--tag", help="Checkpoint tag (last|best)."),
    device: str = typer.Option("auto", "--device", help="Device preference."),
    max_batches: int = typer.Option(50, "--max-batches", help="Batch cap."),
    batch_size: int = typer.Option(8, "--batch-size", help="Rows per batch."),
    seq_len: int = typer.Option(256, "--seq-len", help="Eval window length."),
) -> None:
    """Compute perplexity of a checkpoint over a prepared validation shard."""
    from orkhon.data.pack import PackedDataset
    from orkhon.eval.perplexity import evaluate
    from orkhon.train.checkpoint import load_model_from_checkpoint
    from orkhon.utils.device import resolve_device

    prepared_dir = prepared or Path("data/prepared/smoke")
    bin_path = Path(prepared_dir) / f"{split}.bin"
    if not bin_path.exists():
        raise _fail(f"shard not found: {bin_path} (run `orkhon data prepare` first).")

    _step("Evaluate perplexity")
    dev = resolve_device(device)
    model, _cfg = load_model_from_checkpoint(checkpoint, device=dev, tag=tag)
    model.eval()
    dataset = PackedDataset(bin_path, seq_len=seq_len)
    result = evaluate(
        model,
        dataset,
        max_batches=max_batches,
        batch_size=batch_size,
        seq_len=seq_len,
        device=dev,
    )
    _summary_table(f"perplexity ({split})", result)
    _ok("evaluation complete")


@app.command("bench")
def bench_cmd(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint dir."),
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Tokenizer dir."),
    task: str = typer.Option("builtin", "--task",
                             help="MC: builtin|hellaswag|arc_easy. Generative: gsm8k|mbpp."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Cap examples."),
    device: str = typer.Option("cpu", "--device", help="Device preference."),
    tag: str = typer.Option("last", "--tag", help="Checkpoint tag (last|best)."),
    samples: int = typer.Option(1, "--samples", help="(generative) samples per problem."),
    pass_at: str = typer.Option("1", "--pass-at", help="(generative) k, e.g. 1 or 4."),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens"),
    temperature: float = typer.Option(0.0, "--temperature",
                                      help="(generative) sampling temp; pass@k>1 needs >0."),
    fixture: bool = typer.Option(False, "--fixture", help="Use the offline fixture (no download)."),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="Write a durable JSON report."),
) -> None:
    """Benchmark a checkpoint: multiple-choice (acc) or generative (pass@k)."""
    from orkhon.tokenizer import load_tokenizer
    from orkhon.train.checkpoint import load_model_from_checkpoint
    from orkhon.utils import resolve_device

    _step(f"Benchmark: {task}")
    dev = resolve_device(device)
    model, _ = load_model_from_checkpoint(str(checkpoint), device=dev, tag=tag)
    model.eval()
    tok = load_tokenizer(str(tokenizer))
    report: dict = {"task": task, "checkpoint": str(checkpoint), "tokenizer": str(tokenizer),
                     "tag": tag, "limit": limit, "device": str(dev)}

    if task in ("gsm8k", "mbpp"):
        from orkhon.eval.generative import run_generative_task
        from orkhon.eval.generative_tasks import load_generative_task

        examples = load_generative_task(task, limit, fixture=fixture)
        k = int(pass_at.split(",")[0])
        if samples > 1 and temperature == 0.0:
            console.print("[yellow]warn: --samples > 1 with temperature 0 produces identical "
                          "samples; pass@k collapses to pass@1. Set --temperature > 0.[/yellow]")
        temp = temperature if temperature > 0 or samples == 1 else 0.7

        def gen(prompt: str) -> str:
            from orkhon.model import generate as _gen
            ids = [tok.special.eos] + tok.encode(prompt)
            new = _gen(model, ids, max_new_tokens, temperature=temp, top_k=40,
                       eos_ids=(tok.special.end,), device=dev)
            return tok.decode(new, skip_special=True)

        res = run_generative_task(examples, gen, grader=task, n_samples=samples, k=k,
                                  max_new_tokens=max_new_tokens)
        report.update({"n": res.n, f"pass@{k}": round(res.pass_at_k, 4),
                       "pass@1": round(res.pass_at_1, 4),
                       "mean_reward": round(res.mean_reward, 4),
                       "samples": samples, "temperature": temp,
                       "max_new_tokens": max_new_tokens, "fixture": fixture})
        _summary_table(f"{task} (n={res.n}, samples={samples})", {k: v for k, v in report.items()
                                                                  if isinstance(v, (int, float))})
        _ok(f"pass@{k} {res.pass_at_k:.3f}  pass@1 {res.pass_at_1:.3f}")
    else:
        # Multiple-choice path.
        from orkhon.eval.benchmarks import load_task
        from orkhon.eval.loglikelihood import run_multiple_choice

        examples = load_task(task, limit)
        res = run_multiple_choice(examples, model, tok, device=dev)
        report.update(res)
        _summary_table(f"{task} (n={res['n']})", res)
        _ok(f"acc {res['acc']:.3f}  acc_norm {res['acc_norm']:.3f}")

    if json_out:
        import json
        from datetime import datetime
        report["timestamp"] = datetime.now().isoformat(timespec="seconds")
        json_out = Path(json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        _ok(f"report written to {json_out}")


# --------------------------------------------------------------------------- #
# chat                                                                        #
# --------------------------------------------------------------------------- #


@app.command("agent")
def agent_cmd(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint dir."),
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Tokenizer dir."),
    device: str = typer.Option("auto", "--device"),
    tag: str = typer.Option("last", "--tag"),
    tools: Optional[str] = typer.Option(None, "--tools", help="Comma-separated tools (calculator,read_file)."),
    file_root: Optional[Path] = typer.Option(None, "--file-root", help="Root dir for read_file."),
    rag_index: Optional[Path] = typer.Option(None, "--rag-index", help="RAG index dir."),
    max_steps: int = typer.Option(5, "--max-steps"),
    max_new_tokens: int = typer.Option(160, "--max-new-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
) -> None:
    """Run a bounded agent (plan -> act -> observe) with tools/RAG."""
    from orkhon.serve.chat_cli import agent_repl

    agent_repl(
        checkpoint_dir=str(checkpoint), tokenizer_dir=str(tokenizer), device=device, tag=tag,
        tools=tools.split(",") if tools else None,
        file_roots=[str(file_root)] if file_root else None,
        rag_index=str(rag_index) if rag_index else None,
        max_steps=max_steps, max_new_tokens=max_new_tokens, temperature=temperature,
    )


@app.command("chat")
def chat_cmd(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint dir."),
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Tokenizer dir."),
    device: str = typer.Option("auto", "--device", help="Device preference."),
    tag: str = typer.Option("last", "--tag", help="Checkpoint tag (last|best)."),
    system: Optional[str] = typer.Option(
        None, "--system", help="Optional leading system prompt."
    ),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens"),
    temperature: float = typer.Option(0.8, "--temperature"),
    top_k: Optional[int] = typer.Option(40, "--top-k"),
    top_p: Optional[float] = typer.Option(None, "--top-p"),
    repetition_penalty: float = typer.Option(1.3, "--repetition-penalty"),
    tools: Optional[str] = typer.Option(
        None, "--tools", help="Comma-separated tools to enable (calculator,read_file)."
    ),
    file_root: Optional[Path] = typer.Option(None, "--file-root", help="Root dir for read_file."),
    rag_index: Optional[Path] = typer.Option(None, "--rag-index", help="RAG index dir (enables retrieve)."),
) -> None:
    """Open an interactive chat REPL with a trained checkpoint."""
    from orkhon.serve.chat_cli import chat

    chat(
        checkpoint_dir=str(checkpoint),
        tokenizer_dir=str(tokenizer),
        device=device,
        tag=tag,
        system_prompt=system,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        tools=tools.split(",") if tools else None,
        file_roots=[str(file_root)] if file_root else None,
        rag_index=str(rag_index) if rag_index else None,
    )


@app.command("generate")
def generate_cmd(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint dir."),
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Tokenizer dir."),
    prompt: str = typer.Option("Once upon a time", "--prompt", "-p", help="Prompt to continue."),
    device: str = typer.Option("auto", "--device", help="Device preference."),
    tag: str = typer.Option("last", "--tag", help="Checkpoint tag (last|best)."),
    max_new_tokens: int = typer.Option(200, "--max-new-tokens"),
    temperature: float = typer.Option(0.8, "--temperature"),
    top_k: Optional[int] = typer.Option(40, "--top-k"),
    top_p: Optional[float] = typer.Option(None, "--top-p"),
    repetition_penalty: float = typer.Option(1.3, "--repetition-penalty", help=">1 discourages repeats."),
) -> None:
    """Continue a text prompt with a base (pretrained) model."""
    from orkhon.serve.complete import complete

    text = complete(
        checkpoint_dir=str(checkpoint),
        tokenizer_dir=str(tokenizer),
        prompt=prompt,
        device=device,
        tag=tag,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    console.print(text)


@app.command("import-hf")
def import_hf_cmd(
    repo: str = typer.Option(..., "--repo", help="HF repo id (org/model) or local dir."),
    out: Path = typer.Option(..., "--out", help="Output Orkhon checkpoint dir."),
    revision: Optional[str] = typer.Option(None, "--revision", help="Git revision."),
    device: str = typer.Option("cpu", "--device", help="Device to load on."),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="float32|bfloat16|float16."),
) -> None:
    """Load an HF Llama-architecture model into an Orkhon checkpoint (SFT/serve-ready)."""
    import torch

    from orkhon.model.hf import from_pretrained, save_as_orkhon_checkpoint

    dt = {None: None, "float32": torch.float32, "bfloat16": torch.bfloat16,
          "float16": torch.float16}.get(dtype, None)
    _step(f"Import {repo}")
    model, cfg = from_pretrained(repo, device=device, dtype=dt, revision=revision)
    console.print(f"loaded [green]{cfg.estimate_params()/1e6:.0f}M[/green] params "
                  f"({cfg.n_layers}L · d{cfg.d_model} · vocab {cfg.vocab_size})")
    path = save_as_orkhon_checkpoint(model, cfg, out)
    _ok(f"wrote Orkhon checkpoint to {path}")


@app.command("register")
def register_cmd(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Trained checkpoint dir."),
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Tokenizer dir."),
    name: Optional[str] = typer.Option(None, "--name", help="Codename (auto if omitted)."),
    kind: str = typer.Option("base", "--kind", help="base | instruct."),
    lineage: str = typer.Option("", "--lineage", help="One-line description of how it was made."),
    mode: str = typer.Option("complete", "--mode", help="complete (base) | chat (instruct)."),
    prompt: list[str] = typer.Option([], "--prompt", help="Sample prompt (repeatable)."),
    eval_prepared: Optional[Path] = typer.Option(None, "--eval-prepared", help="Prepared dir for perplexity."),
    device: str = typer.Option("cpu", "--device", help="Device for sampling/eval."),
    root: Path = typer.Option(Path("models"), "--root", help="Model zoo root."),
) -> None:
    """Archive a trained model into the dated, named model zoo."""
    from orkhon.registry import build_index, next_name, register_model

    nm = name or next_name(root)
    prompts = prompt or (["Write a short story about a lost puppy."] if mode == "chat"
                         else ["Once upon a time"])
    _step(f"Register model '{nm}'")
    dest = register_model(
        nm, str(checkpoint), str(tokenizer), kind=kind, lineage=lineage,
        sample_prompts=prompts, generate_mode=mode,
        eval_prepared=str(eval_prepared) if eval_prepared else None,
        device=device, dest_root=str(root),
    )
    build_index(root)
    _ok(f"archived to {dest}")


@app.command("registry")
def registry_cmd(
    root: Path = typer.Option(Path("models"), "--root", help="Model zoo root."),
) -> None:
    """Rebuild and print the model zoo index."""
    from orkhon.registry import build_index

    out = build_index(root)
    console.print(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# serve                                                                       #
# --------------------------------------------------------------------------- #


@app.command("serve")
def serve_cmd(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint dir."),
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Tokenizer dir."),
    device: str = typer.Option("auto", "--device", help="Device preference."),
    tag: str = typer.Option("last", "--tag", help="Checkpoint tag (last|best)."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    tools: Optional[str] = typer.Option(None, "--tools", help="Comma-separated tools to enable."),
    file_root: Optional[Path] = typer.Option(None, "--file-root", help="Root dir for read_file."),
    rag_index: Optional[Path] = typer.Option(None, "--rag-index", help="RAG index dir."),
) -> None:
    """Serve an OpenAI-compatible chat + agent API (FastAPI + uvicorn)."""
    import uvicorn

    from orkhon.serve.api import create_app

    _step(f"Serve API on http://{host}:{port}")
    app_obj = create_app(
        checkpoint_dir=str(checkpoint),
        tokenizer_dir=str(tokenizer),
        device=device,
        tag=tag,
        tools=tools.split(",") if tools else None,
        file_roots=[str(file_root)] if file_root else None,
        rag_index=str(rag_index) if rag_index else None,
    )
    uvicorn.run(app_obj, host=host, port=port, log_level="info")


# --------------------------------------------------------------------------- #
# export                                                                      #
# --------------------------------------------------------------------------- #


@export_app.command("hf")
def export_hf(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint dir."),
    out: Path = typer.Option(..., "--out", help="Export output directory."),
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Tokenizer dir."),
    tag: str = typer.Option("last", "--tag", help="Checkpoint tag (last|best)."),
) -> None:
    """Export a checkpoint to a HuggingFace-style directory (with parity check)."""
    from orkhon.export.to_hf import export

    _step("Export to HuggingFace format")
    console.print(f"checkpoint: [yellow]{checkpoint}[/yellow]  ->  out: [yellow]{out}[/yellow]")
    out_path = export(
        checkpoint_dir=str(checkpoint),
        out_dir=str(out),
        tokenizer_dir=str(tokenizer),
        tag=tag,
    )
    _ok(f"exported to {out_path}")


# --------------------------------------------------------------------------- #
# YAML helper (kept last so command defs read top-down)                       #
# --------------------------------------------------------------------------- #


def _load_yaml_or_empty(path: Optional[Path]) -> dict:
    """Load a small flat YAML for tokenizer/data commands, or {} if no path."""
    if path is None:
        return {}
    from orkhon.config.load import load_yaml

    return load_yaml(path)


def main() -> None:
    """Console-script / ``python -m orkhon`` entry point."""
    app()


if __name__ == "__main__":
    main()
