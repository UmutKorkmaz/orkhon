# The Orkhon Lineage

![Orkhon model lineage: progressively larger inscription stones connected by training signals](assets/model-lineage.png)

> Every Orkhon model is named for a figure or place from Turkic history. Read in order, the names tell a
> single story — the same one the project is named for: from the **Orkhon inscriptions**, the oldest known
> Turkic writing, to a system that writes language itself.

## The arc

> It begins with **founders** (`bumin`, `istemi`), passes through **counselors and princes**
> (`tonyukuk`, `kultigin`, `bilge`), branches into **language and inscription work**
> (`kashgari`, `bengü`, `bengü-göktürk`), rises to the **sky and homeland**
> (`tengri`, `ötüken`), reaches **wisdom itself** (`kutadgu`), then escalates through
> **epic ancestors and world-conquerors** (`oğuz`, `manas`, `tarkan`, `ergenekon`, `atilla`,
> `timur`) — and culminates in `korkut`, the immortal sage, the largest and wisest model
> Orkhon could ever be.

The same hand-written Orkhon code sits at every rung of this ladder. Nothing about the architecture changes
as the models grow — only two numbers do: **how many parameters** the model has, and **how many tokens** it
is trained on. That is the whole of it.

## Two numbers, never confused

- **Parameters** = the size of the model (the "brain"). This fixes the file size on disk:
  `disk ≈ params × bytes-per-weight` (4 bytes in fp32, 2 in bf16, ~0.5 quantized to int4).
- **Tokens** = how much text it reads while training. This fixes how *capable* it becomes, not how *big*.

The rule that drives the whole roadmap: for a model you intend to use, feed it **≈ 100–200× its parameter
count in tokens**. A 50M model wants ~10B tokens; a 1B model wants ~200B. Training longer makes a model
*smarter*, never *larger*.

---

## Models so far — trained and archived

Each lives in `models/<name>-<date>/` with its weights, tokenizer, sample outputs, eval metrics, a snapshot of
the exact code that produced it, and a `run.sh`. See [the model zoo](#the-model-zoo) below.

| Name | Params | Trained on | What it is | Namesake |
|------|--------|------------|------------|----------|
| **bumin** | 4M | synthetic arithmetic | answers `What is 2 plus 2? → Answer: 4.` | **Bumin Qaghan** — founder of the first Göktürk Khaganate; the *first* model |
| **tonyukuk** | 22M | TinyStories (47M tok) | writes fluent, coherent short stories | **Tonyukuk** — the wise statesman-advisor, with his own Orkhon-era inscription |
| **kultigin** | 22M | story instructions | takes a command and writes a story | **Kül Tigin** — Göktürk prince, subject of an Orkhon inscription |
| **istemi** | 51M | FineWeb-Edu, real web text | first model on real-world web text; fluent style, weak facts | **Istemi Qaghan** — co-founder who expanded the realm westward |
| **kashgari** | 135M | imported (SmolLM2) | a real open base loaded into Orkhon with exact logit parity | **Mahmud al-Kashgari** — wrote the first dictionary of the Turkic languages |
| **bengü** | 57M | EN/TR bilingual text | Turkish-capable base with much better Turkish fertility | ***Bengü Taş*** — the “eternal stone” inscriptions |
| **bengü-göktürk** | 57M | deterministic rune→Latin SFT | Old Turkic script transliterator; not a translator | **Göktürk / Orkhon script** — the project’s namesake writing system |

*Metrics:* bumin ppl 5.1 · tonyukuk ppl 4.8 · istemi ppl 46.5 · bengü ppl 47.1.
Small benchmark JSON reports live under [`reports/`](../reports/README.md).

---

## The climb — laptop to cheap cloud

These are the next models: the same architecture, scaled with more parameters and far more tokens. The first
two rungs are reachable on a single laptop (Apple M5 Pro, MPS); the rest need rented spot GPUs. Costs are
planning estimates; re-price GPU providers before each paid run.

| Name | Params | Tokens | Disk (fp32) | Hardware | Cost | The leap | Namesake |
|------|--------|--------|-------------|----------|------|----------|----------|
| **istemi-R1** | 50M | 10B | ~200 MB | M5 Pro, ~2 wks | free | the first *properly-trained* 50M run (60× more data than archived `istemi`) | **Istemi Qaghan** |
| **bilge** | 125M | 25B | ~500 MB | 8×H100, ~19h | ~$190 | first real signal on the cloud | **Bilge Kağan** — *"the wise"*, the project's own namesake |
| **tengri** ★ | 350M | 70B | ~1.4 GB | 8×H100, ~5.6d | ~$1.35k | **the MVP — first model worth sharing** | **Tengri** — the sky-god, supreme deity of the Turks |
| **otuken** ★ | 1B | 200B | ~4 GB | 16×H100, ~18d | ~$17k | **a genuinely useful base + instruction-following** | **Ötüken** — the sacred homeland where the Turkic state is reborn |
| **balasagun** | 3B | 600B | ~12 GB | 32×GPU, ~36d | ~$83k | funded stretch (needs data mixing + μP) | **Balasagun** — birthplace of the author of the *Kutadgu Bilig* |
| **kutadgu** | 7B | 1.3T | ~28 GB | 64×H100, ~68d | ~$315k+ | brackets OLMo-3-7B; a small frontier model | ***Kutadgu Bilig*** — "the wisdom that brings happiness," the great Karakhanid classic |

★ = recommended stopping points. For a solo builder, **tengri (350M)** is the model worth putting your name on.
For a funded team, **otuken (1B)** is the sweet spot.

---

## The frontier — lab and sovereign scale

Beyond 7B you have, in effect, become a frontier lab: thousands of GPUs, months of wall-clock, millions to
hundreds of millions of dollars. The code path is the same up to ~70B with the roadmap's gaps closed; above
that you need Mixture-of-Experts, FP8, and a multi-thousand-GPU cluster. These rungs are aspirational — the
top of the ladder, named for the grandest figures of the Turkic world.

| Name | Params | Tokens | Compute | Cost (spot) | Disk (bf16) | Namesake |
|------|--------|--------|---------|-------------|-------------|----------|
| **oguz** | 13B | 2.6T | 0.14M H100-hr | ~$0.3M | 26 GB | **Oğuz Kağan** — legendary ancestor of all Oghuz Turks |
| **manas** | 32B | 6.4T | 0.85M H100-hr | ~$2M | 64 GB | hero of the **Epic of Manas**, one of the longest epics on Earth |
| **tarkan** | 70B | 14T | 4.1M H100-hr | ~$8M | 140 GB | **Tarkan** — the supreme military commander (≈ Llama-3-70B economics) |
| **ergenekon** | 130B | 15T | 8.1M H100-hr | ~$16M | 260 GB | **Ergenekon** — the iron-mountain myth of Turkic rebirth and breakout |
| **atilla** | 180B | 15T | 11.3M H100-hr | ~$22M | 360 GB | **Attila** — the world-shaking ruler of the Huns |
| **timur** | 405B | 16T | 27M H100-hr | ~$54M | 810 GB | **Timur (Tamerlane)** — builder of a vast empire (≈ Llama-3.1-405B) |
| **korkut** | ~1T (MoE) | 15T+ | 62M H100-hr | ~$125M+ | 2 TB | **Dede Korkut** — the immortal sage-bard who blesses the Oghuz epics |

`korkut` would have to be a Mixture-of-Experts model — a dense trillion-parameter network is impractical even
for frontier labs. It is the wisest and largest Orkhon could ever be: the project's namesake wisdom
(`bilge` → `kutadgu` → `korkut`) carried to its limit.

---

## What runs where

On the **Apple M5 Pro (48 GB, MPS)** that built this project:

| Model | Verdict on the laptop |
|-------|-----------------------|
| up to **istemi-R1 (50M / 10B)** | ✅ ~2 weeks, free — feasible after the scale-readiness gates |
| **bilge (125M)** | 😬 ~3 months at full budget (~9 days if under-trained) |
| **tengri (350M)** | 😬 the practical ceiling — ~2.5 months, compute-optimal only |
| **otuken (1B)** and up | ❌ years of compute; 3B+ will not even fit in 48 GB of memory |

The reason is physics, not code: training cost ≈ `6 × params × tokens` FLOPs, and one H100 is ~150× faster
than the M5 Pro for this. The laptop's real job is to **build and validate everything** — the data pipeline,
the eval harness, all fine-tuning — so that any paid spot run starts from tested code, frozen assumptions, and
benchmark evidence rather than hope.

---

## The model zoo

Every model is archived by [`orkhon register`](../README.md#model-zoo) into a dated, self-contained folder:

```
models/<name>-<YYYYMMDD>/
  checkpoint/        the weights — runnable directly ("executable")
  tokenizer/         the tokenizer it was trained with
  samples.txt        generated outputs
  eval.json          metrics (perplexity / benchmarks)
  code_snapshot.tgz  the exact src/ + configs that produced it
  model_card.md      ·  manifest.json  ·  run.sh
```

The registry hands out the next codename automatically (`next_name`), in the order of the lineage above, so the
story builds itself as each completed model takes the next measured place. The full index lives in
[`models/registry.md`](../models/registry.md).

## The engineering plan

This document is the *what* and the *why*. The *how* — the staged scaling ladder R0→R6, the prioritized list of
concrete code changes that unlock each rung, real GPU economics, and a phased Next/Near/Mid execution plan — is
in [**`docs/roadmap.md`**](roadmap.md). The decision-making behind every path (RAG → fine-tune → pretrain) is in
[`docs/build-your-own-llm-guide.md`](build-your-own-llm-guide.md).

---

*From the first written words of a language, to a system that writes language. From `bumin` to `korkut`.*
