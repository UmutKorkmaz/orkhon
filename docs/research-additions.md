# What Else to Add to Orkhon — Research Synthesis

> A prioritized list of additions **beyond** the existing plans (`roadmap.md` scale axis, `capability-roadmap.md`
> C0–C8, `turkic-languages.md`). The goal is to separate high-leverage engineering work from attractive but
> premature model ambitions.
>
> Lens: a **solo builder** on an M5 Pro + occasional cheap cloud GPU. Almost everything below **builds and tests
> for $0 on the laptop**; only training at scale costs money, and most of that rides *inside* the already-budgeted
> R-ladder runs rather than adding new spend.

## TL;DR — the picture

Across the reviewed directions, the unifying insight is that the highest-value additions are **infrastructure
and measurement that the scale plan already half-needs** (ops spine, data
curation, eval) plus **a few cheap force-multipliers** (`generate_batch`, distillation) — *not* new model
ambitions. The Turkic identity (a Turkish leaderboard, pan-Turkic, the inscriptions) is the differentiator that
justifies a from-scratch project at all.

---

## Tier 1 — highest value, add next

| # | Addition | Why it's top | Effort | Slots into |
|---|----------|--------------|--------|------------|
| **1** | **Ops spine** — `train/monitor.py` (W&B/Trackio + MFU/loss-spike alerts), **`git init` + GitHub CI**, NaN/spike guard, provenance stamped into checkpoints, step-keyed resumable dataloader cursor, DCP async checkpoint | Cross-cutting prerequisite for *every* multi-day cloud run; the roadmap's P0. You can't safely spend on R2+ without it. ~4 of the days are $0 on the laptop. | ~8–9 d (≈$0 laptop; ~$20–60 for one multi-GPU resume-parity test) | roadmap **P0** / capability **C0** |
| **2** | **Data flywheel** — `data/curate/`: sharded streaming tokenize (kills the in-RAM single-`.bin` ceiling), MinHash dedup, 13-gram decontam, quality classifier, provenance → model card | The #1 thing blocking *any* run past ~500M tokens. Unblocks the whole scale ladder; identity-relevant (a clean, auditable Turkish corpus). | ~8–12 d ($0 laptop) | roadmap **P0** |
| **3** | **Evaluation + a Turkish leaderboard** — the `lm-eval` bridge (`eval/harness.py`) + an **Orkhon Open Turkish LLM Leaderboard** HF Space (TR-MMLU/TurkBench + contamination report) | You're flying blind without benchmarks; and *hosting* a Turkish leaderboard is a real community/identity win that costs ~$0. Validate the bridge free on imported `kashgari`. | ~5–7 d ($0) | roadmap **P0** + capability |
| **4** | **`generate_batch()` + serving** — batched generation + paged/evictable KV-cache + a tiny continuous-batch scheduler; plus a **repetition penalty** in `sampling.py` | The secret universal unlock: it's the shared dependency for **reasoning/GRPO, batched serving, and fast eval** — and the repetition penalty *immediately* fixes the loops you saw in `bengü`/`istemi`. | ~8 d ($0 laptop) | capability **C5/C7**, eval |
| **5** | **White-box distillation** — `train/distill.py` (logit/KL distillation from a strong open teacher) | Reaches RL-level capability at **9–30× lower FLOPs than GRPO** (2026 reports). Perfect for a *small-model* project: let `tonyukuk`/`tengri` learn from a big teacher cheaply. | ~4–6 d offline ($0–80) | new train stage |
| **6** | **Open-source & trust layer** — root **LICENSE (Apache-2.0)** + NOTICE/DATA_LICENSES, upgraded model cards (HF front-matter, contamination, limitations, CO₂), GOVERNANCE/CONTRIBUTING, an **R3 tech report** | Cheap, and it's *how a solo project gets noticed and funded*. The `bengü` license gate (Uppsala CC BY-NC-SA) matters legally. | ~6–9 d ($0) | capability **C0** |
| **7** | **Code + Math verticals** — FIM code training (Stack-v2 slice, already a planned R3 10–15%) + a verifiable-reward math line (`\boxed{}`/sympy), both with sandboxed eval | High capability gain for ~$0 incremental — the training **rides inside the R3 token budget**; only the FIM transform + evals are net-new. A Turkish coding/math tutor is a tangible product. | ~14 d each ($0–50 laptop) | rides **R3** |

**The single most underrated addition:** **`generate_batch()`** (#4). It looks like a serving optimization, but
it's quietly the **one missing primitive** that three different roadmap items (GRPO reasoning, batched inference,
fast in-loop eval) all wait on — and the same change carries the repetition-penalty fix that makes *every current
model* read better today. Build it once, unblock four things.

---

## Tier 2 — nice, but later

- **Reasoning / GRPO-RLVR** — the full DeepSeek-R1 stack; Phase A (rewards + rejection sampling) is $0 on laptop, but real GRPO gates on **R4 (otuken 1B)** and ~$500–1.5k. (Distillation #5 captures most of the gain sooner.)
- **Long context (YaRN→32k)** — decouple the RoPE table + `rope_scaling` (a 1–2 day $0 win to *evaluate* 2k models at 8–16k), then a ~$150 32k CPT for a long-doc `bengü`.
- **Pan-Turkic** (Azerbaijani/Kazakh/Uzbek/Uyghur…) — a 5–8% Turkic slice *inside* the R3 mix + multi-script tokenizer seed (must land in the pre-R3 freeze); a strong identity play, but gate a dedicated run on a transfer-proof.
- **Speech/audio** — `orkhon listen` (Whisper-tr ASR → chat → TTS), serve-time, $0, ~5–7 d; native codec-token TTS is a later 12–18 d project.
- **Safety/alignment** — a thin refusal/red-team SFT+DPO layer + a Turkish safety eval; ~7–9 d, mostly $0, folds into R3 post-training.
- **Embeddings/retrieval** — reuse BGE-M3 now; a *Turkic* embedding model only once RAG (C3) proves the need.
- **Domain vertical** — pick **one**: a Turkish-law RAG assistant ("Orkhon Hukuk") on the C3 RAG + free `yargi-mcp`/`mevzuat-mcp` tools.
- **Product/UX** — promote the FastAPI server into a Gradio HF Space + OpenAI-compatible SDK shim (~3–5 d).
- **Funding** — `docs/funding.md`: HF/NVIDIA/TÜBİTAK/EuroHPC grants + a run-cost logger to *offset* the R2–R4 spend with credits.
- **Model merging / continual learning** — `mergekit`-style SLERP/TIES to fuse specialist fine-tunes (a code `bengü` + a chat `bengü`) without retraining; cheap, late.

## Tier 3 — skip / out of scope (for now)

- **Internal MoE** — explicitly out of roadmap v1; revisit only after dense R4/R5 and FP8 land. (Self-speculative *decoding* is the worthwhile efficiency win instead.)
- **Mamba/SSM hybrids** — would rewrite the model + lose Llama-HF parity; SWA+YaRN+RAG covers the real long-context need.
- **From-scratch embedding model / generic frontier chase** — futile vs SOTA; reuse, don't reinvent.

---

## How these map onto the two existing axes

- **Unblock scale (do first):** #1 ops spine + #2 data flywheel + #3 eval = the roadmap's P0, now with concrete owners.
- **Force-multipliers:** #4 `generate_batch` (+ rep-penalty) and #5 distillation make small models punch up cheaply.
- **Identity/differentiation:** the Turkish leaderboard (#3), pan-Turkic, Göktürk transliteration, and the tech
  report (#6) are what make Orkhon *Orkhon* — the from-scratch Turkic stack — rather than another small Llama clone.

*Verdict: do the measurement and infrastructure first; add new model ambitions only when they ride an existing
scale gate or unlock the Turkic identity directly.*
