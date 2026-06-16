# Orkhon's Mother Tongue — Turkish & Göktürk

> A plan addition for teaching Orkhon **modern Turkish** and the **Göktürk / Old Turkic** language of the
> [Orkhon inscriptions](https://en.wikipedia.org/wiki/Orkhon_inscriptions) — the project's own namesake. The
> most thematically fitting feature the model could have: *the system named for the oldest Turkic writing,
> learning to read that writing.*
>
> **Precision:** "Göktürk capability" means script handling, transliteration, sourced translation/glossing, and
> scholarly RAG over attested inscriptions. It does **not** mean fluent free-form Old Turkic generation.
>
> Tokenizer measurements below are from the current Orkhon tokenizer and should be re-run before the R3 freeze.

---

## 1. Two completely different problems

Conflating these is the #1 way this feature goes wrong:

- **Modern Turkish = real multilingual training.** Data is abundant; the binding constraint is the *token budget*
  and the *tokenizer*. Realistic outcome at R3 (`tengri`, 350M): **fluent in Turkish surface form and morphology,
  not deeply knowledgeable** — great for the project's identity and held-out perplexity, not competitive with
  TURNA/Trendyol/Kanarya. Say so in the model card.
- **Göktürk / Old Turkic = an ultra-low-resource transliteration/translation capability + a signature demo.** The
  entire surviving primary corpus (Orkhon + Yenisei + Tonyukuk inscriptions) is **~214 lines / tens of thousands
  of words** — 4–6 orders of magnitude too small to *pretrain a language*. It **cannot** become a fluent Old
  Turkic generator. It **can** become a scholarly **script ⇄ transliteration ⇄ modern-Turkish/English** assistant.

They share exactly **one** hard dependency: the irreversible tokenizer freeze (§2). Everything else is additive.

---

## 2. The ONE irreversible decision — the pre-R3 "tokenizer freeze" ⚠️

The byte-level BPE merge table is irreversible (guide §11): any change — a new special token *or* a changed merge
table — invalidates every checkpoint pretrained on it. Three separate "freeze" flags in the existing docs are
actually **one event** that must happen *before `tengri` (R3, 350M) is pretrained*:

1. `docs/roadmap.md` — "tokenizer 16k→32k(R3)→48k(R4)".
2. `docs/capability-roadmap.md` — append `<|tool|>` (id 8), reserve `<image>` (id 9).
3. **This doc** — multilingual vocab + Turkish + Old-Turkic-script coverage.

**Batch all of it into one commit before R3.** The decision:

| Decision | Choice | Why |
|---|---|---|
| **Vocab size** | **48,000** | Fertility is the whole point, and the cost is paid once. `otuken` (R4, 1B) must inherit `tengri`'s frozen tokenizer, so size for the **1B**, not the 350M. 48k still fits `uint16`. *(Minority view recorded: 32k is defensible in isolation at 350M — more transformer capacity under tied embeddings — but loses once otuken inherits it.)* |
| **dtype** | **uint16, unchanged** | `data/tokenize.py` caps uint16 at vocab < 65536; 48k fits. Don't pay 2× `.bin` storage for `uint32` (an R5+ concern). |
| **Special tokens** | `<\|tool\|>` @ 8, reserve `<image>` @ 9 | append-only; never reorder ids 0–7. Add the two fields to `SpecialIds` + `special_ids()`. |
| **Freeze corpus** | EN (FineWeb) + code + TR (CulturaX/HPLT/wiki) + **Old-Turkic synthetic seed** | the corpus the tokenizer *trains on* is the real lever — over-sample TR/code/runes so they win merges. Objective: **fertility, not loss.** |

**Measured on the live 16k tokenizer (the reason this matters):**

| Language | Now (16k EN) | After (48k ml) target |
|---|---|---|
| Turkish | **~1.65 bytes/tok** (≈2.8–3.3× inflation vs English) | **≥ 3.5 bytes/tok** |
| English | ~4.4 bytes/tok | not below ~4.4 (no regression) |
| Old Turkic runes | **~4.1 tokens/rune** (each rune = 4 UTF-8 bytes) | **≤ 1.5 tokens/rune** |

Fixing Turkish fertility roughly **doubles the effective Turkish a fixed GPU budget buys** — the single largest
*free* win. A new `tokenizer/fertility.py` enforces these targets as the irreversible **go/no-go gate** before the
freeze commits. (Also reconcile the live 16384-vs-32768 config conflict in the same window — verified real.)

> The byte-level BPE **already represents the Old Turkic Unicode block (U+10C00–U+10C4F) losslessly today** —
> verified roundtrip on the live tokenizer, with zero runes ever seen in training (all 256 bytes are in the
> alphabet). The freeze only improves *efficiency*, never *coverage*.

---

## 3. Modern Turkish

### Data (all stream via the existing `data/hf_text.py` path)

| Source | Size | Use |
|---|---|---|
| `uonlp/CulturaX` (`tr`) | **64.3B tokens**, 94.2M docs (gated) | main Turkish web pool |
| `HPLT/HPLT2.0_cleaned` (`tur_Latn`) | **51.7B words**, 390B chars | second main pool |
| `oscar-corpus/OSCAR-2301` (`tr`) | 8.29B words, 73.7 GB | dedup/overlap check |
| Turkish Wikipedia / news / books | small but clean | held-out eval + high-quality replay |
| `malhajar/OpenOrca-tr` (2.35M rows), `TFLai/Turkish-Alpaca` (51.9k) | — | **SFT seed** (quality-filter hard) |

After dedup/quality/license filtering: plan **~50–150B usable Turkish tokens** — more than enough for the 350M/1B
rungs. **Supply is not the constraint; the token budget is.**

### Mix (with English replay so it doesn't collapse)

| Rung | Turkish share |
|---|---|
| R1 50M | English-only baseline + Turkish fertility eval |
| R2 125M | optional 4–8% Turkish proof (~1–2B tok) |
| **R3 350M (`tengri`)** | **5–8% Turkish** (~3.5–5.6B tok) — the freeze rung |
| R4 1B (`otuken`) | 8–12% if Turkish is a headline capability |

Keep Turkish validation **source-separated** (`hplt_tr`, `culturax_tr`, `wiki_tr`, `news_tr`, `sft_tr`).

### SFT + eval
Turkish SFT milestone after R3 (OpenOrca-tr / Turkish-Alpaca, heavily filtered). Blocking Turkish evals once
Turkish enters pretraining: held-out PPL per source, **TR-MMLU** (6,200 Q × 62 sections), **TurkBench** (8,151
samples, 21 subtasks), **TurBLiMP** (16 phenomena × 1,000 minimal pairs for morphology/word-order).

---

## 4. Göktürk / Old Turkic — the signature capability

### The truth (verified)
Core inscriptions = **214 lines** (Bilge Kağan 82, Kül Tigin, Tonyukuk 1/2). The Uppsala *Database of Turkic
Runiform Inscriptions* adds metadata/transliteration/translation per inscription. That's it — **far too little to
pretrain a language.** So Göktürk is **not** a pretraining target; it's a small **SFT fine-tune on top of `tengri`**
that moves text between three representations:

```
Old Turkic script (𐰇𐰏…)  ⇄  Latin transliteration  ⇄  modern Turkish / English gloss
```

### What's achievable (capability ladder)
- **Script handling** — detect/render U+10C00–U+10C4F (already lossless in the tokenizer).
- **Transliteration** — Old Turkic script ⇄ Latin (rule-assisted + learned).
- **Translation/gloss** — attested lines → modern Turkish/English, *source-grounded with a confidence/uncertainty
  field*, using modern Turkish as the bridge language.
- **Scholarly RAG** — retrieve over inscription pages + notes (reuses the capability-roadmap RAG, C3).
- **Refusal mode** — declines free-form Göktürk *composition* (it's not a fluent speaker).

### Data: manufacture signal from almost nothing
Build a small, high-quality parallel corpus `data/old_turkic/inscriptions.jsonl` — fields: `source`,
`inscription`, `line_id`, `old_turkic_unicode`, `transliteration_raw`, `transliteration_normalized`,
`modern_turkish`, `english`, `notes`, `license`, `confidence`. Augment with **deterministic synthetic data**
(script↔translit pairs generated from the rune↔Latin mapping; Turkish→Old-Turkic templated sentences) so the SFT
set is ~10–50× the raw inscription lines.

### The signature demo — *Orkhon reads its namesake*
> Paste a line of the Bilge Kağan inscription (in runes *or* transliteration); Orkhon returns the normalized
> transliteration, the modern Turkish, the English, the source line, and its confidence. The model named after
> the Orkhon inscriptions reading the Orkhon inscriptions — *"from the first written words of a language, to a
> system that writes language,"* closed into a loop.

### Honest ceiling (put this verbatim in the model card)
> *Handles the Old Turkic script, transliteration, and translation of attested inscription lines. It is a
> scholarly transliteration/translation assistant, **not a fluent Göktürk speaker** — no such training data
> exists.*

### License caveat
The Uppsala runiform database is **CC BY-NC-SA**. Fine for research/non-commercial; **commercial release of
trained weights needs review or permission.** Track per-line `license` in the corpus.

---

## 5. The named model — `bengü` / *Bengü Taş*

The Turkish + Old-Turkic specialist (a `tengri`-350M-based fine-tune) gets its own zoo codename:

> **`bengü`** — from ***Bengü Taş***, "the **eternal stone**," the Old Turkic term for the inscribed memorial
> stelae themselves. A model that reads the eternal stones, named for them.

It joins the lineage as the *language* branch (alongside the scale branch). Add `bengü` to `registry.py`'s
`NAME_POOL`.

---

## 6. How it interleaves with the ladders

| When | Scale/Capability context | Turkic work |
|---|---|---|
| Pre-R3 (the freeze) | before `tengri` is pretrained | **48k multilingual tokenizer** (TR + rune seed) + `<\|tool\|>`/`<image>` tokens — one commit |
| R2 125M | proof run | optional 4–8% Turkish; fertility eval |
| **R3 350M `tengri`** | the MVP base | 5–8% Turkish in the mix → Turkish-capable base |
| post-R3 | capability C6 (SFT) | Turkish SFT; then `bengü` = Old-Turkic transliteration/translation fine-tune + the inscription demo (reuses C3 RAG) |
| R4 1B `otuken` | headline base | 8–12% Turkish if it's a headline capability |

Göktürk gates on **R3** (it needs a base strong enough to follow transl/translate instructions) and on **C3**
(RAG over the inscription corpus). Turkish gates only on the **tokenizer freeze**.

---

## 7. Data sources · new files · do next

**Data:** CulturaX-tr · HPLT2-tr · OSCAR-2301-tr · Turkish Wikipedia/Wikisource · OpenOrca-tr · Turkish-Alpaca ·
TR-MMLU/TurkBench/TurBLiMP (eval) · Uppsala Runiform DB + Türk Bitig (Old Turkic, CC BY-NC-SA) · scholarly
editions (Tekin, Erdal, Clauson — references, do not scrape).

**New / touched files:**
```
src/orkhon/tokenizer/special_tokens.py   # append <|tool|>@8, <image>@9
src/orkhon/tokenizer/fertility.py        # NEW — the freeze go/no-go gate
configs/tokenizer/multilingual_48k.yaml  # NEW — EN+code+TR+rune-seed
src/orkhon/data/hf_text.py               # add CulturaX/HPLT/OSCAR/OpenOrca-tr descriptors
configs/data/fineweb_tr_mix.yaml         # NEW — source weights
configs/train/{pretrain_350m_tr_mix,sft_tr,sft_old_turkic}.yaml   # NEW
src/orkhon/data/old_turkic/             # NEW — inscriptions.jsonl + synthetic builder + translit map
src/orkhon/eval/domains.py · configs/eval/{tr,old_turkic}.yaml    # Turkish + Old-Turkic eval
src/orkhon/registry.py                   # add 'bengü' to NAME_POOL
```

**Do next, in order:**
1. **Decide the tokenizer freeze now** (it's the only irreversible thing): 48k, TR + rune-seed corpus, `<|tool|>`/`<image>` tokens. Write `tokenizer/fertility.py` and prove the byte/tok targets *before* committing.
2. Add Turkish dataset descriptors to `hf_text.py`; download a Turkish slice; measure fertility old-vs-new.
3. Fold 5–8% Turkish into the R3 (`tengri`) mix.
4. Build the `data/old_turkic/` parallel corpus + synthetic augmentation.
5. After R3: Turkish SFT, then the **`bengü`** Old-Turkic fine-tune + the *"Orkhon reads its namesake"* demo.

---

*The model named for the Orkhon inscriptions, learning to read them. `bengü` — the eternal stone.*
