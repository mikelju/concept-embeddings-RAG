# Successor project: charter

> Written 2026-09-30, after Phase 16 closed `TRANSFER_REGRESSION`, from the author's decisions of
> that day. **Status: draft for the author.** Nothing here is a spec; the new repository's own
> Phase 0 and specs decide. This file stays in this repository as the hand-over record and is
> copied into the new one as its starting brief.
>
> Labels: **measured** (read from this project's artifacts), **decided** (the author's decision),
> **proposal** (not decided), **projection** (not a measurement).

---

## 1. Why a new project

This project asked one narrow question: *what is the minimum structural signal Dense retrieval
needs to recover multi-hop evidence that similarity alone misses?* It answered it within bounds
(section 7). Phase 16 then showed that the best system of the line, fitted on HotpotQA, loses to
plain Dense + BM25 on news (P14 513 vs 587 of 2,255, measured).

The author's goal is broader: **a cheap, fast way to find the evidence a question needs, in any
corpus, better than what is used today.** That is a different question, with different rules
(the indexing unit becomes a variable, the Entity Hop stops being the object of study, rerankers
and learned models are admitted), and it needs a clean repository where it is the only question.
This project closes with a paper (Phase 17, decided) and hands over what it learned.

**The lesson that motivates the split (decided, recorded 2026-09-30):** this line compared its
systems against BGE-small Dense + BM25, a weak control next to what the literature already does
cheaply. A zero-shot cross-encoder was known and parked as a future phase instead of being the
bar. The successor starts from the literature's best, reproduced, and every candidate is measured
against it.

---

## 2. The question

> For a question over an arbitrary text corpus, which retrieval strategy recovers the full
> evidence most often **at a given cost**, and does it keep that advantage on a corpus nobody
> looked at while choosing it?

- **"Better" is two-dimensional (proposal).** Quality (the share of questions whose full evidence
  enters a fixed reading budget) and cost (compute time and money per question, offline and
  online separately). A strategy is interesting if it is on the quality-cost frontier, not only
  if it wins on quality.
- **"Any corpus", honestly scoped (proposal).** In practice: text corpora with questions whose
  evidence is marked, so that recovery is measurable. News, scientific papers, legal text,
  encyclopedic text, and transcripts if a labelled set exists. Plans, diagrams and images are out
  of scope until a labelled set for them exists; the charter says so from day one.
- **No labels at use time (carried from Phase 15-16).** A real user has no questions with answers
  on their private corpus. A strategy may be *chosen* on open corpora, but it runs on a new corpus
  with nothing fitted to it.

---

## 3. Phase 0: the state of the art (decided)

Before any candidate, map where the field is, in two budgets:

1. **Cheap and fast:** what the best inexpensive pipelines reach. Dense models of several sizes,
   hybrid lexical + dense, learned sparse (SPLADE family), multi-vector late interaction (ColBERT
   family, BGE-M3), zero-shot cross-encoder rerankers, fusion without fitted weights, graph
   methods that need no LLM.
2. **Expensive and slow:** what is reached when budget and time grow. Iterative retrieval with an
   LLM in the loop (IRCoT and successors), agentic RAG, LLM rerankers, LLM-built graphs
   (GraphRAG, HippoRAG and successors).

For each: published figures on the benchmarks this project uses (HotpotQA, MuSiQue, MultiHop-RAG,
2Wiki, BEIR subsets), their metric and unit, latency and cost where reported, and whether code
and weights are public. The output is a map of quality against cost, and a short list of
**reference systems ("ghosts")**, one per cost class, that the new project reproduces on its own
harness before any candidate runs. Where our metric differs from a paper's, the paper's figure is
context, never a comparison.

---

## 4. Protocol (decided unless marked)

- **Terrain, open for choosing:** HotpotQA (FullWiki scale), MuSiQue, MultiHop-RAG, all already
  built here (section 6). Every figure on them may be looked at while choosing.
- **Exam, locked, run once:** a fresh corpus, QASPER or LegalBench-RAG (the choice is the
  author's), with the chosen candidates and the ghosts, under a rule frozen before it is opened.
- **Cost classes (proposal):** a light class (online cost close to Dense + BM25) and a heavy
  class (a model reads question and passage together, or more). Each candidate competes in its
  class; its cost is always shown beside its score.
- **Candidates in principle (decided, 2026-09-30):**
  1. *Judge over a pooled bag* (heavy): Dense, BM25, the hop and neighbouring passages feed one
     pool; a cross-encoder scores every candidate. No fitted weights.
  2. *Multi-seed convergent hop with a refined query* (light): hop from the top 5-10 of Dense and
     BM25, reward candidates reached by several independent seeds, and a second retrieval round
     with the query moved towards the first results.
  3. *Weight-free fusion with source diversity* (light): rank fusion without fitted weights,
     at most a few passages per source document, near-duplicates and boilerplate penalized.
- **Pool of further ideas (proposal):** sentence-level matching inside long passages; lexical
  bridge terms from the first results; an offline association map (passage neighbours by meaning,
  shared rare entities and position); passage score informed by its document's score; reading the
  passages next to each hit; explicit links and citations when a corpus has them; spending the
  expensive step only on uncertain questions. Each is a candidate only through a spec.

---

## 5. The game (decided in principle; built late)

A visual reading of results already measured, for the tournament's competitors: three candidates
and the reference system. It never computes or invents anything, and it never animates questions
one by one.

**Tower defense, waves by difficulty (the author's choice, 2026-09-30):**

| In the game | What it shows |
|---|---|
| Four lanes side by side | the four competitors; the same waves hit every lane |
| A wave | all the questions of one difficulty: 2, 3 or 4 pieces of evidence (light, medium, heavy enemies) |
| An enemy sprite | a fixed share of the wave (for example 1 %), never one question |
| Enemy stopped | the competitor recovered the full evidence within the reading budget |
| Enemy damaged but through | part of the evidence recovered (its health bar is the missing share) |
| Leak, base loses health | question failed |
| A map | a corpus of the terrain; the exam map is locked until its single run |
| Cost of the towers | the competitor's cost per question, in its class |
| Final base health per lane | the aggregate score |

A head-to-head overlay (enemies one lane stopped and another leaked) shows the paired comparison.
The page reads a JSON exported from the measured artifacts; it is built after the terrain results
exist.

---

## 6. What we take from this project

| Asset | Where it is here | How it goes |
|---|---|---|
| Dev/test discipline, frozen specs, deviations, no choice on test | skill `research-protocol`, `CLAUDE.md` | carried as rules |
| Spec -> plan -> implement -> deliver flow, adversarial review | `.claude/commands/`, skill `deliver` | carried |
| Guard hook (no push to main, no force-push, no `.env` read) | `.claude/hooks/guard_main.py` | copied |
| CI security scan and its secret baseline practice | `.github/workflows/security.yml`, `.secrets.baseline` | copied; baseline starts empty |
| Rented GPU recipe, driven by the agent through the RunPod API over SSH | skill `remote-gpu`, `phase_16/16.runpod_recipe.md` | carried, updated with the API flow |
| Corpus loaders: FullWiki (5.2 M paragraphs), MuSiQue, MultiHop-RAG, pinned sources | `corpus/fullwiki.py`, `musique.py`, `multihop_rag.py`, `hf_source.py`, `download.py` | copied, then validated by reproduction (below) |
| Harness: Full Support and gold recall under a token budget, token counter, exact McNemar | `evaluation/harness.py`, `budget.py`, `metrics.py`, `evaluation/phase15.py` (paired test) | copied, validated |
| Retrievers: Dense, BM25, fusion, Entity Hop, relevance-ordered hop | `retrieval/` | copied as the first entrants, validated |
| GLiNER extraction and entity index | `nodes/local_extraction.py`, `nodes/index.py` | copied |
| Expensive caches: FullWiki vectors and GLiNER records, MuSiQue and MultiHop-RAG vectors, BM25 and entity indexes | `data/phase9/`, `data/phase15/`, `data/phase16/` (about 12 GB) | read in place, read-only, each checked against its recorded digest; never moved or rewritten |
| Stored rankings to depth 100, four systems | `data/phase15/`, `data/phase16/` `rankings-*.jsonl.gz`; HotpotQA dev lists in `data/phase10/`, `data/phase14/` | read-only inputs for the judge and for the game prototype |
| Literature and surveys | `docs/refs/bibliografia.md`, `plans/corpora_survey_2026-09-30.md`, the segmentation survey in `phase_16/16.pre_spec_notes.md` | inputs to Phase 0 |
| Plain-Spanish glossary | `docs/GLOSARIO.md` | copied and extended |

**The reproduction gate for everything copied (proposal for the new Phase 0 or 1).** Before any
new measurement, the copied code must reproduce these recorded figures exactly (measured here):

| Check | Recorded figure |
|---|---|
| P10-C, HotpotQA dev, Full Support @2,048 | 4,801 / 7,405 |
| P14, HotpotQA dev | 5,224 / 7,405 |
| Dense / Dense + BM25 / P10-C / P14, MuSiQue validation | 436 / 524 / 669 / 761 of 2,417 |
| Dense / Dense + BM25 / P10-C / P14, MultiHop-RAG answerable | 338 / 587 / 522 / 513 of 2,255 |
| GLiNER configuration digest (pod) | `2f7864661b8ce7ff` |

A miss is a finding about the copy, never a reason to edit code until the number matches.

---

## 7. What stays here

| Left behind | Why |
|---|---|
| The concept-space line (Phases 2-4): `concepts/`, `retrieval/conceptual`, `retrieval/diffusion` | closed with a bounded negative result; no role in the new question |
| Per-phase stage code (`cmd_pN_*`, `evaluation/phaseN.py`) and its about 3,400 tests | written to reproduce this line's records; the new project needs one generic tournament runner |
| The paragraph as the only indexing unit | right for this line's question; in the new one the unit is a declared variable per corpus type |
| "The Entity Hop is intentionally minimal" and the ban on new graph mechanisms without a spec here | the hop becomes one ingredient among many |
| BGE-small as the fixed Dense model and the HotpotQA-fitted weights | Phase 16 showed fitted weights do not travel; the new ghosts come from Phase 0 |
| The 600 / 1,400 HotpotQA-derived set and `test-11` | historical; `test-11` is spent |
| Phase-specific artifacts under `data/phaseN/` other than the caches and rankings listed above | they are this line's record and stay read-only here |

---

## 8. What this project established (for the new one's context)

Measured, each in its phase's `X.results.md`:

- A query-blind, one-hop Entity Hop from Dense's first paragraph complements Dense (Phase 6); GLiNER
  keeps most of that gain for a fraction of an LLM extractor's cost (Phase 7).
- A larger Dense model is not automatically better here: Qwen3-Embedding-0.6B stayed below
  BGE-small at every corpus size up to 500 k paragraphs (Phase 8, deviation 8.1).
- At FullWiki scale the hop lifts Dense (+4.48 pp, Phase 9) and, as a third component, Dense +
  BM25 (+3.92 pp, Phase 10). Ordering the hop's candidates by similarity to the question (P14)
  adds +4.90 pp on the held-out split (Phase 14).
- Frozen at their HotpotQA values, P14 and P10-C still beat Dense + BM25 on MuSiQue (+9.81 and
  +6.00 pp), with the gain on 2-paragraph questions (Phase 15).
- On news (MultiHop-RAG) the same frozen systems lose to Dense + BM25 (P14 513 vs 587,
  `TRANSFER_REGRESSION`, Phase 16). Exploratory, in-sample: the loss is tied to the weight the
  HotpotQA fit moved from BM25 to the hop, and to a seed (Dense's first paragraph) that holds
  evidence for only 17 % of the questions.

---

## 9. Lessons that must not be relearned

- **Measure against the literature's best cheap recipe from the start**, reproduced on the same
  harness (section 1).
- **Fitted weights do not travel between kinds of text** (Phase 16); prefer rules with nothing
  fitted, or report the fitted figure as an upper bound.
- **Check one example before a count enters a spec** (deviation 16.1: "29 facts cross a paragraph
  break" were sentences printed twice).
- **Every corpus gets its own caches:** the question-vector cache key does not include the corpus.
- **Tokenize large corpora in batches**; the Windows ARM64 laptop cannot run spaCy and runs GLiNER
  about 290 times slower than an RTX 4090, so heavy work goes to a rented GPU.
- **The local secret scanner on Windows misses high-entropy hex strings CI flags**; generate
  baseline entries from CI's reported lines.

---

## 10. Open decisions for the author

1. The new repository's name, and whether it is public from the start.
2. The exam corpus: QASPER or LegalBench-RAG.
3. Whether the heavy class admits an LLM in the online loop (agentic RAG as a ghost is needed in
   Phase 0 either way, to know the ceiling; competing with one is a separate decision).
4. Whether offline LLM work (for example synthetic questions per passage) is allowed in the light
   class, since it costs nothing per question at use time.
5. The reading budget of record: 2,048 tokens as here, or a small set of budgets.
