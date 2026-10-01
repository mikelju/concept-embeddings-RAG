# Successor project: hand-over reference

> Written 2026-10-01, after Phase 17 closed and its paper was approved. It is the single
> operational reference for starting the successor project exactly where this repository stops.
> **Status: reference, not a spec.** Nothing here freezes a rule of the new project; its own
> Phase 0 and specs decide. Companion document (the decision record): [`successor_project_charter.md`](successor_project_charter.md).
>
> **Labels** (the same ones this repository uses): **measured** (read from an artifact or a
> command), **derived** (arithmetic on measured values), **exploratory** (gold-informed or fitted
> on the figures it is measured on; decides nothing), **decided** (the author's decision, with its
> date), **proposal** (not decided), **interpretation** (a reading, not tested), **projection**
> (neither measured nor derived). Every statement carries its source in parentheses: a file path
> relative to the repository root, a phase, a date. Figures are copied from the source and are not
> rounded differently. Short digests are as the source prints them (first eight and last four
> characters); full values are in the named file.

---

## 1. Purpose of this document and how to use it

The charter records **what was decided and why** (why a new project, the question, Phase 0, the
protocol, the game, what is taken, what stays). This document records **the state and the
know-how**: where the old project stops, which facts the new one may rely on, which artifacts and
code to take and how, which traps are known, and what to do first. It links to the charter where a
link is enough and repeats a charter statement only where the reader needs it to act.

How to use it. A human reader (the author) reads sections 2 to 4 for the state and the scope, and
section 9 for what is still open. The agent that opens the new repository reads all of it once, then
works from sections 5 (what to copy), 7 (the protocol), 8 (operations) and 10 (first steps). If this
document and the charter disagree, the charter is the decision record for *decisions*, this document
for *state and numbers*; flag the disagreement to the author and do not resolve it silently (section 9 lists the ones found).

The new project has a **different scope, different candidates and a different proposal** from this
one (charter §1-2). Nothing in the old project's rules is binding on it unless section 7 or the
charter says it is carried over.

---

## 2. Where the old project stops (2026-10-01)

### 2.1 Status in one page

- **17 phases are closed** (master plan, "Phase status", `docs/plans/0_master_plan.md`): the
  concept-space line (Phases 1-4, bounded negative result), the entity line (5-14), two transfer
  checks (15 MuSiQue `TRANSFER_SUPPORTED`; 16 MultiHop-RAG `TRANSFER_REGRESSION`) and the judge phase
  (17, "Complete: measured and closed (2026-10-01); no terminal state, no gate").
- **The repository is a closed research line.** What remains is **the publication**, which is the
  author's: a 25-page paper (approved by the author on 2026-10-01), author Mikel Ugarte-Gil,
  independent researcher; **venue path decided 2026-10-01: journals and online publications only,
  no conferences: arXiv preprint first, then TMLR, with Discover Computing as the alternative**
  (`docs/plans/phase_17/17.results.md` §10b; `docs/paper/venues.md`, "The author's choice").
- **Open for the author in the old repository** (17.results.md §12.7, §10b): the arXiv endorsement
  (a first submission without an institutional email needs a personal endorser,
  `docs/paper/venues.md`, "arXiv"), the anonymous mirror for TMLR's double-blind review, the AI
  disclosure wording, and `OpenReview` profile. None blocks the successor project.
- **`test-11` is spent** (the held-out HotpotQA draw of 5,000 questions; Phase 14, opened once).
  The 7,405 HotpotQA validation questions are the line's **dev** (CLAUDE.md, "What is
  established"). No new corpus was opened in Phases 15-17 beyond the two transfer sets.
- **Status of this hand-over's inputs in git:** at the time of writing the working tree has
  uncommitted edits to `CLAUDE.md`, `docs/plans/0_master_plan.md`, the Phase 17 plan, results and
  recipe, and `docs/research_summary.md` (git status at session start, 2026-10-01); the Phase 17
  PR (#28) is merged (`git log`, commit e833d04; memory note `phase-17-judge-paper`).

### 2.2 The paper

`docs/paper/` holds `main.tex`, `sections/*.tex` (nine sections and an appendix: introduction, hop,
setup, scale, concepts, judges, published, limitations, conclusion, appendix), `references.bib`,
`build.py`, and generated `tables/` and `figures/`. Title: "An Untrained Entity Hop for Multi-Hop
Retrieval: Gains, a Regression on News, and Zero-Shot Judges". **25 pages** (main text about 8,
references on pages 9 and 10, appendix pages 10 to 25), **1,339 macros**, two runs give identical
bytes, no hand-written number (rule R5) (17.results.md §10, measured). The message the author
confirmed on 2026-10-01 (17.results.md §10b): an eight-billion-parameter decision model adds
nothing as an evidence judge; a strong cross-encoder judge adds on every corpus; P14 stays the best
of the four systems under that judge on the Wikipedia benchmarks. The paper's systems are placed
beside published figures "under that description [untrained, no LLM at query time, cheap], never as
competitors of trained or LLM-based systems" (17.spec.md D9). Build: `uv run python docs/paper/build.py`
(`cer p17-tables`, then Tectonic 0.17.0). The PDF is not hash-stable; the sources are the record.

### 2.3 The numbers that matter

All Full Support @2,048 tokens (section 11 defines it), whole-paragraph units.

**The ladder** (16.results.md §6; the three columns differ in corpus, questions and size, so only
direction and order compare):

| Rung | HotpotQA `test-11` (5,000) | MuSiQue validation (2,417) | MultiHop-RAG (2,255) |
|---|---:|---:|---:|
| P10-A Dense (BGE-small) | 57.04 % (2,852) | 18.04 % (436) | 14.99 % (338) |
| P10-B Dense + BM25 (0.5 / 0.5) | 61.58 % (3,079) | 21.68 % (524) | **26.03 % (587)** |
| P10-C + Entity Hop (0.5 / 0.3 / 0.2) | 65.70 % (3,285) | 27.68 % (669) | 23.15 % (522) |
| P14 relevance-ordered hop | 70.60 % (3,530) | 31.49 % (761) | 22.75 % (513) |

Paired tests, exact McNemar (wins / losses): P14 against P10-C on `test-11` +4.90 pp, 317 / 72,
p = 9.0e-38 (14.results.md); P14 against P10-B on MuSiQue +9.81 pp, 294 / 57, p = 1.3e-39 (15.results.md);
P14 against P10-B on MultiHop-RAG −3.28 pp, 31 / 105, p = 1.3e-10 (16.results.md §1).

**The judge results** (Phase 17; every retriever frozen, nothing fitted; the judge reorders each
system's own fused top-100; 2,013,745 pairs per judge). Full Support @2,048 of P10-B
(17.results.md §1, measured):

| Set | No judge | J-light | J-strong | J-decision |
|---|---:|---:|---:|---:|
| HotpotQA dev (7,405) | 4,536 | 4,730 | 5,198 | 1,882 |
| MuSiQue validation (2,417) | 524 | 644 | 704 | 174 |
| MultiHop-RAG answerable (2,255) | 587 | 352 | 822 | 208 |

The judges, with their labels. **J-light** = `cross-encoder/ms-marco-MiniLM-L-6-v2`; **J-strong** =
`BAAI/bge-reranker-v2-m3`; **J-decision** = `Contrastive-LM/CLM-v0.1-8B` heads over a frozen
`Qwen/Qwen3-8B`, a bi-encoder (17.results.md §3). Label of "J(P14) against J(P10-B)": does the hop's
pool still add evidence once a judge orders it (17.results.md §1):

| Set | J-light | J-strong | J-decision |
|---|---|---|---|
| HotpotQA dev | `HOP_ADDS_UNDER_JUDGE` (383 / 59, +4.38 pp) | `HOP_ADDS_UNDER_JUDGE` (780 / 67, +9.63 pp) | `HOP_NEUTRAL_UNDER_JUDGE` (264 / 256, +0.11 pp) |
| MuSiQue validation | `HOP_ADDS_UNDER_JUDGE` (68 / 15, +2.19 pp) | `HOP_ADDS_UNDER_JUDGE` (190 / 23, +6.91 pp) | `HOP_ADDS_UNDER_JUDGE` (70 / 22, +1.99 pp) |
| MultiHop-RAG answerable | `HOP_NEUTRAL_UNDER_JUDGE` (14 / 27, −0.58 pp, p = 6.0e-2) | `HOP_HURTS_UNDER_JUDGE` (24 / 49, −1.11 pp) | `HOP_HURTS_UNDER_JUDGE` (21 / 44, −1.02 pp) |

J(P10-B) against P10-B (what the judge adds to the standard recipe): J-strong +8.94 / +7.45 / +10.42
pp on HotpotQA / MuSiQue / MultiHop-RAG (exact p below 1e-28 on each); J-light +2.62 / +4.96 / **−10.42**
pp; J-decision −35.84 / −14.48 / −16.81 pp (17.results.md §1). J-strong lifts all twelve system-set
pairs, +5.59 to +12.59 pp (research_summary.md, point 9).

**The union pool under J-strong** (descriptive, in no preregistered comparison): 5,922 / 7,405
(79.97 %), 809 / 2,417 (33.47 %) and 846 / 2,255 (37.52 %) (17.results.md §1, §12.5). P14 under
J-strong: 5,911, 871, 797. P10-B under J-strong: 5,198, 704, 822. **Ceilings** (questions with every
gold unit inside the system's top-100; exploratory): P10-B 5,267 / 900 / 1,160; P14 6,119 / 1,317 /
1,145; union 6,299 / 1,410 / 1,298, i.e. 85.06 %, 58.34 % and 57.56 % of the questions
(17.results.md §8). Questions with gold outside the union pool: 41.7 % of MuSiQue and 42.4 % of
MultiHop-RAG (derived; 17.results.md §11, limitation 2). The share of the room a judge closes under
J-strong (derived, exploratory): P10-B 90.6 % / 47.9 % / 41.0 %, P14 76.8 % / 19.8 % / 44.9 %
(17.results.md §8).

**Costs.**

| Item | Figure | Source |
|---|---|---|
| Entity extraction, Claude, 19,366 paragraphs | 25.9848 USD measured | research_roadmap.md §4 (Phase 7) |
| Entity extraction, GLiNER, the same paragraphs | 119.4 s, 0.03 USD attributable | research_roadmap.md §4 (Phase 7) |
| GLiNER, all 5,233,329 FullWiki paragraphs | 6.18 h, 4.57 USD attributable (derived from time and rate); the whole Phase 9 session billed 12.71 USD | research_summary.md, point 2 |
| GLiNER, MuSiQue (101,962 units) | 10.3 min, 0.13 USD attributable; session invoiced 0.356 USD | research_summary.md, point 7 |
| GLiNER + BGE, MultiHop-RAG (27,989 units) | 154.77 s extraction, 41.20 s encoding; 0.0404 USD attributable; session invoiced **0.165 USD** | 16.results.md §3 |
| Phase 17, the three judges | 3.28 USD attributable GPU (3.2781 derived); balance delta 5.2325 USD; **invoiced 5.257 USD** (GPU 5.14 + storage 0.117) over a 6 h 56 min 48 s RTX 4090 session | 17.results.md §4 |
| Judge cost at query time (derived, one RTX 4090) | J-light 0.031 / 0.036 / 0.029 s per question; J-strong 0.448 / 0.542 / 0.446 s per question; J-decision query pass 0.0041 / 0.0040 / 0.0073 s per question plus an offline pass of 0.0104 / 0.0130 / 0.0078 s per distinct unit (HotpotQA / MuSiQue / MultiHop-RAG) | 17.results.md §4 |
| Entity Hop latency on the laptop, MultiHop-RAG | P10-A 1.27 ms, P10-B 1.92 ms, P10-C 2.83 ms, P14 3.16 ms mean, no question encoding | 16.results.md §10 (D11) |
| Dense + BM25 + hop, FullWiki scale, mean query latency | 1.33x Dense + BM25 (tail up to 16.5 s); P14 is 1.82x P10-C (laptop) | research_summary.md, points 5-6 |

**Repository status.** A closed research line; code, specs and the paper are committed; the
measured artifacts under `data/` are git-ignored and live on the author's laptop (about 19 GB on
2026-10-01, measured with `du -sh data`; the charter's "about 12 GB" counts only the caches and
rankings it lists, charter §6).

---

## 3. What this line established, as facts the new project may rely on

Each fact has its phase and artifact. "Bounded" means it holds for the setting measured and no
further. The full record of each is in the phase's `X.results.md`.

1. **The pooled-embedding concept space is a bounded negative result** (Phases 2-4, closed by
   `docs/plans/phase_4/4.1_research_line_closure.md`). A sparse concept space induced from pooled
   Dense embeddings adds nothing over Dense; a Dense + BM25 control beats it; diffusion expansion
   re-ranks without promoting a new paragraph (research_summary.md, table). Bounded to the two
   operationalizations tried (embedding-induced concepts, Phases 2-4; text-extracted concepts,
   Phase 5, which were the weakest hop measured and made entities worse when added). CLAUDE.md:
   "do not read it as evidence against text-derived entities". The code stays in this repository
   (charter §7).
2. **A query-blind, one-hop Entity Hop from Dense's first paragraph complements Dense** (Phase 5
   pilot `NAMES_ONLY`, hit@10 0.658 against 0.444; Phase 6 `ENTITY_REPLACEMENT_SUPPORTED` on 1,400
   test questions). Mechanism: P1 = Dense's first unit; candidates are the units outside the first
   10 sharing at least one raw entity node with P1; score = sum of `log(1 + N / (1 + df))` over
   shared nodes; fused with Dense (CLAUDE.md, "Architecture in brief"). On the 19,366-paragraph
   pool, Full Support @2,048 on 1,400 test questions: Dense 0.8250, Dense + BM25 0.8643, Dense +
   Entity Hop (GLiNER) 0.8793, (Claude) 0.8900 (research_summary.md, point 1).
3. **GLiNER keeps most of the gain for a fraction of the cost** (Phase 7, `phase_7/7.results.md`):
   83.5 % of the held-out gain at 0.03 USD against 25.9848 USD for Claude. Selection was by a
   preregistered cost-and-time rule; spaCy scored higher on dev (525 / 600 against 518 / 600) and was
   not selected; it has no test figure and none may be produced retrospectively
   (research_roadmap.md §4, §8.5; skill `research-protocol`). The extractor: `urchade/gliner_medium-v2.1`,
   revision `40ec419335d09393f298636f471328b722c6da9e`, labels person / organization / location / work of art /
   event, threshold 0.5, `max_len` 384, batch 8, configuration digest `2f7864661b8ce7ff`
   (`src/concept_embeddings_rag/config.py`; 16.results.md §3).
4. **BGE-small stayed ahead of Qwen3-Embedding-0.6B at every corpus size up to 500k paragraphs**
   (Phase 8 stopped at its dev gate: 446 / 600 against 487 / 600; deviation 8.1 `STABLE_RANKING`,
   BGE / Qwen 487 / 446 at 19,366, 463 / 420 at C100, 440 / 393 at C250, 423 / 366 at C500; 0_master_plan.md).
   Not established: that a substantially stronger Dense changes anything; none has been tested
   (CLAUDE.md). A "bigger Dense" is therefore not a safe claim, and the baseline lesson in section 7
   applies in both directions.
5. **The hop's gain over Dense survives full FullWiki scale** (Phase 9 `SCALE_SUPPORTED`): 5,233,329
   paragraphs, 5,405 retrieval-unseen questions, Dense 55.62 %, Dense + Entity Hop 60.09 % (+4.48 pp,
   p = 8.0e-28), Dense + BM25 61.30 % (research_summary.md, point 3). At that scale BM25 alone is the
   slightly better complement at the primary budget (p = 0.024, descriptive); 371 questions go the
   hop's way and 436 BM25's, so the signals are not redundant.
6. **Dense + BM25 + hop fusion adds** (Phase 10 `THREE_WAY_SUPPORTED`): Dense 58.32 %, Dense + BM25
   62.58 %, Dense + BM25 + Entity Hop 66.50 % (+3.92 pp, 306 / 110, p = 1.9e-22) on 5,000 held-out
   HotpotQA train questions, which BGE-small saw in fine-tuning (a bias that works against the
   result; contamination probe bound +2.89 pp; 17.results.md §10 table). Weights 0.5 / 0.3 / 0.2 were
   fitted on the 7,405 validation questions.
7. **Phases 11-13 (`DEV_STOP` x3) taught that hard filters on the seed lose** (research_roadmap.md
   §7.2): a DF cap or question-seeded hop (Phase 11: best 4,808 against bar 4,838); question-to-sentence
   similarity choosing which of P1's entities to hop from (Phase 12: best is P10-C's own point, 4,801
   against 4,835); a five-word window around each mention (Phase 13: puts a bridge entity first for
   40.78 % of 2,651 bridge-sharing gold paragraphs against 19.01 % for the best trivial rule, yet
   every `m` below `all` loses questions, 4,643 / 4,740 / 4,765 against 4,801). Gold-informed
   ceilings (exploratory, Phase 13): a perfect seed choice +119 questions, a perfect soft weight
   +639, a perfect reorder of the hop's candidates +1,257, a perfect tie-break +0 (structural under
   min-max fusion). The lesson: the question must enter the hop's **score**, not remove part of it.
8. **P14: candidate relevance helps** (Phase 14 `CANDIDATE_RELEVANCE_SUPPORTED`): the hop scores
   candidates by `(1 − α) · rarity + α · cos(question, candidate)`, both min-max normalized per question,
   α = 0.75 fitted on dev; 5,224 / 7,405 on dev against P10-C's 4,801; on `test-11` 3,530 against
   3,285 (+4.90 pp, 317 / 72). Whether part of the gain re-weights what Dense already ranks cannot be
   separated (on dev, of the won gold paragraphs 260 were in Dense's ranks 11-100 and 242 beyond 100).
9. **MuSiQue transfer, frozen** (Phase 15 `TRANSFER_SUPPORTED`): P14 +9.81 pp and P10-C +6.00 pp
   over Dense + BM25, P14 +3.81 pp over P10-C (151 / 59, p = 1.7e-10), on 2,417 validation questions
   over 101,962 pooled paragraphs. The gain is a two-paragraph gain: 211 of P14's net 237 questions
   come from the 1,252 two-paragraph questions; on the 405 four-paragraph questions no system
   retrieves the full evidence for more than 2 (research_summary.md, point 7). Caveat: MuSiQue mined
   its distractors with BM25, which works against lexical retrieval (interpretation, 15.results.md
   limitation 10).
10. **MultiHop-RAG regression and the refit diagnostic** (Phase 16 `TRANSFER_REGRESSION`): frozen,
    P14 loses to Dense + BM25 (513 vs 587 of 2,255; 31 wins / 105 losses). BM25's step is the largest
    the line measured (+11.04 pp, 276 / 27). Exploratory, in-sample refit (330 points, 16.results.md
    §10): the best point is **hop weight 0**, Dense 0.3 / BM25 0.7, at 662 (29.36 %); BM25 alone
    632; the frozen configuration's loss is at least partly the 0.2 of BM25 weight it hands to the
    hop (interpretation). The loss is mostly on two-gold queries (55 of the net 74) and largest on
    comparison queries (−42), which the spec expected to suit the hop best. The hop's few gains
    were about two thirds from another article (P14, won: 21 of 29). **Fitted weights do not travel
    between kinds of text.**
11. **P1 is gold for 17.07 % of MultiHop-RAG queries, 57.14 % on MuSiQue, 77.4 % on HotpotQA dev**
    (16.results.md §8), though P1's article holds gold for 62.44 % on MultiHop-RAG. Every hop of
    this line starts from P1, so this corpus violates the premise most (interpretation H1,
    16.results.md §12; the diagnostic "hop Full Support split by whether P1 is gold" was not run).
12. **The three judges** (Phase 17, 17.results.md §1-3, §11-12):
    - **A strong cross-encoder lifts everything**: J-strong improves every system on every set.
      The standard recipe was a weak control: P10-B under J-strong (5,198 / 704 / 822) matches or
      beats P14 without a judge on HotpotQA (5,224, in-sample for P14) and news (513), and trails it on
      MuSiQue (761 vs 704) (17.results.md §12.1, derived).
    - **The light judge is two-faced**: J-light helps P10-A and P10-B on the two Wikipedia sets
      (P10-B +2.62 and +4.96 pp), lowers P14 on both (−2.30 and −2.65 pp), and lowers every system on
      news (P10-B −10.42 pp, 324 questions lost against 89 won). Unexplained; truncation touches
      0.23 % of pairs (731 of 317,776) and is unlikely to be the main cause (untested).
    - **CLM-8B (J-decision) is a negative result as an evidence judge** used zero-shot with the
      question alone as its state: it lowers every system on every set by 7.58 to 45.02 pp. "It does
      not rank evidence" and "this usage does not suit it" are not separable. A different state
      (question plus P1, an instruction) is a declared candidate in a new experiment, not a retry.
    - **bfloat16 non-determinism**: J-decision re-scoring (batch 1 against 32) differed by 0.335 /
      0.386 / 0.373 logits (scale 100) and **all 20 re-scored questions reordered on every set**;
      the cross-encoders gave 7e-6 to 6e-5 and 0 reorders. J-decision's figures are one realization;
      a second pass was not measured (decided 2026-10-01, 17.results.md §10b).
    - **The hop's role under a judge is candidate generation, on Wikipedia**: `HOP_ADDS_UNDER_JUDGE`
      in 5 of 9 cells; `HOP_HURTS_UNDER_JUDGE` on news under J-strong (24 / 49) and J-decision. On news
      P10-B's pool holds more gold than P14's (1,160 against 1,145) (hypothesis supported on Wikipedia,
      not on news; §12.2).
13. **The ceiling and the union pool** (Phase 17, D1/D6; 17.spec.md): the **pool** per question is
    the union of the fused top-100 of P10-A, P10-B, P10-C and P14 (mean size 174.9 / 165.8 / 140.9
    units on HotpotQA dev / MuSiQue / MultiHop-RAG; 2,013,745 pairs); the **union line** is the whole
    pool sorted by a judge's score; the **ceiling** counts questions with every gold unit inside a
    list's top-100. The pool is bounded at 100: about four questions in ten on MuSiQue and news are
    out of reach of any reorder, so better retrieval and better judging are both open
    (17.results.md §12.5c). Under J-strong the union still fails 601 and 452 questions inside its
    ceiling on MuSiQue and news (derived).
14. **The hyperlink layer was never used.** The FullWiki archive carries Wikipedia links the loader
    drops. A dev diagnostic of 2026-09-29 (gold-informed, exploratory, scratchpad only): 62.8 % of
    the gold outside Dense's top 10 is linked to P1 (96.9 % when P1 is gold); perfect-order ceiling
    +1,384 questions; link neighbourhood of P1 median 8 against 5,504 hop candidates. 79.9 % of dev
    gold pairs link first to second because HotpotQA built its bridge questions from first-paragraph
    links, so the reach is partly a benchmark property (research_roadmap.md "The corpus's own
    reference structure"; memory `hyperlink-layer-unused`). **The author excluded links as a component
    (2026-09-29)**, since target corpora carry none. A link-free title-mention rule reaches 55.6 % (ceiling
    +1,187; +993 with multi-word titles only). What works is mention to the unit *about* that
    entity, not mention to mention; not scheduled.
15. **Failure anatomy of P10-C on dev** (review of 2026-09-29, `docs/plans/review_2026-09-29.md`,
    exploratory): of 2,604 failures, 40.8 % have a gold unit in none of the three lists (reach) and
    59.2 % have both listed but badly placed (ordering; 932 with both gold in the fused top 100, the
    second at median rank 48 while the context holds about 25 units). P1 is not gold for 22.6 % of
    the questions and 37.3 % of the failures.
16. **Literature numbers are not directly comparable** to Full Support over whole paragraphs under
    a token budget (research_summary.md; `docs/paper/tables/published-*.md`: 55 verified rows). The
    paper carries them as context only.

---

## 4. What the new project is

### 4.1 The question and why

> For a question over an arbitrary text corpus, which retrieval strategy recovers the full
> evidence most often **at a given cost**, and does it keep that advantage on a corpus nobody looked
> at while choosing it? (charter §2)

Why a new repository (charter §1, decided 2026-09-30): the old question was narrow (the minimum
structural signal Dense needs), it answered it within bounds, and the author's goal is broader: **a
cheap, fast way to find the evidence a question needs, in any corpus, better than what is used
today.** The new question needs different rules: **the indexing unit becomes a variable per corpus
type**, the Entity Hop stops being the object of study (it becomes one ingredient), and rerankers
and learned models are admitted. The motivating lesson (decided 2026-09-30, memory
`baseline-must-be-literature-best`): this line compared against BGE-small Dense + BM25, a weak
control; a zero-shot cross-encoder was known and parked. Phase 17 later measured the gap
(section 3, fact 12). The successor starts from the literature's best, reproduced, and measures
every candidate against it.

Scoping proposals in the charter §2 (marked proposal there): "better" is two-dimensional, quality
(share of questions whose full evidence enters a fixed reading budget) and cost (compute time and
money per question, offline and online separately), so a strategy is interesting on the
quality-cost frontier; "any corpus" means text corpora with marked evidence (news, scientific
papers, legal text, encyclopedic text, transcripts if a labelled set exists), and plans, diagrams
and images are out of scope until a labelled set exists; and **no labels at use time**: a strategy
may be chosen on open corpora but runs on a new corpus with nothing fitted to it.

### 4.2 The tournament and the "aggregate-performance game"

**Decided (author, 2026-09-30; memory `new-project-tournament`):** a new repository hosts "a
tournament of retrieval strategies"; a visual game shows the results as aggregate performance,
built late, in the new project. **Decided in principle, charter §5:** a tower-defense, waves by
difficulty:

| In the game | What it shows (charter §5) |
|---|---|
| Four lanes side by side | the four competitors (three candidates and the reference system); the same waves hit every lane |
| A wave | all the questions of one difficulty: 2, 3 or 4 pieces of evidence |
| An enemy sprite | a fixed share of the wave (for example 1 %), never one question |
| Enemy stopped / damaged but through / leak | full evidence recovered within the reading budget / part recovered (health bar is the missing share) / question failed, base loses health |
| A map | a corpus of the terrain; the exam map is locked until its single run |
| Cost of the towers | the competitor's cost per question, in its class |
| Final base health per lane | the aggregate score |

A head-to-head overlay shows the paired comparison (enemies one lane stopped and another leaked).
The page reads a JSON exported from measured artifacts; it never computes or invents anything and
never animates questions one by one. **Rejected by the author:** a "rally" whose track length is the
reading budget ("means nothing to the author"), and per-question mechanics such as a ski run per
question (memory `new-project-tournament`). **Open:** the new repository's name and whether it is
public from the start; the visual form (a static web page is the charter's implication, "the page
reads a JSON", not a stated decision); the export schema; and when to build (after the terrain
results exist, charter §5). The orchestrator's term "videogame" is a paraphrase of the author's
tower-defense choice; the original wording is not in the repository.

### 4.3 Protocol, terrain and exam (charter §4)

- **Terrain, open for choosing:** HotpotQA (FullWiki scale), MuSiQue, MultiHop-RAG, all already built
  here. Every figure on them may be looked at while choosing. (Note for the new project: HotpotQA dev
  is in-sample for P10-C's and P14's weights, 17.spec.md D3.)
- **Exam, locked, run once:** a fresh corpus, QASPER or LegalBench-RAG (**the author's choice, open**),
  with the chosen candidates and the ghosts, under a rule frozen before it is opened.
- **Cost classes (proposal, charter §4):** a light class (online cost close to Dense + BM25) and a
  heavy class (a model reads question and passage together, or more); each candidate competes in its
  class with its cost shown beside its score.

### 4.4 The candidates

What counts as a candidate retriever (proposal for the new project's Phase 0 to settle): any
strategy that produces a ranked list of whole units for a question, runnable on the shared harness,
with its offline and online cost recorded.

**Already available as entrants** (charter §6, measured here):

1. The four frozen systems: P10-A Dense (BGE-small), P10-B Dense + BM25 (0.5 / 0.5), P10-C Dense +
   BM25 + Entity Hop (0.5 / 0.3 / 0.2), P14 (the same with the relevance-ordered hop, α = 0.75). Their
   weights were fitted on HotpotQA dev and **do not travel** (fact 10): in the tournament they are
   reference entrants, and their HotpotQA figures are an upper bound.
2. **The judge-ordered union pool under J-strong as the reference** (decided as the author's
   direction, 17.results.md §12.5a): 5,922 / 7,405, 809 / 2,417 and 846 / 2,255 (descriptive,
   measured), at about 141 to 175 judge pairs per question (about 0.45 to 0.54 s per question on one
   RTX 4090, derived).
3. **The hop as one candidate generator among others** with its indexing and query cost beside every
   result (17.results.md §12.5b), not as the final system.

**Candidates in principle** (decided 2026-09-30, charter §4; each becomes a candidate only through a
spec): (1) *judge over a pooled bag* (heavy class): Dense, BM25, the hop and neighbouring passages feed
one pool and a cross-encoder scores every candidate, no fitted weights (Phase 17 measured the
four-system version of this); (2) *multi-seed convergent hop with a refined query* (light): hop from
the top 5-10 of Dense and BM25, reward candidates reached by several independent seeds, a second
retrieval round with the query moved towards the first results; (3) *weight-free fusion with source
diversity* (light): rank fusion without fitted weights, at most a few passages per source document,
near-duplicates and boilerplate penalized. The author "was unsure about the judge's role and cost"
(memory `new-project-tournament`).

**Pool of further ideas (proposal, charter §4):** sentence-level matching inside long passages;
lexical bridge terms from the first results; an offline association map (neighbours by meaning,
shared rare entities and position); passage score informed by its document's score; reading the
passages next to each hit; explicit links and citations when a corpus has them; spending the
expensive step only on uncertain questions.

**Candidates specific to this hand-over's findings (proposals, not decisions):** (a) CLM-8B as a
**first-stage retriever** rather than a judge, since it is a bi-encoder that can embed a corpus
offline (memory `jev-decision-model-as-judge`: "the natural use in the successor project"; 17.results.md
§12.4: "a different state (question plus the first paragraph, an instruction) is a declared candidate
in a new experiment, not a retry"); (b) the hop to the unit *about* an entity (title or
first-sentence subject), conditional on the corpus having such units (research_roadmap.md; memory
`hyperlink-layer-unused`); (c) the second-hop query reformulation (question + P1 text through Dense),
which the master plan scheduled as Phase 17 and which moved to the successor (research_summary.md,
"Where this leaves the work"; its input is the seed that is gold in only 17 % of news queries, so it
inherits that weakness, 16.results.md "Next research decision", interpretation); (d) multiple seeds
(roadmap 8.3, selected by the author 2026-09-29 as a future phase); (e) a zero-shot cross-encoder
applied to a pool deeper than 100, fused with the first-stage score, or fine-tuned (not tried in
Phase 17, research_roadmap.md).

**Open decisions about candidates** (charter §10): whether the heavy class admits an LLM in the
online loop (agentic RAG is needed as a ghost in Phase 0 either way, to know the ceiling; competing
with one is separate); whether offline LLM work (for example synthetic questions per passage) is
allowed in the light class; the reading budget of record (2,048 tokens as here, or a small set of
budgets). The old line's rule "no LLM in the online loop except as a declared variant" is **not**
automatically carried (charter §1 admits rerankers and learned models; the LLM question is open).

### 4.5 Scope differences from this project

- **Not Wikipedia-only.** MultiHop-RAG showed what changes (fact 10-11). The old unit rule (one
  whole paragraph, never fixed windows) was right for this line's question; in the new one **the unit is
  a declared variable per corpus type** (charter §7). The segmentation survey
  (`docs/plans/phase_16/16.pre_spec_notes.md`) says: for text with its own paragraphs the paragraph
  is the unit the literature falls back to; semantic chunking without an LLM exists but evidence it
  pays is weak (Vectara NAACL 2025: fixed windows of about 200 words match or beat it on real
  documents); learned chunkers use LLMs; segmentation is a real question only for unstructured text
  (transcripts, OCR, flat PDFs) (interpretation).
- **Aggregation, sets and tables (the author's standing curiosity, 2026-09-30; memory
  `author-interest-aggregation-and-tables`; `docs/plans/corpora_survey_2026-09-30.md`).** Two
  situations beyond 2-4 gold paragraphs: (a) queries that need many passages from all over a corpus
  ("all the causes of WWI"), tested with a much larger context (16k-32k): **QUEST** is the author's
  preferred choice (MRecall@K is the literature's Full Support for set queries: BM25 3.7 % and a
  T5-Large dual encoder 14.2 % at K = 100 on 325,505 Wikipedia entity documents, 10.5 gold per query),
  **GlobalQA** the non-Wikipedia option (more than 13,000 questions over more than 2,000 resumes; gold
  sets of 2-50 documents); QAMPARI's unit is closest to ours but "all evidence" will be near zero.
  Hypothesis, not a finding: set queries fan out from constraints, so a one-hop expansion may matter
  less than raw recall of Dense and BM25. It is a new question with its own spec. (b) **Filtered
  numeric aggregates over tables**: no benchmark scores "every matching row retrieved"; TAG (2024)
  reports RAG 0 % against 55 % for hand-written pipelines; the literature's position is that top-k
  similarity retrieval cannot guarantee completeness, so **tables are not retrieval's job: retrieval
  finds the table, code computes** (set aside by the author 2026-09-30, research_roadmap.md §7.1).
  When the author raises them again, frame (a) as a new phase with its own question and (b) as
  "retrieval finds the table, code computes" (memory note's own "how to apply").
- **The corpora survey's ranking** for non-Wikipedia corpora with paragraph-locatable gold
  (corpora_survey §1): 1. MultiHop-RAG (done); 2. **QASPER** (NLP papers, CC BY 4.0, native evidence
  paragraphs; retrieval in the literature is within one paper, a pooled corpus would be a new
  setting); 3. **LegalBench-RAG** (contracts, CC BY 4.0, character spans that map losslessly onto
  paragraphs, mostly single-evidence, a domain-transfer test); 4. WixQA (article gold, small);
  5. BrowseComp-Plus (only if gold could reach paragraph level, which it cannot without new labelling).
  The author fixed the order MultiHop-RAG, QASPER, LegalBench-RAG on 2026-09-30 (research_roadmap.md §7.1).
  For QASPER and LegalBench-RAG, **Full Support degenerates to Recall@budget** on mostly
  single-evidence corpora (survey §1 on BEIR members; for LegalBench-RAG "mostly single-evidence").

### 4.6 Phase 0 (charter §3, decided)

Map the field in two budgets before any candidate: **cheap and fast** (Dense models of several sizes,
hybrid lexical + dense, learned sparse (SPLADE family), multi-vector late interaction (ColBERT family,
BGE-M3), zero-shot cross-encoder rerankers, fusion without fitted weights, graph methods needing no
LLM) and **expensive and slow** (iterative retrieval with an LLM in the loop (IRCoT and successors),
agentic RAG, LLM rerankers, LLM-built graphs (GraphRAG, HippoRAG and successors)). For each: published
figures on HotpotQA, MuSiQue, MultiHop-RAG, 2Wiki and BEIR subsets, metric and unit, latency and cost
where reported, whether code and weights are public. The output is a map of quality against cost and
a short list of **reference systems ("ghosts")**, one per cost class, reproduced on the new harness
before any candidate runs. Where the metric differs from a paper's, the paper's figure is context,
never a comparison. Input already in hand: `docs/refs/bibliografia.md`, the corpora survey, the
segmentation survey, and `docs/paper/tables/published-*.md` (55 figures verified 2026-10-01 against
their papers, with the seven excluded candidates and why: `published-figures.json`, 17.results.md §9).
**Reproduction gate:** section 5.4.

---

## 5. What is carried over, concretely

### 5.1 Code (copy; then validate by the reproduction gate in 5.4)

| Item | Where it lives here | How to take it | Caveats |
|---|---|---|---|
| Retrievers: Dense, BM25, fusion, Entity Hop, relevance-ordered hop | `src/concept_embeddings_rag/retrieval/` (`dense.py`, `bm25.py`, `fusion.py`, `entity_hop.py`, `base.py`); P14's candidate scoring in `evaluation/phase14.py`; the fusion rebuild `fuse_lists` / `_p15_code_identity` in `evaluation/phase15.py` | copy as the first entrants | `conceptual.py` and `diffusion.py` stay behind (charter §7). BM25 is `bm25s` 0.3.11, English stopwords (16.results.md §3). Fusion is min-max weighted sum (17.results.md §10, `retrieval/fusion.py`) |
| Harness and budget metric | `evaluation/harness.py`, `budget.py`, `metrics.py`, `fullwiki.py` (`measure_system`) | copy; validate | `TokenCounter.count_units` tokenizes its whole input in one call: batch it for large corpora (CLAUDE.md "Gotchas"). Token counter is the BGE-small tokenizer (the Phase 9 counter) |
| Paired test and labels | `evaluation/phase15.py` (`paired`, `comparisons`, `label`), `phase16.py` (`hits_at`, `ladder`, `group_summary`) | copy | **Take the fixed exact McNemar**: `exact_two_sided_p` overflowed past 1,024 discordant questions; integer division fixed it in commit `9a9ad3f`, and every recorded p of Phases 10-16 reproduced bit for bit (17.results.md §11) |
| Judges and the pool code | `evaluation/judge.py`, `decision_judge.py`, `phase17.py`; CLI `p17-pool`, `p17-score`, `p17-outcome`, `p17-tables` | copy if the judges are reused | the pod-only `pod` dependency group (`contrastive-lm`, `vllm`; Linux x86_64; never synced on the laptop, deviation 17.1) |
| Paper generator | `evaluation/paper.py`, `docs/paper/build.py`, `tests/evaluation/test_paper.py` | copy as the pattern for "every number generated from artifacts" (R5) | needs Tectonic 0.17.0 (section 8) |
| Corpus loaders | `corpus/fullwiki.py` (reads the three corpus formats), `musique.py`, `multihop_rag.py`, `hf_source.py`, `download.py`, `scale_corpus.py` | copy; validate against the pins in 5.3 | `hf_source.py` pins by revision. MuSiQue: licence not read or verified at its pin (15.results.md); MultiHop-RAG: ODC-BY |
| GLiNER extraction and entity index | `nodes/local_extraction.py`, `nodes/index.py`, `nodes/normalization.py` (normalization-v1) | copy | laptop GLiNER digest is `390d0d8ae603fd1d` (deviation 11.1) and **differs from the pod's `2f7864661b8ce7ff`** (config.py): the pod digest is the one all recorded indexes carry |
| Embedding backend and cache | `embeddings/backend.py`, `cache.py` | copy | **`question_cache_key` does not include the corpus**: every corpus gets its own cache directory (CLAUDE.md "Gotchas") |
| CLI shape | `cli.py` (`uv run cer <stage>`) | take the pattern only | per-phase stages and about 3,400 tests stay behind; the new project needs one generic tournament runner (charter §7) |
| Config as the home of every deciding constant | `config.py` | take the pattern and the pins | |
| Guard hook | `.claude/hooks/guard_main.py` | copy | blocks pushes to main, force-push, `.env` reads; also blocks inline text that looks like such a command, so commit messages and PR bodies go through `-F` / `--body-file` files (deliver skill §8) |
| CI | `.github/workflows/security.yml`, `.secrets.baseline`, `.github/detect_secrets_filters.py` | copy; the baseline **starts empty** (charter §6) | CI floors: urllib3 2.8.0 and `PYSEC-2026-3447` ignored by id while vllm pins `setuptools<81` (commit 3160a66; 17.results.md §10b). SEC-018: the dependency scan must sync the optional groups it audits |
| Skills, commands, templates | `.claude/skills/{research-protocol,deliver,remote-gpu}`, `.claude/commands/`, `docs/templates/` | copy (section 10, step 2) | the commands are in Spanish; the templates keep Spanish filenames; `excalidraw-diagram` and `gdoc` skills are generic helpers |
| Glossary | `docs/GLOSARIO.md` | copy the file | **it is git-ignored on purpose** (memory `translate-ir-jargon-into-plain-spanish`; `.gitignore` line 139), so a `git clone` does not carry it: copy it from the laptop by hand. The charter's "copied and extended" (§6) needs that |

### 5.2 Artifacts (read-only inputs, never moved or rewritten)

The artifacts are **git-ignored** (`data/` is not committed, except the few versioned JSONs such as
`integrity.json` and `outcome.json`). They live on the author's laptop and, for Phase 17, also in
tarballs brought back from the pod. The charter's rule (§6): **read in place, read-only, each checked
against its recorded digest; never moved or rewritten.** Proposal: the new project reads a data root
from one config constant (or an environment variable) pointing at this repository's `data/`, and
refuses to write under it.

| Artifact | Path here | What / size / digest | Source |
|---|---|---|---|
| FullWiki corpus and caches (5,233,329 paragraphs): BGE vectors, GLiNER records, BM25 and entity indexes | `data/phase9/` (`corpus.jsonl.gz`, `cache/`) | GLiNER configuration digest `2f7864661b8ce7ff`; source archive SHA-256 `1acca1c5cc93c4890ea51091d2bad7c3ef6987aead127ab88728dc9e26555729` | config.py (`PHASE_9_ARCHIVE_SHA256`); charter §6 |
| FullWiki source archive (carries the unused link layer) | `data/phase8_1/source/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2` | `text_with_links` on every record | memory `hyperlink-layer-unused`; research_roadmap.md |
| MuSiQue corpus, vectors, BM25, entity index, rankings | `data/phase15/` (`corpus.jsonl.gz`, `rankings-*.jsonl.gz`) | 101,962 units; unit-set hash `15e2770c8ceb70f7`, ordered digest `91fccb24…857a` | 15.results.md |
| MultiHop-RAG corpus, vectors, BM25, entity index, rankings | `data/phase16/` | 27,989 units; unit-set hash `4aa1f9aab57fd101`, ordered `1520cc88…1040`; vectors `daf68122…3b44`; BM25 `ba90bbe8…09e4`; extraction records `18dcb6f7…5489`; entity index `e0b0ab25…a43e`; tarball `phase16-build.tgz` 46,090,493 bytes, SHA-256 `b7cd4c61…9163` | 16.results.md §2-3, §11 |
| HotpotQA dev component lists to depth 100 | `data/phase10/dev-lists.jsonl.gz` (digest `d2e5c2d6…b7e9`), `data/phase14/dev-lists.jsonl.gz` (`006b2768…8ad2`) | dense, bm25, entity hop and relevance hop at five α | 16.results.md §4, §11; 17.spec.md "Existing code context" |
| Stored rankings to depth 100, four systems | `data/phase15/`, `data/phase16/` `rankings-*.jsonl.gz` | read-only inputs for judges and the game prototype | charter §6 |
| Frozen fits | Phase 10 fit `66dfcec1…dd96`; Phase 14 fit `7ce07a17…e224` (`data/phase14/fit.json`) | weights 0.5 / 0.5; 0.5 / 0.3 / 0.2, α = 0.75 | 16.results.md §11 |
| **Phase 17 pools** | `data/phase17/pool-<set>.jsonl.gz`, `pool.json`, `integrity.json` | pool sizes in 5.2.1 | 17.results.md §2 |
| **Phase 17 pairs** | `data/phase17/pairs-<set>.jsonl.gz` | 2,013,745 pairs; sizes and SHA-256 in 5.2.1 | 17.results.md §2; 17.runpod_recipe.md §1 |
| **Phase 17 scores and reordered lists** | `data/phase17/scores-<judge>-<set>.jsonl.gz`, `scoring-<judge>.json`, `reordered-<judge>-<set>-<system>.jsonl.gz` (45 reordered files) | the pod tarball `phase17-scores.tgz`, 114,814,830 bytes, SHA-256 `b2e6708c…fd34`, 196 files verified | 17.results.md §2, §11 |

Where a figure is shortened above, the full value is in the named file (for example the committed
manifests `scoring-light.json`, `scoring-strong.json`, `scoring-decision.json`, `outcome.json`,
`metrics.json`, `published-figures.json`).

#### 5.2.1 The Phase 17 pool and pairs files (17.results.md §2, measured)

| Set | Questions | Distinct units | Pairs | `pairs-*.jsonl.gz` bytes, SHA-256 | `pool-*.jsonl.gz` bytes, SHA-256 |
|---|---:|---:|---:|---|---|
| HotpotQA dev | 7,405 | 841,339 | 1,295,224 | 192,079,213; `a8218505…dff3` (full: `a82185054cadb4ace32d757a7e55a9cf398a029242a115c8e4d206f54368dff3`) | 20,831,773; `b019f58f…34ec` |
| MuSiQue validation | 2,417 | 76,148 | 400,745 | 82,463,395; `fef878ae…63a1` (full: `fef878aeba159afd159305804675a144d111459b22c8b186a523d0a4e79863a1`) | 5,635,884; `9911f3a1…37b0` |
| MultiHop-RAG answerable | 2,255 | 23,116 | 317,776 | 40,234,938; `939132e9…db71` (full: `939132e93e0da3a604caf9569bdd94d1e8e7584f32c9d6f7335dae62baf3db71`) | 5,201,122; `0acd5039…d72d` |

Pool size mean / median / p90 / min / max: 174.9 / 173 / 203 / 114 / 276; 165.8 / 163 / 194 / 113 / 237;
140.9 / 138 / 165 / 105 / 215. Digest chain: `integrity.json` content digest `3fb0b513…1f35` is carried
in `pool.json`; each scores file's SHA-256 and content digest are in its manifest; `outcome.json`
records the integrity digest, the 45 reordered files by digest and the metrics file's SHA-256. The
committed JSONs' SHA-256 (working copies at commit `6c81170`): `integrity.json` `86884368…dcc8`;
`pool.json` `ac6a1239…c5f2`; `scoring-light.json` `6266b676…b7ee`; `scoring-strong.json`
`89651fac…cadb`; `scoring-decision.json` `73b21f1d…fc3f`; `outcome.json` `3b01fcdf…bd47`;
`metrics.json` `b4891d87…6624`; `published-figures.json` `3a0a0e63…de8c` (17.results.md §2).
The pools carry unit ids, not texts; texts come from the corpus files of Phases 9, 15 and 16
(17.spec.md "Data contracts"). Every artifact is JSON, JSONL or NPZ; no pickle (the one exception is
the CLM-8B heads checkpoint, a `torch.save` file checked by SHA-256 before anything reads it,
config.py).

### 5.3 The frozen pins

| What | Pin | Source |
|---|---|---|
| Dense | `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, weights SHA-256 `3c9f3166…21ad`, 384 d, L2-normalized, empty query prompt (also the budget tokenizer's revision) | config.py (`EMBEDDING_REVISION`); 16.results.md §3 |
| Entities | `urchade/gliner_medium-v2.1`, revision `40ec419335d09393f298636f471328b722c6da9e`; weights SHA-256 `922214c0…c023`; tokenizer `microsoft/deberta-v3-base` revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`; gliner 0.2.29; five labels, threshold 0.5, `max_len` 384, batch 8, seed 42; configuration digest `2f7864661b8ce7ff` | config.py; 16.results.md §3, §11 |
| J-light | `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, weights SHA-256 `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`, 22,713,601 parameters, max length 512, float32 (matmul `highest`), batch 256 | config.py (`PHASE_17_JUDGES`); 17.results.md §3 |
| J-strong | `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, weights SHA-256 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`, 567,755,777 parameters, max length 8,192, float32, batch 32 | same |
| J-decision | `Contrastive-LM/CLM-v0.1-8B`, heads revision `e939398d4556fcd9400c76fa8c5a513202f42b0a`, file `CLM_v0.1-8B.pt`, SHA-256 `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`; encoder `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218` (five shards verified by SHA-256); bfloat16 served by vLLM 0.30.0, `--max-model-len 2048`, score `100.0 * dot(z_c, z_s)` | same; 17.runpod_recipe.md §9 |
| Library stack on the pod | torch 2.13.0+cu126, transformers 5.16.1, sentence-transformers 6.0.1, tokenizers 0.23.1, huggingface-hub 1.33.0 (1.29.0 in earlier phases), safetensors 0.8.0 | 17.results.md §3; 16.results.md §4 |
| Frozen fusion | P10-B 0.5 / 0.5; P10-C 0.5 / 0.3 / 0.2; P14 same with α = 0.75, depth 100 | 16.results.md §1 |

### 5.4 Corpora and question sets

| Set | Pin and size | Gold mapping and notes |
|---|---|---|
| **HotpotQA FullWiki** | the processed English Wikipedia archive `enwiki-20171001-...-withlinks-abstracts.tar.bz2`, 5,233,329 paragraphs; the 7,405 official validation questions are dev; `test-11` (5,000 train questions, level `hard`) is spent; the 5,405 "retrieval-unseen" cohort was Phase 9's primary | gold = the two supporting paragraphs; BGE-small was fine-tuned on HotpotQA (C-Pack), so train questions read 2.6-2.9 points better for Dense (research_summary.md, point 5) |
| **MuSiQue** | Hugging Face `bdsaglam/musique`, revision `22873a405dd809893b22ada0b499299fb612d2df`; `musique_ans_v1.0_train.jsonl` 241,046,755 bytes SHA-256 `83a75b1e…490a`, `musique_ans_v1.0_dev.jsonl` 30,439,728 bytes SHA-256 `15fa6379…f3b`; 2,417 validation questions with 2 / 3 / 4 supporting paragraphs (1,252 / 760 / 405); 101,962 pooled units (the literature's "about 84,000" was not reproduced; unexplained); byte identity with the official release **not verified** (author decision 1, Phase 15) | gold = marked paragraphs; 2,629 distinct gold units; 0 unmapped. Near-duplicate gold: 72 gold units, 204 of 6,404 pairs, 172 questions; measured post hoc to move no comparison (15.results.md). Distractors mined with BM25 |
| **MultiHop-RAG** | Hugging Face `yixuantt/MultiHopRAG`, revision `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`, ODC-BY; `corpus.json` 6,785,567 bytes SHA-256 `20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f`, `MultiHopRAG.json` 5,171,312 bytes SHA-256 `03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff` (the GitHub `dataset/` folder served 404 on 2026-09-30); 609 articles, 2,556 queries (2,255 answerable, 301 `null`); every newline paragraph a unit, title prefixed, **boilerplate kept** (4,538 of 28,711 paragraphs are five words or fewer), 722 verbatim repeats collapsed: 27,989 units | gold = the paragraph holding each fact; 2 / 3 / 4 gold units: 1,079 / 780 / 396; 962 distinct gold units; 0 unmapped. **Deviation 16.1**: the first reading "29 facts cross a paragraph break" was wrong; they are sentences a published article prints twice (29 queries, 6 distinct sentences): gold is the first occurrence (`16.1_repeated_facts_not_straddling.md`). The `answer` field was never read. Queries written by an LLM (GPT-4 paraphrase of extracted facts) |

**Proposal for the dev/test split of new corpora** (section 10 step 7): the new exam corpus is opened
once; for the terrain, decide per corpus whether a dev subset is carved before any candidate runs,
or whether the whole set is "open for choosing" as charter §4 says; record the rule before opening.
**The reproduction gate** (charter §6, "A miss is a finding about the copy, never a reason to edit
code until the number matches"):

| Check | Recorded figure | Source |
|---|---|---|
| P10-C, HotpotQA dev, Full Support @2,048 | 4,801 / 7,405 | charter §6; 17.results.md §2 |
| P14, HotpotQA dev | 5,224 / 7,405 | same |
| Dense / Dense + BM25 / P10-C / P14, MuSiQue validation | 436 / 524 / 669 / 761 of 2,417 | same; 15.results.md |
| Dense / Dense + BM25 / P10-C / P14, MultiHop-RAG answerable | 338 / 587 / 522 / 513 of 2,255 | same; 16.results.md |
| GLiNER configuration digest (pod) | `2f7864661b8ce7ff` | charter §6 |
| *Additional, from Phase 17's D3 and plan decision 3 (measured)* | HotpotQA dev P10-A 4,125 and P10-B 4,536 | 17.results.md §2 |
| *Proposal: the judge, if reused* | J-strong(P10-B) 5,198 / 704 / 822; J-strong(union) 5,922 / 809 / 846; the three scores files' SHA-256 equal the manifests'. J-strong's attributable cost was 1.1576 USD (derived) | 17.results.md §1, §4 |

---

## 6. What stays here and why

From charter §7, plus Phase 17's additions:

| Left behind | Why |
|---|---|
| The concept-space line (Phases 2-4): `concepts/`, `retrieval/conceptual`, `retrieval/diffusion` | closed with a bounded negative result; no role in the new question |
| Per-phase stage code (`cmd_pN_*`, `evaluation/phaseN.py`) and its about 3,400 tests | written to reproduce this line's records; the new project needs one generic tournament runner |
| The paragraph as the only indexing unit | right for this line's question; in the new one the unit is a declared variable per corpus type |
| "The Entity Hop is intentionally minimal" and the ban on new graph mechanisms without a spec here | the hop becomes one ingredient among many |
| BGE-small as the fixed Dense model and the HotpotQA-fitted weights (0.5 / 0.3 / 0.2, α = 0.75) | Phase 16 showed fitted weights do not travel; the new ghosts come from Phase 0. **Added by Phase 17:** under a judge the hop's weights are irrelevant to the order but the *pool* they build is what the judge reads, so the weights still shape the union pool |
| The 600 / 1,400 HotpotQA-derived set and `test-11` | historical; `test-11` is spent and stores no rankings (17.spec.md D3) |
| Phase-specific artifacts under `data/phaseN/` other than the caches and rankings in 5.2 | this line's record; read-only here |
| **Added by Phase 17:** the paper, its LaTeX and its venue process | the publication is the old repository's closing task and the author's; the new project may reuse the generator pattern, not the content |
| **Added by Phase 17:** J-decision's figures as a "judge" result | negative and one realization (not batch-deterministic); the new project may declare CLM-8B again only as a new experiment (section 3, fact 12) |
| **Added by Phase 17:** `phase_17/17.results.md` §9's published-figures table | context for the paper; Phase 0 will build its own map |
| The Spanish source documents `docs/refs/descripcion-proyecto.md`, `bibliografia.md` | `bibliografia.md` goes as a Phase 0 input; the original hypothesis stays in this repository |

---

## 7. Protocol the new project keeps

These come from `CLAUDE.md`, the skills `research-protocol` and `deliver`, and the author's memory
notes. The charter marks the first six "decided unless marked" (§4).

1. **Dev/test discipline.** Choose on dev only; the held-out split is opened once, for the
   configuration dev chose, under a rule frozen before any candidate ran; a candidate that fails a
   pre-declared gate is not replaced by another model inside the same phase (a new written
   experiment instead; Phase 8 and the spaCy case are the worked examples). Never tune or choose by
   looking at test results. Never pre-measure a candidate on dev before its spec: only oracle
   ceilings may run (memory `next-direction-candidate-side`). In the new project the **terrain/exam**
   pair replaces dev/test (charter §4): the exam corpus is run once.
2. **Preregistration and frozen specs.** A spec the author approves is frozen; a real problem
   outside the plan is a written deviation `X.Y_name.md` for the author to decide; no new
   threshold, gate, outcome or variant after approval. The flow: spec (`/4-especificar`) then plan
   (`/5-planear`) then autonomous implementation (`/6-implementar`) then `deliver`; **the author's
   attention goes to the spec before and the PR after** (memory `autonomous-branch-workflow`, 2026-09-23).
   Stop and ask only for ASK FIRST items, a problem that would change the frozen spec, a step that
   opens the exam or spends money, a reproduction check that misses its figure, a gate or stop state, or
   two attempts without progress.
3. **Simplification rule (Phase 7 onward).** One main scientific question per phase; roughly 5-8
   implementation steps; no `X.tasks.md` unless needed; reuse code, do not generalize pre-emptively;
   no branches for failures that have not occurred; test code that can change a result; no elaborate
   statistical framework when a direct comparison answers the question; complexity is a research cost.
4. **Write-once artifacts with digests.** Expensive deterministic outputs are cached on disk and
   reused; a different model or configuration writes a distinct cache and never overwrites; a digest
   chain from extraction to final run so a held-out figure cannot be detached from its dev reading;
   non-executing formats only; pin model identity and revision for every download.
5. **Labels by code.** A terminal state or label is written mechanically by the outcome code from the
   run files (`stop_reasons` empty), never by hand (Phases 14-17; 17.spec.md D5).
6. **Measured / derived / exploratory / interpretation / projection labelling.** Estimates and
   projections are never reported as measurements; every phase result labels every value
   (Phase 7's 5M-paragraph 8.56 h / 6.34 USD is a projection, not a budget).
7. **The two-comparisons rule** (author, 2026-09-24; memory `step-2-two-comparisons`): every new
   system states its gain against the project's own bar (P10-C, keep or drop) **and** against the
   community reference (P10-B). In the new project the community reference becomes the Phase 0
   ghost (section 4.6): state both.
8. **"The baseline must be the literature's best cheap recipe."** At every spec or charter, name the
   strongest cheap literature baseline for the task, with its published figure, and either make it
   the control or say in one sentence why not; tell the author **plainly and first** when the working
   control is weaker (memory `baseline-must-be-literature-best`: the author was angry, 2026-09-30,
   that 16 phases compared against a weak control). Also check the project's own measurements
   before recommending a stronger model (Phase 8).
9. **Never rewrite frozen artifacts.** A provenance finding against a closed phase gets a
   prospective fix in the producer, never an edit of the artifact, even a one-line one (memory
   `never-rewrite-frozen-artifacts`, 2026-09-19: SEC-031). Example here: the `allocation.cgroup_*`
   fields are null in the three Phase 17 manifests; the recipe holds the values; the fix is in the
   producer (17.results.md §11, found 5).
10. **Spec drift: fix the spec, visibly.** When verification finds the code right and the spec
    stale, correct the spec as a "Corrected during implementation" note, in its own commit, with no
    code change (memory `spec-drift-update-the-spec`).
11. **Explain purpose before asking decisions** (memory `explain-purpose-before-asking-decisions`):
    say what a phase builds, which later phase consumes it and whether it is measurable against the
    baselines, before presenting any choice.
12. **Spanish glossary for the author; no unglossed jargon** (memory
    `translate-ir-jargon-into-plain-spanish`): gloss every term of art the first time in a session; for
    results, state the finding in outcomes first ("acierta 35 de cada 100"), the metric names belong in the written
    report; offer at most three real paths with cost and what is learned, never lettered variants;
    check the author still holds the high-level picture.
13. **English in code and docs.** The two source documents in `docs/refs/` stay in Spanish; the
    framework commands in `.claude/commands/` are in Spanish; ASCII only in Python console and
    logging strings (`[OK]`, `->`); type hints; `mypy src` and `ruff check .` pass.
14. **Check one example before a count enters a spec** (charter §9; deviation 16.1).
15. **Every corpus gets its own caches; tokenize large corpora in batches** (charter §9).
16. **Reproduce a bug through the real CLI stage or artifact before fixing it**, and when choosing
    between technical options do not weigh implementation effort heavily: agents make code cheap, a
    weak experimental design is expensive (CLAUDE.md).
17. **A repository that is public** publishes whatever is committed (CLAUDE.md); never commit `.env`
    or credentials or the raw corpus; the author's name is on the repository, so double-blind
    submissions need an anonymous mirror (17.spec.md D9).

**Not automatically carried** (flag at Phase 0): "no LLM in the online retrieval loop except as a
declared variant" (charter §1 admits learned models; charter §10 item 3 leaves the heavy class
open); the ban on multiple seeds, hops, relations or canonicalization without a spec (the hop is
now an ingredient; each such lever is a candidate only through a spec, charter §4).

---

## 8. Operational knowledge

**The laptop.** Windows 11 Home ARM64 (10.0.26200), Qualcomm ARMv8, 12 cores, 33,896,992,768 bytes
RAM, Python 3.12.10, torch 2.13.0+cpu, no GPU (16.results.md §11). Python is pinned `>=3.12,<3.13`.
`pyproject.toml` selects `pytorch-cu126` on linux/x86_64 and `pytorch-cpu` elsewhere: respect it.
**No spaCy** (`blis` has no `win_arm64` wheel); **GLiNER on CPU measured 0.558 paragraphs/s against
162.178 on an RTX 4090 (about 290 times slower)** (skill `remote-gpu`); heavy work goes to a rented
GPU, and an experiment is never distorted to fit the laptop. A plain `uv sync` on the laptop removes
the loose `gliner` install; reinstall with `uv pip install gliner==0.2.29` (the `phase7` group cannot install
on win_arm64 because of `blis`) (memory `vllm-cu13-vs-torch-cu126-on-the-pod`). The first
`uv run pytest` takes about 35 s because torch loads: it is not a hang. Git Bash: use Unix syntax.

**RunPod via API and SSH** (decided and working since Phase 16, 2026-09-30; memory `runpod-via-api-and-ssh`; skill `remote-gpu`;
`docs/plans/phase_17/17.runpod_recipe.md`):
- The API key is in the **Windows user-level environment variable `RUNPOD_API_KEY`** (a PowerShell
  `$env:` export is not inherited by the VS Code extension; restart VS Code after setting it). Check
  presence with `env | grep -o "^RUNPOD_API_KEY="`, **never print the value.** The SSH key is
  `~/.ssh/runpod_ed25519` (public half registered in RunPod Settings, SSH Public Keys).
- Flow: `curl` to `https://api.runpod.io/graphql` with `Authorization: Bearer $RUNPOD_API_KEY`;
  `myself { clientBalance pubKey }` (read-only check); `gpuTypes(input:{id:"NVIDIA GeForce RTX 4090"})
  { securePrice }`; `podFindAndDeployOnDemand` with `cloudType: SECURE`, image
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, ports `22/tcp,8888/http`, `startSsh:
  true`, `supportPublicIp: true`, volume 80 GB and container disk 40 GB in Phase 17 (40 / 20 in
  Phase 16); poll `pod(input:{podId}) { runtime { ports } }` for the public port 22 mapping;
  `ssh -i ~/.ssh/runpod_ed25519 -p <port> root@<ip>`; `scp` for uploads and the tarball;
  `podTerminate` at the end (**terminate, not stop**: a stopped pod bills its volume) and
  `myself { clientBalance pods { id } }` to confirm nothing is left.
- **Rate:** 1x RTX 4090, Secure Cloud, **0.74 USD/h** (2026-09-30 and 2026-10-01). Record `costPerHr`
  at creation and enter it as `--hourly-rate-usd`. Phase 16's session: 13 min, 0.165 USD invoiced;
  Phase 17's: 6 h 56 min 48 s, 5.257 USD invoiced.
- **Always:** freeze and push the exact commit; check the three uploads' SHA-256 on the pod; verify
  CUDA with a real kernel (printed `2.13.0+cu126 NVIDIA GeForce RTX 4090 1073741824.0`); probe a small
  sample first; `sha256sum` every expensive artifact on both ends and **download before terminating**
  (`/workspace` dies with the pod); decide on the laptop, not the pod; write `pod_setup.sh` in the
  scratchpad, `scp` it, run it with `nohup`, wait with a background `until grep` loop. Stop the pod
  and ask the author if the session nears the recipe's cap (10 USD in Phase 17; the effective cap was
  the account balance by the author's decision at stop point 1).
- **The manifest `hardware` block reports the host** (for example 256 cores, 1.08 TB RAM), not the
  container allocation; record the allocation separately (skill `remote-gpu`). The Phase 17
  manifests' `allocation.cgroup_*` are null (helper did not read cgroup v2); the recipe holds the
  values (cgroup memory 61,999,996,928 bytes, 128 vCPU visible).
- **Cost accounting lag.** The account balance and the Billing page lag about an hour behind a
  terminated pod: Phase 16's balance delta was 0.1029 USD against an invoiced 0.165 USD; Phase 17's
  was 5.2325 USD against 5.257 USD (0.0245 USD short). The author copies the invoiced total from
  Billing later; record time x rate, the balance delta and the invoice as three distinct figures,
  each labelled (17.results.md §4).
- **The vLLM cu13 / torch cu126 trap** (memory `vllm-cu13-vs-torch-cu126-on-the-pod`; 17.runpod_recipe.md §7):
  PyPI `vllm` 0.30.0 is a CUDA 13 build while the project pins `torch==2.13.0+cu126`. `import vllm`
  failed (no `libcudart.so.13` on the loader path) and `vllm serve` failed (PyPI's torchvision 0.28.0
  is CUDA 13). The fix the author chose: `torchvision==0.28.0` and `torchaudio==2.11.0` declared in the
  `pod` group and routed to `pytorch-cu126` via `tool.uv.sources` (commit `6aa111b`), plus
  `export LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH` in the
  pod's `env.sh`. A cross-check against a transformers CPU bf16 pass gave cosines 0.99983 / 0.99997 /
  0.99995. `tool.uv.sources` only routes packages the project declares directly.
- **The `uv run` re-sync trap.** `uv run` re-syncs the venv from the lock, so a pod-local `uv pip
  install` fix is undone by the next `uv run`; change `pyproject.toml` and `uv.lock` and move the pod
  to that commit.
- **The pkill trap.** `pkill -f "vllm serve"` killed the SSH session whose own command line held the
  pattern; use `pgrep -f "[v]llm serve"` (17.runpod_recipe.md §7).
- **vLLM serving command used** (17.runpod_recipe.md §9): `vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
  --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --port 8090 --dtype bfloat16`;
  check the pooler line reads `pooling_type=LAST` before scoring.

**Subagents.** Spawn only when the author asks for one; a request to fix or review is done inline.
**Model rule** (memory `subagents-always-opus-5`): every subagent the author asks for runs on the
model named explicitly in the call (`opus`), never inherited, except where the author names another
model for a stretch (Phase 17 from 2026-10-01: Sonnet 5.5; the rule reverts to Opus for later work unless the author says
otherwise). Sonnet 5 on Phase 12 (2026-09-27) implemented correctly but stated interpretations as
facts and cited a finding that did not exist: supervise results writing. **Subagents hit a wall-clock
limit near 1,000 s: brief them in pieces that fit, never a whole long-running stage plus its wait.**
Check that the author has not asked for plan changes before launching an implementer. Commit
trailers follow the harness's attribution reminder; rewriting local history was denied by the
permission classifier (2026-10-01), so note a trailer mismatch in the PR instead.

**Secret scanning.** `detect-secrets` on this Windows laptop misses the Hex High Entropy strings the
ubuntu CI flags (memory `secret-scan-windows-blind-spot`): **a local green scan is not evidence; read the
CI job.** Add baseline entries from CI's reported lines: generate them with `detect-secrets scan`
on the **new files only** and merge by script; never `scan --baseline` with a file list (it prunes the
rest); never regenerate on Windows (backslash paths); use forward-slash paths and sha1
`hashed_secret`. Inserting lines above an audited entry rewrites the baseline: fix `line_number`.
Write digests in docs as backticked text without quotes (a hex digest inside double quotes was
flagged). Phase 17 added 327 baseline entries this way. The new project's baseline starts empty
(charter §6); expect to add entries for every published digest.

**Other practice.**
- **Write files with the Write tool, not a heredoc** (memory `write-files-with-write-not-heredoc`): a large `cat <<'EOF'` through
  Bash fails. The guard hook also reads inline shell text: commit messages and PR bodies go in files
  (scratchpad) via `git commit -F` and `gh pr create --body-file`.
- **GitHub reads** through `npx -y gh-axi@0.1.35` (`pr view|list|checks`, `run list|view` only); writes use `gh`.
  Stage with explicit paths, never `git add -A`. Never push to `main`, never merge, never
  force-push; commits and pushes to non-`main` branches are a standing authorization for this
  repository (2026-09-23) and **must be re-granted by the author for the new repository**, since it is
  a separate repository (proposal; the memory note names this one).
- **LaTeX.** There is no TeX on PATH; **Tectonic 0.17.0** (x86_64 Windows build under Windows 11's x64
  emulation, no native ARM64 asset) at `C:/Users/mikel/.local/bin/tectonic-0.17.0/tectonic.exe`
  (17.0 plan; 17.results.md §10). `uv run python docs/paper/build.py` runs `cer p17-tables` then Tectonic.
- **Attribution trailers.** Commits end with the trailer the harness names (this session's:
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`); PR bodies end with the
  "Generated with Claude Code" line. The author's own instructions about these lines take precedence.
- **Security audit** (`/8-auditar`) only when a phase adds a real surface (untrusted input, risky
  dependencies, subprocess, unsafe deserialization, credentials, network APIs); findings go to `docs/security/README.md`.
- **Dependencies:** adding or changing one is an ASK FIRST item (CLAUDE.md); Phase 17 pre-authorized
  its own (the `pod` group). The CI dependency scan syncs every optional group, which now includes
  installing vLLM (open question, memory `phase-17-judge-paper`).
- **`docs/USER_GUIDE.md` is an older pipeline and not authoritative** for Phases 5-9 (CLAUDE.md).

---

## 9. Open questions and risks the new project inherits

**Scientific.**
1. **The hyperlink layer / the unit about an entity.** Excluded as a component because target
   corpora carry no links; the corpus-agnostic version (title-mention, first-sentence subject)
   keeps most of the reach (55.6 % vs 62.8 %). Not tested as a retriever. Declare as conditional on
   the corpus (section 3, fact 14).
2. **The cause of the news regression** is not settled. H1 (the seed is the wrong paragraph of the
   right article), H2 (frozen weights take weight from BM25) and H3 (entity sharing is weak evidence in a small, hub-dense
   corpus) are hypotheses; the diagnostics that would separate them were not run (16.results.md §12).
   One tiny corpus (187 times fewer units than FullWiki), one kind of text, questions written by an
   LLM.
3. **What a judge's cost means at query time.** A cross-encoder scores every pool pair for every
   question: about 175 pairs per question on HotpotQA; J-strong about 0.45-0.54 s per question on one
   RTX 4090 (derived). The old line's systems "are never claimed to match a judge's quality at their
   cost" (17.spec.md limitation 5). Quality at a given cost is the new question; the cost curve of
   judge depth and pool size is unmeasured.
4. **The union pool's bound at 100.** About four questions in ten on MuSiQue and news have gold outside
   it (41.7 % and 42.4 %); a deeper pool is not tried. Whether a reorder of a deeper list or better
   first-stage retrieval closes the larger gap is open (17.results.md §12.5c).
5. **CLM-8B: not a judge, maybe a first-stage retriever.** Its negative result is for one usage (the
   question alone as the state, no instruction). As a bi-encoder it can embed a corpus once offline;
   a different state (question plus the first paragraph, an instruction) is a declared candidate
   for a new experiment, never a retry (17.results.md §12.4). Per-text compute is about 12 times a
   J-strong pair (derived: 8 B against 0.57 B parameters), but each unit is embedded once per corpus
   (memory `jev-decision-model-as-judge`).
6. **J-decision is not batch-deterministic** (bfloat16 plus vLLM scheduling). Two of its three
   labels sit near the threshold (HotpotQA 264 / 256, p = 0.76; MultiHop-RAG 21 / 44, p = 5.9e-3);
   a second pass was not measured. If CLM-8B returns, plan for a determinism check as a gate and
   consider float32 or a fixed batch (a float32 8 B model does not fit a 24 GB card, 17.results.md
   §11 / deviation 17.1).
7. **Jev** (TypeSafe AI, released 2026-09-15): API-only, no open weights, waitlisted, determinism and
   benchmark publishing terms undocumented; **deferred to the successor** (memory
   `jev-decision-model-as-judge`: "revisit it in the successor project"). A closed, mutable API is a
   poor fit for a reproducible tournament; if included, it is a declared, labelled variant.
8. **J-light's news regression is unexplained** (17.results.md §12.3): a reading, untested, is that its
   order discards what the lexical component found.
9. **Fitted weights do not travel**; the successor should prefer rules with nothing fitted, or
   report the fitted figure as an upper bound (charter §9).
10. **Entity Hop at FullWiki scale** is a candidate generator whose cost the new project must carry
    beside every result: indexing 4.57 USD of GPU for FullWiki (derived) but a tail up to 16.5 s per
    query on the laptop at scale.
11. **A stronger Dense** has never been tested (Phase 8 failed its premise); Phase 0's ghosts will
    probably include one, so the old hop gains need re-measuring beside it.

**Process and bookkeeping.**
12. **The invoice lag**: record time x rate, balance delta and the invoice separately and label each;
    the author copies the invoice from Billing after about an hour (sections 8, 2.3).
13. **The ruff-format debt.** The sources read here do not record a list of files failing
    `ruff format --check`; the `deliver` skill's rule is that a check that already failed on the base
    counts as passing when the change does not worsen it, and that such debt is listed in the PR
    (deliver skill §3). Check `uv run ruff format --check .` once in this repository before copying
    code, and format the copy in the new repository's first commit so the debt does not travel
    (proposal; not located in the sources).
14. **Source inconsistencies found while writing this document** (not resolved here; the
    author decides): (a) `17.results.md` §10 still lists the paper's author as the placeholder "M.
    Jurado" while §10b (later the same day) records **Mikel Ugarte-Gil**; `main.tex` already says the
    latter. (b) `docs/plans/research_roadmap.md` §7.2 says "the MDR-style query vector (question + P1)
    is Phase 17" and §9 says the same, while Phase 17 became the judges and the paper and the query
    reformulation moved to the successor. (c) The charter's header still says "Status: draft for the
    author" while the charter was merged (commit `7f89744`; memory: PR #26). (d) The charter says
    the glossary is "copied and extended" while `docs/GLOSARIO.md` is git-ignored. (e) The charter
    §6 says the caches and rankings are "about 12 GB"; `data/` is about 19 GB in all on
    2026-10-01 (it includes Phase 17 and everything else). (f) Memory `MEMORY.md`'s index line for
    Phase 17 still reads "implement in a clean session" (superseded by the later `phase-17-judge-paper` note).
15. **Author-only items still open in the old repository**: arXiv endorsement, the anonymous mirror,
    the AI-disclosure wording, an OpenReview profile (venues.md); none blocks the new project.
16. **The charter's five open decisions** (charter §10): the new repository's name and whether public
    from the start; the exam corpus (QASPER or LegalBench-RAG); whether the heavy class admits an LLM
    in the online loop; whether offline LLM work is allowed in the light class; the reading budget
    of record (2,048 tokens or a small set).

---

## 10. First steps for the new repository (proposal, in order)

1. **Create it with `/nuevo-proyecto`** (skill: "Crea un proyecto SDD Lite en la carpeta actual (git
   init, plantilla, dependencias, comprobaciones y primer commit)"), after the author names it and
   decides public or private. Re-grant the standing branch-commit authorization for the new
   repository (section 8).
2. **Copy the framework**: `.claude/skills/{research-protocol,deliver,remote-gpu}` (edit
   `research-protocol` for terrain/exam in place of dev/test; edit `remote-gpu` with the API flow in
   section 8), `.claude/commands/*.md`, `.claude/hooks/guard_main.py` and its settings entry,
   `docs/templates/*` (`0_plan_maestro.md`, `CLAUDE_proyecto.md`, `X.0_plan_fase.md`, `X.Y_desviacion.md`,
   `X.spec.md`, `X.tasks.md`, `fix-N_nombre.md`), `.github/workflows/security.yml` and
   `detect_secrets_filters.py` (with an **empty** `.secrets.baseline`), `docs/GLOSARIO.md` by hand
   (git-ignored here). Copy this document and the charter into the new repository as the starting brief.
3. **Master plan and CLAUDE.md** via `/1-crea-plan-maestro` and `/3-init-project`, from the charter;
   keep the new `CLAUDE.md` short (it loads every session), put state in the plan.
4. **Write Phase 0's spec from the charter §3** with `/4-especificar`: the map of quality against
   cost in the two budgets, the ghosts, the published-figures ledger (each paper's metric and unit
   stated, never compared with Full Support). Name the strongest cheap literature recipe with its
   published figure and make it the control (the lesson, section 7 item 8). Explain purpose before
   asking decisions.
5. **Copy the code (5.1) and run the reproduction gate (5.4)** before any new measurement; a miss is
   a finding about the copy. This needs the Phase 9, 15 and 16 caches (read in place) and the stored
   rankings; if the new repository is on another machine, the tarballs and digests of 5.2 are the
   transfer record.
6. **Reproduce the ghosts** on the new harness, one per cost class. Phase 17 already provides part of
   the heavy-cheap-judge answer (J-strong over the union pool); Phase 0 decides which ghosts to add
   (agentic RAG for the ceiling is needed either way, charter §10 item 3).
7. **Decide the dev/test (terrain/exam) split for new corpora.** Terrain = HotpotQA, MuSiQue,
   MultiHop-RAG, open for choosing; exam = QASPER or LegalBench-RAG (the author's choice), a rule
   frozen before it is opened, run once. For a corpus whose gold is single-evidence, the metric
   degenerates to Recall@budget (survey §1): state the metric before opening. QUEST or GlobalQA, if
   admitted, are a separate question with their own spec (section 4.5).
8. **What to measure first** (proposal): the ghosts and the four frozen systems on the terrain with
   cost columns (offline and online, separately); then the Phase 17 union-pool-under-J-strong as the
   reference; then the first declared candidate (charter §4 candidates 1-3, one phase each, five to
   eight steps).
9. **Game prototype late** (charter §5): after the terrain results exist; the page reads an exported
   JSON; the four lanes are three candidates and the reference.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Unit** | what a retriever ranks and indexes; here one whole paragraph of a document, with its title prefixed (`indexable_text`: title, then paragraph); never a fixed-size window in this line. In the new project the unit is a declared variable per corpus type |
| **Gold / gold unit** | a unit marked as supporting a question's answer (2-4 per question here) |
| **Full Support @2,048 tokens** | the share of questions for which **every** gold unit fits in a 2,048-token context built from the ranking (token counter: the BGE-small tokenizer). The metric of record; it measures what a reader model would receive, not a rank position (research_summary.md) |
| **Gold recall (GR@k)** | the share of a question's gold units inside the first k units, averaged over questions; **Full support at k units (FS@k)**: every gold unit inside the first k; **nDCG@10**: BEIR's metric with binary relevance over gold units (17.spec.md D6) |
| **Dense / BM25** | the BGE-small embedding retriever and the lexical (keyword) retriever; **Dense + BM25 (P10-B)** fuses them 0.5 / 0.5 by min-max weighted sum |
| **P1** | the first unit of Dense's ranking; the hop's seed |
| **Entity Hop (hop)** | candidates are the units outside the first 10 that share at least one raw entity (a GLiNER-extracted name) with P1, scored by rarity `log(1 + N / (1 + df))` |
| **P10-A / P10-B / P10-C / P14** | the four frozen systems: Dense; Dense + BM25; Dense + BM25 + Entity Hop (0.5 / 0.3 / 0.2); P10-C with the relevance-ordered hop |
| **Candidate relevance** | P14's change to the hop's score: `(1 − α) · rarity + α · cos(question, candidate)`, α = 0.75 |
| **Judge** | a model that scores each (question, unit) pair and reorders a system's top-100, used zero-shot, with no threshold, calibration or fusion. J-light, J-strong (cross-encoders: read question and unit together) and J-decision (a bi-encoder: embeds them separately) |
| **Pool** | per question, the union of the fused top-100 of the four systems (mean 140.9-174.9 units) |
| **Union line** | the whole pool sorted by a judge's score; descriptive, not a system of the line |
| **Ceiling** | the number of questions with every gold unit inside a list's top-100 (or inside the whole pool): the most any reorder of that list could reach; gold-informed, exploratory |
| **Labels** | per-comparison, written by code: `HOP_ADDS_UNDER_JUDGE`, `HOP_HURTS_UNDER_JUDGE`, `HOP_NEUTRAL_UNDER_JUDGE` (Phase 17, wins > losses and p < 0.05, the reverse, otherwise); terminal states `TRANSFER_SUPPORTED`, `TRANSFER_REGRESSION` (15, 16), `CANDIDATE_RELEVANCE_SUPPORTED` (14), `THREE_WAY_SUPPORTED` (10), `SCALE_SUPPORTED` (9), `DEV_STOP` (11-13, a dev gate not passed), `DATA_STOP` (data integrity check missed) |
| **Exact McNemar** | the paired test of two systems on the same questions, over the discordant ones (wins and losses), two-sided, exact binomial |
| **Dev / test, terrain / exam** | dev: where choices are fitted; test: opened once. The new project's equivalents: terrain open for choosing; exam locked, run once |
| **Ghost** | a literature reference system reproduced on the new harness, one per cost class (charter §3) |
| **Measured / derived / exploratory / interpretation / projection** | the five labels of section header |
