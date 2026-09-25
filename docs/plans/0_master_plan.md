# Master Plan: concept-embeddings-RAG

## What is this product?

A research experiment that puts to the test the hypothesis in `docs/refs/descripcion-proyecto.md` §69: that a document corpus can be represented in a sparse semantic space whose dimensions are **concepts discovered from the corpus itself**, and that expanding retrieval iteratively through those concepts recovers context a conventional dense RAG never reaches.

No application is being built. **The deliverable is measured evidence**: comparable retrieval systems evaluated on multi-hop benchmarks with annotated ground truth, together with explicit positive, negative or inconclusive findings.

The project has evolved through evidence. The original concept-space line produced a bounded negative result. A second line, based on entities extracted directly from text, produced a strong positive result: a one-hop entity-navigation signal can complement dense retrieval and replace the BM25 component of the existing dense hybrid under the Phase 6 protocol.

The current research question is therefore narrower and more empirical:

> **What is the minimum structural signal that Dense retrieval needs in order to recover multi-hop evidence that semantic similarity alone does not express?**

> Source documents in `docs/refs/` are in Spanish and stay that way: they are the author's material. Everything this project produces — plans, specs, code, reports — is written in English.

The longer-term research roadmap, including work that is deliberately **not yet scheduled as a phase**, lives in [`research_roadmap.md`](research_roadmap.md).

---

## Documentation convention

The project keeps a lightweight written record of what each phase asks, how it will answer it and what it found.

```text
docs/plans/
├── 0_master_plan.md           # This file - global view
├── research_roadmap.md        # Longer-term research directions
├── phase_1/
│   ├── 1.spec.md              # Functional spec (WHAT + WHY)
│   ├── 1.0_phase_name.md      # Implementation plan (HOW)
│   ├── 1.tasks.md             # Optional; only when genuinely useful
│   └── 1.Y_name.md            # Deviation/problem, if one is actually needed
├── phase_2/
│   └── ...
└── fixes/
    └── fix-N_name.md
```

The historical workflow has been:

`/4-especificar` → `/5-planear` → `/6-implementar` → `/7-verificar` → `/8-auditar` → `/9-documentar`

That workflow remains available, but **from Phase 7 onward it must be applied proportionally rather than mechanically**.

### Simplification rule from Phase 7 onward

Phase 6 demonstrated that experimental engineering can itself become a major cost: roughly 2,500 tests, 24 implementation tasks and substantial machinery were required to answer one comparatively narrow research question.

From Phase 7 onward:

1. **One main scientific question per phase.**
2. Prefer **5–8 implementation steps**, not dozens.
3. Do not create a `tasks.md` unless the phase genuinely needs one.
4. Reuse existing infrastructure instead of generalizing it pre-emptively.
5. Do not build branches, guards, artifact systems or abstractions for hypothetical situations that have not occurred.
6. Add tests for new code that can materially change the experimental result; do not exhaustively encode every process invariant as a test.
7. Keep the essential experimental discipline — dev for choices, explicit configuration, reproducible results and honest limitations — without recreating Phase 6's full protocol machinery.
8. Treat implementation complexity, runtime, storage and preprocessing cost as research costs in their own right.
9. When a simple experiment can answer the question, prefer it over a more comprehensive framework.
10. A phase is finished when its research question has been answered well enough to decide the next experiment; it does not need to become a reusable platform.

---

## Phase status

| Phase | Name | Spec | Status |
|------|--------|------|--------|
| 1 | Corpus, indexing units and dense baseline | Available | **Complete** |
| 2 | Concept space: dictionary + matrix X | Available | **Complete** |
| 3 | Hybrid conceptual retrieval (System B) | Available | **Complete** |
| 4 | Query-aware iterative expansion (System C) | Available | **Complete — negative result** |
| 5 | Text-derived concepts: navigation pilot | Available | **Complete — NAMES ONLY** |
| 6 | Dense + Entity Navigation — end-to-end comparison | Available | **Complete — ENTITY_REPLACEMENT_SUPPORTED** |
| 7 | Cheap entity extraction | Available | **Complete — GLINER SELECTED** |
| 8 | Strong Dense + Entity Hop | Available | **STOP at dev gate — premise failed** |
| 8.1 | **Deviation** — Dense scale sensitivity | Available | **Complete — STABLE_RANKING (rule A)** |
| 9 | HotpotQA FullWiki at literature-comparable scale | [Approved](phase_9/9.spec.md) | **Complete — SCALE_SUPPORTED** |
| 10 | Dense + BM25 + Entity Hop at FullWiki scale (step 1 of 2) | [Approved](phase_10/10.spec.md) | **Complete — THREE_WAY_SUPPORTED** |
| 11 | A better use of the entities at FullWiki scale (step 2 of 2) | [Approved](phase_11/11.spec.md) | **Planned** — [plan](phase_11/11.0_entity_use.md) |

**The first research line is closed** (2026-09-15): concepts induced from pooled embeddings, tested through Phases 2-4, gave a negative and bounded result. The original Phase 5 — a comparative evaluation of that method — was not run; the plan was renumbered so Phase 5 tests the representation the proposal actually described. See [`phase_4/4.1_research_line_closure.md`](phase_4/4.1_research_line_closure.md).

**Phase 5 is measured and closed at the pilot gate:** text-derived entities produce a strong positive second-hop signal on the HotpotQA bridge diagnostic, while text-derived concepts do not. The gate returned **NAMES ONLY**.

**Phase 6 is measured and closed** (2026-09-17): under one protocol frozen and committed before the test split was opened, and read exactly once, `data/replacement/decision.json` records the state **`ENTITY_REPLACEMENT_SUPPORTED`** with no anomaly and no open question. The Entity Hop replaces the BM25 component of the dense hybrid without loss of retrieval quality, and the resulting system also improves on dense alone. See [`phase_6/6.results.md`](phase_6/6.results.md).

**Phase 7 is measured and closed** (2026-09-19): `data/phase7/selection.json` records `selected: "gliner"`, and the single held-out run scores **0.8793 Full Support @2,048** on the 1,400 test questions, keeping **83.5%** of what the 25.9848 USD Claude extraction bought, for an attributable **0.03 USD** of rented GPU compute. The entity representation can be built cheaply. See [`phase_7/7.results.md`](phase_7/7.results.md).

**Phase 8 stopped at its pre-declared dev gate** (2026-09-19): `Qwen/Qwen3-Embedding-0.6B` scored **446 / 600** dev Full Support @2,048 against the inherited BGE-small Dense reading of **487 / 600**, below the 488 the rule required. The premise failed — the model chosen a priori as substantially stronger is weaker on this corpus — so **the held-out split was neither encoded nor measured** and no second encoder was tried. Phase 8 therefore reaches **none** of its three planned outcomes: its research question remains open. See [`phase_8/8.results.md`](phase_8/8.results.md).

**Deviation 8.1 is measured and closed** (2026-09-22) with **`STABLE_RANKING` (rule A)**. On the frozen 600 dev questions, BGE/Qwen Full Support @2,048 was **487/446 at C19, 463/420 at C100, 440/393 at C250 and 423/366 at C500**. The deficit therefore widened from 41 to 57 questions rather than converging. Both headline Full Support reproduction checks matched exactly and neither scale run hit the retrieval-feasibility ceiling. See [`phase_8/8.1_results.md`](phase_8/8.1_results.md).

Phase 9 now carries **BGE-small as the task-validated primary Dense candidate**. Its specification, approved 2026-09-23, excludes Qwen from FullWiki measurement and makes Full Support @2,048 tokens on the 5,405 retrieval-unseen questions the primary endpoint ([`phase_9/9.spec.md`](phase_9/9.spec.md)). The broader Phase 8 question — Entity Hop beside a substantially stronger Dense retriever — remains open because deviation 8.1 did not produce or test such a retriever.

**Phase 9 is measured and closed** (2026-09-24) with **`SCALE_SUPPORTED`**. Over the complete HotpotQA processed Wikipedia (5,233,329 paragraphs), on the 5,405 retrieval-unseen questions, Dense + Entity Hop reaches **3,248 / 5,405** Full Support @2,048 against Dense's **3,006** (+4.48 pp; exact McNemar 374 wins / 132 losses, p = 8.0e-28). Dense + BM25 reaches 3,313, 1.2 pp above the Entity Hop (descriptive, p = 0.024): at this scale BM25 is the slightly stronger complement at the primary budget. The whole FullWiki GLiNER extraction took 6.18 h for an attributable 4.57 USD; the phase billed 12.71 USD. See [`phase_9/9.results.md`](phase_9/9.results.md).

**Phase 10 is measured and closed** (2026-09-24) with **`THREE_WAY_SUPPORTED`**. On 5,000 held-out HotpotQA train questions (level `hard`, seen by BGE-small in fine-tuning) over the same 5.23M-paragraph corpus, Dense + BM25 + Entity Hop (0.5 / 0.3 / 0.2, fitted on the 7,405 validation questions) reaches **3,325 / 5,000** Full Support @2,048 against Dense + BM25's **3,129** (+3.92 pp; exact McNemar 306 wins / 110 losses, p = 1.9e-22), at 1.33× the mean query latency on the laptop. The entity signal is not redundant with BM25. Dense on the train questions reads 2.6-2.9 points above its validation figure, within the 5-point contamination margin. Step 2 (a better use of the entities, on the reserved `test-11`) is the author's decision. See [`phase_10/10.results.md`](phase_10/10.results.md).

**Phase 11 is specified and planned** (2026-09-24; spec amended 2026-09-25 before any measurement, awaiting re-approval). It asks whether a better use of the same entity signal (P1's entities without its hubs, cut by a DF cap; entities GLiNER reads in the question; or both, in a four-component fusion fitted on the 7,405 dev questions over 1,430 configurations) beats P10-C on `test-11`. Before the fit, the new code must reproduce P10-C's 4,801 dev count exactly. The selected configuration must reach 4,838 on dev, or the phase stops with `test-11` unopened. The label comes from P11 against P10-C; P11 against P10-B is reported with its own preregistered test. See [`phase_11/11.spec.md`](phase_11/11.spec.md) and [`phase_11/11.0_entity_use.md`](phase_11/11.0_entity_use.md).

The remaining scheduled work deliberately does **not** try to improve the Entity Hop with canonicalization, relations, multiple seeds or additional hops. Phase 9 asks whether the simple mechanism already discovered survives **corpus scale** with BGE-small as the validated primary Dense candidate. The separate question of Entity Hop beside a substantially stronger Dense retriever remains open.

---

## Phase 1: Corpus, indexing units and dense baseline

Delivers **System A** of §70 and, above all, the measurement harness everything else will be compared against. Without this phase there is no baseline and no way to know whether the rest contributes anything.

- [x] Benchmark subset downloaded and frozen in `data/` (version and hash recorded)
- [x] **Unified pool**: every paragraph of the subset forms a single corpus (**19,366 units**, measured), not the 10 candidates per question. With no space to traverse there is no hypothesis to test
- [x] Indexing units by **complete meaning, never by fixed length**: the article paragraph is the unit and is never split. Stable IDs mapping unit ↔ annotated benchmark sentence
- [x] Swappable embedding backend, with an on-disk cache versioned by (model, indexing unit)
- [x] Reproducible vector index and top-K retrieval (fixed seed)
- [x] Evaluation harness: supporting-fact Recall and Context Precision measured **at a fixed context budget** — same tokens for every system, not same K
- [x] System A (dense) and the **BM25** baseline measured, their numbers recorded as the baseline

## Phase 2: Concept space — dictionary + matrix X

Builds the corpus's own semantic space. It stands on its own even before retrieval uses it: it produces a concept map of the corpus that can be inspected and criticized.

Four spaces were induced (K = 512, 1,024, 2,048, 4,096) and **none of them is chosen here**: the choice of K belongs to Phase 3 and is made against dev recall. The numbers, read from the artifacts rather than recomputed for the report, live in [`phase_2/2.results.md`](phase_2/2.results.md). The security audit closed with no Critical and no High finding; its eight findings and one observation are catalogued in [`docs/security/README.md`](../security/README.md).

- [x] Dictionary induced by **non-negative sparse coding** over the embeddings (`MiniBatchDictionaryLearning` with `positive_code=True` and `positive_dict=False`), fixed seed. The dictionary constraint was dropped on measured evidence, see decision D1 of [`phase_2/2.0_concept_space.md`](phase_2/2.0_concept_space.md): half the energy of the embeddings is negative, so a non-negative dictionary drives the space to one activation per unit — the very regime that disqualified clustering
- [x] Matrix `X = chunks × concepts` as the direct output of that coding: multi-activation with continuous, non-negative weights, **with the score semantics written down**
- [x] Each concept's embedding taken from its own dictionary atom
- [x] Concept labelling via LLM **for interpretability and reporting only**, outside the retrieval critical path
- [x] Dictionary normalization and deduplication
- [x] Structural inspection of the space: activations per chunk, chunks per concept, orphan concepts, dead atoms, co-activation
- [x] **Dictionary quality measured, not assumed**: per-concept semantic coherence against a random-unit null baseline, atom alignment and activation-mass concentration

## Phase 3: Hybrid conceptual retrieval (System B)

Uses the concept space to retrieve and measures it. Answers the first half of the hypothesis: does the interpretable representation retrieve at least as well as the dense one?

- [x] Query → concepts mapping by dot product against the dictionary, no textual intermediary
- [x] Chunk scoring over the active dimensions of `X`
- [x] K, view and query operator chosen against dev recall and frozen before test
- [x] Fusion of dense and conceptual signals into System B
- [x] A **Dense + BM25 control** built through the same fusion machinery
- [x] IDF rarity damping measured
- [x] System B measured against dense and BM25

### What Phase 3 found

The selected concept space was K = 512, `raw`, `projection_full`, with IDF damping.

The decisive result was negative: System B's fitted fusion weight was `w = 1.0`, all weight on dense and none on the concept space. The conceptual signal added no measurable value in the fused retriever.

The Dense + BM25 control, fitted by the same procedure, did improve on dense:

- Dense Full Support @2,048 test: **0.8250**
- Dense + BM25: **0.8643**

The control therefore established that useful complementary signal existed; the concept representation simply was not providing it.

A small complementary conceptual signal remained: conceptual-only retrieval answered five dev questions dense did not, four of them among the questions both dense and BM25 failed. That motivated Phase 4.

---

## Phase 4: Query-aware iterative expansion (System C)

Phase 4 tested the distinctive part of the original proposal: instead of stopping after the first retrieval, use the concept space to navigate from retrieved chunks toward additional context.

- [x] Expansion by diffusion over `X`
- [x] Partial restart on the query
- [x] Stopping criterion and iteration cap
- [x] Per-query traces
- [x] Dense-seeded and conceptual-seeded arms
- [x] Same-budget comparison against the existing baselines

### What Phase 4 found

The result was negative.

System C scored **0.8214 Full Support @2,048 on test**, against:

- Dense: **0.8250**
- Dense + BM25: **0.8643**

The failure mechanism was measured directly: at every usable restart value, the diffusion returned the same 100 seed units in a different order and **never promoted a new paragraph**.

At `restart = 0`, where diffusion could move freely, retrieval collapsed to **0.0117 Full Support**.

The pooled-embedding concept representation therefore failed both as:

- a direct complementary retrieval signal; and
- a substrate for iterative expansion.

### Research line closed — negative result, limited scope

Recorded in [`phase_4/4.1_research_line_closure.md`](phase_4/4.1_research_line_closure.md).

**What it establishes:** a concept space induced by sparse coding of pooled paragraph embeddings adds no measurable retrieval value over dense on this HotpotQA setup.

**What it does not establish:** whether concepts or entities extracted directly from the text can provide useful navigation.

That distinction opened Phase 5.

---

## Phase 5: Text-derived concepts — navigation pilot

Phase 5 extracted **entities and concepts directly from the text** of the complete 19,366-paragraph pool and tested whether those nodes could navigate from Dense's first retrieved paragraph toward evidence Dense had missed.

- [x] Whole-corpus extraction with `claude-sonnet-5`
- [x] Deterministic normalization and node index
- [x] Entity-only, concept-only and entity+concept hops
- [x] Comparison against inherited second-hop baselines
- [x] Pre-declared GO / NAMES ONLY / STOP gate

### What Phase 5 found

The extraction cost **25.98 USD** for 19,366 paragraphs and produced:

- 96,416 entity nodes
- 40,846 concept nodes
- 137,262 total nodes

On the 152-question bridge pilot:

| Second hop | hit@10 per question |
|---|---:|
| Continue dense | 0.3026 |
| BM25 on question | 0.4441 |
| BM25 from `p1` text | 0.4507 |
| pooled-embedding concept hop | 0.0789 |
| **Entity Hop** | **0.6579** |
| text-derived concept hop | 0.0592 |
| entity + concept | 0.4803 |

The gate returned **NAMES ONLY**.

Entities were therefore the first representation in the project to show a strong second-hop navigation signal.

Concepts were not only weak by themselves; adding them to the entity hop degraded the shallow ranking.

The mechanism also showed a clear structural ceiling:

- 118 of 158 missing paragraphs shared an entity node with `p1`
- Entity Hop found 116 of those 118 by depth 100

Once a bridge was expressible by a shared entity, the simple node hop found almost all of it.

---

## Phase 6: Dense + Entity Navigation — end-to-end comparison

Phase 6 asked the narrow end-to-end question opened by Phase 5:

> **Can an entity-navigation second hop replace the BM25 component of the existing Dense + BM25 system without reducing retrieval quality?**

The comparison was:

```text
A: Dense + BM25
B: Dense + Entity Hop
```

Initial scope deliberately excluded:

- concepts;
- canonicalization;
- relation or attribute nodes;
- multiple seeds;
- multiple hops;
- query-aware entity expansion;
- Dense + BM25 + Entity.

The Phase 5 entity mechanism was used unchanged.

### What Phase 6 found

The Entity system selected on dev was:

```text
Dense weight       = 0.7
Entity Hop weight  = 0.3
```

At Full Support @2,048 on the 1,400 test questions:

| System | Full Support | Successes |
|---|---:|---:|
| Dense | 0.8250 | 1,155 |
| Dense + BM25 | 0.8643 | 1,210 |
| **Dense + Entity Hop** | **0.8900** | **1,246** |

The preregistered Phase 6 decision passed all three tests:

| Test | Result |
|---|---|
| B vs Dense | 103 wins / 12 losses, p = 1.676e-19 |
| A vs Dense | 85 wins / 30 losses, p = 1.430e-07 |
| B non-inferior to A | lower bound +0.011135 > −0.019643 |

**State: `ENTITY_REPLACEMENT_SUPPORTED`.**

This establishes, under the Phase 6 protocol, that Entity Hop can replace the BM25 component without losing retrieval quality and that the resulting system improves on Dense.

The observed aggregate improvement is:

- **+6.50 percentage points Full Support over Dense**
- **+2.57 percentage points over Dense + BM25**, reported descriptively rather than as the preregistered superiority hypothesis

Dense failed Full Support on 245 questions; Dense + Entity Hop fails on 154. The system therefore removes approximately **37% of Dense's Full Support failures** on this test population.

The mechanism diagnosis also supports that the gain comes from actual entity expansion:

- all 103 B successes that Dense misses contain at least one paragraph introduced by Entity Hop;
- 67 depend entirely on entity-introduced units rather than merely reranking Dense;
- 85 of B's 96 wins over Dense + BM25 contain entity-introduced units.

### Cost

Measured retrieval latency in the Phase 6 harness:

- Dense: 0.891 ms/query
- Dense + BM25: 1.508 ms/query
- Dense + Entity Hop: 10.447 ms/query

These values are **milliseconds**, not seconds.

The online latency remains small in absolute terms. The much larger scaling concern is offline extraction: Phase 5 used a generative LLM over every paragraph, which cannot simply be extrapolated to a five-million-paragraph FullWiki corpus without first replacing or cost-reducing that preprocessing step.

### What Phase 6 does not establish

It does not establish:

- that the effect survives a substantially stronger dense retriever;
- that it scales to HotpotQA FullWiki;
- that it generalizes to other multi-hop datasets;
- that canonicalization would help;
- that explicit relations would help;
- that more seeds or more hops would help;
- the broader Type C satellite-context claim;
- the wider concept-embeddings architecture.

Those questions are deliberately separated.

---

## Phase 7: Cheap entity extraction

Phase 7 asked the economic question that stood between Phase 6 and any FullWiki build:

> **Can the expensive Phase 5 LLM entity extraction be replaced by a cheap or local extractor without materially losing the Entity Hop retrieval gain?**

Phase 5 processed 19,366 paragraphs for 25.98 USD. HotpotQA FullWiki is on the order of millions. Naively multiplying a frontier-LLM extraction was never an acceptable scaling strategy.

- [x] Two local candidates extracted over the frozen 19,366-paragraph pool: **GLiNER** (`urchade/gliner_medium-v2.1`, revision `40ec419335d09393…`) and **spaCy** (`en_core_web_sm` 3.8.0)
- [x] The same normalization, the same entity incidence index, the same P1-only one-hop Entity Hop, the same Dense retriever — only the reader of the paragraphs changed
- [x] Both candidates measured on the 600 dev questions, fusion weight refitted on dev
- [x] A retention bar and a ranking rule frozen in `config.py` **before any candidate ran**
- [x] The held-out split opened once, for the selected extractor only
- [x] Throughput, failures, index size and cost measured; FullWiki figures projected and labelled as projections
- [x] Targeted security review of the three declared surfaces: no Critical, no High

### What Phase 7 found

**It can.** The recorded selection is `gliner` (`data/phase7/selection.json` → `selected`).

At Full Support @2,048 on the 1,400 test questions:

| System | Full Support | Successes |
|---|---:|---:|
| Dense | 0.8250 | 1,155 |
| Dense + BM25 | 0.8643 | 1,210 |
| Dense + Entity Hop (Claude) | 0.8900 | 1,246 |
| **Dense + Entity Hop (GLiNER)** | **0.8793** | **1,231** |

GLiNER recovers **76 of the 91 questions** Claude's extraction added over Dense — **83.5%** of the held-out gain. It beats the Dense + BM25 control by 21 questions and sits 15 below Claude.

The cost collapse is the point of the phase, and it is measured, not estimated:

| Extraction of 19,366 paragraphs | Value |
|---|---|
| GLiNER wall clock | 119.4 s at **162.178 paragraphs/s**, 1× RTX 4090 (rented, Linux x86_64) |
| GLiNER attributable compute | **0.03 USD** |
| Claude (Phase 5, measured) | **25.9848 USD** |

Projected linearly from that measured throughput, 5M paragraphs would take **8.56 h and 6.34 USD**. Those two numbers are **projections, not measurements**, and carry no authorization to build FullWiki; Phase 9 measures its own.

### The secondary finding

**spaCy measured better than GLiNER on dev** — 525 / 600 (0.8750) against 518 / 600 (0.8633), retaining 95.0% of Claude's dev gain against GLiNER's 77.5% — and was not selected. Its held-out split was **never opened**, so no spaCy test figure exists.

Three reasons, recorded so the choice is not relitigated:

1. The preregistered rule ranks candidates that clear both bars on projected FullWiki cost and time. GLiNER's 6.34 USD / 8.56 h beat spaCy's 11.08 USD / 14.97 h.
2. Overriding that rule *after observing dev* would invalidate the only thing that makes the held-out figure mean anything.
3. spaCy buys its quality with a much denser, less scalable representation: 11 labels against 5, a median of 493 hop candidates per question against 40, a largest hub of 4,085 paragraphs against 1,186.

On dev, GLiNER landed exactly on the Dense + BM25 line (518 = 518). On the held-out split it beat that line by 21 questions. **Dev was the more pessimistic reading**, which is the safe direction for a selection rule to err in.

spaCy is **not discarded**. It remains a candidate for a later scale-and-representation experiment, listed under deferred work.

### What Phase 7 does not establish

It does not establish that GLiNER reads entities *correctly* — NER quality was an explicit anti-goal — that its 21-question margin over BM25 is stable beyond one test run (no interval, no test statistic, by design), or that either extractor's representation survives at five million paragraphs.

Full result: [`phase_7/7.results.md`](phase_7/7.results.md).

---

# Phase 8: Strong Dense + Entity Hop

> **STOPPED AT THE DEV GATE** (2026-09-19). `Qwen/Qwen3-Embedding-0.6B` scored 446 / 600 dev against
> the inherited 487 / 600, below the 488 required, so the held-out split was never opened and the
> question below remains **unanswered**. Deviation 8.1 is opened for corpus-scale sensitivity. The
> description that follows is the phase as specified; [`phase_8/8.results.md`](phase_8/8.results.md)
> is what it measured.

## Question

> **Does Entity Hop still add useful complementary signal when the dense retriever itself is substantially stronger than `BAAI/bge-small-en-v1.5`?**

Phase 6 established a +6.5-point Full Support improvement over a relatively small dense encoder. Before claiming a generally useful structural signal, the experiment must determine whether that gain survives a strong modern dense retriever.

## Scope

Stay on the **same 19,366-paragraph corpus**.

Choose **one strong, modern, open dense retrieval model** based on current retrieval literature, model quality and practical availability.

Do not turn the phase into a model leaderboard.

Recompute the dense corpus embeddings and compare only:

```text
Strong Dense
Strong Dense + BM25
Strong Dense + Entity Hop
```

The Entity Hop uses **GLiNER**, the extractor Phase 7 selected (`urchade/gliner_medium-v2.1`, revision `40ec419335d09393…`), and its existing entity index.

> **Phase 8 inherits GLiNER and changes only the Dense retriever.** Changing the Dense model and the extractor simultaneously would make any improvement or degradation unattributable to either.

That is the answer to "why not spaCy, it scored better on dev?". The extractor question is settled for Phase 8; re-opening it belongs to the deferred scale-and-representation experiment.

P1 remains the only seed and the system remains one-hop.

The fusion weight may be fitted on dev because score distributions will change under the new dense model. No other entity hyperparameter is introduced.

## Main questions

Measure:

1. how much Strong Dense improves over the current BGE-small baseline;
2. whether BM25 still adds complementary signal;
3. whether Entity Hop still adds complementary signal;
4. whether Entity Hop still compares favourably with the lexical complement;
5. how much of the original +6.5-point gain survives.

## Important interpretation

Three outcomes are all informative:

- **Entity Hop remains strongly positive:** evidence that structural relatedness supplies information even a much stronger semantic retriever misses.
- **The effect becomes smaller but remains positive:** Entity Hop partly compensates for Dense weakness but still contains complementary signal.
- **The effect disappears:** the Phase 6 gain was largely a property of the smaller encoder rather than a general retrieval principle.

Do not add canonicalization, explicit relations or extra hops to rescue a weak result.

## Outcome

Phase 8 selects the Dense configuration that Phase 9 will carry into FullWiki.

**It did not.** The gate fired before any held-out reading existed, so no Dense configuration was
selected by Phase 8 itself.

**Deviation 8.1 subsequently resolved the scale-sensitive Dense choice without reopening Phase 8.**
Its `STABLE_RANKING` result makes BGE-small the task-validated primary Dense candidate for Phase 9;
the Phase 9 specification decides whether Qwen is retained as an external-comparability control.

---

# Phase 9: HotpotQA FullWiki — literature-comparable scale

## Question

> **Does the simple Dense + Entity Hop architecture remain useful when retrieval is performed over the standard multi-million-paragraph HotpotQA FullWiki search space?**

This is the first phase designed primarily to make the project's numbers directly comparable with published multi-hop retrieval work.

## Preconditions

Phase 9 opens only after:

- Phase 7 identifies an affordable entity extraction path — **done: GLiNER**; and
- deviation 8.1 resolves the Dense candidate after Phase 8's stop — **done: `STABLE_RANKING`,
  BGE-small primary; the Phase 9 spec excludes Qwen**.

## Scope

Move from the 19,366-paragraph experimental pool to the standard HotpotQA FullWiki corpus, on the order of **five million paragraphs**.

Carry forward the mechanism without adding new retrieval ideas:

```text
question
→ Primary Dense (BGE-small)
→ P1
→ one raw Entity Hop
→ Dense + Entity fusion
```

Keep:

- one seed: P1;
- one hop;
- GLiNER, the cheap extractor selected in Phase 7;
- BGE-small, the primary Dense candidate validated by deviation 8.1;
- no canonicalization;
- no explicit relations;
- no query-aware entity reranking;
- no GraphRAG machinery.

## Systems

At minimum measure:

```text
Primary Dense
Primary Dense + BM25
Primary Dense + Entity Hop
```

The purpose is not to reimplement every published retriever.

Instead, use the **same corpus and standard evaluation setting** used by the literature so that our numbers can be placed beside published Dense, MDR, HippoRAG, KG²RAG, HGRAG, LinearRAG, SAG and related results with the appropriate caveat that implementations and models still differ.

## Metrics

Report both:

1. the project's existing retrieval metrics where feasible; and
2. the standard FullWiki retrieval metrics used in the most relevant published work, so that numerical comparison is meaningful.

The approved spec fixes them: Full Support @2,048 tokens on the 5,405 retrieval-unseen questions is the primary endpoint; Full Support and gold recall at @2/5/10/20 are reported descriptively for literature comparison, with a protocol ledger that keeps pooled-corpus results out of FullWiki comparisons.

## Scaling measurements

FullWiki is also an engineering-scale experiment.

Record:

- total extraction time;
- extraction throughput;
- extraction cost;
- entity index size;
- dense index size;
- peak memory;
- query latency;
- Entity Hop candidate counts;
- effect of entity document-frequency distribution at FullWiki scale.

The phase should determine whether the simple entity incidence representation remains operationally attractive relative to richer graph construction.

Phase 7's **8.56 h / 6.34 USD for 5M paragraphs is a projection** extrapolated linearly from a 19,366-paragraph measurement. Phase 9 is where it is confirmed or refuted by measurement.

## Outcome

Phase 9 should tell us whether the Phase 6 mechanism:

- survives corpus scale;
- produces literature-comparable multi-hop retrieval numbers; and
- deserves broader benchmark validation.

It should **not** automatically open more complex graph machinery.

---

## Work deliberately deferred beyond Phase 9

The following are valid research directions but are **not current phases**:

- entity canonicalization / alias resolution;
- query-aware Entity Hop;
- explicit relation or attribute nodes;
- subject–relation–object triples;
- multiple dense seeds;
- multiple entity hops;
- Dense + BM25 + Entity;
- PageRank or other global graph propagation;
- broader GraphRAG structures;
- hierarchical/book-like corpora;
- Type C thematic/satellite-context evaluation;
- direct reimplementations of every competing paper;
- **re-opening the Phase 7 extractor choice at scale**: does spaCy's dev advantage over GLiNER survive at FullWiki scale, or does it collapse under candidate explosion and hub growth? A median of 493 candidates per hop on 19,366 paragraphs is a different proposition on five million. Recorded so the question is not lost; it is **not** promoted into Phase 8 or Phase 9.

Their ordering is discussed in [`research_roadmap.md`](research_roadmap.md).

---

## Founding decisions

Settled during the discovery conversation. Later evidence may supersede a practical implementation choice, but historical decisions remain recorded rather than rewritten.

| Decision | Choice | Rationale |
|---|---|---|
| Nature of the deliverable | Validation experiment | The goal is to prove or refute research hypotheses, not to build a product |
| Corpus | Public multi-hop benchmark with ground truth | It ships annotated supporting facts and enables comparison with literature |
| Working language | English for everything this project produces | Source documents in `docs/refs/` stay in Spanish as authored |
| Indexing unit | The smallest complete unit of meaning, never fixed-length windows | A fixed window can mix unrelated ideas; in HotpotQA the paragraph is the natural unit |
| Concept engine | Non-negative sparse coding over embeddings | Chosen to create multi-activation concept representations rather than one-hot clustering |
| Concept embedding | Dictionary atom itself | It already lives in the query embedding space |
| Original expansion mechanism | Diffusion over `X` with partial restart | Deterministic way to isolate whether the induced concept representation contributed retrieval value |
| Initial compute | Local ARM64 CPU, small embedding model | Zero-cost experimental starting point; **no longer a scientific constraint for Phase 8+, where stronger retrieval models are explicitly part of the research question** |
| Baselines | BM25 in addition to dense | A useful retrieval claim must survive comparison with a cheap lexical complement |
| Measurement protocol | Fixed context budget where applicable | Keeps downstream context opportunity comparable |
| Retrieval evidence | Objective annotated supporting facts | Avoid an LLM judge on the retrieval critical path |
| Complexity principle from Phase 7 | Minimum implementation needed to answer one research question | The project must remain research rather than grow into an experimental framework for its own sake |

### Platform constraint

Development has historically run on **Windows on ARM (`win_arm64`)**.

This remains a local development constraint, not a scientific requirement.

Existing environment notes:

- `torch` is sourced from the official PyTorch index.
- Some common packages have historically lacked `win_arm64` wheels.
- HuggingFace `datasets` was avoided because of `pyarrow`.
- Available packages have included `numpy`, `scipy`, `scikit-learn`, `sentence-transformers`, `transformers`, `tokenizers`, `onnxruntime`, `faiss-cpu`, `model2vec`.
- Python is pinned to 3.12.

Phases 7–9 may use another machine, cloud GPU or x86 environment when doing so materially reduces preprocessing or embedding cost. The experimental question takes precedence over preserving the original hardware limitation.

**Phase 7 exercised that allowance and it was decisive.** Both extraction passes ran on a rented Linux x86_64 machine with one RTX 4090; `pyproject.toml` now resolves `torch` from the `pytorch-cpu` index on Windows and from `pytorch-cu126` on Linux x86_64, and GLiNER moves itself to CUDA when a device is present. spaCy genuinely cannot be installed on the ARM64 laptop — `blis` publishes no `win_arm64` wheel — and ran on the rented machine. Phases 8 and 9 will use the same path.

---

## Declared limitations

The positive entity result is currently specific to a HotpotQA bridge setting.

Phase 5 selected questions where Dense missed supporting evidence and showed that the missing paragraph was often connected to `p1` through a shared entity. Phase 6 then demonstrated that the same mechanism improves end-to-end retrieval on the frozen 19,366-paragraph corpus.

This still does not establish the broader Type C claim from the original proposal, where useful satellite knowledge — nutrition, pacing, recovery, related mechanisms, etc. — may not be linked through a shared named entity.

The current Entity Hop also has a reachability ceiling:

- 118 of 158 Phase 5 missing paragraphs share an entity with `p1`;
- 116 of those 118 are found by depth 100.

The current system is deliberately simple:

- raw extracted entity forms;
- no canonicalization;
- P1 only;
- one hop;
- no explicit relation types;
- no query-aware filtering during the entity hop.

Its current success therefore says something narrow but useful about the value of **shared-entity relatedness** as a complement to semantic similarity.

Phase 7 established that this representation can be constructed cheaply — with GLiNER, on the 19,366-paragraph corpus, keeping 83.5% of the gain. It did **not** establish that the extraction is faithful (NER quality was never measured) or that the representation behaves the same at scale.

Phase 8 must establish whether the effect survives a stronger Dense retriever.

Phase 9 must establish whether it survives a multi-million-paragraph search space.

Until those experiments exist, the project should not claim state-of-the-art performance or general multi-hop superiority.

---

## Project success criterion

The project succeeds by producing **defensible empirical answers to progressively sharper research questions**, including negative answers.

The original pooled-concept research line is complete with a bounded negative result.

The text-derived entity line has produced its first positive result:

- Phase 5 established a strong entity-navigation signal.
- Phase 6 established that the signal works end to end and satisfies the preregistered BM25-replacement criterion.
- Phase 7 established that the representation can be built with a local model for roughly one thousandth of the LLM extraction cost, keeping 83.5% of the held-out gain.

The immediate next success criteria are now sequential:

1. ~~**Phase 7:** make the entity representation economically scalable without destroying its retrieval value.~~ **Met**, with GLiNER.
2. **Phase 8:** show whether the structural signal remains useful beside a modern strong Dense retriever.
3. **Phase 9:** test that architecture on HotpotQA FullWiki and obtain numbers that can be compared meaningfully with the published multi-hop retrieval literature.

Only after those questions are answered should the project decide whether additional structure — canonicalization, query awareness, explicit relations, more hops or GraphRAG-like machinery — is justified.

---

## Corrective fixes

- **[fix-1](fixes/fix-1_audit_phase_1_findings.md)** — the ten findings of the Phase 1 security audit, all of them integrity controls that existed in the code and were never exercised by the pipeline. Closed with 15 regression tests; the full evaluation was re-run and all 60 recorded metrics reproduce exactly, so the Phase 1 numbers are unaffected.