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
| 7 | Cheap entity extraction | Pending | **Planned** |
| 8 | Strong Dense + Entity Hop | Pending | **Planned after Phase 7** |
| 9 | HotpotQA FullWiki at literature-comparable scale | Pending | **Planned after Phases 7–8** |

**The first research line is closed** (2026-09-15): concepts induced from pooled embeddings, tested through Phases 2-4, gave a negative and bounded result. The original Phase 5 — a comparative evaluation of that method — was not run; the plan was renumbered so Phase 5 tests the representation the proposal actually described. See [`phase_4/4.1_research_line_closure.md`](phase_4/4.1_research_line_closure.md).

**Phase 5 is measured and closed at the pilot gate:** text-derived entities produce a strong positive second-hop signal on the HotpotQA bridge diagnostic, while text-derived concepts do not. The gate returned **NAMES ONLY**.

**Phase 6 is measured and closed** (2026-09-17): under one protocol frozen and committed before the test split was opened, and read exactly once, `data/replacement/decision.json` records the state **`ENTITY_REPLACEMENT_SUPPORTED`** with no anomaly and no open question. The Entity Hop replaces the BM25 component of the dense hybrid without loss of retrieval quality, and the resulting system also improves on dense alone. See [`phase_6/6.results.md`](phase_6/6.results.md).

The next three phases deliberately do **not** try to improve the Entity Hop with canonicalization, relations, multiple seeds or additional hops. They ask whether the simple mechanism already discovered can be made cheap, survive a substantially stronger dense retriever, and scale to the standard FullWiki setting used by the literature.

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

# Phase 7: Cheap Entity Extraction

## Question

> **Can the expensive Phase 5 LLM entity extraction be replaced by a cheap or local extractor without materially losing the Entity Hop retrieval gain?**

This is now the immediate bottleneck.

Phase 5 processed only 19,366 paragraphs and cost approximately 25.98 USD. HotpotQA FullWiki contains on the order of millions of paragraphs. Before scaling, the project must determine whether the entity representation can be built without passing every paragraph through a frontier generative LLM.

## Scope

Run candidate entity extractors over the **existing 19,366-paragraph corpus** so that every candidate can be evaluated against the already-established Phase 6 result.

Initial candidates should stay deliberately small in number, for example:

- the existing Claude extraction as the reference;
- **GLiNER** or a comparable local zero-shot/open NER model;
- a conventional local NER baseline such as **spaCy** or an equivalent lightweight model;
- optionally, if the FullWiki source exposes usable Wikipedia links/anchors, a deterministic Wikipedia-derived entity source.

Do not expand this into a general NER benchmark.

The metric that matters is **retrieval**, not generic NER F1.

## Main comparison

For each viable extractor:

```text
extract entities
→ build the same entity incidence index
→ run the same P1, one-hop Entity Hop
→ combine with the current Dense retriever
→ measure retrieval
```

The existing Claude-based result remains the reference:

```text
Dense + Claude-entities Entity Hop
Full Support @2,048 test = 0.8900
```

## Measure

At minimum record:

- Full Support / Gold Recall / Recall@10 with the same current benchmark;
- Entity Hop hit behaviour;
- number of entity forms and incidence edges;
- extraction failures;
- paragraphs per second;
- wall-clock preprocessing time;
- CPU/GPU requirements;
- estimated processing time for ~5M paragraphs;
- estimated monetary cost for ~5M paragraphs;
- resulting index size.

Selection should be pragmatic:

> choose the cheapest extractor that preserves enough of the retrieval signal to make FullWiki scaling scientifically worthwhile.

No elaborate statistical decision framework is required.

## Constraints

Keep unchanged:

- the current 19,366-paragraph corpus;
- P1 as the only seed;
- one entity hop;
- no canonicalization;
- no relations;
- no query-aware filtering;
- no multiple-hop traversal.

## Outcome

Phase 7 should answer only:

1. Can we build the entity index cheaply enough for FullWiki?
2. How much of the Phase 6 gain survives?
3. Which extractor should Phase 8 and Phase 9 inherit?

---

# Phase 8: Strong Dense + Entity Hop

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

The Entity Hop uses the extractor selected in Phase 7.

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

---

# Phase 9: HotpotQA FullWiki — literature-comparable scale

## Question

> **Does the simple Dense + Entity Hop architecture remain useful when retrieval is performed over the standard multi-million-paragraph HotpotQA FullWiki search space?**

This is the first phase designed primarily to make the project's numbers directly comparable with published multi-hop retrieval work.

## Preconditions

Phase 9 opens only after:

- Phase 7 identifies an affordable entity extraction path; and
- Phase 8 selects the strong dense retriever.

## Scope

Move from the 19,366-paragraph experimental pool to the standard HotpotQA FullWiki corpus, on the order of **five million paragraphs**.

Carry forward the mechanism without adding new retrieval ideas:

```text
question
→ Strong Dense
→ P1
→ one raw Entity Hop
→ Dense + Entity fusion
```

Keep:

- one seed: P1;
- one hop;
- the cheap extractor selected in Phase 7;
- the Strong Dense model selected in Phase 8;
- no canonicalization;
- no explicit relations;
- no query-aware entity reranking;
- no GraphRAG machinery.

## Systems

At minimum measure:

```text
Strong Dense
Strong Dense + BM25
Strong Dense + Entity Hop
```

The purpose is not to reimplement every published retriever.

Instead, use the **same corpus and standard evaluation setting** used by the literature so that our numbers can be placed beside published Dense, MDR, HippoRAG, KG²RAG, HGRAG, LinearRAG, SAG and related results with the appropriate caveat that implementations and models still differ.

## Metrics

Report both:

1. the project's existing retrieval metrics where feasible; and
2. the standard FullWiki retrieval metrics used in the most relevant published work, so that numerical comparison is meaningful.

The exact standard metric set should be fixed in the Phase 9 spec after checking the principal comparison papers, rather than guessed now.

Likely metrics include passage/supporting-fact recall at fixed depths such as `Recall@2/5/10/20`.

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
- direct reimplementations of every competing paper.

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

Phase 7 must establish whether this representation can be constructed cheaply.

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

The immediate next success criteria are now sequential:

1. **Phase 7:** make the entity representation economically scalable without destroying its retrieval value.
2. **Phase 8:** show whether the structural signal remains useful beside a modern strong Dense retriever.
3. **Phase 9:** test that architecture on HotpotQA FullWiki and obtain numbers that can be compared meaningfully with the published multi-hop retrieval literature.

Only after those questions are answered should the project decide whether additional structure — canonicalization, query awareness, explicit relations, more hops or GraphRAG-like machinery — is justified.

---

## Corrective fixes

- **[fix-1](fixes/fix-1_audit_phase_1_findings.md)** — the ten findings of the Phase 1 security audit, all of them integrity controls that existed in the code and were never exercised by the pipeline. Closed with 15 regression tests; the full evaluation was re-run and all 60 recorded metrics reproduce exactly, so the Phase 1 numbers are unaffected.