# Master Plan: concept-embeddings-RAG

## What is this product?

A research experiment that puts to the test the hypothesis in `docs/refs/descripcion-proyecto.md` §69: that a document corpus can be represented in a sparse semantic space whose dimensions are **concepts discovered from the corpus itself**, and that expanding retrieval iteratively through those concepts recovers context a conventional dense RAG never reaches.

No application is being built. **The deliverable is measured evidence**: comparable systems (BM25, dense, conceptual hybrid, iterative expansion) evaluated on a multi-hop benchmark with annotated ground truth, and an explicit verdict on the hypothesis — including the negative verdict if the numbers say so.

> Source documents in `docs/refs/` are in Spanish and stay that way: they are the author's material. Everything this project produces — plans, specs, code, reports — is written in English.

## Documentation convention

Every modification, improvement or fix follows this protocol before touching code:

```
docs/plans/
├── 0_master_plan.md           # This file - global view
├── phase_1/
│   ├── 1.spec.md              # Functional spec (WHAT + WHY) - /4-especificar
│   ├── 1.0_phase_name.md      # Implementation plan (HOW) - /5-planear
│   ├── 1.tasks.md             # (Optional) Atomic tasks - complex phases only
│   └── 1.Y_name.md            # Deviation/problem, sequential (Y = 1, 2, 3...)
├── phase_2/
│   └── ...
└── fixes/
    └── fix-N_name.md          # One-off bug fix (globally sequential)
```

- **`X.spec.md`** → Functional spec. Created with `/4-especificar` BEFORE planning.
- **`X.0`** → Implementation plan. Created with `/5-planear` AFTER the spec.
- **`X.tasks.md`** → Atomic tasks. Optional, only if the phase has >10 steps.
- **`X.Y`** → Deviation, unexpected problem or off-plan adjustment (Y sequential).

**Per-phase workflow:** `/4-especificar` → `/5-planear` → `/6-implementar` → `/7-verificar` → `/8-auditar` → `/9-documentar`

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
| 7 | Entity canonicalization / resolution | Pending | **Conditional** — only if Phase 6 justifies it |
| 8 | Conditional extensions and combinations | Pending | **Not opened** |

**The first research line is closed** (2026-09-15): concepts induced from pooled embeddings, tested
through Phases 2-4, gave a negative and bounded result. The original Phase 5 — a comparative
evaluation of that method — was not run; the plan was renumbered so Phase 5 tests the representation
the proposal actually described. See [`phase_4/4.1_research_line_closure.md`](phase_4/4.1_research_line_closure.md).

**Phase 5 is now measured and closed at the pilot gate:** text-derived entities produce a strong
positive second-hop signal on the HotpotQA bridge diagnostic, while text-derived concepts do not.
The gate returned **NAMES ONLY**. This does not establish an end-to-end improvement over dense+BM25;
it opens a new research line focused first on whether entity navigation can replace the BM25 component.
Canonicalization is a separate conditional line, not part of that first comparison.

**Phase 6 is now measured and closed** (2026-09-17): under one protocol frozen and committed before
the test split was opened, and read exactly once, `data/replacement/decision.json` records the state
**`ENTITY_REPLACEMENT_SUPPORTED`** with no anomaly and no open question. The Entity Hop replaces the
BM25 component of the dense hybrid **without loss of retrieval quality**, and the resulting system
also improves on dense alone. See [`phase_6/6.results.md`](phase_6/6.results.md).

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

- [x] Dictionary induced by **non-negative sparse coding** over the embeddings (`MiniBatchDictionaryLearning` with `positive_code=True` and `positive_dict=False`), fixed seed. The dictionary constraint was dropped on measured evidence, see decision D1 of [`phase_2/2.0_concept_space.md`](phase_2/2.0_concept_space.md): half the energy of the embeddings is negative, so a non-negative dictionary drives the space to one activation per unit - the very regime that disqualified clustering
- [x] Matrix `X = chunks × concepts` as the direct output of that coding: multi-activation with continuous, non-negative weights, **with the score semantics written down** (§3: what exactly 0.73 means)
- [x] Each concept's embedding taken from its own dictionary atom: it already lives in the query's space, with no detour through a generated name
- [x] Concept labelling via LLM **for interpretability and reporting only**, outside the retrieval critical path
- [x] Dictionary normalization and deduplication (§47), with an explicit, auditable merge criterion
- [x] Structural inspection of the space: activations per chunk, chunks per concept, orphan
      concepts, dead atoms, co-activation
- [x] **Dictionary quality measured, not assumed**: per-concept semantic coherence over the units
      that activate it most, read against a random-unit null baseline measured on this same pool -
      the absolute cosine means nothing in an anisotropic embedding space - plus the alignment of a
      concept with its own evidence and the concentration of its activation mass. Concepts are
      ranked by it so the report reads the worst ones rather than the average. **Diagnostic only:
      no concept is pruned in this phase**

## Phase 3: Hybrid conceptual retrieval (System B)

Uses the concept space to retrieve, and measures it. Answers the first half of the hypothesis: does the interpretable representation retrieve at least as well as the dense one? Specified in [`phase_3/3.spec.md`](phase_3/3.spec.md) and planned in [`phase_3/3.0_hybrid_conceptual_retrieval.md`](phase_3/3.0_hybrid_conceptual_retrieval.md), which declares every parameter of the phase before a single number is measured.

- [x] Query → concepts mapping by dot product against the dictionary (§6-§7, §50), no textual intermediaries
- [x] Chunk scoring over the active dimensions of X, with the invariant that a unit whose row is all
      zeros is unreachable through this system and is not silently rescued
- [x] **K, view and query operator chosen against dev recall** over the four Phase 2 spaces, and
      frozen in a versioned selection artifact before the test split is read even once. This is the
      choice Phase 2 deliberately left open, and Phases 4 and 5 inherit it
- [x] Fusion of dense and conceptual signals into System B, with weights fitted on dev rather than
      eyeballed, and a parameter-free reference scheme measured beside them
- [x] **A dense + BM25 control** built by the same fusion code and fitted the same way, so that a
      System B win can be attributed to the concept space rather than to hybridization as such
- [x] Hub damping by a rarity weight declared in advance, measured as a variant against the
      undamped system. Concept **pruning** stays out of this phase and remains a declared ablation
- [x] System B measured in the same harness and against both baselines (dense and BM25)

### What Phase 4 inherits, and the verdict it inherits it with

Recorded here because Phase 4 starts from these three and cannot make them for itself. The numbers and
their artifacts are in [`phase_3/3.results.md`](phase_3/3.results.md).

| Inherited | Value |
|---|---|
| **K** | **512**, dictionary `d84c327aa8af0cdc` (chosen on dev recall; the structural tie-break kept it over K = 2,048 when the margin proved smaller than 600 questions can resolve) |
| **View of `X`** | **`raw`** (wins at K = 512; `row_normalized` wins at the other three, so this belongs to the space and not to the method) |
| **Damping** | **`idf`**, the rarity weight declared in advance (+0.043 on the selected space, positive on three of the four) |
| Query operator | `projection_full` - the unclipped projection, which wins in all eight (K, view) cells |

**The verdict is negative and attributable.** System B's fusion weight was fitted at `w = 1.0`: all
the weight on dense, none on the concept space, so System B *is* dense to four decimals at every
budget on both splits. The dense+BM25 control, built by the same code and fitted by the same
procedure, gains +3.9 points of Full Support on test - so the headroom was real and a cheap lexical
signal took it while the concept space did not.

**One result opens the door Phase 4 walks through.** Conceptual-only retrieval answers 5 dev questions
dense does not, 4 of them among the 72 that defeat both baselines. The complementary signal exists and
score-level fusion cannot reach it, which is precisely the gap diffusion over `X` is meant to close.

Read through the labels, those five have a shape: **every one is a conjunction of two topics** - a
Florida place *and* baseball, an Irish biography *and* combat sports, a magazine *and* politics. A
dense query is one point and must land near one paragraph; a concept vector can ask for the paragraph
that is about both subjects at once, which is the shape of a bridge. Five questions are a hypothesis,
not a result, but it points Phase 4 at the conjunctions rather than at retrieval quality in general.

## Phase 4: Query-aware iterative expansion (System C)

The leap of §8: stop following the question and start navigating the corpus. It is the distinctive part of the proposal and the one most likely to fail. Specified in [`phase_4/4.spec.md`](phase_4/4.spec.md) and planned in [`phase_4/4.0_query_aware_iterative_expansion.md`](phase_4/4.0_query_aware_iterative_expansion.md), which declares the operator, the stopping rule and both parameter grids before a single number is measured.

**The bar this phase is judged against is the dense+BM25 control** at 0.8643 Full Support on test @2,048, not dense at 0.8250: a cheap lexical hybrid already reaches that without any of this machinery. Beating dense and not the control is a result, to be reported as exactly that.

- [x] Expansion by **diffusion over X**: initial activation from the query, propagation chunk → concept → chunk through `X` and `Xᵀ`, with partial restart on the question's concepts at every round
- [x] The query-aware condition (§52-53) is guaranteed by that restart: expansion cannot drift toward the corpus's global co-occurrence — and `restart = 0.0` is measured once on dev so the claim rests on a number. **It collapses to 0.0117 Full Support**: the failure mode is now a measurement
- [x] Stopping criterion on newly contributed mass, with an explicit budget of iterations and tokens (§54). At the frozen cell the threshold fires and the cap never does; at the three lower restarts every cell ran to the cap, reported as the finding HU-4 asks for
- [x] Per-query trace: which concept entered at which iteration and which chunk it brought along — 80 dev questions under the sampling rule fixed before the run
- [x] **Two seed arms measured with identical machinery** — dense-seeded (System C proper) and conceptual-seeded (the isolating variant) — so that a gain can be told apart from "the dense retriever already brought almost everything". The configuration is chosen on the dense arm and applied unchanged to the other
- [x] System C measured against four rivals at four budgets on both splits, with its cost, on a configuration frozen before test was read
- [ ] ~~*(Optional)* Variant with an LLM inside the loop~~ — **declared out of scope by the spec**, and not pursued after the closure: the deterministic method it would have been compared against failed on its own, and the founding decision that keeps an LLM out of the retrieval loop stands

### What Phase 4 leaves, and the verdict it leaves it with

The numbers and their artifacts are in [`phase_4/4.results.md`](phase_4/4.results.md).

| Inherited | Value |
|---|---|
| **System C's configuration** | `restart = 0.8`, normalization `symmetric`, `stop_threshold = 1e-3`, `max_iterations = 5`, `seed_top_k = 100`; artifact `data/expansion/expansion.json`, digest `421ceee84e04914e` |
| **Phase 3 decisions** | unchanged and not re-opened: K = 512, `d84c327aa8af0cdc`, view `raw`, damping `idf`, `projection_full` |
| **The bar** | the dense+BM25 control at 0.8643 Full Support on test @2,048 |

**The verdict is negative, and the mechanism is measured rather than inferred.** System C scores
0.8214 Full Support on test @2,048 against dense's 0.8250 and the control's 0.8643. It costs 4.3x
dense per query for it.

**Why, in one line: at every restart in the declared grid the walk returns the seed's own 100 units,
reordered, and never promotes a new one.** The diffusion reaches all 19,366 units on every question,
so each non-seed unit receives about a hundred-thousandth of the mass while the restart holds every
seed unit above 0.0076. System C is therefore a re-ranker of dense's top-100: it gains 5 dev
questions and loses 6, and recovers 3 of the 72 that defeat both baselines against System B's 0 and
the control's 6. One of those three is recovered by nothing else in this project.

**The arithmetic also says where the grid could not look.** A non-seed unit could only out-score the
weakest seed unit at `restart < 0.10`, below the declared floor of 0.2 — and `restart = 0.0`, the one
point measured below it, collapses to 0.0117. Whether anything survives between the two is a declared
gap, listed by the closure as an untested lever rather than scheduled work.

**One thing Phase 3 named survives.** The questions the concept space uniquely answers really are
conjunctions of two topics: all four the conceptual arm gains over dense are — a Florida place *and* baseball, a magazine *and* politics, an Irish biography *and* combat sports, an opera *and* its singer — and three of them are the same questions Phase 3 read, reached by a different mechanism. The signal is real and reproducible. It is worth about four questions in 600, and diffusion with restart cannot convert it into retrieval.

### Research line closed — negative result, limited scope

Recorded in [`phase_4/4.1_research_line_closure.md`](phase_4/4.1_research_line_closure.md), which reads every figure from a versioned artifact.

**What it establishes.** A concept space induced by sparse coding of pooled paragraph embeddings adds no measurable retrieval value over dense on HotpotQA — not by conceptual retrieval, not by score fusion, not by diffusion — and a dense+BM25 control is clearly stronger. An exploratory dev-only diagnostic adds that a hop through those concepts is the weakest second hop tested at every K, below simply reading further down the dense list.

**What it does not establish.** It does not test the architecture the proposal describes, in which concepts are extracted from the text of each unit. A pooled embedding recoded as sparse atoms is a lossy copy of the vector the dense baseline already uses; that operationalization is what failed, and it is the variable the next phase changes.

**What became of this plan's Phase 5.** The full bench, the fusion and expansion ablations, the K sweep and the cost per query were already measured by Phases 3 and 4. The non-negativity and pruning ablations were not run and are listed as untested levers. MuSiQue moves to Phase 6.

## Phase 5: Text-derived concepts — navigation pilot

Opened by the Phase 4 closure and now **complete**. Specified in [`phase_5/5.spec.md`](phase_5/5.spec.md) and planned in [`phase_5/5.0_text_derived_concepts.md`](phase_5/5.0_text_derived_concepts.md). The phase extracted entities and concepts from the text of the full pool, then measured one-hop navigation from dense's first paragraph on the 152-question diagnostic defined by Phase 4.

- [x] Whole-corpus extraction of entities and concepts, offline with `claude-sonnet-5`; failures recorded, with no post-hoc repair or re-extraction
- [x] Deterministic normalization and a node index covering the 19,366-paragraph pool
- [x] Entity-only, concept-only and entity+concept node hops, with BM25-question as the frozen comparator
- [x] Continuity check against all inherited non-concept hops from the Phase 4 diagnostic
- [x] Pre-declared GO / NAMES ONLY / STOP gate, computed from per-question hit@10

### What Phase 5 found

The extraction covered all **19,366 paragraphs**. The final artifact records **25.98 USD** for the full run, 8 bounds failures (0.0413%, below the 1% finding threshold), and full pool coverage. Normalization produced **96,416 entity nodes**, **40,846 concept nodes**, and **137,262 nodes** total. The node index was highly fragmented: 82.7% of entity forms and 68.9% of concept forms occur in only one paragraph. The eight extraction failures did not include any of the 158 missing-gold paragraphs; one was the `p1` of one pilot question and was retained as an explicitly documented conservative limitation.

The gate returned **NAMES ONLY** on the frozen 152-question pilot:

| Test | Wins | Losses | Ties | p | Passed |
|---|---:|---:|---:|---:|---|
| **Entity hop vs BM25-question** | **56** | 24 | 72 | **0.000226** | **yes** |
| Entity+concept hop vs BM25-question | 41 | 37 | 74 | 0.3672 | no |
| Entity+concept hop vs entity hop | 1 | 29 | 122 | approx. 1.0 | no |
| *Concept hop vs BM25-question* (reported only) | 5 | 66 | 81 | approx. 1.0 | no |

At hit@10 per question, entity navigation scored **0.6579**, versus 0.4441 for BM25 on the question. At hit@100 it scored 0.7434 versus 0.7533 for BM25 on `p1` text: the entity result is therefore a shallow ranking advantage, not evidence of greater ultimate reach. Concepts scored 0.0592 at hit@10 and 0.1743 at hit@100, and combining concepts with entities reduced the entity-only hit@10 to 0.4803.

**What this establishes:** text-derived entities are a strong second-hop navigation signal on this HotpotQA bridge diagnostic. **What it does not establish:** that an end-to-end dense + entity system beats the existing dense+BM25 system, or that concepts provide useful satellite-context discovery. The founding decisions remain unchanged. Full provenance and scope are in [`phase_5/5.results.md`](phase_5/5.results.md).

### How the phase closed, and what it leaves open

`/7-verificar` checked all 47 acceptance criteria against the artifacts rather than against the documents that quote them: 47 verified, none failing, none uncovered, with 1,386 tests passing and no skip, and ruff and mypy clean. It recomputed the gate from the per-question scores and reproduced it to the digit. The six divergences it found were documentation, not measurement, and each was corrected in its own commit.

`/8-auditar` closed with **no Critical and no High**: four findings, one Medium and three Low, catalogued as SEC-020 to SEC-023 in [`docs/security/README.md`](../security/README.md). One shape runs through all four — the artifacts this phase *measures* from carry a digest and verify it on load, while the artifacts it *runs* from carry none, so the cost ceiling, the failure gate and the batch state are all trusted on read. The fixes are specified and **not applied**: each touches `src/`, and whether to spend a `fix-N` on them before Phase 6 is the author's call. Phases 3 and 4 remain unaudited, as the same catalogue records.

`/9-documentar` has **not** been run, so the phase is closed on its numbers but not on its documentation: `CLAUDE.md` still describes the project as it stood before this phase.

## Phase 6: Dense + Entity Navigation — end-to-end comparison

A **new research line derived from Phase 5's positive entity finding**, now **complete**. It replaces the old conditional Phase 6 that was intended to evaluate the text-derived concept hypothesis. The first question is deliberately narrow:

> **Can an entity-navigation second hop replace the BM25 component of the existing dense+BM25 system without reducing retrieval quality?**

The first comparison is therefore:

```
A: Dense -> BM25
B: Dense -> Entity Hop
```

The evaluation must use the same end-to-end benchmark protocol that produced the existing dense+BM25 control, including the same split, seed, context budget, primary metric and evaluation population. The historical **0.8643 Full Support @2,048** dense+BM25 result remains the reference bar from Phase 4, but Phase 6 must measure both systems under one explicitly frozen protocol rather than compare unlike metrics from Phase 5's second-hop diagnostic.

Initial scope deliberately excludes concept nodes, entity canonicalization, relation/attribute nodes, multi-round expansion, and a three-way `Dense + BM25 + Entity` combination. Those are separate research questions and are not introduced until the basic replacement test is understood.

Specified in [`phase_6/6.spec.md`](phase_6/6.spec.md) and planned in [`phase_6/6.0_dense_entity_navigation.md`](phase_6/6.0_dense_entity_navigation.md), with the atomic tasks in [`phase_6/6.tasks.md`](phase_6/6.tasks.md). The plan declares every design decision, the dev-then-test sequence and the decision parameters before a single Phase 6 number is measured, and records four items against the spec that the author decided on 2026-09-17 without amending it.

- [x] Functional spec approved before planning
- [x] End-to-end Dense + BM25 control frozen under the Phase 6 protocol
- [x] Dense + Entity Hop implemented with no canonicalization or other new signal
- [x] Same-budget comparison run on the declared evaluation population
- [x] Result reviewed and decision recorded: continue to canonicalization / combinations, or stop

### What Phase 6 found

The decision procedure — the metric, the budget, the margin, α, the methods and the population — was frozen in `data/replacement/freeze.json` and **committed before the test split was opened**. `cer replace-test` then ran **once**, with no flags, in the order dense → A → first read of the historical test figures → reproduction check → B. The historical dense and dense+BM25 test figures reproduced exactly (32 checks, 0 failed), so the control was reused rather than re-measured, and the margin's source was confirmed from those files at 1,210 − 1,155 = 55 questions over n = 1,400.

At Full Support @2,048 on the 1,400 test questions: **dense 0.8250** (1,155), **A = Dense + BM25 0.8643** (1,210), **B = Dense + Entity Hop 0.8900** (1,246). B's fusion (`weighted`, `{dense: 0.7, entity-hop: 0.3}`) was selected on dev only and frozen.

| Preregistered test | Result | Passed |
|---|---|---|
| **S** — B beats dense | 103 wins, 12 losses, 1,285 ties, p = 1.676e-19 | yes |
| **V** — A beats dense (assay sensitivity) | 85 wins, 30 losses, 1,285 ties, p = 1.430e-07 | yes |
| **N** — B non-inferior to A at δ = 55/2800 | b = 96, c = 60, n = 1,400, Tango 4.9464, lower limit +0.011135 > −0.019643, p = 3.780e-07 | yes |

**State: `ENTITY_REPLACEMENT_SUPPORTED`**, no route, no anomalies, no open question (`data/replacement/decision.json`).

**What this establishes:** under the Phase 6 protocol, an entity-navigation second hop **replaces BM25 in the dense hybrid without reducing retrieval quality**, and the resulting system improves on dense alone. The mechanism is auditable: all 103 questions B supports and dense does not carry at least one paragraph the entity hop introduced, 67 of them with no reordering of dense's own list at all.

**What it does not establish:** the phase asked a replacement question, and B's 2.57-point Full Support margin over A is recorded as **descriptive and deciding nothing** — it is not a superiority finding and does not become a new hypothesis here. The result does not establish the Type C claim, does not validate the wider concept-embeddings architecture, and does not speak to canonicalization: it uses Phase 5's raw uncanonicalized entities. It is specific to HotpotQA, to this frozen corpus and protocol, and it costs roughly 7x the control's retrieval latency (10.4 ms against 1.5 ms per query). The founding decisions remain unchanged. Full provenance, limitations and declared gaps are in [`phase_6/6.results.md`](phase_6/6.results.md).

**This does not reopen the first research line.** Phase 4's negative result concerned concepts *induced from pooled paragraph embeddings*; Phase 6 measures a second-stage generator over entity names *read from the text*. They are different representations, and the Phase 4 closure stands exactly as scoped.

### How the phase closed, and what it leaves open

`/7-verificar`, `/8-auditar` (**mandatory** for this phase: nine artifact types read back from disk and a loader path into Phase 5's node index) and `/9-documentar` have **not** been run. The phase is closed on its numbers and its results document; the audit and the project documentation are still pending, and Phases 3 and 4 remain unaudited as `docs/security/README.md` records.

**No follow-up phase is opened by this result.** Phase 7 stays conditional and Phase 8 stays closed; whether the decision justifies opening either is the author's to decide.

## Phase 7: Entity canonicalization / resolution

**Conditional on Phase 6.** The question is whether independently extracted entity mentions can be canonicalized so that genuine aliases and naming variants share a stable node, without merging distinct entities. The observed singleton rate in Phase 5 motivates this line, but does not by itself imply that singletons should be removed.

The baseline comparison should be:

```
Raw Entity Hop
vs
Canonicalized Entity Hop
```

Candidate techniques may include embedding-based candidate generation followed by an explicit resolution step. Simple clustering or deleting every entity with document frequency 1 is not assumed to be valid: a genuine one-off entity can still be retrieval-critical. The phase should measure both node compression and retrieval effect, and preserve an auditable mapping from raw mentions to canonical nodes.

- [ ] Open only after a Phase 6 decision justifies it
- [ ] Define canonicalization spec before implementation
- [ ] Freeze the raw entity baseline and candidate-generation/resolution procedure on dev
- [ ] Measure the effect on Entity Hop with the same end-to-end evaluation protocol
- [ ] Record whether canonicalization improves, leaves unchanged, or harms retrieval

## Phase 8: Conditional extensions and combinations

**Not opened unless Phase 6 and/or Phase 7 justify further work.** This phase collects the remaining open research questions from the original plan and from the Phase 5 findings. Possible directions are:

- [ ] Dense + BM25 + Entity, if the two-signal replacement test shows complementary value worth studying
- [ ] Multi-round entity navigation or query-aware expansion, after the one-hop mechanism is established
- [ ] Relation and attribute nodes for bridges that are not carried by shared names
- [ ] A benchmark or evaluation set for thematic / satellite-context questions (Type C), rather than only entity-linked HotpotQA bridges
- [ ] Document hierarchy or book-like corpora, where hierarchical context can actually be evaluated
- [ ] Cross-benchmark evaluation (for example MuSiQue) to test whether any successful mechanism generalizes
- [ ] Graph or higher-order structures only if the simpler entity representation demonstrates durable value

The old exploratory Phase 7 items — hierarchy, variable resolution, graph derivation and finer-grained representations — belong here now, rather than being assumed to follow from a successful concept-space experiment.

---

## Founding decisions

Settled during the discovery conversation. Any change propagates here only after being verified.

| Decision | Choice | Rationale |
|---|---|---|
| Nature of the deliverable | Validation experiment | The goal is to prove or refute §69, not to build a product |
| Corpus | Public multi-hop benchmark with ground truth | It ships annotated supporting facts: no dataset has to be fabricated, and the numbers stay comparable with the literature |
| Working language | English for everything this project produces | Source documents in `docs/refs/` stay in Spanish as authored |
| Indexing unit | The smallest complete unit of meaning, the one describable with few concepts. Never fixed-length windows | A 400-character window cuts through the middle of an idea and yields rows of X mixing the end of one concept with the start of another. In the benchmark that unit is the article paragraph, which is never split |
| Concept engine | **Non-negative sparse coding** over the embeddings (`MiniBatchDictionaryLearning`, scikit-learn), not clustering | Clustering assigns each chunk to **one** cluster and leaves X nearly one-hot, destroying co-activation (§14), chunk↔chunk similarity in concept space (§15) and expansion itself. Sparse coding yields multi-activation with continuous weights by construction, which is what §2-§3 require. Lesson taken from SAE-SPLADE, the closest reference in the surveyed prior art. Documented alternative if finer control over sparsity is needed: a ReLU SAE in torch |
| Concept embedding | The dictionary atom itself | It already lives in the query's space, so the §6-§7 mapping is a dot product. Avoids the double loss of embedding an LLM-generated name, which is a poor projection of the concept |
| Expansion mechanism | **Diffusion over X** (`X` and `Xᵀ` with partial restart on the query), not an agentic LLM loop | Deterministic, reproducible and negligible in cost. Above all it **isolates the contribution**: if System C wins with an LLM in the loop, there is no telling whether the concept space worked or the model guessed well. Lesson taken from HyCE-RAG (§25 of the bibliography) |
| Compute | Local ARM64 CPU, small embedding model | Zero cost and no network dependency; forces the on-disk cache to be part of the architecture. **Revisited after the concept-engine change**: the ARM64 limitation stopped biting once BERTopic was dropped, and emulating x64 would penalize exactly the bottleneck (embedding ~10,000 units). To be reconsidered only if an indispensable dependency turns out to have no wheel |
| Baselines | BM25 in addition to dense | On HotpotQA, BM25 is a hard rival and frequently beats dense retrieval. Claiming an improvement "over dense" without BM25 present is a result that does not survive the first question |
| Measurement protocol | **At fixed context budget**, not fixed K | System C retrieves more chunks by design; comparing at equal K is unfair one way and at different K unfair the other. The comparison that holds is at equal tokens sent to the LLM (§73) |
| Primary metric | Supporting-fact recall within the budget | Objective and annotated; no LLM judge on the critical path |
| Dictionary quality | Measured per concept against a random null baseline; reported in Phase 2, never enforced there | Coherence measured in the same embedding space the atoms were fitted to is partly guaranteed by construction: it ranks concepts against each other, it does not validate the method. And an absolute cosine is unreadable without the null, since random units of this corpus are far from orthogonal. Pruning changes what retrieval sees, so it belongs to a dev-recall decision in Phase 3, not to an unvalidated threshold in Phase 2 |

Phase 1 assumptions, to be settled in its spec: a HotpotQA *distractor* subset of 500-1,000 questions, which yields the 5,000-10,000 chunks of the scale suggested in §70; `bge-small-en-v1.5` or `all-MiniLM-L6-v2` embeddings.

### Platform constraint (verified during scaffolding)

Development runs on **Windows on ARM (`win_arm64`)**, which rules out part of the usual ecosystem:

- `torch` is not on PyPI for this platform. `2.13.0+cpu` is installed from the official PyTorch index, declared as an explicit source in `pyproject.toml`.
- `bertopic`, `umap-learn`, `hdbscan`, `numba`, `llvmlite`, `fastembed`: no wheel. Not used.
- HuggingFace `datasets`: no wheel (because of `pyarrow`). The corpus is downloaded as raw JSON and frozen with its hash, which also serves the reproducibility Phase 1 demands.
- Available: `numpy`, `scipy`, `scikit-learn`, `sentence-transformers`, `transformers`, `tokenizers`, `onnxruntime`, `faiss-cpu`, `model2vec`.
- Python pinned to 3.12 (`>=3.12,<3.13`).


---

## Declared limitation

Phase 5's positive entity result is specific to a HotpotQA bridge diagnostic: questions are selected when dense misses a supporting fact outside its top-10, and the missing fact is often linked through a shared named entity. It therefore does not by itself establish the broader Type C claim from §72, where satellite knowledge such as nutrition, pacing or recovery is not explicit in the query. A separate evaluation population is required for that claim.

The current entity hop also has a reachability ceiling: only 118 of the 158 missing paragraphs share an entity node with `p1`, and it finds 116 of those 118 by depth 100. The canonicalization problem is open, and the high singleton rate in Phase 5 may partly reflect legitimate one-off entities and partly naming variation; the experiment has not yet separated those causes.

Phase 6's end-to-end result inherits that scope. It is measured on HotpotQA, on the frozen Phase 1 pool, with Phase 5's raw uncanonicalized entities, under one frozen protocol whose metric, budget, margin, α, method and population cannot be varied on the same test split. It establishes replacement of the BM25 component under that protocol; it does not establish the Type C claim, does not validate the wider concept-embeddings architecture, and says nothing about what canonicalization would add. Its retrieval cost is roughly 7x the control's per query.

The corpus itself remains Wikipedia-style and offers no genuine chapter/section hierarchy. Any claim about hierarchical retrieval therefore belongs to a later phase on a corpus where that structure exists.

## Project success criterion

The project succeeds by producing **defensible, reproducible answers to the active research question**, with its numbers, controls, limitations and provenance. The original pooled-concept research line is closed with a negative, bounded result. Phase 5 separately established a positive entity-navigation signal, and Phase 6 has now shown end to end that it improves on the complete dense+BM25 system's own terms.

The immediate success criterion for the new line was Phase 6: determine under one comparable end-to-end protocol whether Entity Hop can replace BM25 in the dense+BM25 pipeline. **It is met, and the answer is recorded as `ENTITY_REPLACEMENT_SUPPORTED`.** Phase 7 and Phase 8 remain conditional follow-ups rather than prerequisites; opening either is the author's decision, and this result does not open them.

---

## Corrective fixes

- **[fix-1](fixes/fix-1_audit_phase_1_findings.md)** — the ten findings of the Phase 1 security audit, all of them integrity controls that existed in the code and were never exercised by the pipeline. Closed with 15 regression tests; the full evaluation was re-run and all 60 recorded metrics reproduce exactly, so the Phase 1 numbers are unaffected.
