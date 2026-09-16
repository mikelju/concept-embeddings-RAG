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
| 5 | Text-derived concepts: navigation pilot | Available | **Complete — gate: NAMES ONLY** |
| 6 | Dense + Entity Navigation: end-to-end comparison | Pending | **Open by the author's decision** — awaits `/4-especificar` |
| 7 | Exploratory extensions (conditional) | Pending | Not opened |

**The first research line is closed** (2026-09-15): concepts induced from pooled embeddings, tested
through Phases 2-4, gave a negative and bounded result. The original Phase 5 — a comparative
evaluation of that method — was not run; the plan was renumbered so Phase 5 tests the representation
the proposal actually described. See [`phase_4/4.1_research_line_closure.md`](phase_4/4.1_research_line_closure.md).

**The second reading of "concept" has now been tested too** (2026-09-16): the Phase 5 pilot extracted
entities and concepts from the text of all 19,366 paragraphs and measured one hop over each. The
pre-declared gate returned **NAMES ONLY** — entities beat the frozen comparator decisively, concepts
are the weakest hop this project has measured, and adding concepts to entities makes the joint hop
worse. **The concept hypothesis is closed and the founding decisions table is unchanged**; the
result and its scope are in [`phase_5/5.results.md`](phase_5/5.results.md).

**Phase 6 was then redefined, on the author's decision and not by the gate** (2026-09-16). The gate
opens Phase 6 only on GO, and it did not; what opens the phase below is the author's reading of the
entity finding — the positive half of a NAMES ONLY — as worth measuring end to end. The phase the
plan used to carry there, a comparative evaluation of the pooled-embedding concept representation,
is gone with the line it belonged to.

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
conjunctions of two topics: all four the conceptual arm gains over dense are — a Florida place *and*
baseball, a magazine *and* politics, an Irish biography *and* combat sports, an opera *and* its
singer — and three of them are the same questions Phase 3 read, reached by a different mechanism.
The signal is real and reproducible. It is worth about four questions in 600, and diffusion with
restart cannot convert it into retrieval.

### Research line closed — negative result, limited scope

Recorded in [`phase_4/4.1_research_line_closure.md`](phase_4/4.1_research_line_closure.md), which
reads every figure from a versioned artifact.

**What it establishes.** A concept space induced by sparse coding of pooled paragraph embeddings adds
no measurable retrieval value over dense on HotpotQA — not by conceptual retrieval, not by score
fusion, not by diffusion — and a dense+BM25 control is clearly stronger. An exploratory dev-only
diagnostic adds that a hop through those concepts is the weakest second hop tested at every K, below
simply reading further down the dense list.

**What it does not establish.** It does not test the architecture the proposal describes, in which
concepts are extracted from the text of each unit. A pooled embedding recoded as sparse atoms is a
lossy copy of the vector the dense baseline already uses; that operationalization is what failed,
and it is the variable the next phase changes.

**What became of this plan's Phase 5.** The full bench, the fusion and expansion ablations, the K
sweep and the cost per query were already measured by Phases 3 and 4. The non-negativity and pruning
ablations were not run and are listed as untested levers. MuSiQue moves to Phase 6.

## Phase 5: Text-derived concepts — navigation pilot

Opened by the closure above. Specified in [`phase_5/5.spec.md`](phase_5/5.spec.md) and planned in
[`phase_5/5.0_text_derived_concepts.md`](phase_5/5.0_text_derived_concepts.md), which fix every
choice below before a number exists: **the whole corpus extracted offline by Sonnet 5**, **all 152
dev questions** the Phase 4 diagnostic defined, a hop from dense's first paragraph over
**entity-only, concept-only and entity + concept** nodes, the **question** as the statistical unit,
and **BM25 on the question** as the frozen comparator. The gate: GO if entity + concept beats both
the comparator and entity-only; NAMES ONLY if entity-only itself beats the comparator; STOP
otherwise.

The question: **do concepts and entities extracted from the text give a more useful navigation
structure than atoms induced from pooled embeddings?** Concretely, when dense has found the first
paragraph of a bridge and missed the second, can a hop over text-derived nodes recover it better than
reading further down the dense list, or than BM25?

- [x] Specified with `/4-especificar` before any extraction runs: the pilot question set, drawn from
      dev by a deterministic rule and never hand-picked; the node types; the hops compared; the
      metrics; and a GO/STOP gate declared before the first number exists
- [x] The extractor (LLM or otherwise) builds the representation **offline**; it never sits inside
      the retrieval loop, by the same founding decision that governed Phase 4
- [x] Everything else held fixed: corpus, indexing unit, embedding model, both baselines, budgets
- [x] A design that can tell "concepts navigate" apart from "names navigate": an entity-only win
      answers a narrower question than §69 — and that is the win the gate found
- [x] A documented GO/STOP decision - scale to the full corpus and iterative expansion, or close the
      hypothesis

### What the pilot found, and the stop it records

Every figure below is read from a versioned artifact and listed with its command in
[`phase_5/5.results.md`](phase_5/5.results.md). The representation was built by extracting entities
and concepts from **all 19,366 paragraphs** with `claude-sonnet-5`, offline, for **25.98 USD** and
8 failures in 19,366 (0.0413%, below the 1% finding threshold) — the ceiling having been raised from
25 to 35 USD at the measured-sample checkpoint, [deviation 5.1](phase_5/5.1_extraction_cost_ceiling.md).
Normalization v1 turned that into a node index of **137,262 nodes**. `uv run cer navigate` then ran
every hop once over the frozen pilot of 152 dev questions and 158 missing paragraphs.

**The gate returned `NAMES_ONLY`**, computed by code from the per-question hit@10 scores, with
`anomalies` empty:

| Test | Wins | Losses | Ties | p | Passed |
|---|---:|---:|---:|---:|---|
| E beats BM25-on-question | **56** | 24 | 72 | **0.000226** | **yes** |
| EC beats BM25-on-question | 41 | 37 | 74 | 0.3672 | no |
| EC beats E | 1 | 29 | 122 | ≈ 1.0 | no |
| *C beats BM25-on-question* (reported only) | 5 | 66 | 81 | ≈ 1.0 | no |

**Entities navigate.** Hit@10 per question: **0.6579** for the entity hop against 0.4441 for BM25 on
the question, 0.4507 for BM25 on `p1`'s text, and 0.3026 for reading further down the dense list. It
places **18 missing paragraphs** in its top 10 that lie outside both baselines' own reach. It is the
strongest shallow second hop this project has measured.

**Concepts read from the text do not.** 0.0592 hit@10 and 0.1743 hit@100 per question — **below the
pooled-embedding concept hop the first research line already closed as negative** (0.0789 / 0.3684 on
the same questions), and below every other hop at both depths. Only **47 of the 158 missing
paragraphs share a single concept node with `p1`**, against 118 for entities: the arm mostly cannot
reach, and where it reaches it ranks badly.

**Joining the two makes the hop worse, which is the failure the gate was built to detect.** EC scores
0.4803 hit@10 against E's 0.6579. The union can never lower the gold paragraph's own score — it
raises every competitor's more: the candidate field grows from a median of 34.5 paragraphs to 391.5,
and of the 108 paragraphs both arms find, EC ranks 72 worse and drops **23 out of the top 10**. A hop
that finds the *Moby* paragraph at rank 1 on the name `moby` cannot find it within a hundred once
generic concepts join the sum.

**The headline is a depth-10 result and is stated as one.** At hit@100 the entity hop (0.7434) is
**behind BM25 on `p1`'s text** (0.7533): it finds paragraphs *sooner*, not paragraphs the older hops
cannot reach. Its ceiling is reachability — 118 of 158 missing paragraphs share an entity node with
`p1`, and it finds 116 of those 118.

**What the stop closes.** §7 item 1 of the research-line closure — *concepts extracted from the text
rather than directions found in a pooled embedding* — is no longer open in the form this pilot tested
it: that reading of "concept" is the weaker of the two measured. **The concept hypothesis stays
closed**, and nothing below reopens it.

**What it does not close.** HotpotQA bridges are built around an entity named in one paragraph and
described in another, so an entity win is close to a property of the benchmark, as the Phase 4
closure said before this pilot ran. Untested and still open: a corpus whose bridges are thematic
rather than nominal (Type C, §72, the declared gap below); multi-round expansion and any hop but the
single one specified; relation and attribute nodes, which this spec's anti-goals excluded and which
are the node type most likely to carry a non-nominal bridge; a normalization that resolves the
synonymy this one leaves (measured and declared as working against the concept arm); and the levers
§7 of the closure already lists.

**The founding decisions on the concept engine and the concept embedding are unchanged.** They change
only once a phase has verified an alternative, and this pilot verified none: it measured one and
found it worse than what it replaced.

### Phase 5 is complete

**Verified** (2026-09-16): 1,386 tests pass with no failure and no skip, `ruff` and `mypy` clean, and
**47 of 47 acceptance criteria verified**, none failing and none uncovered. Verification reported six
divergences before ticking anything; all six are settled, the two substantive ones by correcting
documents rather than code — an arithmetically wrong cost note in
[`5.results.md`](phase_5/5.results.md), and two data contracts in
[`5.spec.md`](phase_5/5.spec.md) that had fallen behind the artifacts they describe.

**Audited** (2026-09-16): `/8-auditar fase 5` over `nodes/`, the pilot's four `evaluation/` modules
and the four new CLI stages. **No Critical and no High finding**, so the phase closes. Four findings
stand open — one Medium and three Low, SEC-020 to SEC-023 — because each needs a change under `src/`
and the author closed this round to source changes; they are catalogued with their fixes in
[`docs/security/README.md`](../security/README.md), which also records that decision. Phases 3 and 4
remain unaudited, as the same file records under "Deferred audits".

**`/9-documentar` has not been run.** The phase's flow is `/7-verificar` → `/8-auditar` →
`/9-documentar`, and the author chose to close after the audit. That step is outstanding and is
recorded here rather than assumed done.

## Phase 6: Dense + Entity Navigation — end-to-end comparison

**Opened by the author's decision on the entity finding, not by the gate.** The Phase 5 gate opens a
phase only on GO; it returned NAMES ONLY, which by its own table opens nothing. What opens this phase
is a choice: the entity hop is the strongest shallow second hop this project has measured — 0.6579
hit@10 per question against the frozen comparator's 0.4441, 56 wins to 24 with p = 0.000226 — and the
author decided that a positive finding of that size is worth measuring as a system rather than
leaving as one number in a pilot. **The concept hypothesis stays closed**, in both its readings, and
nothing here reopens it.

**Not yet specified.** The next action in this repository is `/4-especificar` for this phase. Nothing
below is a design decision: the questions, the protocol and the gate belong to the spec, and the
pilot's cautions are recorded here so that the spec inherits them rather than rediscovering them.

**The question**, narrower than §69 by construction: does a second retrieval stage that navigates
from dense's first paragraph over text-derived **entity** nodes improve supporting-fact recall **at
equal context budget** over dense, and over the dense+BM25 control that is the real bar — 0.8643 Full
Support on test @2,048, the figure Phase 4 was judged against? A win over dense that does not clear
the control is a result, to be reported as exactly that.

**What Phase 5 hands it, and what it withholds.** The entity arm's ceiling is reachability: 118 of
158 missing paragraphs share an entity node with `p1`, and it finds 116 of those 118, so no depth
buys the other 40. At hit@100 it is already **behind** BM25 on `p1`'s text (0.7434 against 0.7533):
it finds paragraphs sooner, not paragraphs the older hops cannot reach. And all of it was measured on
**dev only**, over **one hop**, on HotpotQA bridges built around a name — which is why the entity win
is close to a property of the benchmark, as the Phase 4 closure said before the pilot ran.

- [ ] The entity-navigation system measured end to end in the existing harness, **at equal context
      budget** rather than at fixed K, against BM25, dense and the dense+BM25 control, on a
      configuration frozen before the test split is read once
- [ ] Cost per query: tokens sent, latency, extraction cost amortized over the corpus (§73 demands
      the gain not be paid in noise or tokens)
- [ ] Cross-validation on a second benchmark (**MuSiQue**) to rule out overfitting to the first —
      which this phase needs more than its predecessor did, because the pilot's own verdict is that
      an entity hop suits nominal bridges
- [ ] Ablations declared in its spec, of the navigation system and not of the closed representation:
      at minimum the hop's origin, its depth and whether the entity stage adds anything BM25 on
      `p1`'s text does not — the hop it loses to at depth 100
- [ ] A report with an explicit verdict on **this** question, negative verdict included

**Carried over from the Phase 6 this replaces**: measurement at equal context budget, cost per query,
and MuSiQue as a guard against overfitting to one benchmark. Those three were never about the concept
space; they are how this project compares anything.

**Dropped, and why.** The full bench of concept systems and the ablations of the text-derived
representation: both the representations they would have benched are closed — the pooled-embedding
space by Phases 3-4 and its closure, the text-derived concept space by the Phase 5 gate — and
benching a closed representation measures nothing. Its iterative expansion: Phase 4 measured
diffusion over `X` and the walk never promotes a new paragraph. The explicit verdict on §69: it has
been answered twice, negatively and with bounds, and this phase does not re-ask it — the project's
answer to §69 stands where the closure and the pilot left it, and what is being measured here is a
narrower claim about names.

## Phase 7: Exploratory extensions (conditional)

**Not opened unless the Phase 6 results justify it.** The decision is documented either way.

- [ ] Documented decision to open the phase or not, grounded in the Phase 6 numbers
- [ ] Document hierarchy as a fourth scoring signal (§34, §56)
- [ ] Variable resolution of the concept dictionary (§48)
- [ ] Graph derived from X, only if the emergent associations prove useful (§58)
- [ ] Token-level or multi-vector concept representations, finer than one vector per unit

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

Multi-hop benchmarks measure **Type B** questions from §72 (multi-concept, with the hops already laid out by the annotator). **Type C** — the 200 km training plan that never mentions nutrition or pacing — is where the proposal promises most, and no standard benchmark measures it: there, satellite knowledge would have to emerge from the corpus rather than from annotation.

No phase of this project can therefore claim, or refute, that the system discovers satellite context, absent a separately built set of Type C questions. This is a declared gap, not a filled one. Covering it is Phase 7 material and requires first deciding how to build such a set without fabricating convenient ground truth. It applies unchanged to both negative results: the Phase 5 pilot's entity win and concept loss are measured on Type B bridges, where the second paragraph is reached by a name, and neither figure transfers to Type C.

**A second limitation of the same family:** the indexing unit is chosen by complete meaning rather than length, but the benchmark offers only a shallow three-level hierarchy (article → paragraph → sentence). A real document hierarchy — chapter, section, subsection — does not exist in Wikipedia and cannot be exercised here. That is why hierarchy is Phase 7 material and not a Phase 3 signal, and why a corpus of books would be the natural setting for that part. A point of precision: RAPTOR does not read document structure either — it starts from fixed-length chunks and **fabricates** the tree by recursive clustering and summarization.

## Project success criterion

The project succeeds if there exists a **defensible and reproducible** answer to the hypothesis of §69, with its numbers, its ablations and its cost. A negative result documented rigorously — that iterative conceptual expansion does not improve recall at equal token budget — is a valid result and closes the project just as well as the opposite.

**That criterion is already met, and it was met before Phase 6 was redefined.** The wording used to read "at the end of Phase 6 — or at the Phase 5 gate, if the pilot says STOP", written when Phase 6 was the comparative evaluation of the concept representation. The pilot did not say STOP, it said NAMES ONLY, and the redefined Phase 6 does not answer §69 at all: it measures a narrower claim about navigating by names. So §69's answer stands where the research-line closure and the Phase 5 pilot left it, and Phase 6 adds to the project without being what decides it.

The first research line has such an answer for its own operationalization (see the closure under Phase 4), and the Phase 5 pilot has one for the reading that closure left open — concepts extracted from the text, which measured worse than the representation it was meant to replace. Both answers are negative, both are bounded, and both are reproducible from versioned artifacts. What neither answers is §69 in general: every measurement of this project is on HotpotQA's Type B bridges, where the second paragraph is reached by a name, and the Type C setting the proposal promises most for is still the declared gap below.

---

## Corrective fixes

- **[fix-1](fixes/fix-1_audit_phase_1_findings.md)** — the ten findings of the Phase 1 security
  audit, all of them integrity controls that existed in the code and were never exercised by the
  pipeline. Closed with 15 regression tests; the full evaluation was re-run and all 60 recorded
  metrics reproduce exactly, so the Phase 1 numbers are unaffected.
