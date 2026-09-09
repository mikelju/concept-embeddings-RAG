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
| 2 | Concept space: dictionary + matrix X | Available | Pending |
| 3 | Hybrid conceptual retrieval (System B) | Pending | Pending |
| 4 | Query-aware iterative expansion (System C) | Pending | Pending |
| 5 | Comparative evaluation and verdict | Pending | Pending |
| 6 | Exploratory extensions (conditional) | Pending | Not opened |

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

- [ ] Dictionary induced by **non-negative sparse coding** over the embeddings (`MiniBatchDictionaryLearning` with `positive_code=True` and `positive_dict=False`), fixed seed. The dictionary constraint was dropped on measured evidence, see decision D1 of [`phase_2/2.0_concept_space.md`](phase_2/2.0_concept_space.md): half the energy of the embeddings is negative, so a non-negative dictionary drives the space to one activation per unit - the very regime that disqualified clustering
- [ ] Matrix `X = chunks × concepts` as the direct output of that coding: multi-activation with continuous, non-negative weights, **with the score semantics written down** (§3: what exactly 0.73 means)
- [ ] Each concept's embedding taken from its own dictionary atom: it already lives in the query's space, with no detour through a generated name
- [ ] Concept labelling via LLM **for interpretability and reporting only**, outside the retrieval critical path
- [ ] Dictionary normalization and deduplication (§47), with an explicit, auditable merge criterion
- [ ] Structural inspection of the space: activations per chunk, chunks per concept, orphan
      concepts, dead atoms, co-activation
- [ ] **Dictionary quality measured, not assumed**: per-concept semantic coherence over the units
      that activate it most, read against a random-unit null baseline measured on this same pool -
      the absolute cosine means nothing in an anisotropic embedding space - plus the alignment of a
      concept with its own evidence and the concentration of its activation mass. Concepts are
      ranked by it so the report reads the worst ones rather than the average. **Diagnostic only:
      no concept is pruned in this phase**

## Phase 3: Hybrid conceptual retrieval (System B)

Uses the concept space to retrieve, and measures it. Answers the first half of the hypothesis: does the interpretable representation retrieve at least as well as the dense one?

- [ ] Query → concepts mapping by dot product against the dictionary (§6-§7, §50), no textual intermediaries
- [ ] Chunk scoring over the active dimensions of X
- [ ] Fusion of dense and conceptual signals, with justified weights rather than eyeballed ones
- [ ] If low-coherence concepts are pruned, the decision is made against dev recall and declared as
      a variant, never inferred from the Phase 2 diagnostic alone
- [ ] System B measured in the same harness and against both baselines (dense and BM25)

## Phase 4: Query-aware iterative expansion (System C)

The leap of §8: stop following the question and start navigating the corpus. It is the distinctive part of the proposal and the one most likely to fail.

- [ ] Expansion by **diffusion over X**: initial activation from the query, propagation chunk → concept → chunk through `X` and `Xᵀ`, with partial restart on the question's concepts at every round
- [ ] The query-aware condition (§52-53) is guaranteed by that restart: expansion cannot drift toward the corpus's global co-occurrence
- [ ] Stopping criterion on newly contributed mass, with an explicit budget of iterations and tokens (§54)
- [ ] Per-query trace: which concept entered at which iteration and which chunk it brought along
- [ ] System C measured
- [ ] *(Optional, time permitting)* Variant with an LLM inside the loop, as a comparison, to estimate how much headroom the deterministic method leaves

## Phase 5: Comparative evaluation and verdict

The phase that answers the hypothesis. Its output is a report, not code.

- [ ] Full bench on the same split and seed, **at equal context budget**: BM25, dense (A), conceptual (B) and iterative (C)
- [ ] Ablations: no dense fusion, no expansion, no non-negativity, varying dictionary size,
      pruning the least coherent concepts
- [ ] Cost per query: tokens sent, latency, number of iterations (§73 demands the gain not be paid in noise or tokens)
- [ ] Cross-validation on a second benchmark (MuSiQue) to rule out overfitting to the first
- [ ] Report with an explicit verdict on §69 and on the document's second hypothesis, including the negative verdict where warranted

## Phase 6: Exploratory extensions (conditional)

**Not opened unless the Phase 5 results justify it.** The decision is documented either way.

- [ ] Documented decision to open the phase or not, grounded in the Phase 5 numbers
- [ ] Document hierarchy as a fourth scoring signal (§34, §56)
- [ ] Variable resolution of the concept dictionary (§48)
- [ ] Graph derived from X, only if the emergent associations prove useful (§58)

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

Phase 5 will therefore be able to claim an improvement in multi-hop retrieval and **not** to claim that the system discovers satellite context, absent a separately built set of Type C questions. This is a declared gap, not a filled one. Covering it is Phase 6 material and requires first deciding how to build such a set without fabricating convenient ground truth.

**A second limitation of the same family:** the indexing unit is chosen by complete meaning rather than length, but the benchmark offers only a shallow three-level hierarchy (article → paragraph → sentence). A real document hierarchy — chapter, section, subsection — does not exist in Wikipedia and cannot be exercised here. That is why hierarchy is Phase 6 material and not a Phase 3 signal, and why a corpus of books would be the natural setting for that part. A point of precision: RAPTOR does not read document structure either — it starts from fixed-length chunks and **fabricates** the tree by recursive clustering and summarization.

## Project success criterion

The project succeeds if, at the end of Phase 5, there exists a **defensible and reproducible** answer to the hypothesis of §69, with its numbers, its ablations and its cost. A negative result documented rigorously — that iterative conceptual expansion does not improve recall at equal token budget — is a valid result and closes the project just as well as the opposite.

---

## Corrective fixes

- **[fix-1](fixes/fix-1_audit_phase_1_findings.md)** — the ten findings of the Phase 1 security
  audit, all of them integrity controls that existed in the code and were never exercised by the
  pipeline. Closed with 15 regression tests; the full evaluation was re-run and all 60 recorded
  metrics reproduce exactly, so the Phase 1 numbers are unaffected.
