---
name: research-protocol
description: Experimental protocol of this project - dev/test separation, preregistered selection, the simplification rule, artifact integrity and the research anti-patterns. Load before specifying, planning, implementing or reviewing anything that produces, selects on or reports a measured retrieval result. Not needed for read-only questions.
---

# Research protocol

The short, non-negotiable version lives in `CLAUDE.md`. This is the detail behind it.

## Dev/test separation

- Fitting or choosing between alternatives happens on the 600 dev questions only.
- The 1,400 test questions are opened once, for the configuration dev chose, under a rule frozen
  before any candidate ran.
- When comparing alternatives, dev picks the winner and only the winner gets a held-out figure.
  That is how Phase 7 selected GLiNER, and why spaCy has no test figure. Never produce one
  retrospectively.
- Overriding a preregistered rule after observing dev destroys the meaning of the held-out
  figure. Phase 7 is the worked example: spaCy scored higher on dev (525 vs 518 of 600) and was
  still not selected, because the frozen rule ranked on projected FullWiki cost and time. Record
  such findings as deferred work, do not act on them. Full record: `docs/plans/phase_7/7.results.md`.
- A Dense model or extractor that fails a pre-declared gate is not replaced by trying another one
  inside the same phase. A different model is a declared candidate in a new, written experiment
  (Phase 8 is the worked example: `docs/plans/phase_8/8.results.md`).

## Simplification rule (Phase 7 onward)

Phase 6 reached ~2,500 tests and 24 tasks. That is not the default any more.

1. One main scientific question per phase.
2. Roughly 5-8 implementation steps, not dozens.
3. No `X.tasks.md` unless genuinely needed.
4. Reuse existing infrastructure; do not generalize pre-emptively.
5. No branches or guards for failures that have not occurred.
6. Test code that can materially change the experimental result; do not encode every process
   invariant as a test.
7. Keep dev/test separation whenever choices are fitted.
8. Record configuration, provenance and results clearly, proportionate to the experiment.
9. No elaborate statistical decision framework when a simpler comparison answers the question.
10. Implementation complexity is a real cost.

Preferred shape: research question -> minimal experiment -> result -> next question. Historical
Phase 6 rigor stays valid and is never simplified retroactively.

## Specifying a phase

A spec fixes the question, what stays fixed, what is measured, how dev decides (bars, ranking,
stop states) and the anti-goals. It does not design the implementation; the plan that follows
does, in roughly 5-8 steps. Phase 7 is the reference for this shape. Once the author approves a
spec it is frozen: no new threshold, gate, outcome or variant is introduced, and no scientific rule
is relaxed to solve an implementation difficulty. A real problem outside the plan is a written
deviation `X.Y_name.md` for the author to decide.

Apply `/8-auditar` only when the phase adds a real security surface: untrusted external input,
risky dependencies, subprocess execution, unsafe deserialization, credentials, network APIs.

## Artifact integrity

- Expensive deterministic computations (embeddings, indexes, extractions) are cached on disk and
  reused. A different model or configuration writes a distinct cache; it never overwrites the
  artifact behind an existing result.
- Use the smallest mechanism that answers "which extractor/model/configuration produced this
  result?". Phase 7's shape to reuse: a configuration digest per extractor plus a digest chain
  extraction -> node index -> dev -> selection -> held-out run, so a test figure cannot be detached
  from the dev reading and selection that authorized it.
- Non-executing formats only (JSON, NPZ, the formats already in use). No pickle for data.
- Pin or record model identity and revision for every download.
- Provenance problems found in a closed phase get a prospective fix in the producer, never a
  retroactive edit of the frozen artifact.
- Estimates and projections are always labelled as such (for example Phase 7's 5M-paragraph
  8.56 h / 6.34 USD is a projection, not a measurement or a budget).

## Deferred directions

Canonicalization, query-aware hop, relations and triples, multiple seeds or hops, Dense + BM25 +
Entity, PageRank, GraphRAG structures, other benchmarks, QA end to end, re-opening the extractor
choice at scale. They are listed with their rationale in `docs/plans/research_roadmap.md`. None of
them is promoted into an active phase without a written, approved spec.

## Anti-patterns

- **Fixed-length chunking.** The indexing unit is the whole paragraph; changing it changes the
  experimental question.
- **Assuming a richer graph is better.** Relations, PageRank, hypergraphs and multi-hop traversal
  are candidate experiments, not upgrades. The simple Entity Hop has a measured positive result.
- **An LLM in the online retrieval loop by default.** It hurts attribution and adds cost and
  non-determinism; only as an explicit experimental variant.
- **Reimplementing the literature early.** MDR, HippoRAG, KG2RAG, HGRAG, SAG, LinearRAG are not
  rebuilt unless a scientific question requires it. Phase 9 first moves to comparable scale and
  metrics.
- **Overengineering.** A small research phase stays small.
