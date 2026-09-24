# Research summary — minimum structure for multi-hop retrieval

> Status on 2026-09-24: the planned research line (Phases 1-9) is complete. This page summarises
> what was asked, what was measured and what it establishes. Every figure links back to the
> phase document that records it with its artifact; nothing here is new measurement.

## The question

Dense retrieval finds paragraphs that *look like* the question. Multi-hop questions also need
paragraphs that do not: the second document of a HotpotQA bridge question is linked to the first
by a fact, not by wording. The working question became:

> What is the **minimum structural signal** Dense retrieval needs to recover multi-hop evidence
> that semantic similarity alone does not express?

"Minimum" is the point. The project did not set out to build a GraphRAG system. It set out to find
the cheapest structure that measurably helps, and to measure its cost, so that richer structure
has a baseline it must justify itself against.

## How it was measured

- **Benchmark.** HotpotQA. A frozen 600 dev / 1,400 test question split over a pooled corpus of
  19,366 whole Wikipedia intro paragraphs (Phases 1-8), then the full HotpotQA processed Wikipedia,
  5,233,329 paragraphs, with all 7,405 validation questions (Phase 9).
- **Unit.** One complete paragraph, never a fixed-length window.
- **Metric of record.** Full Support @2,048 tokens: the share of questions for which **every** gold
  paragraph fits in a 2,048-token context built from the ranking. It measures what a reader model
  would actually receive, not a rank position.
- **Discipline.** Every choice was fitted on dev only. Each held-out split was opened once, for the
  configuration dev chose, under a rule written down before any candidate ran. Negative results
  are kept.

## What was done, phase by phase

| Phase | Question | Outcome |
|---|---|---|
| [1](plans/phase_1/1.results.md) | Build the instrument: corpus, Dense and BM25 baselines, budget harness | Dense beats BM25 at every budget |
| [2](plans/phase_2/2.results.md)-[4](plans/phase_4/4.results.md) | Can a concept space induced from pooled Dense embeddings guide retrieval? | **Negative.** Concepts add nothing over Dense; a Dense + BM25 control beats them; diffusion expansion re-ranks without promoting a new paragraph. Line closed ([4.1](plans/phase_4/4.1_research_line_closure.md)) |
| [5](plans/phase_5/5.results.md) | Do concepts or entities *read from the text* navigate to the missing paragraph? | **`NAMES_ONLY`.** Entities do (hit@10 0.658 vs 0.444 for the comparator); text-derived concepts are the weakest hop measured and make entities worse when added |
| [6](plans/phase_6/6.results.md) | Can a one-hop Entity Hop replace BM25 in the Dense hybrid? | **`ENTITY_REPLACEMENT_SUPPORTED`** on 1,400 test questions |
| [7](plans/phase_7/7.results.md) | Can the costly LLM extraction be replaced by a local one? | **GLiNER selected.** It keeps 83.5 % of the gain for about a thousandth of the cost |
| [8](plans/phase_8/8.results.md) / [8.1](plans/phase_8/8.1_results.md) | Does the hop still help beside a stronger Dense? | **Premise failed.** Qwen3-Embedding-0.6B was weaker than BGE-small on this task at every corpus size up to 500k; no stronger Dense was tested |
| [9](plans/phase_9/9.results.md) | Does it survive the real FullWiki search space? | **`SCALE_SUPPORTED`**, with Dense + BM25 slightly ahead at scale |

## The mechanism that worked

```text
question ─► Dense ranking ─► P1 = its first paragraph
                               │
                    entities named in P1  (GLiNER, zero-shot, 5 labels)
                               │
             every paragraph sharing ≥ 1 raw entity with P1
                               │
        score = Σ log(1 + N / (1 + df)) over the shared entities
                               │
               fused with Dense, 0.7 / 0.3 (fitted on dev)
```

It is deliberately small. It does not look at the question after the first Dense call, has no
relations and no canonicalization, uses one seed and one hop, and calls no LLM at query time.

## What it established

**1. At pool scale, the Entity Hop beats the standard lexical complement.** On the 1,400 held-out
questions (19,366 paragraphs), Full Support @2,048 was:

| System | Full Support |
|---|---:|
| Dense (BGE-small) | 0.8250 |
| Dense + BM25 | 0.8643 |
| Dense + Entity Hop, GLiNER entities | 0.8793 |
| Dense + Entity Hop, Claude entities | 0.8900 |

With GLiNER, the Entity Hop is ahead of Dense + BM25 by 21 questions (+1.50 points) and at every
token budget.

**2. The structure is cheap to build.** GLiNER extracted entities for all 5,233,329 FullWiki
paragraphs in 6.18 h on one rented RTX 4090, for an attributable **4.57 USD** (derived from
measured time and the hourly rate). The whole Phase 9 session was billed 12.71 USD.

**3. The gain over Dense survives full scale.** On the 5,405 FullWiki questions no earlier phase
had used, over 5.2 million paragraphs:

| System | Full Support @2,048 |
|---|---:|
| Dense | 55.62 % |
| Dense + Entity Hop | 60.09 % (+4.48 pp over Dense, exact McNemar p = 8.0e-28) |
| Dense + BM25 | 61.30 % |

The pool-scale gain over Dense was +5.17 points on dev, and it is almost entirely kept at 270 times
the corpus size, while Dense itself falls by about 25 points.

**4. Cheap zero-shot extraction is enough to carry most of the signal.** Claude entities gave the
larger gain at pool scale, but GLiNER kept 83.5 % of it.

## What did not work, and what is not established

- **Concepts, in two operationalizations**, add nothing. The first used concepts induced from
  embeddings (Phases 2-4) and the second concepts extracted from the text (Phase 5). The negative
  result is bounded to those two operationalizations on this benchmark.
- **At FullWiki scale, BM25 is the slightly better complement to Dense at the primary budget**
  (61.30 % against 60.09 %, p = 0.024, descriptive). The Entity Hop is ahead at 4,096 tokens and
  at pool scale. The two signals are not redundant: at scale, 371 questions go the Entity Hop's way
  and 436 go BM25's. **The frozen Entity Hop is therefore not, as it stands, a better system than
  Dense + BM25 over the full search space.**
- **No stronger Dense was tested.** The one candidate tried (Qwen3-Embedding-0.6B) was weaker on
  this task at every corpus size up to 500k paragraphs. Whether the hop still adds value beside a
  substantially stronger Dense retriever is open.
- **No GraphRAG system was compared.** The project measured the floor, not the ceiling.
  Relation-based GraphRAG systems are mostly evaluated on pooled corpora of about 9k–67k
  passages, not on the FullWiki space. For scale, extracting with an LLM (the Phase 5 Claude
  extraction cost 25.98 USD for 19,366 paragraphs) would cost on the order of 7,000 USD over
  FullWiki. That is a **linear projection**, not a measurement, and about a thousand times the
  local entity extraction.
- **Literature numbers are not directly comparable.** Trained multi-hop retrievers report higher
  rank-based recall on FullWiki (MDR: R@2 65.9 / R@20 80.2) under different supervision and metric
  definitions. The ledger in [`9.results.md`](plans/phase_9/9.results.md) states each difference.

## Where this leaves the work

The line answers its question with a measured floor. **A query-blind, one-hop entity signal from a
cheap local extractor recovers a large share of the multi-hop evidence Dense misses, at every
scale tested.** At pool scale that is enough to beat the standard Dense + BM25 hybrid. Over the
full 5.2-million-paragraph space it is not, by 1.2 points at the primary budget.

What the next line has to show is therefore concrete: **a system that is better than Dense + BM25 at
FullWiki scale.** Any added structure must beat that bar, not just Dense. The measured disagreement
between the two complements is the first evidence to use. The 7,405 validation questions are now
spent as held-out data, so a new experiment needs a fresh, untouched question set.

## Reproducibility

Each phase document names its artifacts, digests, model revisions and commands. The rented-GPU
procedure is in [`plans/phase_9/9.runpod_recipe.md`](plans/phase_9/9.runpod_recipe.md). Security
findings of every audited phase are in [`security/README.md`](security/README.md). Deferred
directions and their rationale are in [`plans/research_roadmap.md`](plans/research_roadmap.md).
