# Research summary — minimum structure for multi-hop retrieval

> Status on 2026-09-29: the planned research line (Phases 1-9) is complete, and Phases 10-15 of
> the next line are measured. This page summarises
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
  5,233,329 paragraphs, with all 7,405 validation questions (Phase 9). Phase 15 adds MuSiQue: its
  2,417 validation questions (2-4 supporting paragraphs) over 101,962 pooled MuSiQue paragraphs,
  with every weight frozen at its HotpotQA value.
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
| [10](plans/phase_10/10.results.md) | Does the Entity Hop add anything on top of Dense + BM25 at FullWiki scale? | **`THREE_WAY_SUPPORTED`**: +3.92 pp on 5,000 held-out train questions (seen by BGE in fine-tuning) |
| [11](plans/phase_11/11.results.md)-[13](plans/phase_13/13.results.md) | Can the question choose which of P1's entities to hop from (DF cap, question entities, sentence or mention-window similarity)? | **`DEV_STOP` three times.** Every hard filter on the seeds loses on dev; `test-11` stayed unopened |
| [14](plans/phase_14/14.results.md) | Does ordering the hop's candidates by their similarity to the question help? | **`CANDIDATE_RELEVANCE_SUPPORTED`**: +4.90 pp over Dense + BM25 + Entity Hop on 5,000 new held-out train questions |
| [15](plans/phase_15/15.results.md) | Do the Entity Hop systems still beat Dense + BM25 on MuSiQue, with nothing refitted? | **`TRANSFER_SUPPORTED`**: P14 +9.81 pp over Dense + BM25 and +3.81 pp over the Entity Hop system; the gain is on 2-paragraph questions |

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

**5. At full scale, the Entity Hop adds to BM25 rather than competing with it.** Phase 10 fused
the same frozen hop as a third component (Dense 0.5 / BM25 0.3 / Entity Hop 0.2, fitted on the
7,405 validation questions). On 5,000 held-out HotpotQA train questions, level `hard` (seen by BGE-small in fine-tuning), over the same
5.2 million paragraphs:

| System | Full Support @2,048 |
|---|---:|
| Dense | 58.32 % |
| Dense + BM25 | 62.58 % |
| Dense + BM25 + Entity Hop | 66.50 % (+3.92 pp over Dense + BM25; 306 wins, 110 losses, exact McNemar p = 1.9e-22) |

The mean query latency on the laptop rose 1.33×, with a tail of up to 16.5 s. BGE-small was
fine-tuned on HotpotQA, and Dense reads these train questions 2.6-2.9 points better than the
validation ones. That bias leaves less for a complement to add, so it works against the result.

**6. The hop's order, not only its reach, was holding it back.** Phases 11-13 used the question
to choose which of P1's entities to hop from, as a hard filter, and all three stopped at their dev
gate. Phase 14 kept the seed, the candidate set and the fusion weights, and changed only how the
hop scores its candidates: `(1 − α) · rarity + α · cos(question, candidate)`, both terms min-max
normalized per question, `α = 0.75` fitted on dev. On a second held-out draw of 5,000 `hard` train
questions (`test-11`), Full Support @2,048 was:

| System | Full Support @2,048 |
|---|---:|
| Dense | 57.04 % |
| Dense + BM25 | 61.58 % |
| Dense + BM25 + Entity Hop | 65.70 % (+4.12 pp over Dense + BM25: Phase 10 replicates) |
| Dense + BM25 + relevance-ordered hop (P14) | 70.60 % (+4.90 pp over the Entity Hop system; 317 wins, 72 losses, exact McNemar p = 9.0e-38; +9.02 pp over Dense + BM25) |

The mean query latency on the laptop is 1.82× that of the Entity Hop system. The similarity is
Dense's own signal, and part of the gain may re-weight what Dense already ranks: on dev, half of
the gold paragraphs the new order brought into the context were in Dense's ranks 11-100 and half
beyond its top 100.

**7. The gain is not HotpotQA's alone: it transfers, frozen, to MuSiQue.** HotpotQA built its
bridge questions from first-paragraph hyperlinks, and every weight above was fitted on its dev set.
Phase 15 ran the four systems once, with nothing refitted, on the 2,417 validation questions of
MuSiQue (Wikipedia; questions composed from single-hop ones, filtered against shortcuts, needing
2, 3 or 4 paragraphs) over a corpus of the 101,962 paragraphs of MuSiQue-Ans train and validation.
Full Support @2,048, beside
the same systems on HotpotQA `test-11`:

| System | MuSiQue validation | HotpotQA `test-11` |
|---|---:|---:|
| Dense | 18.04 % | 57.04 % |
| Dense + BM25 | 21.68 % | 61.58 % |
| Dense + BM25 + Entity Hop | 27.68 % (+6.00 pp over Dense + BM25) | 65.70 % |
| P14 | 31.49 % (+9.81 pp over Dense + BM25, 294 wins / 57 losses, p = 1.3e-39; +3.81 pp over the Entity Hop system) | 70.60 % |

Every step adds on both benchmarks; the columns differ in corpus, questions and size, so only the
direction and order of the gains compare. The gain is a two-paragraph gain: 211 of P14's net 237
questions over Dense + BM25 come from the 1,252 questions with two supporting paragraphs. It is
small on three-paragraph questions, and on the 405 four-paragraph questions no system retrieves
the full evidence for more than 2. Refitting the weights on the same questions, an exploratory
figure, adds 45 questions at the best of a 330-point grid. The GLiNER extraction took 10.3 min on one rented
RTX 4090 (0.13 USD attributable).

## What did not work, and what is not established

- **Concepts, in two operationalizations**, add nothing. The first used concepts induced from
  embeddings (Phases 2-4) and the second concepts extracted from the text (Phase 5). The negative
  result is bounded to those two operationalizations on this benchmark.
- **At FullWiki scale, BM25 is the slightly better complement to Dense at the primary budget**
  (61.30 % against 60.09 %, p = 0.024, descriptive). The Entity Hop is ahead at 4,096 tokens and
  at pool scale. The two signals are not redundant: at scale, 371 questions go the Entity Hop's way
  and 436 go BM25's. **The frozen Entity Hop is therefore not, as it stands, a better
  replacement for BM25 over the full search space.** Added beside it, it is (Phase 10, point 5).
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

What the next line had to show was concrete: **a system that is better than Dense + BM25 at
FullWiki scale.** Phase 10 shows one: the unchanged entity signal, fused as a third component,
beats Dense + BM25 by 3.92 points on held-out train questions. From step 2 on, every new system is reported
against both: Dense + BM25 + Entity Hop, the project's own bar, which decides whether a new use of
the entities is kept; and Dense + BM25, the reference the literature understands. The 7,405 validation
questions are now dev.

Phase 14 then moved the bar: the relevance-ordered hop (P14) beats that system by 4.90 points on a
second held-out set, which is now spent. P14 is the line's best system. Phase 15 then checked that
the gain was not only an artifact of how HotpotQA was built: frozen, P14 and the Entity Hop system still
beat Dense + BM25 on MuSiQue, on two-paragraph questions. That is a second Wikipedia benchmark, a
small corpus and a one-hop design against 2-4 hop questions: evidence, not proof, of generality.
Next, by the author's decision, Phase 16 runs the same frozen systems on a corpus that is not
Wikipedia, and Phase 17 tests an untrained second-hop query, the vector of "question + P1",
against Dense ([`plans/0_master_plan.md`](plans/0_master_plan.md)).

## Reproducibility

Each phase document names its artifacts, digests, model revisions and commands. The rented-GPU
procedure is in [`plans/phase_9/9.runpod_recipe.md`](plans/phase_9/9.runpod_recipe.md). Security
findings of every audited phase are in [`security/README.md`](security/README.md). Deferred
directions and their rationale are in [`plans/research_roadmap.md`](plans/research_roadmap.md).
