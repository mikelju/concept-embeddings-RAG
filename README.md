# concept-embeddings-RAG

A research experiment on **retrieval over a corpus-induced concept space**.

Instead of representing each chunk only as an opaque dense embedding, a concept dictionary is
induced from the corpus itself and every chunk becomes a sparse vector over those interpretable
dimensions (`X = chunks × concepts`). Retrieval does not stop at the top-K for the question: it
discovers satellite concepts in the chunks already retrieved and expands the search iteratively,
conditioned on the query.

The full hypothesis, the surveyed prior art and the experimental design are in
[`docs/refs/descripcion-proyecto.md`](docs/refs/descripcion-proyecto.md) and
[`docs/refs/bibliografia.md`](docs/refs/bibliografia.md) (both in Spanish; they are source material).

> That was the starting hypothesis, and it did not hold. The work then narrowed to the question
> the later phases answer: what is the minimum structural signal Dense retrieval needs for
> multi-hop evidence? See [`docs/research_summary.md`](docs/research_summary.md).

**The deliverable of this project is measured evidence, not an application**: comparable
systems (BM25, dense, conceptual, iterative) evaluated on a multi-hop benchmark with annotated
ground truth, and an explicit verdict on the hypothesis.

## Status

**The planned research line (Phases 1-9) is complete.** The summary, with what was established,
what was not and where it leaves the work, is
[`docs/research_summary.md`](docs/research_summary.md).

In short, the concept-space idea failed (Phases 2-5), but a minimal entity signal worked. The
Entity Hop is a query-blind, one-hop step from the first Dense paragraph to paragraphs sharing a
named entity with it, using entities from a cheap local extractor (GLiNER). Full Support at 2,048
context tokens:

| System | Pool, 1,400 test q. (19,366 paragraphs) | FullWiki, 5,405 unseen q. (5.23M paragraphs) |
|---|---:|---:|
| Dense (BGE-small) | 0.8250 | 0.5562 |
| Dense + BM25 | 0.8643 | **0.6130** |
| Dense + Entity Hop (GLiNER) | **0.8793** | 0.6009 |

The Entity Hop's gain over Dense survives full scale (+4.48 points, p = 8.0e-28). Over the full
search space, though, Dense + BM25 is 1.2 points ahead. The whole FullWiki entity extraction cost
4.57 USD of rented GPU time.

## The phases

**Phase 1 — the instrument.** A frozen corpus of 19,366 paragraphs, dense and BM25 baselines, and an
evaluation harness that compares systems at a fixed context budget rather than at a fixed number of
chunks. Dense beats BM25 at every budget, and 12% of dev questions defeat both.
[`1.results.md`](docs/plans/phase_1/1.results.md)

**Phase 2 — the concept space.** Four spaces induced over that same pool by non-negative sparse
coding (K = 512, 1,024, 2,048, 4,096), with structural and coherence diagnostics. At K = 4,096 one
atom activates 99.4% of the corpus while passing every coherence check.
[`2.results.md`](docs/plans/phase_2/2.results.md)

**Phase 3 — retrieving through concepts (System B).** K = 512 chosen on dev; the fusion weight fitted
on dev put all of it on dense, while the same fitting procedure gave the dense+BM25 control +3.9
points. [`3.results.md`](docs/plans/phase_3/3.results.md)

**Phase 4 — expanding by diffusion (System C).** A re-ranker of dense's top-100 that ties dense and
loses to the control at every budget, for arithmetic reasons the report measures.
[`4.results.md`](docs/plans/phase_4/4.results.md)

**Phase 5 — reading entities and concepts from the text.** Entities navigate to the missing
paragraph; text-derived concepts do not. [`5.results.md`](docs/plans/phase_5/5.results.md)

**Phase 6 — Dense + Entity Hop.** The Entity Hop can replace BM25 in the Dense hybrid. Held out, it
reaches 0.8900 against 0.8643. [`6.results.md`](docs/plans/phase_6/6.results.md)

**Phase 7 — cheap extraction.** GLiNER keeps 83.5% of the gain the LLM extraction bought, for
about a thousandth of the cost. [`7.results.md`](docs/plans/phase_7/7.results.md)

**Phase 8 / 8.1 — a stronger Dense.** The candidate was weaker than BGE-small at every corpus size
up to 500k, so the phase stopped at its dev gate. [`8.results.md`](docs/plans/phase_8/8.results.md),
[`8.1_results.md`](docs/plans/phase_8/8.1_results.md)

**Phase 9 — HotpotQA FullWiki.** Over 5.23M paragraphs the gain over Dense survives
(`SCALE_SUPPORTED`), and Dense + BM25 is slightly ahead. [`9.results.md`](docs/plans/phase_9/9.results.md)

Security findings of every audited phase are catalogued in
[`docs/security/README.md`](docs/security/README.md).

## Requirements

- Python 3.12 (not 3.13: the numerical stack wheels are pinned to 3.12)
- [uv](https://docs.astral.sh/uv/) as dependency manager

## Getting started

```bash
uv sync          # creates .venv and installs dependencies
uv run pytest    # runs the tests (the first run takes ~35 s while torch loads)
uv run ruff check .
```

Then run the experiment, in order:

```bash
uv run cer fetch      # download and freeze the benchmark        (~4 min)
uv run cer build      # select the subset, build the pool        (~9 s)
uv run cer embed      # embed the corpus, cached on disk         (~36 min, once)
uv run cer induce     # induce one concept space per K           (hours for the sweep)
uv run cer label      # name the concepts, for the report        (optional, needs ANTHROPIC_API_KEY)
uv run cer select     # choose the space on dev and freeze it    (~3 min of sweep)
uv run cer evaluate   # measure the five Phase 1-3 systems on dev and test   (~1 min)
uv run cer expand     # sweep the expansion grid on dev and freeze one cell
uv run cer evaluate --systems expansion,expansion-conceptual   # measure System C
```

**Both decisions ship with the repository**: `data/selection/selection.json` and
`data/expansion/expansion.json` are versioned, `select` and `expand` never overwrite them, and
`evaluate` refuses to read test without them. On a fresh clone, skip those two stages — `fetch`,
`build`, `embed`, `induce` and the two `evaluate` calls reproduce the recorded numbers.

`label` is the only stage that calls a paid API, is cached per concept, and produces nothing
retrieval depends on. Each stage refuses to run if the previous one has not, and says which to run
first.

The closure's second-hop diagnostic, exploratory and dev only, regenerates with
`uv run python scripts/second_hop_diagnostic.py`.

[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) covers every stage, the results, and what to do when
something stops.

## Platform note

Development happens on **Windows on ARM (ARM64)**. `torch` publishes no `win_arm64` wheels on PyPI,
so `pyproject.toml` declares the official PyTorch index as an explicit source for that package. For
the same reason this project uses neither `umap-learn`, `hdbscan`, `bertopic` nor `datasets`: none
of them has a wheel for this platform.
