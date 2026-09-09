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

**The deliverable of this project is measured evidence, not an application**: four comparable
systems (BM25, dense, conceptual, iterative) evaluated on a multi-hop benchmark with annotated
ground truth, and an explicit verdict on the hypothesis.

## Status

**Phase 2 of 6 — built, verified and audited.** Two things exist: the measuring instrument, and the
concept space it will judge.

**Phase 1 — the instrument.** A frozen corpus of 19,366 paragraphs, dense and BM25 baselines, and an
evaluation harness that compares systems at a fixed context budget rather than at a fixed number of
chunks. The numbers, and an honest reading of them, are in
[`docs/plans/phase_1/1.results.md`](docs/plans/phase_1/1.results.md). In one line: dense beats BM25
at every budget (0.871 vs 0.822 Full Support at 4,096 tokens on test), and **12% of questions
defeat both** — the system finds the entity the question names and misses the bridge entity it does
not. That failure mode is exactly what Phase 4 proposes to fix, and whether it does is the
experiment.

**Phase 2 — the concept space.** Four spaces were induced over that same pool by non-negative sparse
coding (K = 512, 1,024, 2,048, 4,096) and **none of them is chosen here**: the choice of K belongs to
Phase 3 and is made against development recall, never against a number on the results page. All four
land inside the 8-16 active-concepts-per-unit band, so the one-hot regime that disqualified
clustering never appears, and per-concept coherence sits far above the random-unit null measured on
this same pool (0.4292 ± 0.0073) — the worst concept at K = 512 is still 3 standard deviations above
chance.

The finding that matters is one no coherence score catches. From K = 2,048 a single atom activates
66.7% of the corpus, and at K = 4,096 it reaches 99.4% while passing every coherence check with a
score above its own dictionary's mean. Only the over-generality diagnostic sees it. Diffusion travels
along shared concepts, so an atom shared by essentially every unit is an edge from everything to
everything — a warning Phase 3 has to carry into its choice of K rather than discover afterwards.
The full reading, including what these numbers are *not*, is in
[`docs/plans/phase_2/2.results.md`](docs/plans/phase_2/2.results.md).

Concept labelling (interpretability only — retrieval never reads a name) cost **11.73 USD**, 45%
above the forecast, and the discrepancy is argued rather than rounded away. The phase's security
audit closed with no Critical and no High finding; every finding is catalogued in
[`docs/security/README.md`](docs/security/README.md).

Roadmap: [`docs/plans/0_master_plan.md`](docs/plans/0_master_plan.md).

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
uv run cer fetch      # download and freeze the benchmark   (~4 min)
uv run cer build      # select the subset, build the pool   (~9 s)
uv run cer embed      # embed the corpus, cached on disk    (~36 min, once)
uv run cer evaluate   # measure dense and BM25              (~1 min)
uv run cer induce     # induce one concept space per K      (~3.7 h for the sweep)
uv run cer label      # name the concepts, for the report   (needs ANTHROPIC_API_KEY)
```

`induce` and `label` are Phase 2 and are not needed to reproduce the baselines. `label` is the only
stage that calls a paid API, is cached per concept, and produces nothing retrieval depends on.

Each stage refuses to run if the previous one has not, and says which to run first.
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) covers the stages, the results and what to do when
something stops.

## Platform note

Development happens on **Windows on ARM (ARM64)**. `torch` publishes no `win_arm64` wheels on PyPI,
so `pyproject.toml` declares the official PyTorch index as an explicit source for that package. For
the same reason this project uses neither `umap-learn`, `hdbscan`, `bertopic` nor `datasets`: none
of them has a wheel for this platform.
