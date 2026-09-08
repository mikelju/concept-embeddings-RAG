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

**Phase 1 of 6 — complete.** The measuring instrument exists: a frozen corpus of 19,366 paragraphs,
dense and BM25 baselines, and an evaluation harness that compares systems at a fixed context budget
rather than at a fixed number of chunks.

The baseline numbers, and an honest reading of them, are in
[`docs/plans/phase_1/1.results.md`](docs/plans/phase_1/1.results.md). In one line: dense beats BM25
at every budget (0.871 vs 0.822 Full Support at 4,096 tokens on test), and **12% of questions
defeat both** — the system finds the entity the question names and misses the bridge entity it does
not. That failure mode is exactly what Phase 4 proposes to fix, and whether it does is the
experiment.

Phase 2 (the concept space) has not started. Roadmap:
[`docs/plans/0_master_plan.md`](docs/plans/0_master_plan.md).

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
```

Each stage refuses to run if the previous one has not, and says which to run first.
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) covers the stages, the results and what to do when
something stops.

## Platform note

Development happens on **Windows on ARM (ARM64)**. `torch` publishes no `win_arm64` wheels on PyPI,
so `pyproject.toml` declares the official PyTorch index as an explicit source for that package. For
the same reason this project uses neither `umap-learn`, `hdbscan`, `bertopic` nor `datasets`: none
of them has a wheel for this platform.
