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

Phase 1 of 6 — not started. See [`docs/plans/0_master_plan.md`](docs/plans/0_master_plan.md).

## Requirements

- Python 3.12 (not 3.13: the numerical stack wheels are pinned to 3.12)
- [uv](https://docs.astral.sh/uv/) as dependency manager

## Getting started

```bash
uv sync          # creates .venv and installs dependencies
uv run pytest    # runs the tests
uv run ruff check .
```

## Platform note

Development happens on **Windows on ARM (ARM64)**. `torch` publishes no `win_arm64` wheels on PyPI,
so `pyproject.toml` declares the official PyTorch index as an explicit source for that package. For
the same reason this project uses neither `umap-learn`, `hdbscan`, `bertopic` nor `datasets`: none
of them has a wheel for this platform.
