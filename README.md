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

**The deliverable of this project is measured evidence, not an application**: comparable
systems (BM25, dense, conceptual, iterative) evaluated on a multi-hop benchmark with annotated
ground truth, and an explicit verdict on the hypothesis.

## Status

**Current research line — closed with a bounded negative result.** The tested formulation was a
concept space induced by sparse dictionary learning over pooled dense paragraph embeddings, evaluated
with conceptual retrieval, dense+concept fusion, and diffusion-based iterative expansion on HotpotQA.
The evidence does **not** justify claiming that the broader hypothesis about text-derived interpretable
concepts is disproven.

The consolidated closure is in
[`docs/plans/phase_5/5.results.md`](docs/plans/phase_5/5.results.md), with its specification and
closure plan in `docs/plans/phase_5/5.spec.md` and `docs/plans/phase_5/5.0_research_closure.md`.

At 2,048 context tokens on 1,400 test questions, the reported Full Support values are: concepts
0.355, BM25 0.759, dense 0.825, dense+concepts 0.825, diffusion expansion 0.821, and dense+BM25
0.864. The principal Phase 4 failure was mechanical: the tested diffusion configuration did not
promote previously unseen paragraphs beyond the 100-item seed set.

**Important repository note.** The closure source supplied on 2026-09-15 references later Phase 4
result commits and artifacts that are not present in the visible `phase-4-iterative-expansion` Git
history used as this branch's base. The closure records that synchronization gap explicitly rather
than claiming fresh-clone reproducibility for those later results.

## Earlier phases

**Phase 1 — the instrument.** A frozen corpus of 19,366 paragraphs, dense and BM25 baselines, and an
evaluation harness that compares systems at a fixed context budget rather than at a fixed number of
chunks. The phase report is in [`docs/plans/phase_1/1.results.md`](docs/plans/phase_1/1.results.md).

**Phase 2 — the concept space.** Four spaces were induced over that same pool by non-negative sparse
coding (K = 512, 1,024, 2,048, 4,096), with structural and coherence diagnostics. The phase report is
in [`docs/plans/phase_2/2.results.md`](docs/plans/phase_2/2.results.md).

Roadmap and historical phase decisions remain in [`docs/plans/0_master_plan.md`](docs/plans/0_master_plan.md).

## Requirements

- Python 3.12 (not 3.13: the numerical stack wheels are pinned to 3.12)
- [uv](https://docs.astral.sh/uv/) as dependency manager

## Getting started

```bash
uv sync          # creates .venv and installs dependencies
uv run pytest    # runs the tests (the first run takes ~35 s while torch loads)
uv run ruff check .
```

The original experiment stages are:

```bash
uv run cer fetch      # download and freeze the benchmark
uv run cer build      # select the subset, build the pool
uv run cer embed      # embed the corpus, cached on disk
uv run cer evaluate   # measure dense and BM25
uv run cer induce     # induce the concept spaces
uv run cer label      # label concepts for interpretation only
```

The closure documents what those stages established and what remains open; it does not add a new
retrieval mechanism.

## Platform note

Development happens on **Windows on ARM (ARM64)**. `torch` publishes no `win_arm64` wheels on PyPI,
so `pyproject.toml` declares the official PyTorch index as an explicit source for that package. For
the same reason this project uses neither `umap-learn`, `hdbscan`, `bertopic` nor `datasets`: none
of them has a wheel for this platform.
