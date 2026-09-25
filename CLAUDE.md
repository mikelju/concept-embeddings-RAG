# CLAUDE.md — concept-embeddings-RAG

Loaded into every session, so every line here costs tokens on every request. Keep it to rules that
apply to almost any task. State, figures and history live in the documents below, each in one
place. Conditional knowledge lives in skills. Add a lesson here only when a mistake would repeat
without it, in one line.

## What this is

A research experiment in multi-hop retrieval, not an application. The deliverable is measured
evidence: reproducible retrievers, comparable baselines, explicit limitations, negative results
included. No product, API, server or UI is built.

Working question: what is the minimum structural signal Dense retrieval needs to recover
multi-hop evidence that semantic similarity alone does not express?

What is established (figures and full records in each phase's `X.results.md`):

- the pooled-embedding concept-space line (Phases 2-4) is closed with a bounded negative result;
  do not read it as evidence against text-derived entities;
- a query-blind, one-hop Entity Hop from the first Dense paragraph complements Dense (Phase 6);
- GLiNER keeps most of that gain for a fraction of Claude's extraction cost (Phase 7);
- BGE-small stayed ahead of Qwen3-Embedding-0.6B on dev at every corpus size up to 500k
  paragraphs (Phase 8 and deviation 8.1). No substantially stronger Dense has been tested yet.
- the Entity Hop keeps its gain over Dense at full FullWiki scale (+4.48 pp), where Dense + BM25
  is slightly ahead of it (Phase 9);
- added as a third component, the Entity Hop lifts Dense + BM25 at FullWiki scale (+3.92 pp on
  5,000 train questions, Phase 10). The 7,405 validation questions are now dev.
- a DF cap on P1's entities or a question-seeded hop does not beat that on dev (Phase 11,
  `DEV_STOP`). The bottleneck is choosing which of P1's entities to follow.

**Current state (2026-09-25):** Phase 11 (a better use of the entities) closed with `DEV_STOP`
(`phase_11/11.results.md`): P10-C stays the best system and `test-11` stays unopened. The next
phase is the author's decision. If this line and the master plan disagree, the master plan wins; fix this line.

## Where things live

| Need | Source |
|---|---|
| Status of every phase, the plan ahead | `docs/plans/0_master_plan.md` ("Phase status") |
| What the whole line established, in one page | `docs/research_summary.md` |
| Deferred directions and their rationale | `docs/plans/research_roadmap.md` |
| What a phase measured | `docs/plans/phase_X/X.results.md` |
| Active spec and plan | `docs/plans/phase_X/`: `X.spec.md`, plan `X.0_*.md`, deviations `X.Y_*.md` (a deviation may carry its own plan, results and recipe) |
| Original hypothesis and literature (Spanish, stays Spanish) | `docs/refs/descripcion-proyecto.md`, `docs/refs/bibliografia.md` |
| Plain-Spanish glossary for the author | `docs/GLOSARIO.md` |
| Dev/test discipline, selection, integrity, simplification rule | skill `research-protocol` |
| Rented GPU / ARM64 limits | skill `remote-gpu` |
| Validating finished work and opening its PR | skill `deliver` |

`docs/USER_GUIDE.md` describes an older pipeline and is not authoritative for Phases 5-9.

## How work flows

The author's attention goes to the two ends: the spec before and the PR after. The middle is
autonomous.

1. Anything that can change a measured result starts from a spec the author approves
   (`/4-especificar`, then `/5-planear`). Once approved, a spec is frozen.
2. Once the author approves the plan, `/6-implementar` runs it whole without step-by-step
   confirmation, on a working branch, committing each step. Commits and pushes to non-`main`
   branches are a standing authorization from the author (2026-09-23) for this repository, which
   is not a public-zone replica in the root zone map; the root rules on private data still apply.
3. The `deliver` skill closes the work: checks, end-to-end evidence, adversarial review in a
   fresh context, doc pass, PR with a risk assessment.
4. **The PR waits for the author. Never push to `main`, never merge, never force-push.**
5. Stop and ask only for ASK FIRST items, a problem that would change the frozen spec, a step that
   opens the test split or spends money, a reproduction check that misses its recorded figure, a
   spec gate or stop state being reached, or two attempts without progress.

Small fixes and doc changes need no spec: branch, fix, `deliver`.

## Commands

```bash
uv sync
uv run pytest            # first run ~35 s: torch loads, it is not a hang
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run cer <stage>       # fetch build embed evaluate induce label select expand pilot extract
                         # nodes navigate replace-check|freeze|test cheap-extract cheap-eval
                         # strong-embed strong-dense scale-corpus scale-repro scale-run scale-outcome
npx -y gh-axi@0.1.35 pr view|list|checks, run list|view   # only these; writes use gh
```

Do not rerun one-shot historical stages to inspect them: their versioned artifacts are the record.

## Architecture in brief

`src/concept_embeddings_rag/`: `cli.py` (stages), `config.py` (every constant that decides what an
experiment measures), `corpus/`, `embeddings/`, `concepts/`, `nodes/` (entity extraction and
index), `retrieval/` (`dense`, `bm25`, `fusion`, `entity_hop`, historical `conceptual`,
`diffusion`), `evaluation/` (harness, budget, per-phase runners).

- **Indexing unit:** the whole article paragraph. Never fixed-size windows in this research line.
- **Caches:** embeddings and other expensive deterministic outputs live on disk and are reused. A
  different model or configuration writes a distinct cache; it never overwrites one behind a result.
- **Shared harness:** a new retriever, extractor or Dense backend plugs into the existing
  evaluation harness; generalize only when an experiment requires it.
- **Entity Hop (intentionally minimal):** D(q) = Dense ranking; read(q) = first 10 units;
  p1(q) = first unit. Candidates: paragraphs outside read(q) sharing >= 1 raw entity node with
  p1(q). Score: sum of `log(1 + N / (1 + df))` over shared nodes. Rank by score, then unit id. No
  padding. Fused with Dense (weight fitted on dev). No canonicalization, relations, query
  relevance, multiple seeds or hops unless an approved spec asks for them.
- Benchmark: frozen HotpotQA-derived set, 600 dev / 1,400 test, 19,366-paragraph pool. Dense:
  `BAAI/bge-small-en-v1.5` (384 d, L2-normalized). Metric of record: Full Support @2,048 tokens.
  Entities: GLiNER `urchade/gliner_medium-v2.1` rev `40ec419335d09393f298636f471328b722c6da9e`, the Phase 7 index, read-only.

## Rules

**Always:** read the active spec/plan before changing implementation; reuse existing code and
artifacts; keep explicit seeds and deterministic tie-breaking; record model, version and
configuration behind every meaningful metric; keep dev/test separation; run the checks before
calling work done.

**Ask first:** adding or changing dependencies; changing the embedding model outside a phase built
for it; changing the indexing unit; deleting caches; invalidating or modifying a versioned
artifact or a recorded experimental parameter; changing the scope of the active phase; adding a
new representation or graph mechanism the spec does not request.

**Never:** tune or choose anything by looking at test results; run a candidate on test that dev
did not select, or retry a failed gate with another model inside the same phase, even
"descriptively"; rewrite historical results or
artifacts (provenance problems get a prospective fix in the producer); report estimates or
projections as measurements; invent missing data; read, print or commit `.env` or credentials;
commit the raw corpus; silently expand a phase into canonicalization, relations, more seeds or
hops, or GraphRAG; add complexity because a paper uses it; put an LLM in the online retrieval
loop except as a declared variant.

When choosing between technical options, do not weigh implementation effort heavily: agents make
code cheap, and a weak experimental design is expensive. Reproduce a bug through the real CLI
stage or artifact before fixing it.

## Conventions

- English in all project-produced code, docs, comments and commit messages. The two source
  documents in `docs/refs/` stay in Spanish. The framework commands in `.claude/commands/` are in
  Spanish and use the English project paths; templates in `docs/templates/` keep Spanish filenames.
- Type hints on functions; `mypy src` and `ruff check .` pass.
- ASCII only in Python console and logging strings (`[OK]`, `->`); Markdown may use Unicode.
- Readable research code over framework abstractions.
- The repository is **public**: whatever is committed or pushed is published.

## Gotchas

- **Windows ARM64 laptop.** Some packages have no `win_arm64` wheel (spaCy); GLiNER is far too
  slow on CPU. Heavy or GPU work runs on a rented machine: skill `remote-gpu`. Never distort an
  experiment to fit the laptop.
- **Torch source.** `pyproject.toml` selects `pytorch-cu126` on linux/x86_64 and `pytorch-cpu`
  elsewhere. Respect that uv source configuration.
- **Python** is pinned `>=3.12,<3.13`.
- **`question_cache_key` does not include the corpus.** A run over a different corpus must use its
  own cache directory or it overwrites the question cache behind a published result.
- **`TokenCounter.count_units` tokenizes its input in one call.** Batch it for large corpora.
- **Secrets:** only `ANTHROPIC_API_KEY`, used by historical labelling and Phase 5 extraction; it
  lives in `.env`, loaded at runtime.
