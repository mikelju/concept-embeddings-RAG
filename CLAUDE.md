# CLAUDE.md — concept-embeddings-RAG

Project rules and current research context for the agent.

**Always read this file before making changes.**

---

## What is this project?

This is a **research experiment in multi-hop retrieval**, not an application.

The project started from the hypothesis that a document corpus could be represented in a sparse,
interpretable concept space and navigated through that space to recover evidence that conventional
dense retrieval misses.

That original research line was tested and closed with a bounded negative result.

A second research line then emerged from the evidence:

> **Dense similarity and entity-based structural relatedness can provide complementary retrieval
> signals.**

The current working question is:

> **What is the minimum structural signal that Dense retrieval needs in order to recover multi-hop
> evidence that semantic similarity alone does not express?**

The deliverable is **measured evidence**: reproducible retrieval systems, comparable baselines,
explicit limitations, positive or negative findings, and enough provenance to understand what each
number means.

No product, API, server or UI is being built.

The full historical and future roadmap is in:

- `docs/plans/0_master_plan.md`
- `docs/plans/research_roadmap.md`

The original hypothesis and literature notes live in:

- `docs/refs/descripcion-proyecto.md`
- `docs/refs/bibliografia.md`

Those two source documents are in Spanish and stay that way. Project-produced code, plans, reports,
comments and commit messages are written in English.

---

# Current research state

## Phase 1 — corpus and baselines

Complete.

Built:

- a frozen HotpotQA-derived benchmark;
- 2,000 questions;
- 600 dev / 1,400 test;
- one unified searchable pool of **19,366 whole paragraphs**;
- Dense retrieval;
- BM25;
- the fixed-context-budget evaluation harness.

Embedding model:

```text
BAAI/bge-small-en-v1.5
384 dimensions
L2-normalized
```

Headline test result at 2,048 tokens:

```text
Dense Full Support = 0.8250
```

The key failure pattern was already visible here:

```text
question
→ Dense finds a relevant first paragraph
→ that paragraph reveals a bridge entity not present in the question
→ Dense often misses the second required paragraph
```

---

## Phase 2 — pooled-embedding concept space

Complete.

Four sparse concept spaces were induced with:

```text
MiniBatchDictionaryLearning
positive_code=True
positive_dict=False
```

Dictionary sizes:

```text
512
1,024
2,048
4,096
```

The spaces were structurally measurable and semantically coherent enough to inspect, but larger
spaces developed problematic hubs and thin concepts.

This phase did **not** establish retrieval usefulness.

---

## Phase 3 — conceptual retrieval and fusion

Complete.

The selected concept configuration was:

```text
K = 512
view = raw
query operator = projection_full
damping = idf
```

The decisive result was negative:

```text
Dense + Concept fitted weight:
dense = 1.0
concept = 0.0
```

The concept signal added no measurable value to Dense.

The control, built through the same fusion machinery, did:

```text
Dense                  Full Support @2048 test = 0.8250
Dense + BM25           Full Support @2048 test = 0.8643
```

That established that complementary signal existed; the pooled concept representation simply did not
provide it.

---

## Phase 4 — iterative diffusion over the concept space

Complete — negative result.

System C diffused activation through:

```text
chunk → concept → chunk
```

with partial restart on the query.

Headline test result:

```text
Dense                  0.8250
Dense + BM25           0.8643
Concept diffusion      0.8214
```

The failure mechanism was measured directly:

- diffusion reached the corpus mathematically;
- at usable restart values it never promoted a new non-seed paragraph;
- it effectively reranked Dense's own top-100;
- `restart = 0` removed query anchoring and collapsed to `0.0117` Full Support.

This closed the **pooled-embedding concept research line**.

See:

`docs/plans/phase_4/4.1_research_line_closure.md`

Do not reinterpret this result as evidence against text-derived entities or every possible
concept-based representation.

---

## Phase 5 — text-derived entities and concepts

Complete — **NAMES ONLY**.

Entities and concepts were extracted directly from all **19,366 paragraphs** using
`claude-sonnet-5`.

Recorded extraction cost:

```text
25.98 USD
```

Final node counts:

```text
96,416 entity nodes
40,846 concept nodes
137,262 total nodes
```

The navigation pilot used 152 dev questions where Dense left supporting evidence outside top-10.

Second-hop hit@10 per question:

```text
Continue Dense                  0.3026
BM25 on question                0.4441
BM25 on p1 text                 0.4507
Pooled-embedding concept hop    0.0789
Entity Hop                      0.6579
Text-derived concept hop        0.0592
Entity + Concept                0.4803
```

The gate returned:

```text
NAMES_ONLY
```

Important interpretation:

- entities produced a strong second-hop signal;
- concepts did not;
- adding concepts to entities degraded shallow ranking.

Reachability diagnosis:

```text
158 missing gold paragraphs
118 share >=1 entity with p1
116 of those 118 are found by Entity Hop by depth 100
```

Once a bridge is expressible by a shared raw entity, the current simple hop nearly saturates what it
can reach.

---

## Phase 6 — Dense + Entity Hop end to end

Complete — **ENTITY_REPLACEMENT_SUPPORTED**.

Question:

> Can Entity Hop replace the BM25 component of the existing Dense + BM25 hybrid without reducing
> retrieval quality?

Systems:

```text
Dense

A = Dense + BM25
    weights: dense 0.5 / bm25 0.5

B = Dense + Entity Hop
    weights selected on dev:
    dense 0.7 / entity-hop 0.3
```

The Entity Hop itself remained deliberately simple:

```text
Question
  ↓
Dense
  ↓
top-100
  ↓
read top-10
  ↓
p1 = first Dense paragraph
  ↓
entities occurring in p1
  ↓
other corpus paragraphs sharing those entity forms
  ↓
rarity-weighted score
  ↓
one Entity Hop
  ↓
fusion with Dense
```

Entity score uses rarity:

```text
log(1 + N / (1 + df))
```

The hop is **query-blind after Dense produces p1**.

It does not use:

- canonicalization;
- entity linking;
- relation triples;
- attributes;
- PageRank;
- an LLM at retrieval time;
- query reformulation;
- multiple seeds;
- multiple hops;
- learned entity traversal.

Headline test result, 1,400 questions, Full Support @2,048:

```text
Dense                    0.8250 = 1,155 / 1,400
Dense + BM25             0.8643 = 1,210 / 1,400
Dense + Entity Hop       0.8900 = 1,246 / 1,400
```

Observed improvement:

```text
Entity vs Dense          +6.50 percentage points
Entity vs Dense+BM25     +2.57 percentage points
```

The second number is descriptive. Phase 6 was preregistered as a replacement/non-inferiority
experiment, not as a superiority experiment.

Decision tests:

```text
S: B beats Dense
103 wins / 12 losses
p = 1.676e-19
PASS

V: A beats Dense
85 wins / 30 losses
p = 1.430e-07
PASS

N: B non-inferior to A
b = 96
c = 60
lower bound = +0.011135
margin boundary = -0.019643
PASS
```

Final state:

```text
ENTITY_REPLACEMENT_SUPPORTED
```

Mechanism diagnosis:

- all 103 questions where B succeeds and Dense fails contain at least one Entity-Hop-introduced unit;
- 67 depend entirely on entity-introduced units rather than only reranking Dense;
- 85 of the 96 B-only wins over A contain entity-introduced units.

Measured retrieval latency:

```text
Dense                    0.891 ms/query
Dense + BM25             1.508 ms/query
Dense + Entity Hop      10.447 ms/query
```

These are **milliseconds**, not seconds.

The main scaling concern is therefore not current online traversal latency. It is the offline cost of
extracting entities from millions of paragraphs.

Full Phase 6 result:

`docs/plans/phase_6/6.results.md`

---

## Phase 7 — cheap entity extraction

Complete — **GLINER SELECTED** (`data/phase7/selection.json` → `selected: "gliner"`).

Question:

> Can the expensive Claude-based entity extraction be replaced by a cheap/local extractor without
> materially losing the Entity Hop retrieval gain?

Two local candidates were extracted over the same 19,366-paragraph pool, both measured on dev, a
preregistered rule selected one, and that one was measured once on test. Nothing else changed: same
pool, same `BAAI/bge-small-en-v1.5`, same P1-only, one-hop, query-blind Entity Hop. Only the reader
of the paragraphs changed.

Selected extractor:

```text
urchade/gliner_medium-v2.1
revision 40ec419335d09393f298636f471328b722c6da9e
5 labels
threshold 0.5
batch size 8
config digest 2f7864661b8ce7ff
```

Headline held-out result, 1,400 questions, Full Support @2,048:

```text
Dense                           0.8250 = 1,155 / 1,400
Dense + BM25                    0.8643 = 1,210 / 1,400
Dense + Entity Hop (Claude)     0.8900 = 1,246 / 1,400
Dense + Entity Hop (GLiNER)     0.8793 = 1,231 / 1,400
```

GLiNER retains **83.5%** of the Claude Entity Hop's held-out gain over Dense: 76 of the 91 questions
Claude added. It beats the Dense + BM25 control by 21 questions and sits 15 questions below Claude.

Extraction, measured on a rented RTX 4090 (Linux x86_64):

```text
19,366 paragraphs in 119.4 s
162.178 paragraphs/s
0.03 USD attributable compute
```

against Claude's measured 25.9848 USD for the same 19,366 paragraphs.

Projection to 5M paragraphs, linear from the measured throughput:

```text
8.56 h / 6.34 USD
```

That pair is a **projection, never a measurement**, and must be labelled as such wherever it is
quoted.

### The secondary finding, which must not be lost

**spaCy (`en_core_web_sm` 3.8.0) measured better than GLiNER on dev**: 525 / 600 (0.8750) against
GLiNER's 518 / 600 (0.8633), retaining 95.0% of Claude's dev gain against GLiNER's 77.5%. spaCy was
not selected, and the held-out split was **never opened for it**. Three reasons:

- the preregistered rule ranks on projected FullWiki cost and time among the candidates clearing
  both bars, and GLiNER's 6.34 USD / 8.56 h beat spaCy's 11.08 USD / 14.97 h;
- overriding that rule *after observing dev* would invalidate the only thing that makes the held-out
  figure mean anything;
- spaCy buys its quality with a much denser, less scalable representation: 11 labels against 5, a
  median of 493 hop candidates against 40, a largest hub of 4,085 paragraphs against 1,186.

On dev, GLiNER tied the Dense + BM25 line exactly (518 = 518); on the held-out split it beat that
line by 21 questions. Dev was the more pessimistic of the two readings.

spaCy is **not discarded**. It remains a candidate for a later scale-and-representation experiment,
recorded under deferred work — not as an active phase.

Full Phase 7 result:

`docs/plans/phase_7/7.results.md`

---

# Next phases

Two sequential questions remain in the immediate roadmap.

## Phase 8 — Strong Dense

**STOP at dev gate — premise failed. The question below is still open.**

`Qwen/Qwen3-Embedding-0.6B` (revision `97b0c614…`, 1024 dim) scored **446 / 600** dev Full Support
@2,048 against the inherited BGE-small Dense reading of **487 / 600**; the pre-declared rule required
**488**. The model chosen a priori as substantially stronger is weaker on this corpus, so:

- the **held-out split was neither encoded nor measured** — `embedding-test.json` and `test.json` do
  not exist, by design;
- **no second encoder was tried**, because choosing a model after seeing dev is the leaderboard the
  master plan forbids;
- Phase 8 reaches **none** of its three planned outcomes, and Phase 9 inherits **no Dense model**.

On dev only, and descriptively: BM25 recovered +43 questions over Qwen Dense (489 / 600) and Entity
Hop +48 (494 / 600), Entity Hop +5 over BM25. Neither approached the inherited BGE-small hybrids at
518 / 600. Full record: `docs/plans/phase_8/8.results.md`.

**Deviation 8.1 is opened for corpus-scale sensitivity**: the ordering is bounded to the frozen
19,366-paragraph HotpotQA distractor pool, and nothing measured says it survives a change of corpus
scale or shape. The specification that follows is the phase as planned.

Question:

> Does Entity Hop still provide complementary signal when Dense itself is substantially stronger?

Stay on the same 19,366-paragraph corpus.

Choose **one** modern strong open Dense model. Do not create a model leaderboard.

Compare:

```text
Strong Dense
Strong Dense + BM25
Strong Dense + Entity Hop
```

Use the entity extractor Phase 7 selected: **GLiNER**, `urchade/gliner_medium-v2.1` at revision
`40ec419335d09393f298636f471328b722c6da9e`.

> **Phase 8 inherits GLiNER and changes only the Dense retriever.** Changing the Dense model and the
> extractor simultaneously would make any improvement or degradation unattributable to either.

That is also the answer to "why not spaCy, it scored better on dev?": see the Phase 7 record above.
Re-opening the extractor choice belongs to a later scale-and-representation experiment, not here.

Keep:

```text
GLiNER entity index from Phase 7
P1 only
one hop
no canonicalization
no relations
no query-aware filtering
```

Refitting the Dense/second-signal fusion weight on dev is allowed because the score distribution
changes under the new Dense model.

No other entity parameter should be introduced merely to rescue a weak result.

---

## Phase 9 — HotpotQA FullWiki

Planned after Phase 8.

Question:

> Does the simple Strong Dense + Entity Hop architecture remain useful when retrieval happens over the
> standard multi-million-paragraph HotpotQA FullWiki search space?

Carry forward:

```text
GLiNER (urchade/gliner_medium-v2.1), the extractor selected in Phase 7
Strong Dense selected in Phase 8
P1 only
one Entity Hop
```

Phase 7's 5M extraction figures — 8.56 h / 6.34 USD — are **projections** from a measured
162.178 paragraphs/s, not a budget and not an authorization. Phase 9 measures its own.

At minimum compare:

```text
Strong Dense
Strong Dense + BM25
Strong Dense + Entity Hop
```

Do not add:

- canonicalization;
- relation triples;
- PageRank;
- query-aware reranking;
- multiple seeds;
- multiple hops;
- GraphRAG machinery.

Phase 9 should report standard FullWiki retrieval metrics used by the relevant literature so that the
project can finally place its numbers beside published work.

It must also measure scale:

- extraction time and cost;
- embedding time;
- index sizes;
- memory;
- query latency;
- entity candidate counts.

---

# Work deliberately deferred

These are research directions, **not current phases**:

- entity canonicalization / alias resolution;
- query-aware Entity Hop;
- explicit relation and attribute nodes;
- subject-relation-object triples;
- multiple Dense seeds;
- multiple entity hops;
- Dense + BM25 + Entity;
- PageRank/global graph propagation;
- richer GraphRAG structures;
- 2WikiMultiHopQA;
- MuSiQue;
- QA end to end;
- hierarchical/book-like corpora;
- Type C / satellite-context evaluation;
- direct reproduction of every competing paper;
- **re-opening the Phase 7 extractor choice at scale** — does spaCy's dev advantage over GLiNER
  survive at FullWiki scale, or does it collapse under candidate explosion and hub growth? Recorded
  so it is not lost; writing it down does not promote it into Phase 8 or 9.

Their rationale and possible ordering live in:

`docs/plans/research_roadmap.md`

Do not silently promote one of them into the active phase.

---

# Simplification rule from Phase 7 onward

This is a project-level rule.

Phase 6 reached roughly 2,500 tests and 24 implementation tasks. That level of experimental
engineering is no longer the default.

From Phase 7 onward:

1. **One main scientific question per phase.**
2. Aim for roughly **5–8 implementation steps**, not dozens.
3. Do not create `X.tasks.md` unless it is genuinely needed.
4. Reuse existing infrastructure instead of generalizing it pre-emptively.
5. Do not implement branches or guards for hypothetical failures that have not occurred.
6. Test new code that can materially change the experimental result; do not encode every process
   invariant as a test.
7. Keep dev/test separation whenever choices are fitted.
8. Record configuration, provenance and results clearly, but keep artifact machinery proportional to
   the experiment.
9. Do not create an elaborate statistical decision framework when a simpler comparison answers the
   research question.
10. Treat implementation complexity as a real cost.

Preferred workflow:

```text
research question
→ minimal experiment
→ result
→ next question
```

Avoid:

```text
research question
→ generic experimental framework
→ abstractions for hypothetical future cases
→ large maintenance burden
```

Historical Phase 6 rigor remains valid and should not be retroactively simplified. This rule applies
to new work.

---

# Development commands

Core environment:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
```

Add dependency:

```bash
uv add <package>
```

**Ask first before adding or changing dependencies.**

Current CLI stages:

```text
fetch
build
embed
evaluate
induce
label
select
expand
pilot
extract
nodes
navigate
replace-check
replace-freeze
replace-test
cheap-extract
cheap-eval
strong-embed
strong-dense
```

Historical pipeline:

```bash
uv run cer fetch
uv run cer build
uv run cer embed

uv run cer induce
uv run cer label
uv run cer select
uv run cer evaluate

uv run cer expand
uv run cer evaluate --systems expansion,expansion-conceptual

uv run cer pilot
uv run cer extract
uv run cer nodes
uv run cer navigate

uv run cer replace-check
uv run cer replace-freeze
uv run cer replace-test

uv run cer cheap-extract --extractor gliner
uv run cer cheap-eval
uv run cer cheap-eval --test
```

Do not re-run one-shot historical experimental stages merely to inspect them. Their versioned artifacts
are the record.

---

# Architecture

## File structure

```text
src/concept_embeddings_rag/
├── cli.py
├── config.py
├── artifacts.py
├── corpus/
├── embeddings/
├── concepts/
├── nodes/
├── retrieval/
│   ├── dense.py
│   ├── bm25.py
│   ├── conceptual.py
│   ├── diffusion.py
│   ├── fusion.py
│   └── entity_hop.py
└── evaluation/
    ├── harness.py
    ├── budget.py
    ├── second_hop.py
    ├── navigation.py
    ├── pilot.py
    ├── replacement_inputs.py
    ├── replacement_run.py
    ├── replacement_freeze.py
    ├── replacement_decision.py
    └── ...

tests/
scripts/
data/

docs/
├── plans/
│   ├── 0_master_plan.md
│   ├── research_roadmap.md
│   ├── phase_1/
│   ├── ...
│   └── phase_7/
├── refs/
├── security/
├── templates/
└── USER_GUIDE.md

.claude/
├── commands/
└── skills/
```

---

# Main architectural patterns

## 1. Complete-meaning indexing units

The benchmark indexing unit is the whole article paragraph.

Never replace it with arbitrary fixed-size windows inside this research line without explicitly
changing the experimental question.

---

## 2. Expensive computations are cached

Corpus embeddings and other expensive deterministic computations live on disk and are reused.

Do not delete caches casually.

A different model/configuration should produce a distinct cache rather than overwrite the artifact
behind an existing result.

---

## 3. Retriever implementations share the evaluation harness

The project compares retrieval signals.

A new extractor or Dense backend should not require rewriting the metric/evaluation machinery if the
existing interface can support it.

Reuse first.

Generalize only when the experiment actually requires it.

---

## 4. Entity Hop is currently intentionally minimal

Current semantics:

```text
D(q) = Dense ranking
read(q) = first 10 Dense units
p1(q) = first Dense unit

Entity candidates:
paragraphs outside read(q)
sharing >=1 raw entity node with p1(q)

candidate score:
sum of rarity weights of shared entity nodes

rarity(node):
log(1 + N / (1 + df))

ranking:
score descending
then unit id
```

No padding is added if fewer positive candidates exist.

This is currently a one-hop second-stage generator.

Do not add canonicalization, relations, query relevance or more hops unless the active phase explicitly
asks for them.

---

# Data / experimental discipline

The historical project used strong contamination controls because the same fixed test split was used
across several tightly coupled phases.

For new work:

- fitting or choosing between alternatives happens on dev;
- test is used for the final evaluation of the chosen configuration;
- do not tune a new model/extractor by repeatedly looking at its test score;
- when comparing alternatives, use dev to choose the practical winner, then report only that
  winner's held-out test result — this is how Phase 7 selected GLiNER, and why spaCy has no test
  figure;
- record which model/version/configuration generated every important result;
- do not invent missing figures.

Do not reproduce Phase 6's full statistical machinery unless the new research question actually
requires it.

---

# Tech stack

Current core stack includes:

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Dependency manager | uv |
| Numerics | numpy / scipy |
| ML | scikit-learn |
| Embeddings | sentence-transformers / torch |
| Transformers | transformers |
| Lexical retrieval | bm25s |
| Vector retrieval | local numpy/faiss-compatible machinery |
| Entity extraction (`phase7` group) | `gliner==0.2.29`, `spacy==3.8.16`, `en_core_web_sm` 3.8.0 |
| Tests | pytest |
| Lint | ruff |
| Types | mypy |

GLiNER is now part of the stack: it produced the entity index Phase 8 inherits, and it is installed
through the optional `phase7` dependency group rather than the core dependencies.

The local development machine is Windows on ARM64.

That is a development constraint, **not a scientific constraint**, and Phase 7 exercised the
escape hatch: both extraction passes ran on a rented Linux x86_64 RTX 4090 machine. `pyproject.toml`
resolves `torch` from the `pytorch-cpu` index on Windows and from `pytorch-cu126` on Linux x86_64,
and GLiNER moves itself to CUDA when a device is present. Phases 8 and 9 will use that path again.

Do not reject a scientifically appropriate Phase 8/9 model solely because it does not run efficiently
on the original laptop.

---

# Environment variables

The Anthropic API is used by historical LLM-based stages.

```env
ANTHROPIC_API_KEY=
```

It has been used for:

- concept labelling;
- Phase 5 entity/concept extraction.

That is the whole list. The Phase 7 local extractors required no key, and neither extraction pass
made an Anthropic API call.

Secrets:

- live only in `.env`;
- are loaded at runtime;
- are never printed;
- are never committed.

---

# Code conventions

- English everywhere in project-produced code and documentation.
- Type hints on functions.
- `mypy src` should pass.
- `ruff check .` should pass.
- ASCII only in Python console/logging strings; Markdown may use Unicode.
- Explicit seed for stochastic experimental operations.
- Stable deterministic tie-breaking where rankings depend on equal scores.
- Preserve prior versioned experimental artifacts.
- Never silently overwrite the configuration behind a published result.
- Prefer readable research code over generic framework abstractions.

---

# Agent permissions

## ALWAYS

Do without asking:

- read the relevant phase spec/plan before changing implementation;
- reuse existing code and artifacts where possible;
- preserve explicit seeds;
- record the configuration behind meaningful metrics;
- run focused tests for changed code;
- run the project's normal validation before declaring implementation complete;
- preserve dev/test separation.

## ASK FIRST

Stop and ask before:

- adding or changing dependencies;
- changing the embedding model outside a phase whose explicit purpose is to do so;
- changing the indexing unit;
- deleting caches;
- invalidating a versioned artifact;
- modifying an already-recorded historical experimental parameter;
- changing the scope of the active phase;
- adding a major new representation or graph mechanism that the active spec does not request.

Phase 8 is explicitly intended to change the Dense model, so that change will be permitted by its
approved spec rather than requiring a surprise mid-implementation decision.

## NEVER

- tune hyperparameters by looking at held-out test results;
- rewrite historical results because a newer method performs better;
- report estimates as measured results;
- invent missing data;
- commit API keys or credentials;
- commit the downloaded raw corpus unless an existing project rule explicitly versions a particular
  manifest/artifact;
- silently expand a phase into canonicalization, relations, multiple seeds/hops or GraphRAG;
- add complexity merely because another paper uses it.

---

# Project anti-patterns

## Fixed-length chunking

Do not replace whole semantic units with arbitrary 400/1000-character chunks.

---

## Assuming a richer graph is automatically better

Relations, triples, PageRank, hypergraphs and multi-hop graph traversal are candidate experiments,
not default upgrades.

The current simple Entity Hop already has a measured positive result.

---

## Putting an LLM in the online retrieval loop by default

This makes attribution harder and adds cost/non-determinism.

Only do it as an explicit experimental variant.

---

## Reimplementing the literature before it is necessary

The immediate plan is **not** to rebuild MDR, HippoRAG, KG2RAG, HGRAG, SAG, LinearRAG, etc.

Phase 9 first moves to the same benchmark scale/metrics so published numbers become meaningfully
comparable.

Only reproduce specific prior mechanisms later if a scientific question requires it.

---

## Overengineering the experiment

Do not reproduce Phase 6's 24-task / ~2,500-test pattern as the default.

A small research phase should stay small.

---

# Known gotchas

## 1. Windows ARM64

The local machine is ARM64.

Some packages lack `win_arm64` wheels, and Phase 7 hit that for real: **spaCy cannot be installed on
this laptop** because `blis` publishes no `win_arm64` wheel. It ran on the rented x86_64 machine
instead. GLiNER installs locally but is far too slow on ARM64 CPU to be useful — the laptop probe
measured 0.558 paragraphs/s against 162.178 p/s on the rented RTX 4090.

Before adding an extractor or model dependency, check:

- whether it installs on the current environment;
- whether ONNX or another backend is available;
- whether it is more sensible to run the work on another machine.

Do not distort the scientific experiment solely to accommodate the laptop. Renting compute is an
established, approved path in this project, not an exception.

---

## 2. Torch source

`torch` requires the official PyTorch package source rather than PyPI.

`pyproject.toml` declares two explicit indexes and selects between them by marker:

```text
pytorch-cu126   sys_platform == 'linux' and platform_machine == 'x86_64'
pytorch-cpu     everything else, including win_arm64
```

That is what lets the same pinned code run on the ARM64 laptop and on a rented CUDA machine.
Respect the existing `pyproject.toml` / `uv` source configuration.

---

## 3. Python version

Python is pinned to:

```text
>=3.12,<3.13
```

Do not change it casually.

---

## 4. `.claude/`

`.claude/commands/` and `docs/templates/` are shared framework material and may still use Spanish
generic filenames.

This project uses the English mapping:

| Framework wording | Project path |
|---|---|
| `docs/plans/0_plan_maestro.md` | `docs/plans/0_master_plan.md` |
| `docs/plans/fase_X/` | `docs/plans/phase_X/` |
| `X.0_nombre_fase.md` | `X.0_phase_name.md` |
| `X.Y_nombre.md` | `X.Y_name.md` |
| `fixes/fix-N_nombre.md` | `fixes/fix-N_name.md` |

When a framework command mentions the Spanish generic path, use the English project path.

---

## 5. Historical one-shot artifacts

Phases 5 and 6 contain one-shot experimental artifacts.

Do not rerun them just because a report can be regenerated.

Read their existing outputs.

---

## 6. Phase 7 selection — how it was actually made

Recorded because it will be questioned again.

The extractor was **not** chosen on test performance. Both candidates were measured on the 600 dev
questions; a bar and ranking frozen in `config.py` before any candidate ran admitted both and ranked
GLiNER first on projected FullWiki cost and time; only then was the held-out split opened, once, for
GLiNER alone. spaCy's test figure does not exist and should not be produced retrospectively.

spaCy scored higher on dev. That does not reopen the choice: overriding a preregistered rule after
observing dev would destroy the meaning of the held-out figure. The same discipline applies to any
later phase that compares alternatives.

---

# Planning workflow

For a new phase:

```text
/4-especificar
→ /5-planear
→ implement
→ verify
→ document
```

Apply `/8-auditar` when the phase introduces an actual security-relevant surface such as:

- external untrusted input;
- new dependencies with material risk;
- subprocess execution;
- unsafe deserialization;
- credentials/authentication;
- network APIs.

Do not run elaborate process steps mechanically when they add no value to the research phase.

## What Phase 7 established about this workflow

Phase 7 followed the simplification rule and it worked: one research question, a spec that fixed the
question, the constraints, the measurements, the selection rule and the anti-goals without designing
the implementation, then a short implementation plan, then the experiment. The targeted `/8-auditar`
pass covered three declared surfaces rather than the whole codebase.

Phase 8 should be specified the same way: state the question, what stays fixed, what is measured and
how dev decides; leave the implementation to the plan that follows; aim for roughly 5–8 concrete
steps.

---

# Security / artifact integrity

Historical artifact verification remains valid and should not be removed.

New artifacts from Phase 7 onward do not automatically need the same machinery as Phase 6.

Use the smallest integrity mechanism that lets us answer:

```text
Which extractor/model/configuration produced this result?
```

Phase 7's answer was a configuration digest per extractor plus a digest chain linking extraction →
node index → dev → selection → held-out run, so a test figure cannot be detached from the dev
reading and the selection that authorized it. Reuse that shape.

Avoid executable serialization for untrusted data.

Prefer JSON / NPZ / other non-executing formats already used by the project.

New model downloads are external inputs. Pin or record model identity/version/revision when practical.

---

# Documentation status

Authoritative research state:

```text
docs/plans/0_master_plan.md
docs/plans/research_roadmap.md
```

Historical phase results:

```text
docs/plans/phase_1/1.results.md
docs/plans/phase_2/2.results.md
docs/plans/phase_3/3.results.md
docs/plans/phase_4/4.results.md
docs/plans/phase_5/5.results.md
docs/plans/phase_6/6.results.md
docs/plans/phase_7/7.results.md
```

`docs/USER_GUIDE.md` currently describes an older execution pipeline and should not be treated as
authoritative for Phases 5–9 until it is updated.

For current research decisions, trust the master plan, roadmap, approved phase specs and measured
result artifacts.

---

# Immediate next action

Phase 8 is measured and **stopped at its dev gate**: `Qwen/Qwen3-Embedding-0.6B` scored 446 / 600 dev
against the inherited 487 / 600, below the 488 required. The held-out split was never opened and the
phase's research question is still open. See `docs/plans/phase_8/8.results.md`.

The current work is:

```text
Deviation 8.1 — Dense scale sensitivity
APPROVED 2026-09-20 — in implementation, nothing measured
```

The Phase 8 ordering between the two encoders is bounded to the frozen 19,366-paragraph HotpotQA
distractor pool — a haystack built from selected near-misses, which is an unusual environment and not
obviously predictive of an open corpus. 8.1 asks whether that ordering is a property of the models or
of the corpus, by growing the retrieval space to 100k / 250k / 500k official HotpotQA FullWiki
paragraphs on dev only, with corpus size as the single variable.

Both documents are approved and the specification is **frozen**:

```text
docs/plans/phase_8/8.1_dense_scale_sensitivity.md    deviation + functional spec (frozen)
docs/plans/phase_8/8.1_implementation_plan.md        code-level plan, S1-S7
```

What that means for any session working on 8.1:

- no new threshold, gate, outcome or experimental variant is introduced, and no scientific rule is
  relaxed to solve an implementation difficulty — if a real problem outside the plan appears, **stop
  and report** rather than redesign;
- the fixed values are the ones already written down: unmatched-unit ceiling 50, salt `42`, prefixes
  80,634 / 230,634 / 480,634, reproduction counts 487 / 600 and 446 / 600, re-encode tolerance of
  1 question, BGE retention floor 244 / 600;
- the five implementation questions the plan raised are **closed**: `data/phase8_1/` stays untracked,
  reproduction level 1 also runs on the measurement host, `8.1_runpod_recipe.md` is written before the
  pod is opened, the new caches come off the pod before it dies, and **no new dependency** may be added
  without asking first;
- two traps in the inherited code, both verified: `question_cache_key` does not include the corpus, so
  8.1 must encode into `data/phase8_1/cache/` or it would overwrite the question cache behind the
  487 / 600; and `TokenCounter.count_units` tokenizes its whole input in one call, which does not
  survive 480,634 paragraphs.

Nothing is measured yet, and neither `DATA_STOP`/`REPRODUCTION_STOP` nor any of the four terminal
states has been reached.

Rules that still hold, unchanged by the stop:

- the Phase 7 GLiNER entity index is inherited as it is, and is read-only;
- no measured Phase 5/6/7/8 artifact is modified or regenerated;
- no second encoder is retrofitted into Phase 8 — a different model is a declared candidate in a
  new, written experiment, never a silent retry inside a closed one.