# CLAUDE.md — concept-embeddings-RAG

Project rules and context for the agent. Always read before making changes.

---

## What is this project?

A **research experiment** on retrieval over a corpus-induced concept space: every chunk is represented as a sparse vector over concepts discovered from the corpus itself (`X = chunks × concepts`), and retrieval expands iteratively through that space instead of stopping at the top-K of the question.

**The deliverable is measured evidence, not an application**: four comparable systems (BM25, dense, conceptual, iterative) over a multi-hop benchmark with annotated ground truth, and an explicit verdict on the hypothesis. A well-documented negative result closes the project just as well as a positive one.

**Current state: Phases 1 and 2 complete.** The frozen corpus, both baselines and the evaluation harness are built, audited and measured; the numbers live in `docs/plans/phase_1/1.results.md` and are what every later phase is judged against. Phase 2 induced four concept spaces (K = 512, 1,024, 2,048, 4,096) and **chose none of them**: the choice of K belongs to Phase 3 and is made against dev recall. Its numbers, and the one concept that activates on 99.4% of the corpus at K = 4,096, are in `docs/plans/phase_2/2.results.md`. The full hypothesis and the surveyed prior art live in `docs/refs/descripcion-proyecto.md` and `docs/refs/bibliografia.md` (both in Spanish, they are source material); the roadmap lives in `docs/plans/0_master_plan.md`, and `docs/USER_GUIDE.md` explains how to run the pipeline.

---

## Development commands

```bash
uv sync                  # Create/update the environment from pyproject.toml + uv.lock
uv run pytest            # Run the tests
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run mypy src          # Type check
uv add <package>         # Add a dependency (ASK FIRST - see agent permissions)
```

There is no server or UI to start: entry points are experiment scripts run through `uv run`. The first `pytest` of each session takes around 35 s because torch has to load; that is not a hang.

---

## Architecture

### File structure

```
src/concept_embeddings_rag/
├── cli.py                    # The six stages: fetch, build, embed, evaluate, induce, label
├── config.py                 # Every constant that decides what an experiment measures
├── corpus/                   # download (hash-verified), hf_source, split, pool, manifest
├── embeddings/               # Swappable backend + .npz cache keyed by configuration
├── retrieval/                # Retriever protocol, dense and BM25 behind it
├── evaluation/               # Budget filling, the four metrics, the harness
├── concepts/                 # dictionary, coding, dedup, diagnostics, labeling
└── artifacts.py              # Atomic writes: every artifact lands whole or not at all
tests/                        # Mirrors the source layout. test_scaffold.py checks ARM64
data/                         # Corpus, pool and caches: git-ignored. Manifest and results: versioned
docs/
├── plans/                    # SDD: master plan, phase specs, phase plans and fixes
├── refs/                     # Immutable reference material (hypothesis, bibliography)
├── security/                 # /8-auditar findings catalogue (security/README.md)
├── templates/                # Framework document templates
└── USER_GUIDE.md             # How to run the pipeline
.claude/commands/             # The 10 commands of the SDD-WAT workflow
```

Entry point: `uv run cer <stage>`. Each stage refuses to run if its input is missing and names the
stage to run first. `embed` (~36 min on this machine) and `induce` (hours for the sweep) are the
expensive ones; `label` is the only one that spends money, and nothing depends on it.

### Main pattern: staged pipeline with on-disk artifacts

The experiment is a chain of transformations where **every stage persists its output, versioned by the configuration that produced it**:

```
frozen corpus -> indexing units -> embeddings (cached)
   -> concept dictionary -> matrix X -> retrieval -> metrics
```

Consequences that shape all the code:

- Every artifact carries its configuration key (model, indexing unit, dictionary size, seed). Changing a parameter produces a different artifact; it never overwrites the previous one.
- No stage recomputes what is already on disk. On CPU, embedding the corpus is the expensive operation of this project.
- Swappable backends (embeddings, retriever) sit behind a common interface: the experiment compares systems, so replacing one must never force changes in the evaluation harness.

### Design decisions already settled

The reasoning lives in the decisions table of `docs/plans/0_master_plan.md`. Operationally:

- **Indexing unit is a complete unit of meaning**, never a fixed-length window.
- **Non-negative sparse coding** (`MiniBatchDictionaryLearning` with `positive_code=True`) induces the concepts, not clustering: clustering leaves X nearly one-hot and destroys co-activation.
- **A concept's embedding is its dictionary atom**, not the embedding of its generated name.
- **Expansion is diffusion over X** (`X` and `Xᵀ` with partial restart on the query), not an LLM loop.
- **Measurement at a fixed context budget**, not at fixed K.

---

## Tech stack

| Layer | Technology | Version |
|------|-----------|---------|
| Language | Python (native ARM64) | 3.12.10 |
| Dependency manager | uv | 0.10.9 |
| Numerics | numpy / scipy | 2.5.2 / 1.18.1 |
| Classical ML and concept space | scikit-learn | 1.9.0 |
| Embeddings | sentence-transformers / torch | 6.0.1 / 2.13.0+cpu |
| Transformers | transformers | 5.16.1 |
| HTTP and progress | httpx / tqdm | 0.28.1 / 4.70.0 |
| Retrieval (lexical) | bm25s | 0.3.11 |
| Tests | pytest | 9.1.1 |
| Lint and types | ruff / mypy | 0.16.5 / 2.3.1 |
| Deployment | — (local experiment) | — |

---

## Environment variables

One, and only the `label` stage reads it. Every other stage runs with no key and no account.

```env
ANTHROPIC_API_KEY=       # `cer label` only: naming concepts for the report
```

Secrets live only in `.env`, are loaded at runtime, and are never printed or quoted.

---

## Code conventions

- **English everywhere**: code, comments, docstrings, documentation, commit messages and generated reports.
- **Type hints on every function.** `mypy src` must pass clean.
- **ASCII only in `print()`, logging and Python strings.** No emoji, arrows or bullets: use `[OK]`, `[ERROR]`, `[WARN]`, `->`. Markdown files may use Unicode.
- **Explicit seed** for anything stochastic. Without `random_state` a result is not a result.
- **Artifacts versioned by configuration**: the cached file name derives from the parameters that generated it.
- Tests in `tests/`, named after the criterion they verify.
- 100-character lines; ruff with `E, F, I, UP, B, SIM, S`. The `S` rules are the bandit port:
  a non-cryptographic hash must say so with `usedforsecurity=False`. `S101` is ignored under
  `tests/`, where asserts are the point.
- **Verify artifacts on load, do not merely record them.** Every on-disk artifact carries a hash;
  the code that reads it compares. This was the whole content of the Phase 1 audit — see
  `docs/security/` and `docs/plans/fixes/fix-1_audit_phase_1_findings.md`.

---

## Agent permissions

✅ **ALWAYS** (do without asking):
- Run `uv run pytest` before calling a task done, and never report it done if tests fail
- Set an explicit seed on every stochastic process
- Cache every expensive computation on disk and reuse it instead of recomputing
- Record, next to every metric, the configuration that produced it (model, unit, dictionary, seed, budget)

⚠️ **ASK FIRST** (stop and request confirmation):
- Adding or changing dependencies — on ARM64 a new dependency may have no wheel
- Changing the embedding model or the indexing unit — invalidates the cache and makes previously obtained metrics incomparable
- Deleting or invalidating caches — rebuilding them costs hours of CPU
- Touching parameters of an experiment whose numbers are already recorded

🚫 **NEVER** (no exceptions):
- **Tune hyperparameters by looking at the evaluation set.** It contaminates the test set and voids the whole experiment. Tuning happens on a separate development split
- Report metrics that cannot be regenerated with a command (no config, no seed, or no code)
- Use non-ASCII Unicode in `print()`, logging or Python strings
- Fill gaps with estimates, placeholders or "expected" results. If the datum is missing, declare the gap
- Commit secrets, credentials or the downloaded corpus

---

## Project anti-patterns

🚫 **Splitting by fixed length** (400/1000 characters) — cuts through the middle of an idea and produces rows of X that mix the end of one concept with the start of another. The unit is a complete unit of meaning.

🚫 **Hard chunk → concept assignment** (KMeans, HDBSCAN and family) — leaves X nearly one-hot and destroys co-activation, chunk↔chunk similarity in concept space, and expansion. Always non-negative sparse coding. Clustering is admissible only as a declared ablation.

🚫 **Putting an LLM inside the expansion loop by default** — makes results non-deterministic and expensive, and above all makes it impossible to tell whether the concept space won or the model was clever. Only as an explicit comparison variant.

🚫 **Comparing systems at fixed K** — the iterative system retrieves more chunks by design. The valid comparison is at equal tokens sent to the LLM.

🚫 **Recomputing cached embeddings** — the expensive operation of the project, and what makes iteration unaffordable.

---

## Known gotchas

1. **Windows on ARM (ARM64).** The machine is a Snapdragon ARMv8 and both installed Pythons are native ARM64. Verified and accepted: we work natively, not emulated. Emulating x64 would penalize exactly the bottleneck (embedding ~10,000 units) in exchange for libraries we have already decided not to use.

2. **`torch` is not on PyPI for `win_arm64`.** `pyproject.toml` declares the official PyTorch index as an **explicit** source for that package only (`[[tool.uv.index]]` + `[tool.uv.sources]`). A bare `uv add torch` will fail; respect that configuration.

3. **Packages ruled out by the platform, no `win_arm64` wheel:** `bertopic`, `umap-learn`, `hdbscan`, `numba`, `llvmlite`, `fastembed`, and HuggingFace `datasets` (because of `pyarrow`). Do not try to install them. Substitutes in use: scikit-learn for reduction and sparse coding; direct download of the benchmark JSON instead of `datasets` — which is also more reproducible, since it is frozen with its hash.

4. **Python pinned to `>=3.12,<3.13`.** Not a whim: it constrains the numerical stack wheels. `.python-version` says `3.12`.

5. **The shell lies about the architecture.** In this environment `PROCESSOR_ARCHITECTURE` reports `AMD64` even though the machine is ARM64, because bash runs emulated. To diagnose, use `python -c "import platform; print(platform.machine())"`.

6. **ruff excludes `.claude/`**: it holds third-party skill scripts with 10 warnings that are not project code and are not to be touched.

7. **Security tooling.** `ruff --select=S` (the bandit port) is in the project's lint selection, and `pip-audit` lives in the `security` dependency group: `uv run pip-audit --desc`. `detect-secrets` is still CI-only. **`pip-audit` skips `torch`** because the pinned `2.13.0+cpu` carries a local version identifier PyPI does not know, and it prints that skip beside "No known vulnerabilities found" — so check the upstream version against OSV separately rather than reading the clean line as full coverage. For the same reason `pip-audit -r <requirements>` cannot work here at all: it resolves every pin against PyPI and dies on torch. Audit the environment (`uv sync --group security --group labeling && uv run pip-audit`), which is what CI does — and name every runtime group, because `uv sync` prunes whatever the groups listed do not ask for, so `--group security` alone audits an environment `anthropic` is absent from (SEC-018).

8. **The first `pytest` of a session takes ~35 s** because of torch startup. Not a hang.

9. **Planning documents are named in English here, the framework commands are not.** `.claude/commands/` and `docs/templates/` are shared across all the author's projects and still refer to the Spanish names. In this project the mapping is:

   | Framework command says | This project uses |
   |---|---|
   | `docs/plans/0_plan_maestro.md` | `docs/plans/0_master_plan.md` |
   | `docs/plans/fase_X/` | `docs/plans/phase_X/` |
   | `X.0_nombre_fase.md` | `X.0_phase_name.md` |
   | `X.Y_nombre.md` | `X.Y_name.md` |
   | `fixes/fix-N_nombre.md` | `fixes/fix-N_name.md` |

   The commands are read by an agent, not by a parser, so this does not break them — but write new planning documents with the English names, never the ones the command text quotes literally.

---

## Evolution plan

The full roadmap is in `docs/plans/0_master_plan.md`.

**Before any change:**
- New phase → `/4-especificar` → `/5-planear` → `/6-implementar` → `/7-verificar` → `/8-auditar` → `/9-documentar`
- Bug fix → `/5-planear` (if not trivial) → `/6-implementar` → `/7-verificar` → `/8-auditar` (if it touches auth, crypto, external input, subprocess or deserialization)
- Before packaging/release → `/8-auditar completo` as the release gate

Run `/prime` at the start of every session to load context.

---

## Security audit

The `/8-auditar` command runs a security review before `/9-documentar`. It is self-contained — the whole protocol lives in `.claude/commands/8-auditar.md` and depends on no skill. It is a **mandatory** step of the phase workflow whenever the code touches:

- Authentication / authorization
- Cryptography or credential storage
- External input (email, web, user-uploaded files)
- `subprocess` / command execution
- Deserialization (pickle, yaml.load, json.loads over unvalidated input)
- New or updated dependencies

In this project the relevant surface will mainly be **the corpus download** (external input: verify the hash, do not trust the downloaded JSON) and **deserialization of cached artifacts** (never `pickle` over files not produced by the pipeline itself; prefer non-executing formats such as `.npz` or JSON).

The audit presents its findings in chat — severity, CWE, `file:line`, what an attacker gains, and the fix — and then fixes them in the same session, each fix with its regression test. It writes no report: the only persistent record is one row per finding in `docs/security/README.md`. Every blocking finding (Critical/High) is closed before the phase closes; a non-trivial fix follows the project's fix protocol (`fix-N` in `docs/plans/fixes/`).

See `CLAUDE_GLOBAL.md` → "Auditoría de seguridad" section for the detailed rules.
