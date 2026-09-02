# CLAUDE.md — concept-embeddings-RAG

Project rules and context for the agent. Always read before making changes.

---

## What is this project?

A **research experiment** on retrieval over a corpus-induced concept space: every chunk is represented as a sparse vector over concepts discovered from the corpus itself (`X = chunks × concepts`), and retrieval expands iteratively through that space instead of stopping at the top-K of the question.

**The deliverable is measured evidence, not an application**: four comparable systems (BM25, dense, conceptual, iterative) over a multi-hop benchmark with annotated ground truth, and an explicit verdict on the hypothesis. A well-documented negative result closes the project just as well as a positive one.

**Current state: scaffold.** The manifest, the verified environment and the smoke tests exist. There is no pipeline code yet — that is built in Phase 1. The full hypothesis and the surveyed prior art live in `docs/refs/descripcion-proyecto.md` and `docs/refs/bibliografia.md` (both in Spanish, they are source material); the roadmap lives in `docs/plans/0_master_plan.md`.

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
src/concept_embeddings_rag/   # Main package (today only __init__.py; populated in Phase 1)
tests/                        # Tests. test_scaffold.py verifies the stack works on ARM64
data/                         # Downloaded, frozen corpus. Git-ignored except .gitkeep
docs/
├── plans/                    # SDD: master plan, phase specs and phase plans
├── refs/                     # Immutable reference material (hypothesis, bibliography)
├── security/                 # /8-auditar reports
└── templates/                # Framework document templates
.claude/commands/             # The 10 commands of the SDD-WAT workflow
```

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
| Tests | pytest | 9.1.1 |
| Lint and types | ruff / mypy | 0.16.5 / 2.3.1 |
| Deployment | — (local experiment) | — |

---

## Environment variables

None yet. Phase 2 will need `ANTHROPIC_API_KEY` to label concepts with an LLM (one cached call per concept, and purely for interpretability: retrieval does not depend on it).

```env
ANTHROPIC_API_KEY=       # Phase 2 only: naming concepts for the report. Pending.
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
- 100-character lines; ruff with `E, F, I, UP, B, SIM`.

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

7. **Security tooling postponed.** `bandit`, `pip-audit` and `detect-secrets` are not installed locally; the `.github/workflows/security.yml` workflow already runs them in CI. When `/8-auditar` needs its Phase 2 automated scan, add the `security` group to the pyproject.

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

The `/8-auditar` command (skill `audit-code` in `.claude/skills/`) runs a professional security audit before `/9-documentar`. It is a **mandatory** step of the phase workflow whenever the code touches:

- Authentication / authorization
- Cryptography or credential storage
- External input (email, web, user-uploaded files)
- `subprocess` / command execution
- Deserialization (pickle, yaml.load, json.loads over unvalidated input)
- New or updated dependencies

In this project the relevant surface will mainly be **the corpus download** (external input: verify the hash, do not trust the downloaded JSON) and **deserialization of cached artifacts** (never `pickle` over files not produced by the pipeline itself; prefer non-executing formats such as `.npz` or JSON).

The audit produces a report in `docs/security/audit-YYYY-MM-DD-<mode>.md` with severity, CWE, OWASP, file:line and proposed fix. Every blocking finding (Critical/High) is resolved with a `fix-N` before closing the phase. The consolidated catalogue lives in `docs/security/README.md`.

See `CLAUDE_GLOBAL.md` → "Auditoría de seguridad" section for the detailed rules.
