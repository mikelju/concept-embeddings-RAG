# Security audits

Consolidated catalogue of the audits run with `/8-auditar`, and the only persistent record they
leave: one row per finding. The audit presents its findings in chat and fixes them in the same
session — it writes no report.

## Reports (historical)

The two Phase 1 passes predate that change and left a full report each. They are kept as they were
written; later audits add rows below, not documents.

| Date | Mode | Scope | Critical | High | Medium | Low | Info |
|---|---|---|---:|---:|---:|---:|---:|
| [2026-09-08](audit-2026-09-08-fase-1.md) | fase 1 | `src/` (20 files), `pyproject.toml`, `.gitignore`, CI workflow | 0 | 0 | 4 | 4 | 2 |
| [2026-09-08 (re-audit)](audit-2026-09-08-fase-1-2.md) | fase 1, 2nd pass | the above, plus the `fix-1` changes themselves | 0 | 0 | 0 | 0 (2 found and fixed in-pass) | 3 |

## Open findings

None **of the phases that have been audited**, which are Phases 1 and 2. Every finding from the
three passes is closed, each held closed by a regression test, and neither phase has blocking
security work outstanding. Phases 3 and 4 have not been audited — see "Deferred audits" below.

### Phase 1 — 2026-09-08

| ID | Severity | Title | Status |
|---|---|---|---|
| SEC-001 | Medium | Corpus integrity check recorded, never enforced | Fixed ([fix-1](../plans/fixes/fix-1_audit_phase_1_findings.md)) |
| SEC-002 | Medium | Embedding model resolved from a branch, not a pinned commit | Fixed (fix-1) |
| SEC-003 | Medium | `load_pool` did not re-derive the ids it loaded | Fixed (fix-1) |
| SEC-004 | Medium | Unbounded pagination and response size when fetching the corpus | Fixed (fix-1) |
| SEC-005 | Low | SHA-1 for content addressing without `usedforsecurity=False` | Fixed (fix-1) |
| SEC-006 | Low | Result filename built from unvalidated `split` / `system` | Fixed (fix-1) |
| SEC-007 | Low | CI secret scan could not fail; dependency gate ran on Python 3.11 | Fixed (fix-1) |
| SEC-008 | Low | `CorpusManifest` validated field names but not types | Fixed (fix-1) |
| SEC-009 | Low | The SEC-001 fix covered one branch of `cmd_fetch` and not the other | Fixed in the re-audit |
| SEC-010 | Low | `load_pool` verified units but not that gold ids resolved | Fixed in the re-audit |
| SEC-011 | Low | The SEC-007 fix pointed CI at a `pip-audit` command that cannot resolve this project's pinned torch | Fixed in the re-audit |

Accepted by design, not fixed: OBS-002 (trust-on-first-use), OBS-003 (`resolved_revision` proves
less than its name suggests), OBS-004 (`_check_pool_against_manifest` fails open without a
manifest), OBS-005 (superseded embedding cache kept on disk). Each is argued in its report.

### Phase 2 — 2026-09-09

Scope: the five modules the phase added (`concepts/dictionary.py`, `coding.py`, `dedup.py`,
`diagnostics.py`, `labeling.py`), the two new CLI stages, the new `anthropic` and `python-dotenv`
dependencies, and the CI security workflow. One theme runs through all of it: this phase writes
five new artifact types and reads them back, and Phase 1 had already established that recording a
hash is not verifying one.

| ID | Severity | Title | Status |
|---|---|---|---|
| SEC-012 | Medium | Diagnostics bound to neither their pool nor their own bytes, yet they choose the evidence the paid labelling call sees | Fixed in-session |
| SEC-013 | Medium | The matrix sidecar was written on save and never checked on load: shape, sparsity and digest all unverified | Fixed in-session |
| SEC-014 | Medium | The label cache was keyed by model and prompt but bound to neither, and read with an unguarded `json.loads` | Fixed in-session |
| SEC-015 | Low | The model's answer reached an artifact without being validated: a missing field raised on subscript, an overlong one was written whole | Fixed in-session |
| SEC-016 | Low | `load_dotenv()` walked parent directories, so a `.env` outside the project could supply the key | Fixed in-session |
| SEC-017 | Low | The dictionary sidecar and the merge log never checked the key they were filed under | Fixed in-session |
| SEC-018 | Low | The CI dependency scan synced `--group security` alone, so `uv sync` pruned `labeling` — the only dependencies that open a network connection — before auditing | Fixed in-session |
| SEC-019 | Low | `.secrets.baseline` predated the phase, so the gate would have failed the first pull request on four unaudited findings | Fixed in-session |

Raised as an observation and fixed anyway: **OBS-006** — `Path.write_text` and
`np.savez_compressed` truncate the target before writing, so an interrupted run left a truncated
artifact where a valid one had been. Every "verify on load" check added above assumes the file it
reads was written whole, so the two are one piece of work.
`src/concept_embeddings_rag/artifacts.py` now writes through a temporary and `os.replace`.

Every fix is backward compatible by construction: new integrity fields are **written always,
verified only when present**, and the label cache upgrades legacy entries in place. That was
checked against the artifacts already on disk rather than argued — the eight dictionaries, eight
matrices, four diagnostics, four merge logs, the labels artifact and all 2048 cached answers in
`data/concepts/` load clean under the new checks. No recomputation, no re-spend, and the numbers
in `docs/plans/phase_2/2.results.md` stand.

**Phase closure: no Critical and no High finding, so the phase may close.** The three Medium
findings are fixed and held closed by regression tests, so no `fix-N` document is owed.

## Deferred audits

An audit that was not run is a decision on record here, never a gap nobody noticed.

### Phase 3 — deferred on 2026-09-14, until the Phase 4 results are in

**Update, 2026-09-15: the condition has been met and the decision is still open.** Phase 4 has its
numbers, it is negative, and its code (diffusion, the expansion selection and freeze, the traces) is
unaudited too, so the surface below now covers both phases. The research line they implement was
closed the same day ([4.1](../plans/phase_4/4.1_research_line_closure.md)); the project was not, so
"What closes this" still applies: a Phase 3 + 4 audit, or an explicit exemption recorded here. That
choice belongs to the author and has not been made.

**Status: not audited.** Phase 3 was verified against its specification and merged to `main`
([PR #4](https://github.com/mikelju/concept-embeddings-RAG/pull/4)) with `/8-auditar` outstanding.

**Reason.** Phase 3 returned a negative result — the concept space retrieves worse than either
baseline and contributes nothing to a fused system — so the code it audits may not survive to the
end of the project. Phase 4 asks the question that decides this: whether the second paragraph can be
reached through concepts at all. Auditing now would spend a session on a surface that Phase 4 either
promotes to the project's main result or retires. The decision is to run the audit **after** Phase 4
has a number, covering Phases 3 and 4 together, and to scale its depth to what that number says.

**What this costs, stated rather than implied.** The Phase 3 surface that an audit would look at is
artifact deserialization: `json.loads` over the selection artifact and the result files, plus the
`.npz` matrices. All of it is produced by this pipeline, verified on load by digest, and carries no
`pickle` and no `allow_pickle=True`. The risk being carried is therefore a review that has not
happened, not a known weakness left open.

**What closes this.** Either a Phase 3 + Phase 4 audit pass with its rows added above, or - if the
project ends on a negative verdict and the code is never used outside this repository - an explicit
decision to close it as exempt, recorded here with that reasoning. Whichever happens, it is written
down before the project closes.

**Also outstanding from Phase 3, unrelated to security**: `tests/test_cli.py` is not
`ruff format`-clean (one block at line 958). No CI check enforces formatting, so nothing failed.

## Dependency scanning

`pip-audit` 2.10.1 is installed in the `security` dependency group. Audit the environment, and
sync every runtime group first — `uv sync` prunes whatever the groups named do not ask for, which
is how SEC-018 produced a clean result over an environment `anthropic` was absent from:

```bash
uv sync --group security --group labeling
uv run pip-audit --desc
```

**Last run: 2026-09-09 — 0 known vulnerabilities**, over an environment containing `anthropic`
1.4.0 and `python-dotenv` 1.2.3 (2026-09-08's run did not).

One caveat worth knowing before reading a future clean result: `pip-audit` **skips `torch`**,
because the pinned `2.13.0+cpu` carries a local version identifier PyPI does not recognise. It
prints the skip in a separate table under "No known vulnerabilities found", which is easy to read
as coverage it does not have. The upstream release was checked separately against OSV and is
clean. Any future scan needs the same second step for `torch`.

A clean scan is a statement about the day it ran, not a property of the code.

## Secret scanning

CI gates on `detect-secrets-hook --baseline .secrets.baseline`, which fails on any finding the
baseline does not already carry. The baseline is therefore a list of **audited** false positives,
and regenerating it to make a red job go green is the one way this gate can be defeated.

Regenerated 2026-09-09 for Phase 2 (SEC-019): 28 findings over 19 files. Five are new and all
five are false positives, audited here rather than in a commit message:

| File | Finding | Why it is not a secret |
|---|---|---|
| `src/concept_embeddings_rag/concepts/labeling.py:39` | Secret Keyword | `API_KEY_VARIABLE = "ANTHROPIC_API_KEY"` is the *name* of an environment variable. The value is read at runtime and never written anywhere |
| `docs/plans/phase_2/2.results.md:236` | Hex High Entropy String | A dictionary key: a digest of public configuration |
| `tests/concepts/test_dedup.py:50` | Hex High Entropy String | The pool hash in a test fixture, a digest of the public corpus |
| `tests/concepts/test_dictionary_artifact.py:33` | Hex High Entropy String | The same fixture value |
| `docs/security/README.md` (the row above) | Secret Keyword | Documenting the first finding meant quoting the flagged line verbatim, so the catalogue reproduces the pattern it explains. Auditing a false positive should not create one, but writing it down is still worth more than the noise |

Regenerate deliberately, and note the two traps:

```bash
uv run detect-secrets scan $(git ls-files --cached --others --exclude-standard) > .secrets.baseline
```

`--others --exclude-standard` adds files staged for the current change but not yet committed,
which plain `git ls-files` leaves unscanned — the trap that hides a finding in exactly the new
code an audit is looking at. And a baseline regenerated on this Windows machine keys every finding
with backslashes while the job runs on ubuntu: normalise them to forward slashes, or the hook
reads its own audited findings as new ones. `tests/test_security_gate.py` holds both closed.

Regenerated again 2026-09-15 with the research-line closure (commit `360927d`): 111 findings, the
28 above kept and 83 added. Every added one was read before committing, and every one is a digest
of public configuration or artifacts, a dictionary key, the pinned model commit, a fixture digest or
a HotpotQA question id.

### Path exclusion for id-bearing pipeline artifacts (Phase 5, decision D12)

Phase 5 versions artifacts whose content is mostly unit ids and question ids — content hashes of
the public corpus. detect-secrets flags each one as a high-entropy string: the frozen pilot alone
raised **1,385** findings. Auditing thousands of machine-written ids one by one would bury the 111
findings that were read, which is the opposite of what a baseline is for.

So the baseline carries one `should_exclude_file` filter, and it is deliberately narrow:

```
^data[\\/](pilot|navigation)[\\/][^\\/]+[.]json$
```

Only `.json` files directly inside `data/pilot/` and `data/navigation/` are skipped. Both are
written by the pipeline, verified on load by digest, and hold no text beyond ids, figures and
configuration. The pattern accepts either separator, so the check run on this Windows machine is the
check CI runs on ubuntu. Verified before committing: the full tracked tree passes, a hex string in a
file outside those directories still fails, and the same string inside them is skipped.
`tests/test_security_gate.py` holds the pattern to exactly that scope.

**When regenerating**, pass the filter again, or it is silently dropped with the old baseline:

```bash
uv run detect-secrets scan --exclude-files '^data[\\/](pilot|navigation)[\\/][^\\/]+[.]json$' \
  $(git ls-files --cached --others --exclude-standard) > .secrets.baseline
```

## Conventions

- This file is the whole paperwork of an audit: one row per finding, with id, severity, title and
  status. Findings are presented and argued in chat, not written up here.
- Every Critical or High finding is resolved with a `fix-N` in `docs/plans/fixes/` before the
  phase closes, or the decision to defer it is recorded in the master plan with its reason.
- Finding ids are unique per project: continue the `SEC-NNN` sequence rather than restarting it
  per audit. Next free id: **SEC-020** (`OBS-NNN` for observations: next free is **OBS-007**).
- A phase exempted from the audit (the command's escape hatch) is recorded here too, with its
  reason: an exempt phase is a decision on record, not a phase nobody looked at.
