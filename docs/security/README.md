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

**Four, all from Phase 5, none blocking** (one Medium, three Low): SEC-020 to SEC-023. Every one
needs a change under `src/`, and the author closed Phase 5 to source changes for that round, so they
are reported and catalogued rather than fixed — the decision is recorded in the Phase 5 section
below. No Critical and no High finding exists in any audited phase.

Phases 1 and 2 have nothing outstanding: every finding of their three passes is closed and held
closed by a regression test. Phases 3 and 4 have not been audited — see "Deferred audits" below.

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

### Phase 5 — 2026-09-16

Scope: the three modules `nodes/` adds (`extraction.py`, `normalization.py`, `index.py`), the four
`evaluation/` modules of the pilot (`pilot.py`, `navigation.py`, `second_hop.py`, `gate.py`), the
four new CLI stages (`pilot`, `extract`, `nodes`, `navigate`), and the Phase 5 constants. The
sensitive surface is the one `CLAUDE.md` names for this phase: an LLM's answers parsed and bounded
(external input), five new artifact kinds read back from disk (deserialization), a new API client
over an existing dependency, and a stage that reads a credential.

What the scanners said: `ruff` including the whole `S` selection, clean over the scope; `mypy src`
clean; `detect-secrets-hook` clean against the baseline; `pip-audit` over the environment synced
with `--group security --group labeling`, no known vulnerabilities, `torch` skipped as always. No
`pickle`, no `eval`, no `exec`, no `subprocess`, no `yaml.load` anywhere in `src/`; every `np.load`
of the phase passes `allow_pickle=False`.

One theme runs through the four findings, and it is the theme of Phases 1 and 2 again in a new
place: the artifacts this phase *measures* from — the pilot, the node index, the hop run, the
traces, the gate, the extraction archive — all carry a digest and verify it on load, while the
artifacts it *runs* from — the sample report, the extraction summary, the batch state — carry none
and are trusted.

| ID | Severity | Title | Status |
|---|---|---|---|
| SEC-020 | Medium | The sample report is read back with no integrity check, and its token means are what the cost ceiling is tested against | Open — fix specified, not applied |
| SEC-021 | Low | The extraction summary is trusted on load, and its `finding` flag is the gate that stops a failed extraction | Open — fix specified, not applied |
| SEC-022 | Low | A value from that unverified summary reaches a file path unshaped, so the node index can be sought outside its directory | Open — fix specified, not applied |
| SEC-023 | Low | The resumable batch state is reloaded unvalidated and decides which batch's answers enter the extraction cache | Open — fix specified, not applied |

**Why they are open.** The audit protocol fixes its findings in the same session, each with a
regression test. For this round the author forbade any change under `src/` and `data/`, so each
finding was written up with its `file:line`, its exploitability and the fix it needs, and none was
applied. None is Critical or High, so none blocks the phase; the work is carried deliberately, not
overlooked. Two of them (SEC-020, SEC-021) also have a `data/` component, because giving an existing
artifact a digest means rewriting the artifact — unless the Phase 2 pattern is reused, writing the
field always and verifying it only when present, which leaves the committed files valid and
unverified.

Accepted rather than fixed: **OBS-007** — the extraction prompt interpolates corpus text, so a
paragraph can address the model directly (CWE-1427). It is accepted for structural reasons, not by
omission: the corpus is the hash-frozen public subset of Phase 1, the answer is schema-constrained
at the API and re-validated and bounded locally, it is never evaluated, executed or interpolated
into a shell, and node forms are never printed. What an injection buys is a wrong node in the index,
which is a measurement error of the kind the failure counters and fragmentation figures already
report. Written down so a future corpus — which may be neither frozen nor public — inherits the
reasoning instead of the silence. **OBS-008** — `load_extraction` re-hydrates records without
re-applying the bounds `read_cached` applies, so a string where a list belongs would become a tuple
of characters; accepted because the archive is verified against its digest before any record is
built from it.

**Phase closure: no Critical and no High finding, so the phase may close** with the four open Low
and Medium findings carried as recorded above.

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

**Last run: 2026-09-16 — 0 known vulnerabilities**, over the 85 packages of the environment synced
with `--group security --group labeling`, `anthropic` and `python-dotenv` included. Phase 5 added no
dependency: `pyproject.toml` and `uv.lock` are unchanged against `main`, and the extraction client
is new code over the SDK the `labeling` group already provided. Two packages were skipped and both
skips are expected — `torch`, for the reason below, and the project itself, which is not on PyPI.
(Previous run: 2026-09-09, also clean.)

One caveat worth knowing before reading a future clean result: `pip-audit` **skips `torch`**,
because the pinned `2.13.0+cpu` carries a local version identifier PyPI does not recognise. It
prints the skip in a separate table under "No known vulnerabilities found", which is easy to read
as coverage it does not have. The upstream release was checked separately against OSV and is
clean. Any future scan needs the same second step for `torch`.

The second step was run again on 2026-09-16, against the GitHub Advisory Database rather than
osv.dev — OSV's query API takes a POST and the tooling available in that session could only issue a
GET, while the GitHub database is one of OSV's own feeds and is reachable with the CLI already
authenticated here:

```bash
gh api "/advisories?ecosystem=pip&affects=torch@2.13.0" --jq '.[] | "\(.ghsa_id) \(.severity)"'
```

It returned nothing, and the same query on a deliberately old pin (`torch@2.0.0`) returns five
advisories, so the empty answer is coverage and not a broken query.

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

Extended 2026-09-15 by Phase 5 deviation 5.1, which versions `data/extraction/sample-report.json`:
14 findings added, 125 in total. All 14 were read: one is the extraction prompt digest and 13 are
unit ids of the 20 sampled paragraphs, content hashes of the public corpus. The path exclusion below
was deliberately **not** widened to cover the report: one small file is audited entry by entry.

Extended 2026-09-16 by Phase 5 task T13, which versions the extraction archive and its summary:
2 findings added, 127 in total. Both are in `data/extraction/extraction-0107de3ae9b4e4a3.json`: the
sha256 digest of the archive and the extraction prompt digest. The `.jsonl.gz` archive is binary and
raises no finding; it holds the entities and concepts the model read from the public corpus, and no
credential, key or request header was ever written into it.

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
  per audit. Next free id: **SEC-024** (`OBS-NNN` for observations: next free is **OBS-009**).
- A phase exempted from the audit (the command's escape hatch) is recorded here too, with its
  reason: an exempt phase is a decision on record, not a phase nobody looked at.
