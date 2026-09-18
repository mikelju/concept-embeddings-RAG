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

**Eleven, none blocking**: four from Phase 5 (one Medium, three Low), SEC-020 to SEC-023, and
seven Low from Phase 6, SEC-024 to SEC-030. Every one needs a change under `src/`, `.github/` or
the workflow, and the author closed both rounds to source changes, so they are reported and
catalogued rather than fixed — the decision is recorded in each phase's section below. No Critical
and no High finding exists in any audited phase.

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

### Phase 6 — 2026-09-18

Scope: everything the phase added under `src/` (the nine `evaluation/` modules `replacement_run.py`,
`replacement_freeze.py`, `replacement_inputs.py`, `replacement_decision.py`, `outcomes.py`,
`control_reproduction.py`, `continuity.py`, `tango.py`, `entity_diagnostics.py`, `readout.py`, plus
`retrieval/entity_hop.py`, the `fusion.py` and `harness.py` changes, `decision_parameters.py`, the
Phase 6 constants and the three CLI stages), the eleven artifact kinds written under
`data/replacement/` and every loader that reads them back, the inherited loaders they reach
(`nodes/index.py`, `evaluation/navigation.py`, `evaluation/pilot.py`, `nodes/extraction.py`), and
the custom detect-secrets content filter of D20.

What the scanners said: `ruff` with the whole `S` selection, clean; `mypy src` clean over 55 files;
`detect-secrets-hook --baseline .secrets.baseline $(git ls-files)` clean; `pytest` 2,487 passed, 1
skipped. Phase 6 adds no dependency — `pyproject.toml` and `uv.lock` are byte-identical to `main` —
so the 2026-09-16 `pip-audit` result still describes this environment. No `pickle`, `eval`, `exec`,
`subprocess`, `yaml.load` or network call anywhere on the Phase 6 path, and every `np.load` it
reaches passes `allow_pickle=False`.

The phase's own integrity work is unusually complete, and the findings sit in its margins. Every
inherited input is pinned by digest in `config.py` and re-derived on load; the split guard refuses a
test question at every dev door; the freeze is dev-only by construction and refuses a historical
identity carrying a figure anywhere in it; the outcomes recompute their own aggregates and must
equal the run result they name; the diagnostics re-derive each question's metrics from the recorded
context. The committed artifacts were checked against all of it in this session: the sixteen
digest-bearing files under `data/replacement/` match their digests, the decision's `freeze_digest`
and three `outcome_digests` match the files on disk, the test outcomes name the freeze, and
`m1_check` reproduces the pinned M1 (1,210 − 1,155 = 55) from the historical test files.

One theme runs through the seven findings, and it is narrower than the previous phases': **what the
loaders verify is a subset of what the writers recorded**. The decision records the digests of the
outcomes it was computed from and never compares them; it records S, V and N and re-derives only the
state row from them; the freeze records fourteen protocol fields and the test stage compares five;
several file names travel from one artifact into a `Path` without the shape check that `outcomes.py`
applies to the same kind of name.

| ID | Severity | Title | Status |
|---|---|---|---|
| SEC-024 | Low | `load_decision` records the outcome digests and never compares them with the artifacts on disk | Open — reported, not fixed |
| SEC-025 | Low | S, V and N are trusted on load; only the state row is re-derived, so a re-signed verdict verifies against its own contradicting statistic | Open — reported, not fixed |
| SEC-026 | Low | File names taken from artifact content reach `Path` unshaped, where `outcomes.py` shapes the same kind of name | Open — reported, not fixed |
| SEC-027 | Low | The deviation document is checked for existence only, and a relative path may leave `PROJECT_ROOT` through `..` | Open — reported, not fixed |
| SEC-028 | Low | `RunResult.save` still truncates before writing: the six run results are the only Phase 6 artifacts not written atomically | Open (inherited from Phase 1) — reported, not fixed |
| SEC-029 | Low | The test stage re-checks five of the freeze's protocol fields; model, revision, tokenizer, seed, `top_k`, budgets and recall depths are not compared | Open — reported, not fixed |
| SEC-030 | Low | The workflow's baseline-regeneration comment drops both scanner filters, so following it buries the ids in the baseline as audited entries | Open — reported, not fixed |

**Why they are open.** The author scoped this round to audit only: no fix was to be applied, no
`cer` stage re-run and no artifact regenerated, because the test split was read once and re-reading
it would void the experiment. Each finding is written up with its `file:line`, its reachability and
the fix it needs, and none was applied. None is Critical or High, so none blocks the phase.

**None of them affects the validity of `ENTITY_REPLACEMENT_SUPPORTED`.** Every one needs write
access to artifacts that are committed to git, where a change is a visible diff, or is a comment;
SEC-027's deviation and supersession paths are not exercised by the committed freeze (`supersedes`
and `deviation` are null, the control mode is `reused`); and SEC-029's gap is caught indirectly in
`reused` mode by the step-5 reproduction against the historical test figures, which passed.

Recorded rather than fixed: **OBS-009** — every write-once artifact tests `path.exists()` and then
writes through `os.replace`, which overwrites, so two concurrent stage runs could both pass the
check and the second silently replace the first, the freeze included; a local single-operator
pipeline whose stages take minutes to hours. **OBS-010** — the D20 filter's path condition is
`data/replacement/[^/]+[.]json`, directory-wide rather than artifact-specific, so a future `.json`
dropped flat into that directory inherits the skip although the allowlist was reconciled against the
Phase 6 writers only. **OBS-011** — the residue any allowlist of this shape carries: a secret that
*is* a lowercase-hex string of exactly 16, 24, 40 or 64 characters, alone on its line, under an
allowlisted key, inside `data/replacement/*.json`, is skipped. Accepted as the price of not
excluding the directory outright, and written down so the residue is on the record rather than
implied.

**Phase closure: no Critical and no High finding, so the phase may close** with the seven open Low
findings carried as recorded above.

## Deferred audits

An audit that was not run is a decision on record here, never a gap nobody noticed.

### Phase 3 — deferred on 2026-09-14, until the Phase 4 results are in

**Update, 2026-09-18: still open, and the Phase 6 audit deliberately did not widen to it.** That
round was scoped to Phase 6 and none of its seven findings lands in Phase 3 or Phase 4 code. Two
things are worth separating. Phase 4's diffusion path is not on the Phase 6 path at all: Phase 6
inherits the pilot, the node index, the hop run and the navigation traces, and touches no expansion
artifact, so nothing was learned about it either way. Phase 3's `selection.py` **is** on the Phase 6
path — `load_selection`, `selection_digest`, `fit_fusion_weight`, `check_dev_only`,
`check_freeze_precedes` and the shared `serialize_payload` / `digest_of_payload` are all exercised —
and the parts Phase 6 reaches were read in this round without raising a finding. That is incidental
coverage of one module, not the Phase 3 audit: the sweep, the artifact writer and the Phase 3 CLI
stage were not reviewed. The choice between a Phase 3 + 4 pass and a recorded exemption is still the
author's and still unmade.

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

### Content filter for deterministic hex ids (Phase 6, decision D20)

Phase 6 versions its artifacts flat under `data/replacement/`: the dev checks, the freeze, run
results, per-question outcomes, diagnostics, traces, readouts, the test reproduction record and the
decision. Like Phase 5's, they are full of machine-written lowercase hex ids.

**The measurement.** The scanner, run with its default settings and no exclusion over Phase 5's id
artifacts (`pilot.json`, `hop-run.json`, `traces.json`, `gate-decision.json`), reports **1,906**
findings: 1,385 in `pilot.json`, 294 in `hop-run.json`, 225 in `traces.json`, 2 in
`gate-decision.json`. Re-measured on 2026-09-17 with detect-secrets 1.5.0: every one is
`HexHighEntropyString`, and every one is a lowercase hex string of 16, 24, 40 or 64 characters standing
alone as a JSON value (unit ids and short keys; HotpotQA question ids; the resolved model commit;
sha256 digests). `unit_id` lines raise nothing: the scanner's own `is_likely_id_string` skips them.
Phase 6's files carry the same shapes over 600 dev and 1,400 test questions.

**Why not a path exclusion.** Every Phase 6 file mixes those ids with content the scanner must keep
reading: entity forms that came out of an LLM extraction (traces), verbatim texts and file names
(freeze, decision), configuration (run results). A path exclusion would stop scanning all of it.
Phase 5's exclusion above stays byte-identical, is not extended, and covers no Phase 6 file.

**The filter.** `.github/detect_secrets_filters.py::is_deterministic_replacement_id`, standard
library only, registered in `.secrets.baseline` `filters_used` as
`file://.github/detect_secrets_filters.py::is_deterministic_replacement_id` (the one entry added to
the baseline; its results were not regenerated). The hook configures its filters from the baseline,
so CI applies it. It skips a finding **if and only if all four hold**:

1. **Path**: the file name, with `\` normalized to `/`, fully matches
   `data/replacement/[^/]+[.]json` - directly inside the directory, `.json` only.
2. **Detector**: the finding comes from `HexHighEntropyString`.
3. **Line shape**: the line, surrounding whitespace stripped (JSON indentation included), is exactly
   `"<key>": "<hex>"` with an optional trailing comma, `<key>` on the allowlist with `<hex>` of an
   allowed length; or a bare list element `"<hex>"`, optional comma, of length 16 or 24. `<hex>` is
   `[0-9a-f]` only.
4. **Value**: the flagged secret equals that `<hex>`.

**The allowlist** is exactly the set of (key, length) pairs under which the Phase 6 writers put a
lowercase hex value the detector flags, measured on a run of the three stages (completed on
2026-09-17, before T21):
- **24:** `qid`.
- **16:** `p1`, `unit_set_hash`, `prompt_digest`, `corpus_cache_key`, `question_cache_key`,
  `dev_question_cache_key`, and the question-set hashes under `dev` / `test`.
- **40:** `revision`, `resolved_revision`.
- **64:** `digest`, `freeze_digest`, `supersedes`, `node_index_digest`, `extraction_digest`,
  `pilot_digest`, `hop_run_digest`, `traces_digest`, `run_digest`, `outcomes_digest`,
  `first_digest`, `second_digest`, `sha256`, `manifest_sha256`, `token_counts_sha256`, and the
  digests keyed by system name, `dense`, `hybrid-bm25`, `hybrid-entity-hop`.
- **16 or 40:** `expected` / `observed` of the reproduction checks whose figure is itself an id
  (`unit_set_hash`, `revision`).

`unit_id` is deliberately absent: detect-secrets' own `is_likely_id_string` heuristic already skips
it. Keys may contain `-`, for the system names.

**What stays scanned inside `data/replacement/`**: every finding of every other detector; hex values
of any other length or with uppercase letters; a hex value under a key off the list; any line
holding more than that one key and value, so an appended string, a text, an entity form or a file
name is always read; bare 40- or 64-character elements; nested files and non-JSON files. Outside
`data/replacement/` nothing changes.

**How it fails.** detect-secrets 1.5.0 loads a file filter when it scans; if the file or the
function is missing it logs a warning and applies no filter (`settings.get_filters`). A broken
filter brings the findings back and fails the gate; it cannot silence anything. The opposite risk,
the module edited to skip too much, is what `tests/test_security_gate.py` guards (function-level
cases for every allowed and refused shape, an end-to-end run of the real scanner configured from the
real baseline in a repository-shaped temporary tree, and structural checks), and a change to the
module is reviewed the way a baseline change is.

**Reconciled with the writers before T21.** The first allowlist held the contract's key names only.
A toy run of the three stages then showed hex values under keys off the list, which the gate would
have reported on the first real artifacts. The review that D20 calls for completed the list from the
writers themselves.

The completion ran through the whole deviation branch: first freeze, a reused test run stopped by a
failed reproduction, superseding freeze, re-measured test run. On that run it takes every hex-bearing
key, confirms each one in the writers' source, and adds none that no writer emits.

Two tests hold it there:
- `tests/test_cli_replacement.py` repeats that run on the toy pipeline and requires the emitted
  (key, length) pairs to equal the allowlist, no more and no less (`unit_id` excepted, see above);
- `tests/test_security_gate.py` checks every key at its lengths and at every other length, the refused
  shapes for the new keys, and that no exclusion of the baseline was widened.

**Reconciled against the real artifacts, one stage at a time.** Each allowlisted key declares, from
the writers, which artifacts carry it: `checks-dev.json` and the dev runs and outcomes at T21, the
freeze and the dev readings at T22, the test readings and the decision at T23. A data-dependent test
requires the key to occur as soon as one of its artifacts exists, and skips while none does, so a key
no existing artifact emits fails rather than passing unnoticed; a second test requires every hex shape
the real artifacts hold to be covered by the allowlist. Adding a key without declaring where it comes
from fails a third test. `supersedes` carries a digest only in a superseding freeze, so it is required
only if the deviation branch of D14 has produced one.

Verified on the T21 artifacts (2026-09-17): every hex shape written is covered, and the keys whose
artifacts T22 and T23 still have to write are recorded as skipped.

**Inline audits outside `data/replacement/`.** The Phase 6 source and tests hold hex literals the
gate flags and the filter, by design, does not cover: the six pinned digests of D2 in `config.py`
(sha256 digests of public, pipeline-written artifacts) and their copies in `tests/test_config.py`,
the Phase 1 pool hash and the extraction prompt digest in that test, and the toy model revision in
`tests/evaluation/replacement_fixtures.py`. Each carries an inline `# pragma: allowlist secret`
rather than a new baseline entry, so the audit sits on the line it covers and the baseline's results
stay as they were. Verified on 2026-09-17: `detect-secrets-hook --baseline .secrets.baseline
$(git ls-files --cached --others --exclude-standard)` passes over the tracked tree and the new files.

**When regenerating**, pass both filters again, or they are silently dropped with the old baseline:

```bash
uv run detect-secrets scan --exclude-files '^data[\\/](pilot|navigation)[\\/][^\\/]+[.]json$' \
  --filter 'file://.github/detect_secrets_filters.py::is_deterministic_replacement_id' \
  $(git ls-files --cached --others --exclude-standard) > .secrets.baseline
```

The comment beside the CI job in `.github/workflows/security.yml` still prints the command **without
either flag**, which is SEC-030 above: the catalogue has it right and the file a maintainer is most
likely to copy from does not.

**Re-verified adversarially by the Phase 6 audit (2026-09-18).** The four conditions were attacked
one at a time against the real module, and then end to end with the real scanner configured from the
real baseline over a repository-shaped temporary tree. The path condition refuses `./data/...`,
`a/data/...`, a nested directory, `.JSON`, `x.json.bak`, a trailing space, a different case, an
absolute path and `data/replacement/../../.env`, and accepts a Windows backslash spelling — which is
what makes the check run on this machine the check CI runs. The detector condition refuses every
plugin but `HexHighEntropyString`. The line shape refuses a second key on the line, a prefix, a
trailing comment, a doubled comma, a missing or doubled space, a space before the colon, uppercase
hex, a key off the list, an allowlisted key at a length off its list, and a bare 40- or 64-character
element. A flagged secret other than the line's own value is refused. End to end, the hook still
reported a 32-hex under `digest`, a 64-hex under a key off the list, an AWS key id, a basic-auth
URL, a private-key header, a keyword secret, a line holding a second key, and the same allowlisted
line in a file outside the directory — and skipped exactly one case, the allowlisted shape. The two
residues are OBS-010 and OBS-011 in the Phase 6 section above. The Phase 5 path exclusion is
byte-identical and covers no Phase 6 file.

## Conventions

- This file is the whole paperwork of an audit: one row per finding, with id, severity, title and
  status. Findings are presented and argued in chat, not written up here.
- Every Critical or High finding is resolved with a `fix-N` in `docs/plans/fixes/` before the
  phase closes, or the decision to defer it is recorded in the master plan with its reason.
- Finding ids are unique per project: continue the `SEC-NNN` sequence rather than restarting it
  per audit. Next free id: **SEC-031** (`OBS-NNN` for observations: next free is **OBS-012**).
- A phase exempted from the audit (the command's escape hatch) is recorded here too, with its
  reason: an exempt phase is a decision on record, not a phase nobody looked at.
