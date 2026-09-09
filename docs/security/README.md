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

None. Every finding from both passes is closed, each held closed by a regression test.
Phase 1 has no blocking security work outstanding.

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

## Dependency scanning

`pip-audit` 2.10.1 is installed in the `security` dependency group. Run it with:

```bash
uv run pip-audit --desc
```

**Last run: 2026-09-08 — 0 known vulnerabilities.**

One caveat worth knowing before reading a future clean result: `pip-audit` **skips `torch`**,
because the pinned `2.13.0+cpu` carries a local version identifier PyPI does not recognise. It
prints the skip in a separate table under "No known vulnerabilities found", which is easy to read
as coverage it does not have. The upstream release was checked separately against OSV and is
clean. Any future scan needs the same second step for `torch`.

A clean scan is a statement about the day it ran, not a property of the code.

## Conventions

- This file is the whole paperwork of an audit: one row per finding, with id, severity, title and
  status. Findings are presented and argued in chat, not written up here.
- Every Critical or High finding is resolved with a `fix-N` in `docs/plans/fixes/` before the
  phase closes, or the decision to defer it is recorded in the master plan with its reason.
- Finding ids are unique per project: continue the `SEC-NNN` sequence rather than restarting it
  per audit. Next free id: **SEC-012**.
- A phase exempted from the audit (the command's escape hatch) is recorded here too, with its
  reason: an exempt phase is a decision on record, not a phase nobody looked at.
