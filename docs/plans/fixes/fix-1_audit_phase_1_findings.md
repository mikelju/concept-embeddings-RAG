# fix-1: Phase 1 audit findings

> Origin: `docs/security/audit-2026-09-08-fase-1.md` | Phase: 1 | Status: applied

## Problem

The Phase 1 audit found no Critical or High issues, but eight findings, four of them Medium and
all four of the same family: **integrity controls that exist in the code and are never exercised
by the pipeline**. For a project whose deliverable is measured evidence, a silently falsified
measurement is the most expensive failure available, so all eight are fixed together rather than
deferred. The re-audit then found two more, both introduced or left open by this fix itself;
they are listed below and fixed here too.

| ID | Severity | Summary |
|---|---|---|
| SEC-001 | Medium | Corpus hash recorded, never compared. `ensure_corpus` had no production caller |
| SEC-002 | Medium | Embedding model resolved from `main`, not a commit; sidecar recorded `"main"` |
| SEC-003 | Medium | `load_pool` trusted content-derived ids without re-deriving them |
| SEC-004 | Medium | Unbounded pagination and unbounded response size when fetching the corpus |
| SEC-005 | Low | SHA-1 for content addressing without `usedforsecurity=False` |
| SEC-006 | Low | Result filename built from unvalidated `split` / `system` |
| SEC-007 | Low | CI secret scan could not fail; dependency gate ran on Python 3.11 |
| SEC-008 | Low | `CorpusManifest` validated field names but not their types |
| OBS-001 | Info | Missing type hints on four deserialization-boundary functions |
| SEC-009 | Low | *(found in the re-audit)* the SEC-001 fix covered the "corpus present" branch of `cmd_fetch` and not the "re-assemble" one |
| SEC-010 | Low | *(found in the re-audit)* the SEC-003 fix verified units but not that each question's gold ids still resolved |

## Approach

Two constraints shaped the work.

**No metric may change.** The Phase 1 numbers are recorded in `1.results.md` and every later
phase is judged against them. Any fix that altered a ranking, a budget or a gold id would void
them. The fixes are therefore all verification added *around* the existing computation, never
inside it.

**The embedding cache must survive.** `cache_key` derives from the model revision, so pinning
SEC-002 changes the key from `a28365b7ed90ed10` to `bacd74e7f468f0d4` and would orphan the
36-minute artifact. The local HuggingFace snapshot proves `main` resolved to
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` when that cache was built, so the vectors under both
keys are the same vectors. The artifact is copied to the new key rather than recomputed, and the
claim is then verified the only way that counts: by re-running `cer evaluate` and diffing the
metrics against the recorded ones.

## Changes

| File | Change | Finding |
|---|---|---|
| `cli.py` | `cmd_fetch` loads any existing manifest and compares the on-disk hash against it before accepting the file; `cmd_build` verifies the raw corpus against `manifest.sha256` before parsing, and the pool against `manifest.unit_set_hash` after building | SEC-001, SEC-003 |
| `cli.py` | `_cache_key_for` and `cmd_embed` record `resolved_revision`; type hints added | SEC-002, OBS-001 |
| `config.py` | `EMBEDDING_REVISION` pinned to the commit; the false comment about the sidecar removed | SEC-002 |
| `backend.py` | `resolved_revision()` asks the Hub which commit was actually loaded, falling back to the requested revision offline | SEC-002 |
| `cache.py` | `usedforsecurity=False` on both SHA-1 sites; sidecar carries `resolved_revision` | SEC-002, SEC-005 |
| `pool.py` | `load_pool` re-derives every unit id and raises on mismatch; `usedforsecurity=False`; type hints on `save_pool` / `load_pool`; module-level imports | SEC-003, SEC-005, OBS-001 |
| `split.py` | `usedforsecurity=False` | SEC-005 |
| `download.py` | `_http_fetcher` streams with a byte ceiling; `redownload_on_mismatch` now defaults to `False` (fail closed, keep the evidence) | SEC-004, DEF-004 |
| `hf_source.py` | `fetch_split` takes a `max_rows` ceiling and raises past it | SEC-004 |
| `manifest.py` | `__post_init__` validates the shape of `sha256`, the type of `seed` and of the split sizes | SEC-008 |
| `harness.py` | `validate()` rejects `system` / `split` values that are not safe as filename components | SEC-006 |
| `budget.py` | `TokenCounter` passes `revision`; type hint on `count_units` | SEC-002, OBS-001 |
| `.github/workflows/security.yml` | Python 3.12 to match `requires-python`; the secret scan now fails the job on a finding; `pip-audit` runs against the locked tree | SEC-007 |
| `pyproject.toml` | ruff `select` gains `S`, with `S101` ignored under `tests/` | DEF-003 |
| `cli.py` | the re-assemble branch of `cmd_fetch` verifies the new digest too, so both branches converge on one check | SEC-009 |
| `pool.py` | `load_pool` also proves every question's gold ids resolve to a unit in the pool | SEC-010 |

## Deliberately not done

- **SHA-1 was not replaced with SHA-256.** It feeds `cache_key` and `unit_id_for`, so changing it
  would invalidate the embedding cache and make the recorded numbers incomparable with future
  ones. `usedforsecurity=False` closes the audit finding at zero cost. A wider id, if ever wanted,
  belongs in a deliberate cache rebuild between phases.
- **Trust-on-first-use was not engineered away** (OBS-002). Whoever can write `data/` can write
  both the corpus and the manifest that vouches for it. Git history on the versioned manifest is
  the real tamper-evidence, and it already exists.
- **`pip-audit` was still not run.** It is not installed locally and installing it needs
  authorization. The CI job is now capable of running it; the gap stays declared in
  `docs/security/README.md` until it does.

## Verification

- Full suite green: **109 tests**, 15 of them new regression tests written against the
  findings, each verified to fail without its fix.
- `ruff check` (now with `S`), `ruff format --check` and `mypy src` all clean.
- `cer evaluate` re-run end to end against the migrated cache: all **60 metrics** across both
  systems and both splits reproduce the values in `1.results.md` to the last recorded digit.
- Re-audit: [`audit-2026-09-08-fase-1-2.md`](../../security/audit-2026-09-08-fase-1-2.md).
  It found two further issues, both in the fix itself (SEC-009, SEC-010); both are fixed and
  tested here. That is the argument for having run the second pass at all.
