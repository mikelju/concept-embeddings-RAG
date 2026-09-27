# Fix 2: `code_commit` ignored uncommitted code

## Phase / functionality affected
The provenance every measured artifact records since Phase 9: `_git_commit()` in
`src/concept_embeddings_rag/cli.py`, written into each artifact as `code_commit` (and `commit`
in the pass markers). It surfaced in Phases 11 and 12.

## Symptom
When a stage ran from code that was not yet committed, the artifact recorded the previous
commit. Phase 11's `fit.json` records `a43a4df` although it ran from code later committed as
`77f6fb1`. Phase 12's S2/S4 artifacts record `92ce7c1` although they ran from `58d5a84`, and its
`fit.json` records `58d5a84` although it ran from `ec6566b`. Both results documents state the
mismatch. No figure is affected, but the recorded commit did not identify the code that
produced the artifact.

## Root cause
`_git_commit()` returned `git rev-parse HEAD` and did not look at the working tree.

## Fix
`_git_commit()` also runs `git status --porcelain -- src pyproject.toml uv.lock`. If those paths
hold any uncommitted change, untracked files included, it returns `<sha>-dirty`. Changes outside
the code paths (docs, `data/`, local settings) do not decide what code ran, so they do not mark
the run dirty. Outside a checkout it still returns `unknown`.

The fix is prospective. The Phase 11 and Phase 12 artifacts keep what they recorded, and their
results documents already state the lag.

## User-visible change
Only for future runs. A stage run from uncommitted code records `<sha>-dirty` instead of a bare
sha. No stage refuses to run.

## Files changed
- `src/concept_embeddings_rag/cli.py`: `_git_commit()` and `_CODE_PATHS`.
- `tests/test_git_commit.py` (new): a clean tree gives the bare sha; modified or untracked code
  is marked dirty; changes outside the code paths are not; outside a checkout it gives
  `unknown`.

## Cross-reference
- Recorded in: `docs/plans/0_master_plan.md`, "Corrective fixes" section. The function is shared
  by every phase since 9, so no single phase plan owns it.
- Found by the adversarial review of Phase 12 (PR #17).
