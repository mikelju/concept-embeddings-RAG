---
name: deliver
description: Take finished work on a branch to a reviewed, evidenced pull request that waits for the author - local checks, secret scan, end-to-end evidence, adversarial review in a fresh context, doc pass, push and PR with a risk assessment. Load when implementation is done (end of /6-implementar) or when the author says deliver, ship, open the PR or "entrega". Never merges.
---

# Deliver

The author reviews at the end, in the PR, instead of step by step. This skill makes that review
cheap: the PR states the intent, shows evidence that the change does what was asked, lists what
the pipeline already caught and fixed, and says how risky the change is so the author knows how
deep to look.

**Hard limits.** Never push to `main`, never merge, never enable auto-merge, never force-push.
This repository is public: anything pushed is published.

GitHub reads go through `npx -y gh-axi@0.1.35`, whose output is 40-70 % smaller than `gh` on
this repo's PRs. Only `pr view`, `pr checks`, `pr list`, `run list` and `run view` are allowed;
the guard hook blocks every other gh-axi call. Writes (`pr create`, `pr edit`) stay on `gh` with
`--body-file`. Ignore gh-axi's merge suggestions: merging is the author's.

## 1. Frame the intent

Write 3-6 lines: what was asked, from which spec/plan/conversation, and what "done" means. This
goes to the reviewer and into the PR. If the intent is ambiguous, stop and ask; do not guess.

## 2. Branch and base

- Confirm the current branch is not `main`. The PR base is `main`, unless the branch was cut from a
  phase branch that is not merged yet; then the base is that phase branch (stacked PR) and the PR
  says so.
- `git fetch origin`. If the base advanced, merge it into the branch (no rebase of pushed history).
  Resolve trivial conflicts; stop on conflicts that touch experimental logic or results.

## 3. Local checks

All must pass before the PR is opened:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

A check that already failed on the base branch counts as passing if the change does not make it
worse; list that pre-existing debt in the PR instead of fixing it outside scope. Fix failures
inside the scope of the change, up to two rounds. Never weaken, skip or delete a test
to get green. If something still fails, the PR is opened as a draft with the failure stated.

## 4. What is being committed

- `git status --short` and review every path. Stage with explicit paths, never `git add -A`.
- Never stage: `.env` or any credential, `data/` (raw corpus, caches, measured artifacts unless a
  project rule versions that exact artifact), `.sdd-local/`, `.tmp/`, notebooks with outputs, or an
  author's unapproved draft that the task did not produce.
- Secret scan on the staged files, as CI does:
  `uv run --group security detect-secrets-hook --baseline .secrets.baseline $(git diff --cached --name-only)`.
- Nothing copied from `C:\Python Projects\PEIA\` or `portfolio-private\`.

## 5. End-to-end evidence

Unit tests are not enough. Show the change working through the real path:

- a `uv run cer <stage>` run on dev (or a small fixture) exercising the changed code, with the
  command and the lines of output that matter;
- where a change touches retrieval or evaluation, a reproduction check against a recorded figure
  (for example BGE-small Dense 487 / 600 dev Full Support @2,048) that must still match exactly;
- for documentation-only changes: the links and paths it adds resolve.

A reproduction mismatch is a finding to report and stop on. It is never a failure to fix by
changing code until the number matches. Never open the test split, spend money or rerun a one-shot
historical stage to produce evidence.
If evidence cannot be produced, say why in the PR; do not invent it.

## 6. Adversarial review in a fresh context

Spawn one reviewer subagent (model `opus`) that has not seen the implementation conversation.
Give it: the intent from step 1, the diff range (`git diff <base>...HEAD`), the spec/plan paths and
the `research-protocol` skill path. Ask for findings only, each with file:line and a concrete
failure scenario, covering:

- correctness bugs and spec drift (code does something the approved spec did not ask for, or
  misses something it did);
- dev/test leakage, a choice fitted on test, a changed frozen parameter or threshold;
- overwritten caches or historical artifacts, unlabelled projections, missing provenance;
- non-determinism: missing seeds, unstable tie-breaking;
- secrets, unsafe deserialization, new untrusted inputs.

Then: fix clear findings inside the scope and rerun step 3; send findings with scientific or
product implications (anything that would change what is measured or decided) to the author under
"Decisions for the author" instead of fixing them. One review round, plus a second only if the fixes
were substantial.

## 7. Documentation pass

Update what the change made stale, and nothing else: plan checkboxes, the phase results document if
something was measured, the status table in `docs/plans/0_master_plan.md` and the "Current state"
line in `CLAUDE.md` when a phase or deviation changed state. Historical results are never
rewritten.

## 8. Commit, push, PR

Write the commit message and the PR body to files in the scratchpad with the Write tool and pass
them with `-F` / `--body-file`. The guard hook reads inline shell text and blocks, on purpose, any
inline text that looks like a push to main or an env-file read.

```bash
git commit -F <message file>          # English, explicit paths already staged
git push -u origin <branch>
gh pr create --base <base> --title "<title>" --body-file <tmp file in the scratchpad>
```

If a PR for the branch already exists, update it (`gh pr edit --body-file`) instead of opening a
second one. PR body:

```markdown
## Intent
<step 1>

## What changed
<short list by file or area>

## How it was verified
<commands and results from steps 3 and 5; anything not run, and why>

## Review
<fixed findings, one line each; escalated findings under "Decisions for the author">

## Risk
Low | Medium | High - <why>. Where to look: <files or questions worth the author's time>.
Low: docs, tests, isolated code with no effect on any measured number.
Medium: shared retrieval/evaluation code, new dependencies, cache paths.
High: anything that can change a recorded result, a split, a selection rule or a frozen artifact.

## Not done / limitations
```

## 9. Report

Give the author the PR URL, the risk level and any decisions waiting for them. Stop there. The
merge is the author's.
