# concept-embeddings-RAG — User Guide

> How to run the experiment: freeze a benchmark corpus, compute its embeddings, induce the corpus's
> own concept spaces, choose one on development data, and measure every retrieval system against
> the baselines. Written for whoever runs or reproduces the experiment, not for whoever changes it.

---

## 📋 Contents

- [Before you start](#-before-you-start)
- [The stages](#-the-stages)
- [1. Freeze the corpus](#1--freeze-the-corpus-fetch)
- [2. Build the pool](#2--build-the-pool-build)
- [3. Compute the embeddings](#3--compute-the-embeddings-embed)
- [4. Induce the concept spaces](#4--induce-the-concept-spaces-induce)
- [5. Name the concepts](#5--name-the-concepts-label)
- [6. Choose the concept space](#6--choose-the-concept-space-select)
- [7. Measure the systems](#7--measure-the-systems-evaluate)
- [8. Choose the expansion](#8--choose-the-expansion-expand)
- [Reading the results](#-reading-the-results)
- [When something refuses to run](#-when-something-refuses-to-run)
- [Questions](#-questions)
- [Known limits](#-known-limits)

---

## 🚀 Before you start

There is no server, no interface and nothing to log into. Everything is one command, `cer`, with
eight stages you run in order.

1. Install the environment: `uv sync`
2. Check it works: `uv run pytest` — the first run takes about 35 seconds while torch loads. That
   is not a hang.
3. Run the stages below, in order. **Each stage refuses to run if the previous one has not**, and
   tells you which one to run.

You need an internet connection for stages 1, 3 and 5 only. **Stage 5 is also the only one that
needs an API key and the only one that costs money** — and nothing else depends on it, so you can
skip it and still have everything the experiment measures.

📦 **Two decisions already ship with the repository.** The files that stages 6 and 8 freeze —
`data/selection/selection.json` and `data/expansion/expansion.json` — are versioned, because every
recorded number is bound to them. On a fresh clone you normally **do not run** `select` or `expand`:
`evaluate` reads those files. Running either stage again means re-deciding, which is covered in
their sections.

---

## 🧭 The stages

| Stage | What it does | Roughly how long |
|---|---|---|
| `fetch` | Downloads the benchmark and freezes it | ~4 min |
| `build` | Picks the question subset and builds the corpus | ~9 s |
| `embed` | Turns every paragraph into a vector | **~36 min** |
| `induce` | Builds the concept spaces from the corpus | **hours** |
| `label` | Puts names on the concepts, for the report | ~1 h, **costs money** |
| `select` | Chooses the concept space and fits the hybrids, on dev only | ~3 min of sweep |
| `evaluate` | Measures the systems on both splits and writes the results | ~1 min |
| `expand` | Chooses the expansion setting, on dev only | not recorded |

Run them like this:

```bash
uv run cer fetch
uv run cer build
uv run cer embed
uv run cer induce
uv run cer label      # optional, needs a key
uv run cer select     # skip on a clone: the decision ships with the repository
uv run cer evaluate
uv run cer expand     # skip on a clone, for the same reason
uv run cer evaluate --systems expansion,expansion-conceptual
```

💡 **You only pay the long stages once.** Everything is cached, so re-running `evaluate` after
that takes about a minute, and re-running `induce` finds its own output and skips straight past it.

---

## 1. 📥 Freeze the corpus (`fetch`)

Downloads the HotpotQA benchmark and records a fingerprint of it, so every measurement you ever
take refers to exactly the same data.

### How to run it

```bash
uv run cer fetch
```

### What you should know

- It downloads **7,405 questions** page by page, which is why it takes a few minutes. If the
  server asks it to slow down, it waits and retries on its own — you will see `[WARN] rate
  limited`. Let it finish.
- The download happens **once**. Run it again and it verifies what is already on disk instead of
  downloading anything.
- The fingerprint lives in `data/manifest.json`. **Keep that file** — it is what lets you prove
  later that your corpus is the corpus your numbers came from.
- If the file on disk no longer matches its fingerprint, the command **stops** rather than
  quietly accepting it. That is deliberate.

---

## 2. 🧱 Build the pool (`build`)

Picks the questions to work with and turns all their paragraphs into one searchable corpus.

### How to run it

```bash
uv run cer build
```

### What you should know

- It selects **2,000 questions** and splits them into **600 for development** and **1,400 for
  test**. The selection is fixed: same seed, same questions, every time, on any machine.
- All the paragraphs from all those questions become **one shared corpus of 19,366 paragraphs**.
  Both splits search the same corpus, so neither gets an easier problem than the other.
- A paragraph is never cut in half or merged with another. Whole paragraphs only.
- Running it twice produces exactly the same result, so it is safe to repeat.

### 🔒 The rule that matters

**Never make a decision by looking at the test numbers.** The development split exists for
tuning; the test split is measured once, at the end. Looking at test while adjusting anything
invalidates the whole experiment — there is no way to undo it afterwards. Stages 6 and 8, the two
that decide anything, refuse a test question at the door.

---

## 3. 🧮 Compute the embeddings (`embed`)

Turns each of the 19,366 paragraphs into a vector, and counts how long each one is.

### How to run it

```bash
uv run cer embed
```

### What you should know

- ⏳ **This is the slow one: around 36 minutes.** It downloads the language model the first time,
  then works through the corpus. Leave it running.
- It only ever does this **once**. The result is saved in `data/cache/`, and every later run reads
  it instead of recomputing. A second `embed` finishes in seconds.
- The cache is labelled with the settings that produced it. Change the model and you get a new
  cache file — the old one is never overwritten, so previous results stay reproducible.

### ⚠️ Do not delete `data/cache/`

Deleting it costs you 36 minutes of computer time to get back exactly the same numbers — and it
makes stages 4, 6, 7 and 8 unrunnable until you have paid them again.

---

## 4. 🧩 Induce the concept spaces (`induce`)

Discovers the corpus's own concepts and rewrites every paragraph as a short list of them. This is
the representation stages 6 to 8 retrieve over.

### How to run it

```bash
uv run cer induce
```

### What you should know

- It builds **four spaces in one go**, one per dictionary size: **512, 1,024, 2,048 and 4,096
  concepts**. Having four is the point — stage 6 compares them.
- ❗ **No size is chosen here.** That is stage 6's job, made against development recall.
- It **never re-embeds anything**. It reads what stage 3 cached and refuses to start without it.
- ✅ **Safe to interrupt and re-run.** Every piece it writes is filed under the settings that made
  it, so a second run finds its own output and skips past it. You lose only the size that was in
  progress.
- Everything lands in `data/concepts/`. It is not versioned, so **a clone has to run this stage**
  before `evaluate` can use the decisions that ship with the repository.

### How long it takes

| Size | Time on this machine |
|---:|---|
| 512 | ~1 min — an upper bound, it shared the machine with a test run |
| 1,024 | ~4 min |
| 2,048 | ~10 min |
| 4,096 | ⚠️ **not measured** — the machine slept mid-run, so the 3.5 h on the clock is calendar time, not work |

Budget an evening for the full sweep and do not read the recorded total as a benchmark.

### Messages you will see

- `[INFO] coding alpha ... -> 10.01 active concepts per unit` — the tuning that keeps each
  paragraph on **8 to 16 concepts**. Too few and the representation collapses to one concept per
  paragraph, which is the failure this whole approach exists to avoid.
- `[INFO] k=2048: 2100 atoms -> 2048 after deduplication` — near-duplicate concepts merged.
- `[INFO] coherence 0.537 against a null of 0.429` — how alike the paragraphs of a concept are,
  next to what unrelated paragraphs of this corpus score. The second number is what makes the
  first readable.
- `[WARN] the sweep has used ... of its 7200 s budget` — ⚠️ **this guard is checked after a size
  finishes, never during one.** It will not cut a long run short, and seeing it does not mean your
  sweep was truncated partway.

---

## 5. 🏷 Name the concepts (`label`)

Asks a language model to put a name and a one-line gloss on each concept, so you can read the
space instead of squinting at numbers.

### How to run it

```bash
uv sync --group labeling          # once: installs what this stage needs
cp .env.example .env              # then put your key in it
uv run cer label                  # names the 2,048-concept space
uv run cer label --k 512          # or any other size
```

### What you should know

- 💰 **The only stage that spends money.** It prints an estimate before doing anything, and that
  estimate is taken **before the cache is consulted**, so it is an upper bound.
- The recorded run cost **11.73 USD** — 45% above what was forecast. Expect the same order, not
  the forecast.
- By default **one space is named: the 2,048-concept one.** `--k` names another — the reports read
  the 512-concept space this way once stage 6 chose it.
- ✅ **You never pay twice.** Every answer is cached on disk per concept. Interrupt it, re-run it,
  and it resumes where it stopped. A call that failed was never billed.
- If the service is overloaded it retries up to **8 times** on its own, backing off between
  attempts. A **rejected key aborts immediately** instead of burning retries on a problem
  retrying cannot fix.
- 🔁 **The names are not reproducible.** The same concept can come back with a different name on a
  different day. The saved file plus its cache *is* the record — re-running reads it back rather
  than regenerating it.

### 🔒 Nothing depends on this

Retrieval never reads a name. A concept is used as the vector the corpus produced for it, so the
labels exist for you and for the report and for nothing else. **Skip this stage entirely and no
measurement in this project changes.**

Your key goes in `.env` and nowhere else. It is read when a call is made, never printed, never
written into any file the project produces.

---

## 6. 🎯 Choose the concept space (`select`)

Decides, on the development questions alone, which of the four concept spaces the retrieval
systems use and how the two hybrid systems weigh their signals — then freezes those decisions in a
file every later stage reads.

### How to run it

```bash
uv run cer select
```

### What you should know

- It needs stages 3 and 4: the cached embeddings and **all four** concept spaces.
- It makes **four decisions, in a fixed order**: the space (its size, how each paragraph's row is
  scaled, how the question is turned into concepts); then whether rare concepts should count for
  more; then how System B combines dense search with the concept space; then the same fitting for
  the **control**, dense combined with BM25. The control is fitted by the same code so that a win
  for System B could be credited to the concepts rather than to combining two signals as such.
- **Only the 600 development questions are measured.** The test questions are turned into vectors
  and cached in `data/cache/questions/`, so stage 7 does not have to load the model — a vector is
  not a measurement, and the sweep refuses a test question regardless.
- The sweep took **2 min 59 s** on this machine, not counting loading the corpus and caches.
- It writes `data/selection/selection.json`, fingerprinted. Every result built on it records which
  freeze it came from.

### 🔒 It freezes once

**`select` never writes over an existing selection** — and the repository ships one. On a clone it
therefore runs the whole sweep and only then stops with `a selection is already frozen at ...`.
You do not need it: stage 7 reads the shipped file. If you genuinely mean to re-decide, move the
existing file aside yourself first. That is a second freeze, which the project treats as a
deviation to be written down, and every number measured under the old file belongs to the old file.

What the recorded run chose, and why, is in
[`docs/plans/phase_3/3.results.md`](plans/phase_3/3.results.md): the 512-concept space, and a
fusion weight that put everything on dense.

### Messages you will see

- `[INFO] space: ... (runner-up ..., margin ...)` — the winning space and how close it was.
- `the margin was inside what 600 questions resolve, so structure decided it` — the difference was
  too small for 600 questions to tell apart, so the rule declared in advance broke the tie.
- `[INFO] damping: ... over ... by ...` and `[INFO] fusion: ... at w=...` — the other decisions.
- `[OK] 4 decisions frozen at ... for k=...` — done.

---

## 7. 📊 Measure the systems (`evaluate`)

Runs the retrieval systems over every question of both splits and writes down how well each one
did.

### How to run it

```bash
uv run cer evaluate                                          # the five systems of Phases 1-3
uv run cer evaluate --systems expansion,expansion-conceptual # the expansion, after stage 8
```

### What you should know

- **It needs a frozen selection.** It will not read the test split on a configuration that is still
  open, so it runs after stage 6 — or, on a clone, after stage 4 has put back the four spaces the
  shipped selection names.
- Without `--systems`, five systems are measured: **dense** (meaning-based search), **BM25**
  (keyword-based search), **conceptual** (the concept space alone), **hybrid-conceptual**
  (System B: dense + concepts) and **hybrid-bm25** (the control: dense + BM25). BM25 is there to
  keep the comparison honest — it is a genuinely strong rival on this benchmark.
- `--systems` measures only the systems it names. The two expansion systems — `expansion`
  (System C) and `expansion-conceptual` (the same walk started from the concept space) — are
  **built only when named**, and need stage 8's freeze. That is what stops a second test result
  for every other system from being written beside them.
- **Nothing is fitted here.** Every weight and setting is read from the frozen files, and each
  result is checked to have been written after the freeze it was built under.
- All systems are measured **at the same context budget**, not at the same number of paragraphs.
  That is the only fair comparison: a system that returns more paragraphs is not better if it needs
  more room to do it. Budgets used: **512, 1,024, 2,048 and 4,096 tokens**.
- No system is ever shown the candidate paragraphs attached to a question. All of them search the
  full corpus of 19,366.
- Results go to `data/results/`, one file per system and split. **Nothing is ever overwritten** —
  each run writes new files, so a measurement you took last week cannot be silently replaced by
  one you took today. On a clone, a re-run should reproduce the versioned files exactly: compare
  them, never replace them.

---

## 8. 🌊 Choose the expansion (`expand`)

Tries the iterative expansion — System C, which walks from the paragraphs a question retrieved to
the concepts they share and back — across its grid of settings on development data, chooses one
setting, and freezes it.

### How to run it

```bash
uv run cer expand
uv run cer evaluate --systems expansion,expansion-conceptual
```

### What you should know

- **It inherits stage 6's decisions** — the space, the scaling, the rarity weighting, the question
  operator — by reading and verifying the frozen selection, never by retyping them. If that file is
  missing, modified, or names a space that is not on disk, it refuses to start.
- It measures **24 settings on the 600 development questions**: 4 values of how strongly each round
  returns to the starting paragraphs, times 3 ways of normalizing the walk, times 2 starting points
  (dense, which is System C proper, and the concept space, which isolates what the walk itself
  adds). The phase may spend at most 40 such passes on dev, and the frozen file records how many it
  did.
- When the walk stops — a threshold on how much changed, and a cap of five rounds — is fixed in the
  configuration, not tuned.
- It writes `data/expansion/expansion.json`, fingerprinted and bound to the selection it inherited.
- The sweep's wall-clock time was not recorded. Per question, a walk costs about 3.7 ms against
  0.8 ms for dense search.

### 🔒 It freezes once, and says so first

Like `select`, it never writes over an existing freeze, and the repository ships one — but `expand`
checks **before** the sweep and stops at once with `a selection is already frozen at ...`. On a
clone, go straight to `evaluate --systems expansion,expansion-conceptual`. Re-deciding means moving
the file aside deliberately, with the same consequences as in stage 6.

What the recorded run chose, and why the result is negative, is in
[`docs/plans/phase_4/4.results.md`](plans/phase_4/4.results.md).

### Messages you will see

- `[INFO] sweeping 24 cells (4 restarts x 3 normalizations x 2 arms) over 600 dev questions`
- `[INFO] cell: ... at full_support ... (runner-up ..., margin ..., resolution ...)` — the chosen
  setting, its margin, and the smallest difference 600 questions can resolve.
- `the margin was inside what 600 questions resolve, so cost decided it` — the cheaper setting won
  the tie, by the rule declared in advance.
- `[OK] frozen at ... after 24 of 40 dev evaluations -> ...` — done.

---

## 📈 Reading the results

Each result file records the numbers **and the exact configuration that produced them** — model,
corpus fingerprint, seed, tokenizer, code version, and for the later systems the freeze they were
built under. A number without that is not accepted.

### The metrics

- **Gold Recall** — of the 2 paragraphs needed to answer, how many made it into the context.
- **Full Support** — the honest one: the share of questions where **both** needed paragraphs are
  present. Half the evidence answers nothing.
- **Precision** — of what was included, how much was actually needed.
- **Recall@K** — the same idea at a fixed number of paragraphs, for comparison with published work.

### The baselines — Phase 1

Written up in [`docs/plans/phase_1/1.results.md`](plans/phase_1/1.results.md). The short version:
dense beats BM25 at every budget, and 12% of questions defeat both — which is the case this project
was built to attack.

### The concept spaces — Phase 2

Written up in [`docs/plans/phase_2/2.results.md`](plans/phase_2/2.results.md), read from the files
on disk rather than recomputed for the report:

- **Every space works as a representation.** All four keep paragraphs on 8 to 16 concepts, and even
  the least coherent concept of the smallest space is far above what unrelated paragraphs score.
- ⚠️ **One concept swallows the corpus at the larger sizes.** At 2,048 concepts a single one is
  active on 67% of all paragraphs; at 4,096, on 99.4%. It passes every coherence check — it looks
  like one of the *better* concepts — and only the over-generality measure catches it.

### Retrieving through concepts — Phase 3

Written up in [`docs/plans/phase_3/3.results.md`](plans/phase_3/3.results.md). The concept space
alone retrieves far worse than either baseline, and System B's fitted weight is all on dense, so
System B *is* dense. The control, fitted the same way, gains about 4 points of Full Support on test.

### The expansion — Phase 4

Written up in [`docs/plans/phase_4/4.results.md`](plans/phase_4/4.results.md). System C ties dense
and loses to the control at every budget, because the walk returns the paragraphs it started from,
reordered, and never brings a new one in.

### The closure

[`docs/plans/phase_4/4.1_research_line_closure.md`](plans/phase_4/4.1_research_line_closure.md)
says what those results rule out and — just as prominently — what they do not: concepts extracted
from the text itself were never tested. It also quotes an exploratory, dev-only diagnostic of
which "second hop" finds a paragraph dense missed. That one regenerates with:

```bash
uv run python scripts/second_hop_diagnostic.py
```

It needs stages 2-4 on disk, reads no test question, decides nothing, and writes
`data/diagnostics/second_hop-dev.json`.

---

## 🛠 When something refuses to run

Every stage checks its inputs and says what to do. These are the messages you may see.

| Message | What it means | What to do |
|---|---|---|
| `no frozen corpus found: run 'fetch' first` | Stage 1 has not run | `uv run cer fetch` |
| `no pool found: run 'build' first` | Stage 2 has not run | `uv run cer build` |
| `no embeddings or token counts found: run 'embed' first` | Stage 3 has not run | `uv run cer embed` |
| `embeddings are not cached for this pool` | The corpus changed after the embeddings were made | `uv run cer embed` again |
| `no concept space for k=2048 ...: run 'induce' first` | Stage 4 has not run | `uv run cer induce` |
| `the labelling stage needs the optional dependency group` | Stage 5 is not installed | `uv sync --group labeling` |
| `ANTHROPIC_API_KEY is not set` | Stage 5 has no key | Copy `.env.example` to `.env` and put it there |
| `the API key was rejected; fix the key rather than retrying` | The key is wrong or revoked | Fix the key — re-running will not help |
| `rate limited after the SDK's own retries` | The service is busy | Re-run: it resumes from the cache, and you are not charged twice |
| `the diagnostics of ... cite unit ..., which is not in the pool` | A concept space describes a corpus no longer on disk | Re-run `induce`, or restore the pool it was built from |
| `the test split may not be read on a configuration that is still open (...)` | No usable frozen selection — on a clone, usually because the spaces it names are not induced yet | Read the reason in brackets: `uv run cer induce`, or `uv run cer select` if there is no selection at all |
| `the frozen selection names a space that is not on disk` | Stage 4 has not produced the space stage 6 chose | `uv run cer induce` |
| `a selection is already frozen at ...` | Stage 6 or 8 was asked to decide again | Nothing, normally — see "It freezes once" in stage 6 |
| `the expansion systems inherit a Phase 3 configuration and there is none frozen` | `--systems` named the expansion without a stage 6 freeze | `uv run cer select` (or restore the shipped file) |
| `the expansion systems may not be measured on an open configuration` | Stage 8's freeze is missing or does not match | `uv run cer expand` (or restore the shipped file) |
| `unknown system(s) [...]` | `--systems` was given a name it does not know | Use the names the message lists |
| `...has hash X, manifest expects Y` | The corpus file no longer matches its fingerprint | See below |
| `pool hashes to X but the manifest expects Y` | The corpus and the pool no longer agree | `uv run cer build` |
| `unit ... does not hash to its content` | `data/pool.json` was modified | `uv run cer build` |

### 🔴 If the corpus fingerprint does not match

The command stops on purpose. It means the file changed since it was frozen — a bad download, an
interrupted write, or an edit. **Do not work around it.** Either restore the original file, or
delete `data/manifest.json` and re-freeze deliberately — and if you do that, treat every number
you recorded earlier as belonging to the old corpus.

### 🔴 If a concept space or a freeze refuses to load

Same principle, one layer up. Every file stages 4, 6 and 8 write carries a fingerprint of itself
and of the corpus it came from, and the code that reads it checks both. A refusal means the file is
not the file that was written — truncated by an interrupted run, or edited. For a concept space,
delete it and re-run `induce`; it rebuilds only what is missing. For a freeze, restore it from the
repository rather than regenerating it, unless re-deciding is what you mean.

---

## ❓ Questions

**Do I have to run all the stages every time?**
No. Run them in order once. After that, only re-run the stage whose inputs changed — and
`evaluate` as often as you like, it is fast.

**On a fresh clone, what is the shortest path to the recorded numbers?**
`fetch`, `build`, `embed`, `induce`, then `evaluate` and `evaluate --systems
expansion,expansion-conceptual`. The two decisions ship with the repository, so `select` and
`expand` are not needed — and would refuse to overwrite them anyway.

**Can I stop `embed` halfway and resume?**
No. It saves at the end, so an interrupted run starts over. Give it the 36 minutes.

**Can I stop `induce` or `label` halfway and resume?**
Yes, both. `induce` re-runs skip every space already on disk; `label` resumes from its cache and
never pays for the same concept twice.

**Can I stop `select` or `expand` halfway?**
Yes, but they resume nothing: each writes its file once, at the end, whole or not at all. An
interrupted run leaves no freeze behind and starts over.

**Why is my `data/results/` filling up with files?**
By design. Every run writes new files instead of replacing old ones, so a measurement can never
be lost by accident. Old files are safe to read; none of them is stale in a way that matters,
because each carries the configuration it came from.

**Can I use a different embedding model?**
Yes, but it is a deliberate decision, not a tweak: it invalidates the 36-minute cache, makes the
new numbers incomparable with the recorded ones, and every concept space and both freezes have to
be rebuilt. It is a one-line change in `src/concept_embeddings_rag/config.py`.

**Does any of this send data anywhere or need an API key?**
Only stage 5. Stages 1 and 3 download from public sources and upload nothing. Stage 5 sends
paragraphs of the public benchmark corpus to the model that names the concepts, and needs a key in
`.env`. Skip it and the project runs with no key and no network calls beyond the two downloads.

**Which dictionary size should I use?**
The 512-concept space, chosen by stage 6 on development recall and frozen. Picking a different one
by reading the results pages is exactly the mistake the freeze exists to prevent.

---

## ⚠️ Known limits

- **This measures one thing, and only one.** Whether the paragraphs needed to answer a question
  were retrieved — not whether an answer is any good. Nothing writes answers.
- **The results are internally comparable, not comparable with published HotpotQA figures.** The
  standard setup ranks 10 candidates per question; this one searches 19,366 paragraphs. Harder
  than one, far easier than the other.
- **The benchmark only tests questions whose hops were laid out in advance by an annotator.** The
  harder case this project is ultimately aiming at — knowledge the question never hints at — is
  not measured by any standard benchmark, and is not claimed here.
- **`embed` is slower on this machine than it would be elsewhere.** The Windows-on-ARM build of
  PyTorch lacks some optimizations; the same work takes 3-5 minutes on a typical x86 machine.
  It is a one-off cost.
- **The cost of the largest concept space is unknown.** The machine slept during that run, so the
  recorded time is calendar time. Recovering the real figure would mean rebuilding it from
  scratch, which would throw away the space the recorded numbers describe. It was not done, and
  the gap is declared rather than estimated.
- **Concept names are the least trustworthy thing here.** They are not reproducible, they were
  produced by a model reading ten paragraphs, and nothing in the project reads them back. Treat
  them as a reading aid, never as evidence.
- **Some Phase 4 readings come from no stage.** The count of new paragraphs per setting, the run
  with no return to the starting paragraphs at all, the expansion's failure analysis and the
  per-question traces in `data/traces/` were produced by a one-off report script that is not in the
  repository. The figures are quoted in the Phase 4 report; regenerating them from a clone is not
  possible today.
- **The first research line is closed, and negative.** Concepts induced from the embeddings were
  built, measured and ruled out on this benchmark. Concepts extracted from the text — the next
  phase — are not built yet. See [`docs/plans/0_master_plan.md`](plans/0_master_plan.md).
