# concept-embeddings-RAG — User Guide

> How to run the experiment: freeze a benchmark corpus, build the two baseline retrieval systems,
> measure them, and induce the corpus's own concept space. Written for whoever runs or reproduces
> the experiment, not for whoever changes it.

---

## 📋 Contents

- [Before you start](#-before-you-start)
- [The six stages](#-the-six-stages)
- [1. Freeze the corpus](#1--freeze-the-corpus-fetch)
- [2. Build the pool](#2--build-the-pool-build)
- [3. Compute the embeddings](#3--compute-the-embeddings-embed)
- [4. Measure the systems](#4--measure-the-systems-evaluate)
- [5. Induce the concept spaces](#5--induce-the-concept-spaces-induce)
- [6. Name the concepts](#6--name-the-concepts-label)
- [Reading the results](#-reading-the-results)
- [When something refuses to run](#-when-something-refuses-to-run)
- [Questions](#-questions)
- [Known limits](#-known-limits)

---

## 🚀 Before you start

There is no server, no interface and nothing to log into. Everything is one command, `cer`, with
six stages you run in order.

1. Install the environment: `uv sync`
2. Check it works: `uv run pytest` — the first run takes about 35 seconds while torch loads. That
   is not a hang.
3. Run the stages below, in order. **Each stage refuses to run if the previous one has not**, and
   tells you which one to run.

You need an internet connection for stages 1, 3 and 6 only. **Stage 6 is also the only one that
needs an API key and the only one that costs money** — and nothing else depends on it, so you can
stop after stage 5 and still have everything the experiment measures.

---

## 🧭 The six stages

| Stage | What it does | Roughly how long |
|---|---|---|
| `fetch` | Downloads the benchmark and freezes it | ~4 min |
| `build` | Picks the question subset and builds the corpus | ~9 s |
| `embed` | Turns every paragraph into a vector | **~36 min** |
| `evaluate` | Measures both baseline systems and writes the results | ~1 min |
| `induce` | Builds the concept spaces from the corpus | **hours** |
| `label` | Puts names on the concepts, for the report | ~1 h, **costs money** |

Run them like this:

```bash
uv run cer fetch
uv run cer build
uv run cer embed
uv run cer evaluate
uv run cer induce
uv run cer label      # optional, needs a key
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
invalidates the whole experiment — there is no way to undo it afterwards.

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
makes stage 5 unrunnable until you have paid them again.

---

## 4. 📊 Measure the systems (`evaluate`)

Runs both retrieval systems over every question and writes down how well each one did.

### How to run it

```bash
uv run cer evaluate
```

### What you should know

- Two systems are measured: **dense** (meaning-based search) and **BM25** (keyword-based search).
  BM25 is there to keep the comparison honest — it is a genuinely strong rival on this benchmark.
- Both are measured **at the same context budget**, not at the same number of paragraphs. That is
  the only fair comparison: a system that returns more paragraphs is not better if it needs more
  room to do it. Budgets used: **512, 1,024, 2,048 and 4,096 tokens**.
- Neither system is ever shown the candidate paragraphs attached to a question. Both always
  search the full corpus of 19,366.
- Results go to `data/results/`, one file per system and split. **Nothing is ever overwritten** —
  each run writes new files, so a measurement you took last week cannot be silently replaced by
  one you took today.

---

## 5. 🧩 Induce the concept spaces (`induce`)

Discovers the corpus's own concepts and rewrites every paragraph as a short list of them. This is
the representation the later phases will retrieve over.

### How to run it

```bash
uv run cer induce
```

### What you should know

- It builds **four spaces in one go**, one per dictionary size: **512, 1,024, 2,048 and 4,096
  concepts**. Having four is the point — they are compared later.
- ❗ **No size is chosen here.** Which one the project ends up using is decided in Phase 3, against
  development recall. Nothing on this page or in the results picks a winner.
- It **never re-embeds anything**. It reads what stage 3 cached and refuses to start without it.
- ✅ **Safe to interrupt and re-run.** Every piece it writes is filed under the settings that made
  it, so a second run finds its own output and skips past it. You lose only the size that was in
  progress.
- Everything lands in `data/concepts/`.

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

## 6. 🏷 Name the concepts (`label`)

Asks a language model to put a name and a one-line gloss on each concept, so you can read the
space instead of squinting at numbers.

### How to run it

```bash
uv sync --group labeling          # once: installs what this stage needs
cp .env.example .env              # then put your key in it
uv run cer label
```

### What you should know

- 💰 **The only stage that spends money.** It prints an estimate before doing anything, and that
  estimate is taken **before the cache is consulted**, so it is an upper bound.
- The recorded run cost **11.73 USD** — 45% above what was forecast. Expect the same order, not
  the forecast.
- Only **one space is named: the 2,048-concept one.** The other three are read through their
  evidence paragraphs, which costs nothing and shows the same thing.
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

## 📈 Reading the results

Each result file records the numbers **and the exact configuration that produced them** — model,
corpus fingerprint, seed, tokenizer, code version. A number without that is not accepted.

### The baselines

Four measurements matter:

- **Gold Recall** — of the 2 paragraphs needed to answer, how many made it into the context.
- **Full Support** — the honest one: the share of questions where **both** needed paragraphs are
  present. Half the evidence answers nothing.
- **Precision** — of what was included, how much was actually needed.
- **Recall@K** — the same idea at a fixed number of paragraphs, for comparison with published work.

The Phase 1 numbers and what they mean are written up in
[`docs/plans/phase_1/1.results.md`](plans/phase_1/1.results.md). The short version: dense beats
BM25 at every budget, and 12% of questions defeat both — which is the case this project was built
to attack.

### The concept spaces

Written up in [`docs/plans/phase_2/2.results.md`](plans/phase_2/2.results.md), read from the files
on disk rather than recomputed for the report. Three things to take from it:

- **Every space works as a representation.** All four keep paragraphs on 8 to 16 concepts, and even
  the least coherent concept of the smallest space is far above what unrelated paragraphs score.
- ⚠️ **One concept swallows the corpus at the larger sizes.** At 2,048 concepts a single one is
  active on 67% of all paragraphs; at 4,096, on 99.4%. It passes every coherence check — it looks
  like one of the *better* concepts — and only the over-generality measure catches it.
- **That is a warning, not a verdict.** Whether it hurts retrieval is measured in Phase 3 and 4,
  and nothing so far settles it.

---

## 🛠 When something refuses to run

Every stage checks its inputs and says what to do. These are the messages you may see.

| Message | What it means | What to do |
|---|---|---|
| `no frozen corpus found: run 'fetch' first` | Stage 1 has not run | `uv run cer fetch` |
| `no pool found: run 'build' first` | Stage 2 has not run | `uv run cer build` |
| `no embeddings or token counts found: run 'embed' first` | Stage 3 has not run | `uv run cer embed` |
| `embeddings are not cached for this pool` | The corpus changed after the embeddings were made | `uv run cer embed` again |
| `no concept space for k=2048 ...: run 'induce' first` | Stage 5 has not run | `uv run cer induce` |
| `the labelling stage needs the optional dependency group` | Stage 6 is not installed | `uv sync --group labeling` |
| `ANTHROPIC_API_KEY is not set` | Stage 6 has no key | Copy `.env.example` to `.env` and put it there |
| `the API key was rejected; fix the key rather than retrying` | The key is wrong or revoked | Fix the key — re-running will not help |
| `rate limited after the SDK's own retries` | The service is busy | Re-run: it resumes from the cache, and you are not charged twice |
| `the diagnostics of ... cite unit ..., which is not in the pool` | A concept space describes a corpus no longer on disk | Re-run `induce`, or restore the pool it was built from |
| `...has hash X, manifest expects Y` | The corpus file no longer matches its fingerprint | See below |
| `pool hashes to X but the manifest expects Y` | The corpus and the pool no longer agree | `uv run cer build` |
| `unit ... does not hash to its content` | `data/pool.json` was modified | `uv run cer build` |

### 🔴 If the corpus fingerprint does not match

The command stops on purpose. It means the file changed since it was frozen — a bad download, an
interrupted write, or an edit. **Do not work around it.** Either restore the original file, or
delete `data/manifest.json` and re-freeze deliberately — and if you do that, treat every number
you recorded earlier as belonging to the old corpus.

### 🔴 If a concept space refuses to load

Same principle, one layer up. Every file stage 5 writes carries a fingerprint of itself and of the
corpus it came from, and the code that reads it checks both. A refusal means the file is not the
file that was written — truncated by an interrupted run, or edited. Delete it and re-run `induce`;
it rebuilds only what is missing.

---

## ❓ Questions

**Do I have to run all the stages every time?**
No. Run them in order once. After that, only re-run the stage whose inputs changed — and
`evaluate` as often as you like, it is fast.

**Can I stop `embed` halfway and resume?**
No. It saves at the end, so an interrupted run starts over. Give it the 36 minutes.

**Can I stop `induce` or `label` halfway and resume?**
Yes, both. `induce` re-runs skip every space already on disk; `label` resumes from its cache and
never pays for the same concept twice.

**Why is my `data/results/` filling up with files?**
By design. Every run writes new files instead of replacing old ones, so a measurement can never
be lost by accident. Old files are safe to read; none of them is stale in a way that matters,
because each carries the configuration it came from.

**Can I use a different embedding model?**
Yes, but it is a deliberate decision, not a tweak: it invalidates the 36-minute cache, makes the
new numbers incomparable with the recorded ones, and every concept space has to be rebuilt. It is
a one-line change in `src/concept_embeddings_rag/config.py`.

**Does any of this send data anywhere or need an API key?**
Only stage 6. Stages 1 and 3 download from public sources and upload nothing. Stage 6 sends
paragraphs of the public benchmark corpus to the model that names the concepts, and needs a key in
`.env`. Skip it and the project runs with no key and no network calls beyond the two downloads.

**Which dictionary size should I use?**
None of them, yet. That is decided in Phase 3 against development recall, and choosing one now by
reading the Phase 2 numbers is exactly the mistake the results document refuses to make.

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
- **Retrieval over the concept space is not built yet.** Phases 1 and 2 are done: the baselines,
  the harness and the spaces exist. Conceptual retrieval and iterative expansion do not. See
  [`docs/plans/0_master_plan.md`](plans/0_master_plan.md) for what comes next.
