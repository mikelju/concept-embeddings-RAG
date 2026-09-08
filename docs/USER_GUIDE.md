# concept-embeddings-RAG — User Guide

> How to run the Phase 1 experiment: freeze a benchmark corpus, build the two baseline retrieval
> systems, and measure them. Written for whoever runs or reproduces the experiment, not for
> whoever changes it.

---

## 📋 Contents

- [Before you start](#-before-you-start)
- [The four stages](#-the-four-stages)
- [1. Freeze the corpus](#1--freeze-the-corpus-fetch)
- [2. Build the pool](#2--build-the-pool-build)
- [3. Compute the embeddings](#3--compute-the-embeddings-embed)
- [4. Measure the systems](#4--measure-the-systems-evaluate)
- [Reading the results](#-reading-the-results)
- [When something refuses to run](#-when-something-refuses-to-run)
- [Questions](#-questions)
- [Known limits](#-known-limits)

---

## 🚀 Before you start

There is no server, no interface and nothing to log into. Everything is one command, `cer`, with
four stages you run in order.

1. Install the environment: `uv sync`
2. Check it works: `uv run pytest` — the first run takes about 35 seconds while torch loads. That
   is not a hang.
3. Run the four stages below, in order. **Each stage refuses to run if the previous one has not**,
   and tells you which one to run.

You need an internet connection for stages 1 and 3 only. After that the experiment runs offline.

---

## 🧭 The four stages

| Stage | What it does | Roughly how long |
|---|---|---|
| `fetch` | Downloads the benchmark and freezes it | ~4 min |
| `build` | Picks the question subset and builds the corpus | ~9 s |
| `embed` | Turns every paragraph into a vector | **~36 min** |
| `evaluate` | Measures both systems and writes the results | ~1 min |

Run them like this:

```bash
uv run cer fetch
uv run cer build
uv run cer embed
uv run cer evaluate
```

💡 **You only pay the 36 minutes once.** Everything is cached, so re-running `evaluate` after that
takes about a minute.

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

Deleting it costs you 36 minutes of computer time to get back exactly the same numbers.

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

## 📈 Reading the results

Each result file records the numbers **and the exact configuration that produced them** — model,
corpus fingerprint, seed, tokenizer, code version. A number without that is not accepted.

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

---

## 🛠 When something refuses to run

Every stage checks its inputs and says what to do. These are the messages you may see.

| Message | What it means | What to do |
|---|---|---|
| `no frozen corpus found: run 'fetch' first` | Stage 1 has not run | `uv run cer fetch` |
| `no pool found: run 'build' first` | Stage 2 has not run | `uv run cer build` |
| `no embeddings or token counts found: run 'embed' first` | Stage 3 has not run | `uv run cer embed` |
| `embeddings are not cached for this pool` | The corpus changed after the embeddings were made | `uv run cer embed` again |
| `...has hash X, manifest expects Y` | The corpus file no longer matches its fingerprint | See below |
| `pool hashes to X but the manifest expects Y` | The corpus and the pool no longer agree | `uv run cer build` |
| `unit ... does not hash to its content` | `data/pool.json` was modified | `uv run cer build` |

### 🔴 If the corpus fingerprint does not match

The command stops on purpose. It means the file changed since it was frozen — a bad download, an
interrupted write, or an edit. **Do not work around it.** Either restore the original file, or
delete `data/manifest.json` and re-freeze deliberately — and if you do that, treat every number
you recorded earlier as belonging to the old corpus.

---

## ❓ Questions

**Do I have to run all four stages every time?**
No. Run them in order once. After that, only re-run the stage whose inputs changed — and
`evaluate` as often as you like, it is fast.

**Can I stop `embed` halfway and resume?**
No. It saves at the end, so an interrupted run starts over. Give it the 36 minutes.

**Why is my `data/results/` filling up with files?**
By design. Every run writes new files instead of replacing old ones, so a measurement can never
be lost by accident. Old files are safe to read; none of them is stale in a way that matters,
because each carries the configuration it came from.

**Can I use a different embedding model?**
Yes, but it is a deliberate decision, not a tweak: it invalidates the 36-minute cache and makes
the new numbers incomparable with the recorded ones. It is a one-line change in
`src/concept_embeddings_rag/config.py`.

**Does any of this send data anywhere or need an API key?**
No. Stages 1 and 3 download from public sources; nothing is uploaded, and no key is needed.

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
- **This is Phase 1 only.** Concepts, conceptual retrieval and iterative expansion are not built
  yet. See [`docs/plans/0_master_plan.md`](plans/0_master_plan.md) for what comes next.
