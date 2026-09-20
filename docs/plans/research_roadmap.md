# Research Roadmap: from Entity Hop result to scalable multi-hop retrieval

## 1. Why this roadmap exists

The project started from a broad idea:

> A corpus may contain semantic structure that allows retrieval to move beyond the paragraphs most similar to the original question.

The first implementation tried to create that structure through a sparse concept space induced from dense paragraph embeddings.

That research line failed.

The project then tested concepts and entities extracted directly from text. Concepts again failed, but entities produced the first strong positive result.

Phase 6 integrated that Entity Hop into the complete retrieval system:

```text
Question
  ↓
Dense retrieval
  ↓
P1
  ↓
entities in P1
  ↓
other paragraphs sharing those entities
  ↓
rarity-weighted Entity Hop ranking
  ↓
Dense + Entity fusion
```

On the frozen 1,400-question test set at a 2,048-token context budget:

```text
Dense                    0.8250 Full Support
Dense + BM25             0.8643
Dense + Entity Hop       0.8900
```

Entity Hop therefore removed approximately 37% of Dense's Full Support failures on this population and satisfied Phase 6's preregistered replacement criterion against Dense + BM25.

The next question is no longer:

> Can entities help?

That has already been answered positively in this setting.

The next questions are:

1. ~~**Can we build the entity representation cheaply enough to scale?**~~ **Answered by Phase 7: yes, with GLiNER.**
2. **Does the entity signal still matter beside a much stronger Dense retriever?**
3. **Does the mechanism survive a search space of millions of paragraphs?**
4. **Does it generalize beyond this HotpotQA setup and improve downstream answers?**

This document separates the work that is now committed from the much larger set of possible future extensions.

---

# 2. Current working interpretation

The strongest interpretation supported by the evidence so far is:

> Dense similarity and entity relatedness are complementary retrieval signals.

Dense is good at:

```text
Question → semantically similar evidence
```

Entity Hop is good at:

```text
Evidence already found
→ bridge entity revealed inside that evidence
→ other evidence associated through that entity
```

The second signal is valuable precisely because the bridge entity may not occur in the original question.

The important current hypothesis is therefore not that entity graphs are novel.

Entity-based multi-hop retrieval already exists in the literature.

The interesting question is more specific:

> **How little structural machinery is actually needed to complement Dense retrieval?**

The current system uses unusually little:

- no relation graph;
- no triples;
- no PageRank;
- no learned graph traversal;
- no LLM at query time;
- no query reformulation;
- no entity canonicalization;
- one Dense seed (`P1`);
- one entity hop;
- exact normalized entity incidence;
- rarity-weighted scoring;
- fusion back with Dense.

The roadmap should preserve this simplicity until evidence shows that a more complex component is needed.

---

# 3. Immediate committed roadmap

The next three phases form one dependency chain:

```text
Phase 7
Cheap entity extraction        COMPLETE - GLiNER selected
        ↓
Phase 8
Strong Dense                   STOP at dev gate - premise failed
        ↓
Deviation 8.1
Corpus-scale sensitivity       next
        ↓
Phase 9
HotpotQA FullWiki              blocked: no Dense model selected
```

Phase 7 did not fail economically: extraction fell from 25.9848 USD to an attributable 0.03 USD over the same 19,366 paragraphs, while retrieval kept 83.5% of the gain. FullWiki remains practical, subject to Phase 9 measuring its own costs rather than inheriting Phase 7's projection.

If Phase 8 shows Entity Hop adds nothing beside a strong Dense retriever, the interpretation of Phase 6 changes substantially and Phase 9 must be reconsidered accordingly.

If both survive, Phase 9 becomes the central comparison experiment.

**Neither happened.** Phase 8 stopped at its pre-declared dev gate: `Qwen/Qwen3-Embedding-0.6B` scored 446 / 600 dev Full Support @2,048 against the inherited BGE-small reading of 487 / 600, below the 488 the rule required. The held-out split was neither encoded nor measured, no second encoder was tried, and the question above is still open. Deviation 8.1 is opened for corpus-scale sensitivity: the ordering between the two encoders is bounded to the frozen 19,366-paragraph distractor pool. See [`phase_8/8.results.md`](phase_8/8.results.md).

---

# 4. Phase 7 — Cheap entity extraction — COMPLETE

## Result

**GLiNER selected.** `data/phase7/selection.json` → `selected: "gliner"`.

The generative-LLM extraction can be replaced by a local model: retrieval keeps **83.5%** of the held-out gain at roughly one thousandth of the extraction cost.

Full record: [`phase_7/7.results.md`](phase_7/7.results.md).

## What was run

Of the four candidate families this roadmap listed, the two local ones were run end to end against the Claude reference:

### A. Existing Claude extraction — the reference

`claude-sonnet-5`, unchanged and not re-measured. Its value was never affordability; it is the known working representation every cheaper extractor is measured against.

### B. Local zero-shot NER — **selected**

```text
urchade/gliner_medium-v2.1
revision 40ec419335d09393f298636f471328b722c6da9e
5 labels, threshold 0.5, batch size 8
config digest 2f7864661b8ce7ff
```

### C. Conventional local NER — measured, not selected

```text
spaCy en_core_web_sm 3.8.0
11 labels
```

### D. Wikipedia-native links / anchors — **discarded with evidence**

The spec settled this before implementation: the corpus exposes no link, anchor or entity layer to mine. The question can only be reopened where the FullWiki source is actually opened, which is Phase 9.

## What it measured

Held-out, 1,400 questions, Full Support @2,048:

```text
Dense                           0.8250 = 1,155 / 1,400
Dense + BM25                    0.8643 = 1,210 / 1,400
Dense + Entity Hop (Claude)     0.8900 = 1,246 / 1,400
Dense + Entity Hop (GLiNER)     0.8793 = 1,231 / 1,400
```

GLiNER recovers 76 of the 91 questions Claude added over Dense, beats the BM25 control by 21 questions, and sits 15 below Claude.

Extraction, on a rented RTX 4090 (Linux x86_64):

```text
GLiNER    19,366 paragraphs in 119.4 s    162.178 p/s    0.03 USD attributable
spaCy     19,366 paragraphs in 208.8 s     92.766 p/s    0.05 USD attributable
Claude    25.9848 USD measured, wall clock never recorded
```

Projected linearly to 5M paragraphs — **projections, not measurements**:

```text
GLiNER     8.56 h    6.34 USD
spaCy     14.97 h   11.08 USD
```

## Selection, and the finding that came with it

The rule was frozen before any candidate ran: clear a retention bar and an economic bar, then rank on projected FullWiki cost and time. Both candidates cleared both bars; GLiNER ranked first; only GLiNER's held-out split was opened.

**spaCy scored higher on dev** — 525 / 600 against GLiNER's 518 / 600, retaining 95.0% of Claude's dev gain against 77.5% — and was not selected. Three reasons:

1. the preregistered ranking is on cost and time, and GLiNER's 6.34 USD / 8.56 h beat spaCy's 11.08 USD / 14.97 h;
2. overriding a preregistered rule *after observing dev* would invalidate the only thing that makes the held-out figure mean anything;
3. spaCy buys its quality with a much denser, less scalable representation — 11 labels against 5, a median of 493 hop candidates against 40, a largest hub of 4,085 paragraphs against 1,186.

On dev, GLiNER tied the Dense + BM25 line exactly (518 = 518); on test it beat that line by 21 questions. Dev was the more pessimistic reading, which is the safe direction for a selection rule to err in.

spaCy is **not discarded** — see §8.5.

## What it did not settle

Nothing here says which extractor reads entities *correctly*: NER quality was an explicit anti-goal. The 21-question margin over BM25 comes from one held-out run with no interval and no test statistic. And neither extractor's representation has been observed at five million paragraphs.

---

# 5. Phase 8 — Strong Dense

> **Measured, and stopped at the dev gate** (2026-09-19). The goal below was not reached: the chosen
> model was weaker than the inherited baseline on this corpus, so the held-out split was never
> opened. What follows is the phase as planned; [`phase_8/8.results.md`](phase_8/8.results.md) is
> what it found. Deviation 8.1 follows.

## Goal

Determine whether Entity Hop remains complementary when Dense retrieval becomes substantially stronger.

## Why this matters

The current Dense retriever uses:

```text
BAAI/bge-small-en-v1.5
```

Phase 6 showed:

```text
Dense                  0.8250
Dense + Entity         0.8900
```

That +6.5-point gain is large.

But two explanations remain possible:

### Interpretation A

Entity structure supplies genuinely different information:

```text
semantic similarity
+
structural relatedness
```

Even a very strong Dense retriever should still benefit.

### Interpretation B

Entity Hop mainly compensates for limitations of the small embedding model.

A stronger retriever could make much of the gain disappear.

Phase 8 distinguishes those explanations.

## Experiment

Stay on the current **19,366-paragraph corpus**.

Select one modern strong open Dense model.

Do not benchmark many models.

Measure:

```text
Strong Dense
Strong Dense + BM25
Strong Dense + Entity Hop
```

Use the Phase 7 entity representation: the **GLiNER** index, unchanged.

> **Phase 8 inherits GLiNER and changes only the Dense retriever.** Changing the Dense model and the extractor simultaneously would make any improvement or degradation unattributable to either.

Keep fixed:

```text
GLiNER entity index from Phase 7
P1 only
1 entity hop
no canonicalization
no relations
no query-aware filtering
```

Because Dense scores will change, the Dense/Entity fusion weight can be fitted on dev again.

That is the only necessary retrieval tuning.

## Interpret the result

### Strong positive effect remains

This would strengthen the main scientific hypothesis:

> entity relatedness contains retrieval information that even a strong semantic retriever does not capture.

### Smaller but persistent effect

This would indicate that Phase 6 contained both:

- compensation for a weaker Dense model; and
- a real complementary structural signal.

### Effect disappears

This would be equally important.

It would mean that the current Entity Hop is primarily an efficient way of repairing a weaker semantic retriever rather than a generally necessary retrieval channel.

No additional graph complexity should be introduced inside Phase 8 to rescue the result.

---

# 6. Phase 9 — HotpotQA FullWiki

## Goal

Move from the current 19,366-paragraph research corpus to the standard multi-million-paragraph HotpotQA FullWiki retrieval problem.

## Why it matters

Our current results are internally rigorous but are not directly numerically comparable with most published FullWiki multi-hop retrievers.

Published systems frequently search millions of Wikipedia paragraphs.

Our current search space is much smaller.

Rather than reimplementing every prior paper, the first useful comparison is therefore:

> run our method in the same benchmark regime.

## Architecture

Carry forward only the components already selected:

```text
Strong Dense from Phase 8
+
GLiNER, the cheap entity extractor selected in Phase 7
+
P1
+
one Entity Hop
```

No new research mechanism is added while scaling.

## Systems

Measure at least:

```text
Strong Dense
Strong Dense + BM25
Strong Dense + Entity Hop
```

## Evaluation

Use standard HotpotQA FullWiki retrieval metrics reported by the most relevant publications.

Also retain compatible internal metrics where practical.

This gives two views:

```text
internal continuity with Phases 1–8
+
external comparability with the literature
```

The Phase 9 spec should identify the precise published metrics before implementation.

## Scale questions

Phase 9 is not only about accuracy.

Measure the real cost of the architecture at approximately FullWiki scale:

```text
entity preprocessing time
entity preprocessing cost
dense embedding time
entity-index disk size
dense-index disk size
query latency
Entity Hop candidate count
memory
document-frequency distribution
```

This will tell us whether the simplicity of the incidence index gives a practical advantage over richer graph construction.

---

# 7. Likely next experiments after Phase 9

These are **not committed phases yet**.

Their order should depend on what Phases 7–9 show. Phase 7 has now reported; Phases 8 and 9 have not.

## 7.1 Cross-benchmark generalization

Most likely next step if FullWiki remains positive.

Evaluate on:

```text
2WikiMultiHopQA
MuSiQue
```

Why:

HotpotQA bridge questions are especially compatible with entity-mediated navigation.

A method that works only there may be exploiting benchmark structure.

2Wiki and especially MuSiQue would test whether the mechanism generalizes.

Keep the basic architecture unchanged initially.

---

## 7.2 Query-aware Entity Hop

The current Entity Hop becomes query-blind after Dense produces `P1`.

Current:

```text
Q → Dense → P1
             ↓
      shared entities
             ↓
         candidates
```

Possible extension:

```text
Q → Dense → P1
             ↓
      shared entities
             ↓
         candidates
             ↓
   relevance(candidate, Q)
```

This asks whether the structural signal should remain deliberately orthogonal to the query or should be filtered semantically.

Relevant modern graph-retrieval systems often reintroduce query relevance during structural expansion.

This is scientifically interesting because it isolates whether the current gain comes partly from **not** asking the semantic retriever the same question twice.

Do not implement until the simple query-blind variant has been tested at scale.

---

## 7.3 QA end to end

Current positive result is a retrieval result.

The downstream question is:

> Does better supporting-evidence retrieval actually improve answer quality?

Evaluate a fixed reader/LLM over contexts produced by:

```text
Strong Dense
Strong Dense + BM25
Strong Dense + Entity Hop
```

Metrics should preferably include standard benchmark answer metrics rather than only LLM judging.

Retrieval improvement does not automatically guarantee answer improvement, so this is a separate empirical question.

---

# 8. Deferred representation improvements

These should only be introduced after a concrete failure mode motivates them.

## 8.1 Entity canonicalization

Current raw forms may split:

```text
Barack Obama
Obama
President Obama
Barack H. Obama
```

Canonicalization could improve reachability.

But it can also incorrectly merge entities.

That means:

```text
raw entity
vs
canonical entity
```

should eventually be an ablation, not an assumed upgrade.

This work was previously scheduled as Phase 7, which became cheap entity extraction instead.

It stays postponed because the more important questions were, and mostly still are:

```text
Can we build the current system cheaply?   answered in Phase 7: yes, with GLiNER
Does it survive Strong Dense?              Phase 8
Does it scale to FullWiki?                 Phase 9
```

Note also that Phase 7 changed *which raw forms exist* without touching canonicalization: GLiNER's index holds 81,617 distinct nodes against the Claude extraction's 96,416. Any future canonicalization ablation runs on whichever index the project is actually carrying, not on Phase 5's.

---

## 8.2 Relations and triples

A **triple** is usually represented as:

```text
(subject, relation, object)
```

Example:

```text
(Kaley Cuoco, starred_in, The Big Bang Theory)
```

Our current node representation stores entities independently:

```text
paragraph → Kaley Cuoco
paragraph → The Big Bang Theory
```

A relation representation preserves the explicit semantic link:

```text
Kaley Cuoco --starred_in--> The Big Bang Theory
```

Relations were treated differently from entities and concepts because they are a more complex representation.

Entities/concepts can be attached independently to a paragraph.

A relation requires:

```text
subject
predicate
object
```

and normally benefits from:

- entity normalization;
- coreference handling;
- subject/object disambiguation;
- relation extraction.

Phase 6 deliberately excluded relation/attribute nodes so that the simpler entity signal could be tested in isolation.

Explicit relations remain interesting for cases where the correct bridge is **not carried by a shared name**.

But they should not be added merely because GraphRAG systems commonly contain them.

The current evidence says raw entities already carry substantial signal.

---

## 8.3 Multiple seeds

Current:

```text
P1 only
```

Possible future comparison:

```text
P1
vs
top-2
vs
top-5
vs
top-10
```

Multiple seeds increase structural reach but also expose the hop to more irrelevant entities.

The fact that P1 already works strongly suggests precision may matter more than breadth.

For now:

> **P1 remains fixed.**

---

## 8.4 Multiple entity hops

Current:

```text
Dense → P1 → Entity Hop → stop
```

Possible:

```text
Dense
→ P1
→ entity hop 1
→ new paragraph
→ entity hop 2
→ ...
```

More hops may increase reach.

They may also reproduce the drift/hub problem already observed in the failed concept-diffusion line.

Published graph-retrieval work also provides evidence that additional graph depth does not automatically improve retrieval.

For now:

> **one hop remains fixed.**

---

## 8.5 Extractor representation density at scale — spaCy revisited

Phase 7 selected GLiNER by a preregistered cost-and-time ranking, but spaCy was the better **dev** retriever: 525 / 600 against 518 / 600, retaining 95.0% of Claude's dev gain against GLiNER's 77.5%. It was never measured on the held-out split, so no spaCy test figure exists and none should be produced retrospectively.

What separates them is representation density, measured on the 19,366-paragraph corpus:

```text
                        spaCy      GLiNER
labels                     11           5
distinct nodes         97,189      81,617
nodes / paragraph, median   8           6
hop candidates, median    493          40
largest hub (paragraphs) 4,085       1,186
```

The open question:

> **Does spaCy's dev advantage survive at FullWiki scale, or does it collapse under candidate explosion and hub growth?**

A median of 493 candidates per hop over 19,366 paragraphs is a different proposition over five million, and the largest hub is already three times wider. The honest answer is that nobody knows, because the only evidence is a small-corpus dev reading.

This is a **deferred scale-and-representation experiment, not an active phase**. It is recorded so the finding is not lost. Writing it down does not promote it into Phase 8 or Phase 9, and it does not reopen the Phase 7 selection: Phase 8 inherits GLiNER precisely so that Strong Dense is the only variable that moves.

---

# 9. Explicitly not planned now: prior-art reimplementation

A possible benchmark would be:

```text
Dense
Dense + BM25
MDR-style second Dense
canonical Entity Hop
HGRAG/SAG-style graph expansion
raw Entity Hop
```

This could isolate mechanisms very precisely.

It is **not part of the immediate roadmap**.

The priority is first to obtain numbers in a benchmark regime that can already be compared with published results.

If Phase 9 produces a strong result but the reason for the difference remains ambiguous, selective implementation of one or two competing mechanisms may then become worthwhile.

Do not build an internal reproduction suite of every GraphRAG method pre-emptively.

---

# 10. Other deferred directions

## Dense + BM25 + Entity

Phase 6 asked whether Entity could **replace** BM25.

A three-way fusion asks a different question:

```text
Does BM25 still contain complementary information after Entity Hop is present?
```

Potentially useful, but not required for the current scientific story.

---

## Graph propagation / PageRank

Methods such as HippoRAG, HGRAG and LinearRAG use richer graph traversal.

Our current result comes from a single local incidence operation.

Global propagation should only be added if local one-hop reach becomes a measured bottleneck.

---

## Higher-order graph structures

Possible future structures include:

```text
entity graphs
hypergraphs
event graphs
relation graphs
document/entity bipartite graphs
```

Do not assume they are improvements.

Each increases extraction, preprocessing and retrieval complexity.

---

## Hierarchical retrieval

The current corpus is Wikipedia-style and does not contain the genuine book/chapter/section hierarchy that motivated part of the original proposal.

Testing hierarchy therefore requires a different corpus.

It is not part of the current HotpotQA research line.

---

## Type C / satellite-context retrieval

The original proposal also concerned knowledge that may be useful even when no explicit bridge entity exists.

Example:

```text
question about preparing a 200 km cycling event
→ nutrition
→ hydration
→ pacing
→ recovery
```

HotpotQA entity bridges are not a valid benchmark for that claim.

A separate evaluation population or dataset would be required.

This remains open and should not be inferred from the Entity Hop result.

---

# 11. Cost and scalability as first-class research questions

The current online Entity Hop latency is approximately:

```text
10.447 ms / query
```

That is about:

```text
0.0104 seconds
```

The main scalability concern is therefore **not online traversal**.

It is preprocessing.

Phase 5 used a generative LLM to inspect every paragraph.

At FullWiki scale, entity extraction can dominate:

```text
money
time
GPU/CPU resources
storage
```

Phase 7 existed specifically to resolve this before any FullWiki build began, and it did. Over the same 19,366 paragraphs:

```text
Claude     25.9848 USD   measured
GLiNER        0.03 USD   measured, 119.4 s on one rented RTX 4090
```

Extrapolated linearly to five million paragraphs, GLiNER **projects** to 8.56 h and 6.34 USD against roughly 6,700 USD for the same Claude extraction. Both 5M figures are projections; neither is an invoice, and neither authorizes a FullWiki build. Phase 9 measures its own.

Preprocessing is therefore no longer the blocking cost. What remains unmeasured at scale is the *representation*: candidate-set size, hub growth and document-frequency distribution over millions of paragraphs.

Every future representation extension should answer:

> How much retrieval gain does this structure buy per unit of preprocessing and online complexity?

This cost/benefit question is part of the science of the project, not merely deployment engineering.

---

# 12. Simplicity principle

From Phase 7 onward the project deliberately changes implementation style.

The experiment should not grow faster than the scientific knowledge it produces.

Rules:

1. One central question per phase.
2. Prefer 5–8 implementation steps.
3. Avoid `tasks.md` unless genuinely necessary.
4. Reuse code rather than generalize it.
5. Do not build hypothetical failure branches in advance.
6. Test result-changing code, not every procedural detail.
7. Keep configuration and results reproducible.
8. Keep dev/test separation when a choice is being fitted.
9. Do not create a statistical decision framework when a direct comparison answers the question.
10. Stop the phase when its question is answered.

The goal is:

```text
research question
→ minimal experiment
→ result
→ next question
```

not:

```text
research question
→ experimental framework
→ generalized infrastructure
→ hundreds of invariants
→ large maintenance burden
```

---

# 13. Decision tree

The near-term roadmap can be summarized as:

```text
Phase 7: Can entities be extracted cheaply?
        |
        +-- NO
        |    → investigate Wikipedia-native signals or one other extraction approach
        |    → do not scale FullWiki yet
        |
        +-- YES  ← ANSWERED: GLiNER, 0.03 USD, 83.5% of the gain retained
             |
             v
Phase 8: Does Entity Hop survive Strong Dense?
        |
        +-- NO / almost disappears
        |    → reinterpret Phase 6 as mainly compensating for weak Dense
        |    → decide whether FullWiki is still scientifically useful
        |
        +-- YES
             |
             v
Phase 9: Does it work on FullWiki?
        |
        +-- NO
        |    → diagnose corpus-scale/entity-frequency failure
        |
        +-- YES
             |
             v
Generalization:
2Wiki + MuSiQue
             |
             v
QA end-to-end / query-aware ablation
             |
             v
Only then consider:
canonicalization / relations / extra hops / richer graph structure
```

---

# 14. What would constitute a particularly strong result?

The strongest plausible sequence would be:

1. ~~a local/simple extractor preserves most of the current Entity Hop gain;~~ **observed in Phase 7: GLiNER keeps 83.5% of the held-out gain;**
2. Entity Hop still adds measurable value to a modern strong Dense retriever;
3. the effect survives HotpotQA FullWiki;
4. it also appears on 2Wiki and MuSiQue;
5. better retrieval produces better downstream QA;
6. more complicated graph machinery is not required to obtain most of the benefit.

That would support a clear research contribution:

> **A minimal local shared-entity association channel can provide a useful structural complement to strong dense retrieval in multi-hop QA, without requiring a full knowledge graph or learned multi-hop retriever.**

The project does **not yet** establish that claim.

Step 1 is now observed. Phases 8 and 9 are the next steps toward finding out whether the rest is true.