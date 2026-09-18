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

1. **Can we build the entity representation cheaply enough to scale?**
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
Cheap entity extraction
        ↓
Phase 8
Strong Dense
        ↓
Phase 9
HotpotQA FullWiki
```

If Phase 7 fails economically, FullWiki is not practical.

If Phase 8 shows Entity Hop adds nothing beside a strong Dense retriever, the interpretation of Phase 6 changes substantially and Phase 9 must be reconsidered accordingly.

If both survive, Phase 9 becomes the central comparison experiment.

---

# 4. Phase 7 — Cheap entity extraction

## Goal

Replace the Phase 5 generative-LLM extraction with a local or otherwise inexpensive entity extraction path while preserving as much retrieval quality as possible.

## Why it comes first

Phase 5 processed:

```text
19,366 paragraphs
cost ≈ 25.98 USD
```

HotpotQA FullWiki is on the order of millions of paragraphs.

Naively multiplying the existing API extraction is not an acceptable scaling strategy without first measuring alternatives.

The important resource is not only API money:

- preprocessing time;
- GPU hours;
- CPU hours;
- memory;
- output/index size;
- failure rate;
- operational complexity

all matter.

## Candidate extraction families

Start with only a few materially different approaches.

### A. Existing Claude extraction

This is the reference.

It already produced the positive Phase 5/6 result.

Its value is not that it is affordable at scale, but that every cheaper extractor can be measured against a known working representation.

### B. Local zero-shot / open NER

First candidate family:

```text
GLiNER or equivalent
```

Advantages:

- runs locally;
- does not require generative decoding;
- can support configurable entity classes;
- designed specifically for entity extraction.

### C. Conventional local NER

For example:

```text
spaCy transformer / large English pipeline
or another lightweight NER model
```

Advantages:

- mature;
- simple;
- fast;
- potentially CPU-friendly.

It may have a narrower entity ontology, but that does not matter until retrieval is measured.

### D. Wikipedia-native entity signals

Investigate whether the exact FullWiki source preserves useful information such as:

- hyperlinks;
- anchor text;
- article targets;
- titles.

If so, Wikipedia itself may provide a high-precision entity-link signal with almost no model cost.

This should only be implemented if the dataset actually exposes enough information.

## What not to optimize

Do not choose an extractor because it wins a generic NER benchmark.

Our target is not named-entity-recognition F1.

Our target is:

> **Does the resulting Entity Hop still recover the missing evidence?**

The decisive comparison is therefore end to end.

## Experiment

For each viable extractor:

```text
19,366 frozen paragraphs
        ↓
entity extraction
        ↓
same normalization/index shape
        ↓
same Entity Hop
        ↓
same Dense + Entity retrieval
        ↓
retrieval metrics
```

Record:

```text
retrieval quality
entity count
incidence count
failed paragraphs
paragraphs / second
total preprocessing time
hardware
index size
estimated FullWiki time
estimated FullWiki cost
```

## Selection

Choose the cheapest practical extractor that retains enough of the retrieval effect to justify FullWiki.

This need not be turned into a complicated hypothesis test.

The engineering/scientific trade-off is itself the result.

---

# 5. Phase 8 — Strong Dense

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

Use the Phase 7 entity representation.

Keep fixed:

```text
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
cheap entity extractor from Phase 7
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

Their order should depend on what Phases 7–9 show.

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

This work was previously scheduled as Phase 7.

It is deliberately postponed because the more important questions are now:

```text
Can we build the current system cheaply?
Does it survive Strong Dense?
Does it scale to FullWiki?
```

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

Phase 7 exists specifically to resolve this before any FullWiki build begins.

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
        +-- YES
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

1. a local/simple extractor preserves most of the current Entity Hop gain;
2. Entity Hop still adds measurable value to a modern strong Dense retriever;
3. the effect survives HotpotQA FullWiki;
4. it also appears on 2Wiki and MuSiQue;
5. better retrieval produces better downstream QA;
6. more complicated graph machinery is not required to obtain most of the benefit.

That would support a clear research contribution:

> **A minimal local shared-entity association channel can provide a useful structural complement to strong dense retrieval in multi-hop QA, without requiring a full knowledge graph or learned multi-hop retriever.**

The project does **not yet** establish that claim.

Phases 7–9 are the next steps toward finding out whether it is true.