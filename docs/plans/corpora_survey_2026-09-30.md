# Survey (2026-09-30): public corpora with labelled evidence, aggregation queries, tables

> Requested by the author on 2026-09-30, after Phase 15, for three questions: which public corpora
> with labelled evidence and published retrieval figures the line could be compared on; how
> retrieval is evaluated when a query needs many passages (summaries, "all the causes of X"); and
> how it behaves over tables when a filtered aggregate needs every matching row. Compiled from a
> web survey (36 fetches, primary sources where possible). Every figure is quoted from the cited
> page; **(snippet)** marks a figure seen only in a search snippet or secondary summary; "not
> verified" means it could not be checked. Nothing here is a measurement of this project.

## 1. Corpora with labelled evidence and published retrieval numbers

The project's metric, Full Support @2,048 tokens over whole source paragraphs, has no direct
equivalent in the literature; comparisons need the gold at paragraph level and a metric the
published work also reports (Hits@k, Recall@k) recomputed on the same units.

| Dataset | Domain, licence | Corpus / questions | Gold | Published figures | Comparable? |
|---|---|---|---|---|---|
| [MultiHop-RAG](https://arxiv.org/abs/2401.15391) (Tang & Yang 2024) | news; ODC-BY on the HF card | 609 articles; 2,556 queries (301 null), 2-4 evidence pieces | evidence sentences with article title | voyage-02 + bge-reranker-large: MRR@10 0.586, MAP@10 0.480, Hits@4 0.663, Hits@10 0.747 (256-token chunks, per evidence piece) | yes for Hits@k recomputed on paragraphs; Full Support is new |
| [QASPER](https://arxiv.org/abs/2105.03011) (Dasigi 2021) | NLP papers; CC BY 4.0 | 1,585 papers; 5,049 questions, 55.5 % need several paragraphs (snippet) | **evidence paragraphs, native** | Evidence-F1 within one paper; naive RAG 54.28 (snippet, [2602.13647](https://arxiv.org/pdf/2602.13647)) | the cleanest paragraph gold outside Wikipedia; retrieval is within one paper in the literature, a pooled corpus would be a new setting |
| [LegalBench-RAG](https://arxiv.org/abs/2408.10343) (2024) | contracts, NDAs, privacy policies; CC BY 4.0 | 714 documents, 79.7 M characters; 6,889 questions (mini: 776), single-evidence | **character spans** | text-embedding-3-large, mini: Precision@1 6.41 %, Recall@64 62.22 %; MAUD Recall@64 25.62 % | spans map losslessly onto paragraphs; tests domain transfer, not multi-hop |
| [FRAMES](https://arxiv.org/abs/2409.12941) (2024) | Wikipedia; CC BY 4.0 | 824 questions, 2-15 gold articles | whole articles | end-to-end accuracy; BM25 article recall 0.12-0.15 (not verified) | weak: article gold |
| [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/) | Wikipedia | HippoRAG subset 1,000 q / 6,119 passages | supporting paragraphs | HippoRAG 2 Recall@5 90.4 vs NV-Embed-v2 76.5 (snippet, [2502.14802](https://arxiv.org/abs/2502.14802)) | yes, but Wikipedia again |
| [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) (2025) | web pages; CC BY 4.0 | 100,195 documents (avg 5,179 words); 830 queries, 2.9 gold and 6.1 evidence documents each | documents | Recall@5/100/1000: BM25 1.2/4.7/13.7; Qwen3-Embed-0.6B 6.2/26.5/59.7; Qwen3-Embed-8B 14.5/47.7/76.7 | document gold over long pages; no paragraph gold |
| [WixQA](https://arxiv.org/abs/2505.08643) (2025) | enterprise help center; MIT | 6,221 articles; 200 expert questions, 27 % multi-article | article URLs | answer F1 0.43 best; retrieval figures thin | article gold, small |
| [BEIR](https://arxiv.org/abs/2104.08663) members | mixed | e.g. FiQA 57k / 648 q (2.6 relevant); TREC-COVID 171k / 50 q; SciFact 5k / 300 q; NFCorpus 3.6k / 323 q | document / passage qrels | nDCG@10, BEIR Table 2: FiQA BM25 .236, best .347; SciFact .665 / .688; TREC-COVID .656 / .757 | mostly about one relevant item per query, so Full Support degenerates to Recall@budget |
| [BRIGHT](https://arxiv.org/abs/2407.12883) | StackExchange, code, maths | 12 tasks | qrels | nDCG@10 24.3 in the paper; ReasonIR-8B 29.9 / 36.9 with reranker; DIVER 46.8 (snippets) | reasoning-intensive, single-need |
| [TechQA](https://aclanthology.org/2020.acl-main.117/) | IBM support | 801,998 Technotes; 1,400 questions | answer span; 50 candidates given | reading comprehension, not open retrieval | weak |
| [ConditionalQA](https://arxiv.org/abs/2110.06884) | UK policy pages | 436 documents; 3,102 questions | evidence sentences within one document | answer F1 | within one document |
| [FinanceBench](https://arxiv.org/abs/2311.11944) | SEC filings; licence disputed | 150 public questions | evidence page | answer accuracy | page gold, PDFs, tiny |
| [FinQA](https://arxiv.org/abs/2109.00122) / TAT-QA / MultiHiertt | financial reports | one report page per question | supporting sentences and rows | retriever Recall@3 (not verified) | within one document |
| CRAG, RAGBench, MIRAGE/MedRAG, LoTTE, MS MARCO | various | | no passage gold, or single-hop, or answer-string scoring | | no |

**Ranking for a phase like Phase 16** (non-Wikipedia, paragraph-locatable gold, published
numbers): 1. MultiHop-RAG; 2. QASPER; 3. LegalBench-RAG; 4. WixQA; 5. BrowseComp-Plus (only if
the gold could be brought to paragraph level, which it cannot without new labelling).

## 2. Aggregation queries: many passages, summaries

Benchmarks whose gold is a **set** of evidence and whose metric asks for the whole set:

| Benchmark | Query and gold | Metric | Published figures | Large-K Full Support applicable? |
|---|---|---|---|---|
| [QUEST](https://arxiv.org/abs/2305.11694) (Malaviya 2023) | entity-seeking set queries ("films shot in X but not Y"); 325,505 Wikipedia entity documents; 10.5 gold documents per query on average; 1,307 / 323 / 1,727 split | Recall@K and **MRecall@K = share of queries with all gold in the top K**, the literature's form of Full Support | BM25 MRecall@100 0.037, @1000 0.087, Recall@100 0.197; T5-Large dual encoder MRecall@100 0.142, @1000 0.408, Recall@100 0.476 | yes, at document level with a token budget |
| [QAMPARI](https://arxiv.org/abs/2205.12665) (2023) | questions with many answers (test mean 13.2); Wikipedia passages of about 100 words; 1-2 evidence passages per answer | ERecall@K over evidence passages | BM25 ERecall@100/200 47.6/55.6; DPR 25.5/30.2; sparse+dense fusion aRecall 81.23 (snippet, [2602.18425](https://arxiv.org/abs/2602.18425)) | yes; the unit is closest to ours, but "all evidence" will be near zero |
| [GlobalQA](https://arxiv.org/abs/2510.26205) (2025) | counting, extremum, sorting, top-k over >2,000 resumes in 23 domains; >13,000 questions; gold sets of 2-50 documents (42.6 % need >20) | document-set F1@k, answer F1 | standard RAG answer F1 1.51; GlobalRAG 6.63, D-F1@20 12.01 (Qwen2.5-14B) | yes, non-Wikipedia, short documents, 16k-32k budgets |
| [RoMQA](https://arxiv.org/abs/2210.14353) | constraint queries with about 108 answers on average | recall | not verified | Wikidata-derived |
| [SummHay](https://arxiv.org/abs/2407.01370) (2024) | synthetic haystacks where insights repeat across documents | coverage and citation of source documents per insight | long-context models without retrieval under 20 %; human estimate 56 % | document citations, synthetic |
| [QMSum](https://arxiv.org/abs/2104.05938) | 1,808 query-summary pairs over 232 meetings; every relevant span marked | span recall under an extraction budget (1/6, 1/5, 1/4 of the text): 72.5 / 75.2 / 79.1 | | within one meeting; units are utterances |
| [PluriHopWIND](https://arxiv.org/abs/2510.14377) | 48 questions over 191 wind-energy reports | LLM-judged statement F1 | naive RAG 0.38-0.40, GraphRAG 0.21-0.30 | too small; statement gold |
| GraphRAG "global" ([Edge 2024](https://arxiv.org/abs/2404.16130)) | no gold evidence; pairwise LLM-judge win rates | | 72-83 % win rates (snippet); under a debiased judge LightRAG's win rate over naive RAG falls to 39 % ([2506.06331](https://arxiv.org/abs/2506.06331)); against reference answers global search underperforms plain RAG ([2502.11371](https://arxiv.org/abs/2502.11371)) | not an evidence-recall result |
| Loong, HELMET, InfoDeepSeek, CorpusQA | long-context suites | | no passage-level gold sets found, or no fixed corpus | no |

**Reading for the project.** QUEST's MRecall@K is the literature's Full Support for set queries,
with published BM25 and dual-encoder baselines; GlobalQA is its non-Wikipedia counterpart. A
caution (hypothesis): set queries fan out from constraints rather than chaining through one
bridge paragraph, so a one-hop expansion from P1 may matter less than the raw recall of Dense
and BM25; a large-K test of the Entity Hop is a different scientific question, not an extension
of the current one.

## 3. Tables: filtered aggregates by retrieval

| Benchmark | Gold granularity | "All rows" scored? |
|---|---|---|
| [OTT-QA](https://github.com/wenhuchen/OTT-QA) | table segment (row + headers) and linked passages; 410,740 tables, 6.34 M passages | no; block/table Recall@k |
| [NQ-Tables](https://aclanthology.org/2021.naacl-main.43/) | table; single cell answer | no |
| [Open-WikiTable](https://arxiv.org/abs/2305.07288) | table; queries include aggregation, executed by the reader/SQL | no |
| [FEVEROUS](https://arxiv.org/abs/2106.05707) | sentences and/or **table cells**; 87,026 claims; score requires the complete evidence set | closest, at small per-claim scale; baseline 18 % evidence + verdict |
| [HybridQA](https://arxiv.org/abs/2004.07347) | cell + passage, table given | no |
| FinQA / TAT-QA / [MultiHiertt](https://arxiv.org/abs/2206.01347) | rows and sentences within one report | no |
| [TableRAG](https://arxiv.org/abs/2410.04739) (2024) | schema and cell retrieval on million-token tables | no; aggregates by execution |
| [TARGET](https://arxiv.org/abs/2505.11545), [T2-RAGBench](https://arxiv.org/abs/2604.01733) | table-level Recall@k; hybrid + rerank Recall@5 0.816 (snippet) | no |

No benchmark found scores "every matching row retrieved". Where aggregation is studied directly,
[TAG](https://arxiv.org/abs/2408.14717) (Biswal 2024, 80 BIRD-derived queries) reports RAG 0 %,
retrieval + LM rank 2 %, Text2SQL 17 %, hand-written TAG pipelines 55 %, and writes that RAG
"is only able to provide information about some of the races, as most of the relevant races are
not retrieved". The literature's position: top-k similarity retrieval cannot guarantee
completeness, so filtered aggregates belong to SQL or code execution, with retrieval used to find
the right table and columns. A retrieval-only research question does not reach this case.

## What this means for the line (interpretation)

1. Phase 16 stays on MultiHop-RAG; report Hits@4 / Hits@10 recomputed on paragraphs beside Full
   Support, since the published figures are per evidence piece over 256-token chunks.
2. The next non-Wikipedia corpora with native paragraph gold are QASPER (pooled across papers, a
   new setting) and LegalBench-RAG (character spans, mostly single-evidence).
3. A large-context test of the Entity Hop has a ready metric and baselines in QUEST (MRecall@K)
   and a non-Wikipedia option in GlobalQA; it is a new phase with its own question.
4. Numeric aggregates over tables are, by the literature's own account, outside what retrieval
   can guarantee; the honest study there is "retrieval finds the table, code computes", which is
   not this line's question.
