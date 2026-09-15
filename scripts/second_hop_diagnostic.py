"""Exploratory, dev only, read-only: would a second hop surface the missing paragraph?

Run after Phase 4, when the question was no longer how to tune the walk but whether any
hop through the concept space points at the right paragraph at all. It selects and
freezes nothing, never reads the test split, and no number it produces chose anything.
The research-line closure (`docs/plans/phase_4/4.1_research_line_closure.md`) quotes it,
which is why it lives in the repository rather than in a scratch directory:

    uv run python scripts/second_hop_diagnostic.py

Since Phase 5 the protocol itself lives in `concept_embeddings_rag.evaluation.second_hop`,
which the navigation pilot measures its new hops against; this script is one of its callers,
and `tests/evaluation/test_second_hop.py` holds it to its committed output.

Protocol, per dev question:

- A reader has already read dense's top-10 paragraphs (`read`).
- A gold paragraph outside `read` is **missing**: it is what a second hop must find.
- The hop starts from the single best paragraph `p1` - the one a reader reads first -
  and ranks every paragraph not already read. Hit@10 and hit@100 of each missing gold
  paragraph among those candidates is the figure, so the denominator is paragraphs.

Hops compared on exactly the same missing paragraphs:

    dense-continue      keep reading dense's own list (no hop at all)
    bm25-question       BM25 on the question
    dense-neighbours    paragraphs whose embedding is closest to p1's
    bm25-follow         BM25 with p1's *text* as the query ("follow the names")
    concept-hop K/m     paragraphs sharing p1's concepts, each paragraph described by
                        its top-m concepts, idf-damped:  score(v) = sum_c X[v,c] w_c X[p1,c]

Every input is loaded through the pipeline's own verifying loaders, and the output
records the configuration it was computed under. Nothing here is stochastic.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

from concept_embeddings_rag import __version__, config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.concepts.coding import load_matrix
from concept_embeddings_rag.concepts.dictionary import dictionary_key
from concept_embeddings_rag.corpus.pool import load_pool
from concept_embeddings_rag.embeddings.backend import SentenceTransformerBackend
from concept_embeddings_rag.embeddings.cache import (
    EmbeddingCache,
    build_query_backend,
    cache_key,
    unit_set_hash,
)
from concept_embeddings_rag.evaluation import second_hop
from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
from concept_embeddings_rag.retrieval.conceptual import concept_support, rarity_weights

SPLIT = "dev"
READ = config.PILOT_READ_DEPTH
DEPTHS = config.SECOND_HOP_DEPTHS
TRUNCATIONS: tuple[int | None, ...] = (1, 2, 4, None)  # None = every concept the coding kept
DEFAULT_OUT = config.DATA_DIR / "diagnostics" / "second_hop-dev.json"


def variant(k: int, m: int | None) -> str:
    return f"k={k}/m={m or 'all'}"


def main(out: Path) -> None:
    units, questions = load_pool(config.DATA_DIR / "pool.json")
    unit_ids = [unit.unit_id for unit in units]
    index = {unit_id: row for row, unit_id in enumerate(unit_ids)}
    pool_hash = unit_set_hash(unit_ids)
    dev = [question for question in questions if question.split == SPLIT]
    raw_path = config.DATA_DIR / "hotpot_raw.json"
    kinds = {q["_id"]: q["type"] for q in json.loads(raw_path.read_text("utf-8"))}

    backend = SentenceTransformerBackend()
    cached = EmbeddingCache(config.CACHE_DIR).load(
        cache_key(backend.name, backend.revision, pool_hash, normalized=True),
        expected_unit_ids=unit_ids,
    )
    if cached is None:
        raise SystemExit("[ERROR] embeddings are not cached for this pool: run 'embed' first")
    vectors, _ = cached
    query_backend = build_query_backend(dev, backend, EmbeddingCache(config.QUESTION_CACHE_DIR))
    bm25 = BM25Retriever(units)

    # The four deduplicated Phase 2 spaces, keyed exactly as `select` keys them.
    keys = {
        k: dictionary_key(
            k=k,
            seed=config.CONCEPT_SEED,
            alpha=config.INDUCTION_ALPHA,
            unit_set_hash=pool_hash,
            model=config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_REVISION,
            merge_threshold=config.MERGE_COSINE_THRESHOLD,
        )
        for k in config.CONCEPT_K_SWEEP
    }

    spaces: dict[tuple[int, int | None], tuple[sparse.csr_matrix, np.ndarray]] = {}
    structure: dict[str, dict[str, float]] = {}
    for k, key in sorted(keys.items()):
        raw = load_matrix(key, config.CONCEPTS_DIR, view="raw", expected_unit_ids=unit_ids).X
        for m in TRUNCATIONS:
            X = second_hop.truncate_rows(sparse.csr_matrix(raw, dtype=np.float64), m)
            support = concept_support(X)
            weights = rarity_weights(support, X.shape[0])
            spaces[(k, m)] = (X, weights)
            live = support[support > 0]
            structure[variant(k, m)] = {
                "active_per_paragraph": float(X.nnz / X.shape[0]),
                "paragraphs_per_concept_mean": float(live.mean()),
                "paragraphs_per_concept_median": float(np.median(live)),
                "largest_concept_share": float(live.max() / X.shape[0]),
            }

    hits: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    reach: dict[str, list[float]] = defaultdict(list)
    questions_with_missing: dict[str, int] = defaultdict(int)
    missing_paragraphs: dict[str, int] = defaultdict(int)

    def record(method: str, kind: str, ranked: list[int], missing: list[int]) -> None:
        for depth in DEPTHS:
            top = set(ranked[:depth])
            for gold in missing:
                value = float(gold in top)
                hits[method][f"all@{depth}"].append(value)
                hits[method][f"{kind}@{depth}"].append(value)

    for question in dev:
        kind = kinds[question.qid]
        order = second_hop.dense_order(vectors, query_backend.encode([question.question])[0])
        read_list = second_hop.read_rows(order, READ)
        read = set(read_list)
        p1 = second_hop.origin(read_list)
        missing = [index[g] for g in question.gold_unit_ids if index[g] not in read]
        if not missing:
            continue
        for name in ("all", kind):
            questions_with_missing[name] += 1
            missing_paragraphs[name] += len(missing)

        depth = max(DEPTHS)
        continued = second_hop.dense_continue(order, read_depth=READ, depth=depth)
        record(second_hop.DENSE_CONTINUE, kind, continued, missing)
        by_question = second_hop.rank_bm25(
            bm25, index, question.question, read, read_depth=READ, depth=depth
        )
        record(second_hop.BM25_QUESTION, kind, by_question, missing)
        neighbours = second_hop.dense_neighbours(vectors, p1, read, depth=depth)
        record(second_hop.DENSE_NEIGHBOURS, kind, neighbours, missing)
        followed = second_hop.rank_bm25(
            bm25, index, units[p1].indexable_text, read, read_depth=READ, depth=depth
        )
        record(second_hop.BM25_FOLLOW, kind, followed, missing)

        for (k, m), (X, weights) in spaces.items():
            ranked, reached = second_hop.concept_hop(X, weights, p1=p1, read=read, depth=depth)
            reach[variant(k, m)].append(reached)
            record(f"concept-hop(p1) {variant(k, m)}", kind, ranked, missing)

    summary = {
        method: {name: float(np.mean(values)) for name, values in table.items()}
        for method, table in hits.items()
    }
    report = {
        "config": {
            "split": SPLIT,
            "read": READ,
            "depths": list(DEPTHS),
            "truncations": [m or "all" for m in TRUNCATIONS],
            "model": backend.name,
            "revision": backend.revision,
            "unit_set_hash": pool_hash,
            "n_units": len(unit_ids),
            "dictionary_keys": {str(k): key for k, key in sorted(keys.items())},
            "damping": "idf",
            "code_version": __version__,
            "stochastic": False,
        },
        "n_dev_questions": len(dev),
        "n_questions_with_missing": dict(questions_with_missing),
        "n_missing_paragraphs": dict(missing_paragraphs),
        "hit_rates": summary,
        "reach_from_p1": {name: float(np.mean(values)) for name, values in reach.items()},
        "structure": structure,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out, json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"[INFO] {questions_with_missing['all']} of {len(dev)} {SPLIT} questions miss "
        f"{missing_paragraphs['all']} gold paragraphs outside dense top-{READ}"
    )
    print(f"{'method':<34} {'hit@10':>7} {'hit@100':>8} {'reach':>7}")
    for method, table in summary.items():
        tag = method.split(" ", 1)[1] if method.startswith("concept") else None
        r = report["reach_from_p1"].get(tag, float("nan")) if tag else float("nan")
        print(f"{method:<34} {table['all@10']:7.3f} {table['all@100']:8.3f} {r:7.3f}")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the JSON")
    main(parser.parse_args().out)
