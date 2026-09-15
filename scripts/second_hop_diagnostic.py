"""Exploratory, dev only, read-only: would a second hop surface the missing paragraph?

Run after Phase 4, when the question was no longer how to tune the walk but whether any
hop through the concept space points at the right paragraph at all. It selects and
freezes nothing, never reads the test split, and no number it produces chose anything.
The research-line closure (`docs/plans/phase_4/4.1_research_line_closure.md`) quotes it,
which is why it lives in the repository rather than in a scratch directory:

    uv run python scripts/second_hop_diagnostic.py

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
from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
from concept_embeddings_rag.retrieval.conceptual import concept_support, rarity_weights

SPLIT = "dev"
READ = 10
DEPTHS = (10, 100)
TRUNCATIONS: tuple[int | None, ...] = (1, 2, 4, None)  # None = every concept the coding kept
DEFAULT_OUT = config.DATA_DIR / "diagnostics" / "second_hop-dev.json"


def truncate_rows(X: sparse.csr_matrix, m: int | None) -> sparse.csr_matrix:
    """Describe every paragraph by its m strongest concepts and nothing else."""
    if m is None:
        return X
    X = X.tocsr(copy=True)
    for row in range(X.shape[0]):
        start, end = X.indptr[row], X.indptr[row + 1]
        if end - start > m:
            values = X.data[start:end]
            weakest = np.argsort(values)[: (end - start) - m]
            values[weakest] = 0.0
    X.eliminate_zeros()
    return X


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
            X = truncate_rows(sparse.csr_matrix(raw, dtype=np.float64), m)
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
        q = query_backend.encode([question.question])[0]
        order = np.argsort(-(vectors @ q), kind="stable")
        read = {int(row) for row in order[:READ]}
        p1 = int(order[0])
        missing = [index[g] for g in question.gold_unit_ids if index[g] not in read]
        if not missing:
            continue
        for name in ("all", kind):
            questions_with_missing[name] += 1
            missing_paragraphs[name] += len(missing)

        def rank_scores(scores: np.ndarray, *, positive_only: bool) -> list[int]:
            scores = scores.astype(np.float64, copy=True)
            scores[list(read)] = -np.inf  # noqa: B023 - read in the same iteration it is bound
            if positive_only:
                scores[scores <= 0.0] = -np.inf
            depth = max(DEPTHS)
            top = np.argpartition(-scores, depth)[:depth]
            top = top[np.argsort(-scores[top], kind="stable")]
            return [int(row) for row in top if np.isfinite(scores[row])]

        def rank_bm25(text: str) -> list[int]:
            got = bm25.retrieve(text, top_k=max(DEPTHS) + READ + 1)
            rows = [index[u] for u, s in got if s > 0.0 and index[u] not in read]  # noqa: B023
            return rows[: max(DEPTHS)]

        continued = [int(row) for row in order[READ : READ + max(DEPTHS)]]
        record("dense-continue", kind, continued, missing)
        record("bm25-question", kind, rank_bm25(question.question), missing)
        neighbours = rank_scores(vectors @ vectors[p1], positive_only=False)
        record("dense-neighbours(p1)", kind, neighbours, missing)
        record("bm25-follow(p1)", kind, rank_bm25(units[p1].indexable_text), missing)

        for (k, m), (X, weights) in spaces.items():
            row = X.getrow(p1)
            concepts = np.zeros(X.shape[1])
            concepts[row.indices] = row.data * weights[row.indices]
            scores = np.asarray(X @ concepts).ravel()
            reach[variant(k, m)].append(float((scores > 0).sum() / X.shape[0]))
            ranked = rank_scores(scores, positive_only=True)
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
