"""Phase 6, T4: A is bit-identical after the second-stage generalization (invariant 1, D5).

The Phase 3 `FusedRetriever.retrieve` body was one line:
`fuse([c.retrieve(query, top_k) for c in components], ...)`. The generalized body calls dense,
then the second component, and fuses the two lists. For a signal that must be the same two
calls handed to the same `fuse`, so the reference below keeps the Phase 3 body verbatim and
every fused list - units and scores - must equal it:

1. on stub components with irregular scores and ties, for both schemes and every grid weight;
2. on the real dense and BM25 retrievers over the 600 dev questions (skipped without data).
   Rankings only: no harness, no metric. The end-to-end evidence is HU-2's reproduction.
"""

import json
import random
from collections.abc import Mapping, Sequence

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.selection import load_selection
from concept_embeddings_rag.retrieval.base import Hit, Retriever
from concept_embeddings_rag.retrieval.fusion import RRF, WEIGHTED, FusedRetriever, fuse

POOL = config.DATA_DIR / "pool.json"
TOKENS = config.DATA_DIR / "token_counts.json"


def phase_3_reference(
    components: Sequence[Retriever],
    query: str,
    top_k: int,
    scheme: str,
    weights: Mapping[str, float] | None,
) -> list[Hit]:
    """The Phase 3 body, verbatim, over components already in the fixed order."""
    ordered = None if weights is None else tuple(weights[c.name] for c in components)
    return fuse(
        [c.retrieve(query, top_k) for c in components], scheme=scheme, top_k=top_k, weights=ordered
    )


class RandomRetriever:
    """Irregular scores with deliberate ties, fixed by a seed per query."""

    def __init__(self, name: str, seed: int) -> None:
        self.name = name
        self.seed = seed

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        rng = random.Random(f"{self.seed}:{query}")  # noqa: S311 - seeded toy scores, not crypto
        units = rng.sample([f"u{index:03d}" for index in range(60)], k=min(top_k, 40))
        scores = [round(rng.uniform(-2.0, 5.0), 1) for _ in units]
        hits = list(zip(units, scores, strict=True))
        return sorted(hits, key=lambda hit: (-hit[1], hit[0]))


def configurations() -> list[tuple[str, float | None]]:
    return [(RRF, None), *((WEIGHTED, float(w)) for w in config.FUSION_WEIGHT_GRID)]


def assert_identical(components: Sequence[Retriever], queries: Sequence[str], top_k: int) -> int:
    compared = 0
    for scheme, w in configurations():
        weights = None if w is None else {"dense": w, "bm25": 1.0 - w}
        hybrid = FusedRetriever(list(components), scheme=scheme, weights=weights)
        for query in queries:
            expected = phase_3_reference(components, query, top_k, scheme, weights)
            assert hybrid.retrieve(query, top_k) == expected, (scheme, w, query)
            compared += 1
    return compared


def test_the_generalized_control_equals_the_phase_3_body_on_toys_for_every_configuration():
    components = [RandomRetriever("dense", 1), RandomRetriever("bm25", 2)]
    queries = [f"question {index}" for index in range(25)]
    compared = assert_identical(components, queries, top_k=20)
    assert compared == 12 * 25


def test_the_toy_retrievers_really_produce_ties_so_the_tie_break_is_exercised():
    hits = RandomRetriever("dense", 1).retrieve("question 0", 40)
    scores = [score for _, score in hits]
    assert len(set(scores)) < len(scores)


def test_the_generalized_control_equals_the_phase_3_body_on_the_real_dev_questions():
    """D5 (2): real dense and BM25, 600 dev questions, RRF and the 11 weights. Dev only."""
    if not POOL.exists() or not TOKENS.exists():
        pytest.skip("the pool is not on disk")

    from concept_embeddings_rag.corpus.pool import load_pool
    from concept_embeddings_rag.embeddings.backend import SentenceTransformerBackend
    from concept_embeddings_rag.embeddings.cache import (
        EmbeddingCache,
        build_query_backend,
        cache_key,
        unit_set_hash,
    )
    from concept_embeddings_rag.retrieval.bm25 import BM25Retriever
    from concept_embeddings_rag.retrieval.dense import DenseRetriever

    units, questions = load_pool(POOL)
    unit_ids = [unit.unit_id for unit in units]
    backend = SentenceTransformerBackend()
    key = cache_key(backend.name, backend.revision, unit_set_hash(unit_ids), normalized=True)
    corpus = EmbeddingCache(config.CACHE_DIR).load(key, expected_unit_ids=unit_ids)
    if corpus is None:
        pytest.skip("the corpus embeddings are not cached")
    dev = [question for question in questions if question.split == "dev"]
    assert len(dev) == config.N_DEV
    question_cache = EmbeddingCache(config.QUESTION_CACHE_DIR)
    try:
        query_backend = build_query_backend(dev, backend, question_cache)
    except Exception as error:  # a missing cache would call the model: skip, never download
        pytest.skip(f"the dev question embeddings are not usable offline: {error}")

    vectors, cached_ids = corpus
    dense = DenseRetriever(vectors=vectors, unit_ids=cached_ids, backend=query_backend)
    bm25 = BM25Retriever(units)
    compared = assert_identical([dense, bm25], [q.question for q in dev], config.EVALUATION_TOP_K)
    assert compared == 12 * config.N_DEV


def test_the_frozen_phase_3_selection_still_loads_and_verifies():
    path = config.SELECTION_DIR / "selection.json"
    if not path.exists():
        pytest.skip("the Phase 3 selection is not on disk")
    report = load_selection(
        config.SELECTION_DIR, expected_unit_set_hash=config.PHASE_1_UNIT_SET_HASH
    )
    assert json.loads(path.read_text(encoding="utf-8"))["digest"] == config.PINNED_SELECTION_DIGEST
    assert report.control is not None
    assert report.control.components == ("dense", "bm25")
