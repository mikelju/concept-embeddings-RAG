"""T2-T3: the question embedding cache and the backend that serves it.

The dev sweep of Phase 3 makes dozens of passes over the same 600 questions. A
dense encode costs ~30.7 ms on this machine, so re-encoding them every pass would
cost hours and change no result. Two properties are what make the cache safe to
lean on: a question set that does not match the cache must refuse to load rather
than realign silently, and a cache miss must raise rather than fall back to the
model - a silent fallback is the bug that makes the sweep slow without making it
wrong, and therefore invisible.

Every test here embeds with a fake backend: nothing downloads a model and nothing
reads the corpus cache.
"""

import numpy as np
import pytest

from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.cache import (
    CacheAlignmentError,
    CachedQueryBackend,
    EmbeddingCache,
    QueryCacheMiss,
    build_query_backend,
    embed_questions,
    question_cache_key,
    question_set_hash,
)
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.dense import DenseRetriever


class CountingBackend:
    """Deterministic stand-in for the real model, and it counts its calls."""

    name = "fake"
    revision = "v0"
    dim = 3
    normalize = True

    def __init__(self):
        self.calls = 0
        self.texts_seen: list[str] = []

    def encode(self, texts):
        self.calls += 1
        self.texts_seen.extend(texts)
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts], dtype=np.float32)


def some_questions(split: str = "dev") -> list[Question]:
    return [
        Question(
            qid="q1",
            question="who wrote it?",
            answer="a",
            gold_unit_ids=("aaa",),
            supporting_facts=(("A", 0),),
            split=split,
        ),
        Question(
            qid="q2",
            question="where is the river?",
            answer="b",
            gold_unit_ids=("bbb",),
            supporting_facts=(("B", 0),),
            split=split,
        ),
    ]


# --- T2: the cache ------------------------------------------------------------


def test_question_set_hash_ignores_order_but_not_membership():
    assert question_set_hash(["q1", "q2"]) == question_set_hash(["q2", "q1"])
    assert question_set_hash(["q1", "q2"]) != question_set_hash(["q1", "q2", "q3"])


def test_the_key_separates_model_revision_question_set_and_split():
    base = question_cache_key("model", "rev", "qset", "dev", normalized=True)
    assert base != question_cache_key("other", "rev", "qset", "dev", normalized=True)
    assert base != question_cache_key("model", "rev2", "qset", "dev", normalized=True)
    assert base != question_cache_key("model", "rev", "other", "dev", normalized=True)
    assert base != question_cache_key("model", "rev", "qset", "test", normalized=True)
    assert base != question_cache_key("model", "rev", "qset", "dev", normalized=False)


def test_round_trip_returns_the_same_vectors_and_the_same_qid_order(tmp_path):
    backend = CountingBackend()
    cache = EmbeddingCache(tmp_path)
    questions = some_questions()

    vectors_first, qids_first = embed_questions(questions, backend, cache)
    vectors_second, qids_second = embed_questions(questions, backend, cache)

    assert backend.calls == 1
    assert qids_first == qids_second == ["q1", "q2"]
    assert np.array_equal(vectors_first, vectors_second)


def test_the_question_text_is_what_gets_embedded(tmp_path):
    backend = CountingBackend()
    embed_questions(some_questions(), backend, EmbeddingCache(tmp_path))

    assert backend.texts_seen == ["who wrote it?", "where is the river?"]


def test_each_split_gets_its_own_artifact_instead_of_overwriting(tmp_path):
    backend = CountingBackend()
    cache = EmbeddingCache(tmp_path)

    embed_questions(some_questions("dev"), backend, cache)
    embed_questions(some_questions("test"), backend, cache)

    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_a_cache_whose_question_set_does_not_match_refuses_to_load(tmp_path):
    """The failure mode this guards against is every vector silently mislabelled."""
    backend = CountingBackend()
    cache = EmbeddingCache(tmp_path)
    questions = some_questions()
    embed_questions(questions, backend, cache)

    key = question_cache_key(
        backend.name,
        backend.revision,
        question_set_hash([q.qid for q in questions]),
        "dev",
        normalized=True,
    )
    with pytest.raises(CacheAlignmentError):
        cache.load(key, expected_unit_ids=["q1", "q9"])


def test_a_mixed_split_is_refused_rather_than_filed_under_one_of_them(tmp_path):
    mixed = [some_questions("dev")[0], some_questions("test")[1]]

    with pytest.raises(ValueError, match="split"):
        embed_questions(mixed, CountingBackend(), EmbeddingCache(tmp_path))


def test_the_artifact_loads_without_pickle(tmp_path):
    backend = CountingBackend()
    cache = EmbeddingCache(tmp_path)
    embed_questions(some_questions(), backend, cache)

    for path in tmp_path.glob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            assert payload["vectors"].shape[0] == 2


# --- T3: the backend that serves it -------------------------------------------


def test_it_satisfies_the_embedding_backend_interface(tmp_path):
    real = CountingBackend()
    served = build_query_backend(some_questions(), real, EmbeddingCache(tmp_path))

    assert served.name == real.name
    assert served.revision == real.revision
    assert served.dim == real.dim


def test_a_known_question_resolves_to_the_vector_that_was_cached(tmp_path):
    real = CountingBackend()
    cache = EmbeddingCache(tmp_path)
    vectors, qids = embed_questions(some_questions(), real, cache)
    served = build_query_backend(some_questions(), real, cache)

    assert np.array_equal(served.encode(["who wrote it?"])[0], vectors[qids.index("q1")])


def test_an_unknown_query_raises_instead_of_falling_back_to_the_model(tmp_path):
    real = CountingBackend()
    served = build_query_backend(some_questions(), real, EmbeddingCache(tmp_path))
    calls_before = real.calls

    with pytest.raises(QueryCacheMiss):
        served.encode(["a question nobody asked"])

    assert real.calls == calls_before


def test_two_identical_questions_resolve_to_one_stored_vector():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    served = CachedQueryBackend(
        texts=["same text", "same text"],
        vectors=vectors,
        name="fake",
        revision="v0",
    )

    assert len(served) == 1
    assert np.array_equal(served.encode(["same text"])[0], vectors[0])


def test_two_identical_questions_with_different_vectors_are_a_refusal_not_a_choice():
    """Silently keeping one of two answers for the same key is how a cache lies."""
    with pytest.raises(CacheAlignmentError):
        CachedQueryBackend(
            texts=["same text", "same text"],
            vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            name="fake",
            revision="v0",
        )


def test_dense_retriever_runs_against_it_with_dense_py_unmodified(tmp_path):
    """The point of D1: a different backend, not a different retriever."""
    real = CountingBackend()
    served = build_query_backend(some_questions(), real, EmbeddingCache(tmp_path))

    unit_vectors = np.array([[3.0, 1.0, 0.0], [12.0, 1.0, 0.0]], dtype=np.float32)
    retriever = DenseRetriever(unit_vectors, ["aaa", "bbb"], backend=served)
    calls_before = real.calls

    hits = retriever.retrieve("where is the river?", top_k=2)

    assert isinstance(retriever, Retriever)
    assert [unit_id for unit_id, _score in hits] == ["bbb", "aaa"]
    assert real.calls == calls_before


def test_the_served_dimension_is_verified_against_the_cache_metadata(tmp_path):
    """Verify on load, do not merely record: the sidecar is checked, not trusted."""
    real = CountingBackend()
    cache = EmbeddingCache(tmp_path)
    embed_questions(some_questions(), real, cache)

    sidecar = next(tmp_path.glob("*.json"))
    sidecar.write_text(sidecar.read_text(encoding="utf-8").replace('"dim": 3', '"dim": 9'))

    with pytest.raises(CacheAlignmentError):
        build_query_backend(some_questions(), real, cache)
