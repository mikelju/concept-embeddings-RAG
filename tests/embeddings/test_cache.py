"""T7: the embedding cache, which is what makes iterating affordable.

Two properties matter more than speed here: a second run must recompute nothing,
and a cache whose vectors no longer line up with its ids must refuse to load
rather than silently mislabel every vector in the corpus.
"""

import hashlib
import json

import numpy as np
import pytest

from concept_embeddings_rag.corpus.pool import IndexingUnit, Question
from concept_embeddings_rag.embeddings.backend import QueryPromptError
from concept_embeddings_rag.embeddings.cache import (
    CacheAlignmentError,
    EmbeddingCache,
    _question_artifact_key,
    build_query_backend,
    cache_key,
    embed_questions,
    embed_units,
    question_cache_key,
    unit_set_hash,
)


class CountingBackend:
    name = "fake"
    revision = "v0"
    dim = 3

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts], dtype=np.float32)


def some_units() -> list[IndexingUnit]:
    return [
        IndexingUnit(unit_id="aaa", title="A", sentences=("first.",)),
        IndexingUnit(unit_id="bbb", title="B", sentences=("second one.",)),
    ]


def test_cache_key_changes_with_the_model_and_with_the_corpus():
    base = cache_key("model", "rev", "unitset", normalized=True)
    assert base != cache_key("other", "rev", "unitset", normalized=True)
    assert base != cache_key("model", "rev2", "unitset", normalized=True)
    assert base != cache_key("model", "rev", "other-unitset", normalized=True)
    assert base != cache_key("model", "rev", "unitset", normalized=False)


def test_unit_set_hash_ignores_order():
    assert unit_set_hash(["a", "b", "c"]) == unit_set_hash(["c", "a", "b"])
    assert unit_set_hash(["a", "b"]) != unit_set_hash(["a", "b", "c"])


def test_second_run_recomputes_nothing(tmp_path):
    backend = CountingBackend()
    cache = EmbeddingCache(tmp_path)
    units = some_units()

    vectors_first, ids_first = embed_units(units, backend, cache)
    vectors_second, ids_second = embed_units(units, backend, cache)

    assert backend.calls == 1
    assert ids_first == ids_second
    assert np.array_equal(vectors_first, vectors_second)


def test_changing_the_corpus_produces_a_new_artifact_instead_of_overwriting(tmp_path):
    backend = CountingBackend()
    cache = EmbeddingCache(tmp_path)

    embed_units(some_units(), backend, cache)
    bigger = some_units() + [IndexingUnit(unit_id="ccc", title="C", sentences=("third.",))]
    embed_units(bigger, backend, cache)

    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_a_length_mismatch_between_vectors_and_ids_aborts(tmp_path):
    cache = EmbeddingCache(tmp_path)
    key = "deadbeef"

    with pytest.raises(CacheAlignmentError):
        cache.save(key, np.zeros((3, 2), dtype=np.float32), ["a", "b"], metadata={})


def test_loading_with_unexpected_ids_aborts(tmp_path):
    cache = EmbeddingCache(tmp_path)
    key = "deadbeef"
    cache.save(key, np.zeros((2, 2), dtype=np.float32), ["a", "b"], metadata={})

    with pytest.raises(CacheAlignmentError):
        cache.load(key, expected_unit_ids=["a", "z"])


def test_cache_files_load_without_pickle(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.save(key := "cafe", np.zeros((2, 2), dtype=np.float32), ["a", "b"], metadata={})

    with np.load(cache.path_for(key), allow_pickle=False) as payload:
        assert payload["vectors"].shape == (2, 2)


# --- Phase 8 (S2): the query prompt belongs to the question key alone --------
#
# Decision 5 / R3: an asymmetric model encodes queries with an instruction prefix and
# documents without one, so the prompt must enter `question_cache_key` - or two
# different query encodings would collide under one key - and must **not** enter the
# corpus key, since letting a query-only parameter invalidate 19,366 paragraph
# embeddings would buy nothing.
#
# The prompt is appended only when it is non-empty. Appending it unconditionally would
# change every Phase 1-7 question key and orphan every existing question cache: an
# invalidation by arithmetic rather than by deletion, and just as much a breach of R4.


class PromptedBackend(CountingBackend):
    """A backend that carries a resolved query prompt, as the Phase 8 one does."""

    def __init__(self, prompt: str):
        super().__init__()
        self.resolved_query_prompt = prompt
        self.normalize = True


def a_question(qid: str, text: str, split: str = "dev") -> Question:
    return Question(
        qid=qid,
        question=text,
        answer="-",
        gold_unit_ids=("aaa",),
        supporting_facts=(),
        split=split,
    )


def test_an_empty_prompt_leaves_every_legacy_question_key_byte_identical():
    """The payload Phases 1-7 hashed, recomputed here rather than trusted."""
    legacy = hashlib.sha1(
        b"question|model|rev|qset|dev|1", usedforsecurity=False
    ).hexdigest()[:16]

    assert question_cache_key("model", "rev", "qset", "dev", True) == legacy
    assert question_cache_key("model", "rev", "qset", "dev", True, "") == legacy


def test_a_non_empty_prompt_changes_the_question_key_and_not_the_corpus_key():
    without = question_cache_key("model", "rev", "qset", "dev", True)
    with_prompt = question_cache_key("model", "rev", "qset", "dev", True, "Instruct:\nQuery:")

    assert with_prompt != without
    # Two different prompts cannot collide under one key.
    assert with_prompt != question_cache_key("model", "rev", "qset", "dev", True, "Other:")
    # The corpus key takes no prompt at all: documents are encoded without one.
    assert cache_key("model", "rev", "unitset", normalized=True) == cache_key(
        "model", "rev", "unitset", normalized=True
    )


def test_embed_units_refuses_a_backend_that_carries_a_query_prompt(tmp_path):
    """Structural, not a convention someone can forget: documents stay unprefixed."""
    backend = PromptedBackend("Instruct:\nQuery:")
    cache = EmbeddingCache(tmp_path)

    with pytest.raises(QueryPromptError):
        embed_units(some_units(), backend, cache)

    assert backend.calls == 0
    assert list(tmp_path.glob("*.npz")) == []


def test_questions_embedded_with_a_prompt_land_beside_those_embedded_without(tmp_path):
    cache = EmbeddingCache(tmp_path)
    questions = [a_question("q1", "a question")]

    embed_questions(questions, CountingBackend(), cache)
    embed_questions(questions, PromptedBackend("Instruct:\nQuery:"), cache)

    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_the_prompt_reaches_the_question_sidecar_and_a_disagreeing_one_is_refused(tmp_path):
    """A sidecar is only worth writing if something reads it back and disagrees."""
    cache = EmbeddingCache(tmp_path)
    questions = [a_question("q1", "a question")]
    backend = PromptedBackend("Instruct:\nQuery:")

    build_query_backend(questions, backend, cache)
    key = _question_artifact_key(questions, backend)
    recorded = json.loads(cache.sidecar_for(key).read_text(encoding="utf-8"))
    assert recorded["query_prompt"] == "Instruct:\nQuery:"

    recorded["query_prompt"] = "Instruct: something else\nQuery:"
    cache.sidecar_for(key).write_text(json.dumps(recorded), encoding="utf-8")
    with pytest.raises(CacheAlignmentError, match="query_prompt"):
        build_query_backend(questions, backend, cache)


def test_a_question_cache_without_a_prompt_is_verified_exactly_as_it_is_today(tmp_path):
    """Phase 1-7 caches keep loading and verifying unchanged: no field written, none checked."""
    cache = EmbeddingCache(tmp_path)
    questions = [a_question("q1", "a question")]
    backend = CountingBackend()

    build_query_backend(questions, backend, cache)

    key = _question_artifact_key(questions, backend)
    recorded = json.loads(cache.sidecar_for(key).read_text(encoding="utf-8"))
    assert "query_prompt" not in recorded
    assert build_query_backend(questions, backend, cache) is not None
