"""T7: the embedding cache, which is what makes iterating affordable.

Two properties matter more than speed here: a second run must recompute nothing,
and a cache whose vectors no longer line up with its ids must refuse to load
rather than silently mislabel every vector in the corpus.
"""

import numpy as np
import pytest

from concept_embeddings_rag.corpus.pool import IndexingUnit
from concept_embeddings_rag.embeddings.cache import (
    CacheAlignmentError,
    EmbeddingCache,
    cache_key,
    embed_units,
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
