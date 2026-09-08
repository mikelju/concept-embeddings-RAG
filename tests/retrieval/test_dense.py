"""T8: dense retrieval by cosine similarity over the whole pool.

Two properties are non-negotiable: the ranking is reproducible run after run
(ties included), and retrieval always searches the full pool rather than the ten
candidates the benchmark happens to attach to each question.
"""

import numpy as np

from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.dense import DenseRetriever


class DirectionBackend:
    """Encodes a query as the literal vector written in the query string."""

    name = "fake"
    revision = "v0"
    dim = 2
    normalize = True

    def encode(self, texts):
        return np.array([[float(x) for x in t.split(",")] for t in texts], dtype=np.float32)


def a_retriever() -> DenseRetriever:
    vectors = np.array(
        [
            [1.0, 0.0],  # points east
            [0.0, 1.0],  # points north
            [0.7071, 0.7071],  # north-east
        ],
        dtype=np.float32,
    )
    return DenseRetriever(
        vectors=vectors, unit_ids=["east", "north", "ne"], backend=DirectionBackend()
    )


def test_ranking_follows_cosine_similarity():
    hits = a_retriever().retrieve("1,0", top_k=3)
    assert [unit_id for unit_id, _ in hits] == ["east", "ne", "north"]


def test_top_k_limits_the_number_of_hits():
    assert len(a_retriever().retrieve("1,0", top_k=2)) == 2


def test_asking_for_more_than_the_pool_returns_the_whole_pool():
    assert len(a_retriever().retrieve("1,0", top_k=50)) == 3


def test_ranking_is_reproducible_including_ties():
    retriever = a_retriever()
    first = retriever.retrieve("1,1", top_k=3)
    second = retriever.retrieve("1,1", top_k=3)
    assert first == second


def test_it_satisfies_the_retriever_interface():
    retriever: Retriever = a_retriever()
    assert retriever.name == "dense"
