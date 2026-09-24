"""Phase 10, S3: Dense + BM25 + Entity Hop fused as three components.

The three-way hybrid must rank exactly what the existing `fuse` gives over the three lists
it gathers, so that the dev grid (which re-fuses cached lists) and the held-out pass (which
calls `retrieve`) measure the same system. The two-component `FusedRetriever` is untouched.
"""

from collections.abc import Sequence

import pytest

from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.fusion import (
    WEIGHTED,
    FusedRetriever,
    TripleFusedRetriever,
    fuse,
    fuse_lists,
)

DENSE_HITS: list[Hit] = [("d1", 0.9), ("d2", 0.7), ("s1", 0.6), ("d4", 0.2)]
BM25_HITS: list[Hit] = [("b1", 12.0), ("d2", 9.0), ("s1", 3.0)]
HOP_HITS: list[Hit] = [("h1", 8.0), ("s1", 5.0), ("h2", 1.0)]
WEIGHTS = {"dense": 0.5, "bm25": 0.3, "entity-hop": 0.2}


class Dense:
    name = "dense"

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        self.calls += 1
        return list(DENSE_HITS[:top_k])


class BM25:
    name = "bm25"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        self.queries.append(query)
        return list(BM25_HITS[:top_k])


class Hop:
    name = "entity-hop"

    def __init__(self) -> None:
        self.received: list[list[Hit]] = []

    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
        self.received.append(list(first))
        return list(HOP_HITS[:top_k])


def test_retrieve_equals_fuse_over_the_three_lists():
    dense, bm25, hop = Dense(), BM25(), Hop()
    system = TripleFusedRetriever(dense, bm25, hop, weights=WEIGHTS)

    ranked = system.retrieve("q", 5)

    expected = fuse(
        [DENSE_HITS, BM25_HITS, HOP_HITS], scheme=WEIGHTED, top_k=5, weights=(0.5, 0.3, 0.2)
    )
    assert ranked == expected
    assert dense.calls == 1  # Dense is asked once
    assert bm25.queries == ["q"]  # BM25 reads the question
    assert hop.received == [DENSE_HITS[:5]]  # the stage reads Dense's list, never the question


def test_fuse_lists_is_the_same_call_the_retriever_makes():
    system = TripleFusedRetriever(Dense(), BM25(), Hop(), weights=WEIGHTS)
    lists = (DENSE_HITS, BM25_HITS, HOP_HITS)
    assert fuse_lists(lists, (0.5, 0.3, 0.2), top_k=4) == system.retrieve("q", 4)


def test_the_two_way_hybrid_is_fuse_lists_over_its_two_lists():
    """The dev reproduction rebuilds P9-B and P9-C from cached lists with this identity."""
    for second, weights in ((BM25(), {"dense": 0.5, "bm25": 0.5}), (Hop(), None)):
        if weights is None:
            weights = {"dense": 0.7, "entity-hop": 0.3}
        hybrid = FusedRetriever([Dense(), second], scheme=WEIGHTED, weights=weights)
        other = BM25_HITS if second.name == "bm25" else HOP_HITS
        ordered = (weights["dense"], weights[second.name])
        assert hybrid.retrieve("q", 4) == fuse_lists((DENSE_HITS, other), ordered, top_k=4)


def test_a_zero_entity_weight_still_lets_the_hop_list_into_the_union():
    """Declared, not compensated: the stage's units enter the union at the absent floor."""
    zero = TripleFusedRetriever(
        Dense(), BM25(), Hop(), weights={"dense": 0.5, "bm25": 0.5, "entity-hop": 0.0}
    )
    ranked = [unit for unit, _ in zero.retrieve("q", 10)]
    assert "h1" in ranked and "h2" in ranked


def test_weights_are_named_convex_and_complete():
    with pytest.raises(ValueError, match="sum to 1"):
        TripleFusedRetriever(
            Dense(), BM25(), Hop(), weights={"dense": 0.5, "bm25": 0.5, "entity-hop": 0.5}
        )
    with pytest.raises(ValueError, match="every component"):
        TripleFusedRetriever(Dense(), BM25(), Hop(), weights={"dense": 0.5, "bm25": 0.5})
    with pytest.raises(ValueError, match="negative"):
        TripleFusedRetriever(
            Dense(), BM25(), Hop(), weights={"dense": 0.6, "bm25": 0.6, "entity-hop": -0.2}
        )


def test_components_must_be_the_declared_three_in_their_roles():
    with pytest.raises(ValueError, match="bm25"):
        TripleFusedRetriever(Dense(), Dense(), Hop(), weights=WEIGHTS)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entity-hop"):
        TripleFusedRetriever(Dense(), BM25(), BM25(), weights=WEIGHTS)  # type: ignore[arg-type]


def test_describe_names_the_system_and_its_fitted_weights():
    system = TripleFusedRetriever(Dense(), BM25(), Hop(), weights=WEIGHTS)
    assert system.name == "hybrid-bm25-entity-hop"
    assert system.describe() == {
        "name": "hybrid-bm25-entity-hop",
        "components": ["dense", "bm25", "entity-hop"],
        "scheme": "weighted",
        "normalization": "min_max",
        "weights": WEIGHTS,
        "fitted_on": "dev",
    }
