"""Phase 11, S3: the DF-capped P1 hop, the question hop and the four-component fusion.

Uncapped, the new seeded hop must be the Phase 9 hop bit for bit: that identity is what lets
the grid contain P10-C itself. The cap drops exactly the nodes above it; the question hop
seeds from the question's nodes and still never re-admits a read unit; and the fused system
ranks exactly what `fuse_lists` gives over its four lists, as the dev grid assumes.
"""

import random
from collections.abc import Sequence

import numpy as np
import pytest

from concept_embeddings_rag.evaluation.second_hop import (
    node_hop_columnwise,
    node_weights,
    seeded_hop_columnwise,
)
from concept_embeddings_rag.nodes.index import build_node_index
from concept_embeddings_rag.nodes.local_extraction import record_from_forms
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.conceptual import concept_support
from concept_embeddings_rag.retrieval.entity_hop import (
    READ_DEPTH,
    CappedEntityHopStage,
    ColumnwiseEntityHopStage,
    EntityHopError,
    QuestionEntityHop,
    capped_seeds,
)
from concept_embeddings_rag.retrieval.fusion import QuadFusedRetriever, fuse, fuse_lists


def a_random_index(seed: int, n_units: int = 300, vocabulary: int = 40):
    """Few forms over many rows, so shared-node sets repeat and scores tie often."""
    rng = random.Random(seed)  # noqa: S311 - fixture layout, not security
    unit_ids = [f"{rng.getrandbits(64):016x}" for _ in range(n_units)]
    records = {
        unit_id: record_from_forms(
            unit_id,
            [f"Form {rng.randrange(vocabulary)}" for _ in range(rng.randrange(0, 5))],
            model="fixture",
            configuration_digest="f",
        )
        for unit_id in unit_ids
    }
    return build_node_index(records, unit_ids, extraction_digest=f"{seed:064d}")


def dense_list(unit_ids: Sequence[str], start: int, n: int = 12) -> list[Hit]:
    ordered = sorted(unit_ids)
    return [(u, 1.0 - p / 100) for p, u in enumerate(ordered[start : start + n])]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_seeded_with_all_of_p1s_entities_is_the_phase_9_hop(seed):
    index = a_random_index(seed)
    weights = node_weights(index)
    columns = ColumnwiseEntityHopStage.columns_for(index)
    incidence = index.incidence
    for p1 in range(len(index.unit_ids)):
        read = [p1, (p1 + 1) % len(index.unit_ids), (p1 + 7) % len(index.unit_ids)]
        seeds = incidence.indices[incidence.indptr[p1] : incidence.indptr[p1 + 1]]
        expected = node_hop_columnwise(index, weights, columns, p1=p1, read=read, depth=10)
        observed = seeded_hop_columnwise(index, weights, columns, seeds=seeds, read=read, depth=10)
        assert observed == expected


def test_the_uncapped_stage_expands_exactly_as_the_phase_9_stage():
    index = a_random_index(4)
    weights = node_weights(index)
    columns = ColumnwiseEntityHopStage.columns_for(index)
    df = concept_support(index.incidence)
    reference = ColumnwiseEntityHopStage(index, weights)
    uncapped = CappedEntityHopStage(index, weights, cap=None, columns=columns, df=df)
    for start in range(0, len(index.unit_ids) - 12, 5):
        dense = dense_list(index.unit_ids, start)
        assert uncapped.expand(dense, 100) == reference.expand(dense, 100)
    assert uncapped.name == reference.name


def test_the_cap_keeps_every_node_at_or_below_it_and_drops_the_rest():
    df = np.array([1, 5, 10, 11, 1000])
    mask = np.array([True, True, True, True, False])
    nodes = np.array([4, 3, 2, 1, 0])
    assert capped_seeds(nodes, mask, df, cap=10).tolist() == [0, 1, 2]
    assert capped_seeds(nodes, mask, df, cap=None).tolist() == [0, 1, 2, 3]
    assert capped_seeds(nodes, mask, df, cap=0).tolist() == []


def test_a_capped_stage_hops_only_from_the_kept_nodes():
    index = a_random_index(5)
    weights = node_weights(index)
    columns = ColumnwiseEntityHopStage.columns_for(index)
    df = concept_support(index.incidence)
    cap = int(np.median(df[df > 0]))
    stage = CappedEntityHopStage(index, weights, cap=cap, columns=columns, df=df)
    rows = {u: r for r, u in enumerate(index.unit_ids)}
    incidence = index.incidence
    for start in range(0, len(index.unit_ids) - 12, 7):
        dense = dense_list(index.unit_ids, start)
        read = [rows[u] for u, _ in dense[:READ_DEPTH]]
        p1_nodes = incidence.indices[incidence.indptr[read[0]] : incidence.indptr[read[0] + 1]]
        kept = p1_nodes[df[p1_nodes] <= cap]
        expected, _ = seeded_hop_columnwise(
            index, weights, columns, seeds=kept, read=read, depth=100
        )
        assert [(c.unit_id, c.score) for c in expected] == stage.propose(dense, 100)


def test_the_question_hop_seeds_from_the_question_and_excludes_what_dense_read():
    index = a_random_index(6)
    weights = node_weights(index)
    columns = ColumnwiseEntityHopStage.columns_for(index)
    rows = {u: r for r, u in enumerate(index.unit_ids)}
    hop = QuestionEntityHop(
        index, weights, columns=columns, nodes_by_question={"q": [0, 1], "none": []}
    )
    dense = dense_list(index.unit_ids, 3)
    read = [rows[u] for u, _ in dense[:READ_DEPTH]]
    expected, _ = seeded_hop_columnwise(
        index, weights, columns, seeds=np.array([0, 1]), read=read, depth=100
    )
    proposed = hop.propose("q", dense, 100)
    assert proposed == [(c.unit_id, c.score) for c in expected]
    assert proposed  # the fixture shares node 0 or 1 with some unread unit
    assert not {u for u, _ in proposed} & {u for u, _ in dense[:READ_DEPTH]}
    assert hop.propose("none", dense, 100) == []
    with pytest.raises(EntityHopError, match="no recorded"):
        hop.propose("unknown", dense, 100)


# --- the four-component fusion ---------------------------------------------------------

DENSE_HITS: list[Hit] = [("d1", 0.9), ("d2", 0.7), ("s1", 0.6), ("d4", 0.2)]
BM25_HITS: list[Hit] = [("b1", 12.0), ("d2", 9.0), ("s1", 3.0)]
HOP_HITS: list[Hit] = [("h1", 8.0), ("s1", 5.0), ("h2", 1.0)]
QUESTION_HITS: list[Hit] = [("x1", 4.0), ("h1", 2.0), ("x2", 0.5)]
WEIGHTS = {"dense": 0.4, "bm25": 0.3, "entity-hop": 0.1, "question-hop": 0.2}


class Dense:
    name = "dense"

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return list(DENSE_HITS[:top_k])


class BM25:
    name = "bm25"

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return list(BM25_HITS[:top_k])


class Hop:
    name = "entity-hop"
    cap = 3000

    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
        return list(HOP_HITS[:top_k])


class QuestionHop:
    name = "question-hop"

    def __init__(self) -> None:
        self.received: list[tuple[str, list[Hit]]] = []

    def propose(self, query: str, first: Sequence[Hit], top_k: int) -> list[Hit]:
        self.received.append((query, list(first)))
        return list(QUESTION_HITS[:top_k])


def test_retrieve_equals_fuse_over_the_four_lists():
    question = QuestionHop()
    system = QuadFusedRetriever(Dense(), BM25(), Hop(), question, weights=WEIGHTS)
    expected = fuse(
        [DENSE_HITS, BM25_HITS, HOP_HITS, QUESTION_HITS],
        scheme="weighted",
        top_k=5,
        weights=(0.4, 0.3, 0.1, 0.2),
    )
    assert system.retrieve("q", 5) == expected
    lists = (DENSE_HITS, BM25_HITS, HOP_HITS, QUESTION_HITS)
    assert fuse_lists(lists, (0.4, 0.3, 0.1, 0.2), top_k=5) == expected
    assert question.received == [("q", DENSE_HITS[:5])]  # the question and Dense's list


def test_a_zero_question_weight_still_lets_its_list_into_the_union():
    weights = {"dense": 0.5, "bm25": 0.3, "entity-hop": 0.2, "question-hop": 0.0}
    system = QuadFusedRetriever(Dense(), BM25(), Hop(), QuestionHop(), weights=weights)
    ranked = [unit for unit, _ in system.retrieve("q", 20)]
    assert "x1" in ranked and "x2" in ranked


def test_components_hold_their_roles_and_describe_records_the_cap():
    with pytest.raises(ValueError, match="question-hop"):
        QuadFusedRetriever(Dense(), BM25(), Hop(), Hop(), weights=WEIGHTS)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="every component"):
        QuadFusedRetriever(
            Dense(), BM25(), Hop(), QuestionHop(), weights={"dense": 0.5, "bm25": 0.5}
        )
    system = QuadFusedRetriever(Dense(), BM25(), Hop(), QuestionHop(), weights=WEIGHTS)
    assert system.name == "hybrid-bm25-entity-hop-question-hop"
    assert system.describe()["components"] == ["dense", "bm25", "entity-hop", "question-hop"]
    assert system.describe()["weights"] == WEIGHTS
    assert system.describe()["p1_df_cap"] == 3000
