"""Phase 12, S3: the seeded hop stage and the three-component retriever.

At `s = all (None), exclude = False` every one of P1's entity nodes is a seed, so
`SentenceSeededHopStage` must equal `ColumnwiseEntityHopStage` bit for bit - that identity is
what lets the D3 grid contain P10-C itself, and D4 checks it again on the real index. At
`s = 1` only the best sentence's entities seed the hop; an empty seed set (every entity
excluded) gives an empty candidate list with `positives = 0`, no padding.
"""

import random
from collections.abc import Sequence

import numpy as np
import pytest

from concept_embeddings_rag.evaluation.second_hop import node_weights
from concept_embeddings_rag.nodes.index import build_node_index
from concept_embeddings_rag.nodes.local_extraction import record_from_forms
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.entity_hop import (
    ColumnwiseEntityHopStage,
    EntityHopError,
    OnlineSentences,
    SentenceSeededHopStage,
)
from concept_embeddings_rag.retrieval.fusion import (
    WEIGHTED,
    QueryTripleFusedRetriever,
    fuse,
    fuse_lists,
)

# --- the identity: s = all, exclude = False is the Phase 9 hop, bit for bit -----------------


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


class NoVectors:
    """A sentence source the identity path must never ask for a vector."""

    def sentences_of(self, unit_id: str) -> Sequence[str]:
        return []

    def vectors_of(self, unit_id: str) -> np.ndarray:
        raise AssertionError(f"vectors_of({unit_id!r}) called: s = None never needs a vector")


def _no_question_vector(text: str) -> np.ndarray:
    raise AssertionError(f"question_vector({text!r}) called: s = None never needs one")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_identity_configuration_equals_the_columnwise_phase_9_hop(seed):
    index = a_random_index(seed)
    weights = node_weights(index)
    columns = ColumnwiseEntityHopStage.columns_for(index)
    reference = ColumnwiseEntityHopStage(index, weights)
    forms = {node.node_id: node.form for node in index.nodes}
    stage = SentenceSeededHopStage(
        index,
        weights,
        columns=columns,
        s=None,
        exclude=False,
        forms=forms,
        sentences=NoVectors(),
        question_vector=_no_question_vector,
    )
    for start in range(0, len(index.unit_ids) - 12, 5):
        dense = dense_list(index.unit_ids, start)
        expected = reference.expand(dense, 100)
        got = stage.hop("q", dense, 100)
        assert got.candidates == expected.candidates
        assert got.positives == expected.positives


# --- s = 1 seeds only from the single best sentence ------------------------------------------

FILLERS = [f"f{i}" for i in range(1, 10)]


def _a_seeded_fixture():
    forms_by_unit = {
        "p1": ["Alpha", "Beta", "Gamma"],
        **{unit_id: ["Filler"] for unit_id in FILLERS},
        "c1": ["Alpha"],
        "c2": ["Beta"],
        "c3": ["Gamma"],
    }
    records = {
        unit_id: record_from_forms(unit_id, forms, model="fixture", configuration_digest="f")
        for unit_id, forms in forms_by_unit.items()
    }
    unit_ids = ["p1", *FILLERS, "c1", "c2", "c3"]
    index = build_node_index(records, unit_ids, extraction_digest="0" * 16)
    forms_by_id = {node.node_id: node.form for node in index.nodes}
    return index, forms_by_id


SEEDED_SENTENCES = ["About alpha here.", "About beta and gamma here."]
SEEDED_VECTORS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class FixedSentences:
    """P1 always has the same two sentences and vectors, whichever unit is asked."""

    def sentences_of(self, unit_id: str) -> Sequence[str]:
        return SEEDED_SENTENCES

    def vectors_of(self, unit_id: str) -> np.ndarray:
        return SEEDED_VECTORS


def _question_vector_favoring_sentence_0(_text: str) -> np.ndarray:
    return np.array([1.0, 0.0], dtype=np.float32)


def _make_stage(
    index,
    forms_by_id,
    *,
    s,
    exclude,
    question_vector=_question_vector_favoring_sentence_0,
):
    weights = node_weights(index)
    columns = ColumnwiseEntityHopStage.columns_for(index)
    return SentenceSeededHopStage(
        index,
        weights,
        columns=columns,
        s=s,
        exclude=exclude,
        forms=forms_by_id,
        sentences=FixedSentences(),
        question_vector=question_vector,
    )


def _dense_list_with_p1_first(index) -> list[Hit]:
    unit_ids = list(index.unit_ids)
    return [(u, 1.0 - i / 1000) for i, u in enumerate(unit_ids)]


def test_s1_seeds_only_the_best_sentences_entities():
    index, forms_by_id = _a_seeded_fixture()
    stage = _make_stage(index, forms_by_id, s=1, exclude=False)
    dense = _dense_list_with_p1_first(index)

    result = stage.hop("who directed it?", dense, 100)

    alpha_id = next(n.node_id for n in index.nodes if n.form == "alpha")
    assert result.seeds == (alpha_id,)
    assert [c.unit_id for c in result.candidates] == ["c1"]
    assert result.p1_entity_nodes == 3
    assert result.located_entity_nodes == 3


def test_excluding_every_entity_gives_an_empty_list_with_no_padding():
    index, forms_by_id = _a_seeded_fixture()
    stage = _make_stage(index, forms_by_id, s=None, exclude=True)
    dense = _dense_list_with_p1_first(index)

    result = stage.hop("alpha beta and gamma are all named here", dense, 100)

    assert result.seeds == ()
    assert result.candidates == ()
    assert result.positives == 0


def test_propose_returns_the_candidates_as_hits():
    index, forms_by_id = _a_seeded_fixture()
    stage = _make_stage(index, forms_by_id, s=1, exclude=False)
    dense = _dense_list_with_p1_first(index)
    expected = stage.hop("who directed it?", dense, 100)
    proposed = stage.propose("who directed it?", dense, 100)
    assert proposed == [(c.unit_id, c.score) for c in expected.candidates]


# --- OnlineSentences: one encode per unit, timed --------------------------------------------


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.zeros((len(texts), 2), dtype=np.float32)


def test_online_sentences_encodes_each_unit_once_and_times_the_encode():
    backend = FakeBackend()
    online = OnlineSentences({"u1": ["s1", "s2"]}, backend)

    first = online.vectors_of("u1")
    second = online.vectors_of("u1")

    assert np.array_equal(first, second)
    assert len(backend.calls) == 1  # encoded once
    assert online.calls[0] >= 0.0
    assert online.calls[1] == 0.0  # the second call is a cache hit: no time spent
    with pytest.raises(EntityHopError, match="not in the corpus"):
        online.sentences_of("missing")


# --- QueryTripleFusedRetriever: the third slot is asked with the query too -----------------

DENSE_HITS: list[Hit] = [("d1", 0.9), ("d2", 0.7), ("s1", 0.6), ("d4", 0.2)]
BM25_HITS: list[Hit] = [("b1", 12.0), ("d2", 9.0), ("s1", 3.0)]
HOP_HITS: list[Hit] = [("h1", 8.0), ("s1", 5.0), ("h2", 1.0)]
FUSION_WEIGHTS = {"dense": 0.5, "bm25": 0.3, "seeded-hop": 0.2}


class Dense:
    name = "dense"

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return list(DENSE_HITS[:top_k])


class BM25:
    name = "bm25"

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return list(BM25_HITS[:top_k])


class SeededHop:
    name = "seeded-hop"
    s = None
    exclude = False

    def __init__(self) -> None:
        self.received: list[tuple[str, list[Hit]]] = []

    def propose(self, query: str, first: Sequence[Hit], top_k: int) -> list[Hit]:
        self.received.append((query, list(first)))
        return list(HOP_HITS[:top_k])


def test_retrieve_equals_fuse_over_the_three_lists():
    hop = SeededHop()
    system = QueryTripleFusedRetriever(Dense(), BM25(), hop, weights=FUSION_WEIGHTS)

    ranked = system.retrieve("q", 5)

    expected = fuse(
        [DENSE_HITS, BM25_HITS, HOP_HITS], scheme=WEIGHTED, top_k=5, weights=(0.5, 0.3, 0.2)
    )
    assert ranked == expected
    assert hop.received == [("q", DENSE_HITS[:5])]  # asked with the query and dense's list


def test_fuse_lists_is_the_same_call_the_retriever_makes():
    system = QueryTripleFusedRetriever(Dense(), BM25(), SeededHop(), weights=FUSION_WEIGHTS)
    lists = (DENSE_HITS, BM25_HITS, HOP_HITS)
    assert fuse_lists(lists, (0.5, 0.3, 0.2), top_k=4) == system.retrieve("q", 4)


def test_components_must_hold_their_declared_roles():
    with pytest.raises(ValueError, match="bm25"):
        QueryTripleFusedRetriever(Dense(), Dense(), SeededHop(), weights=FUSION_WEIGHTS)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seeded-hop"):
        QueryTripleFusedRetriever(Dense(), BM25(), BM25(), weights=FUSION_WEIGHTS)  # type: ignore[arg-type]


def test_describe_records_s_exclude_and_the_weights():
    hop = SeededHop()
    hop.s, hop.exclude = 2, True
    system = QueryTripleFusedRetriever(Dense(), BM25(), hop, weights=FUSION_WEIGHTS)
    assert system.name == "hybrid-bm25-seeded-hop"
    assert system.describe() == {
        "name": "hybrid-bm25-seeded-hop",
        "components": ["dense", "bm25", "seeded-hop"],
        "scheme": "weighted",
        "normalization": "min_max",
        "weights": FUSION_WEIGHTS,
        "s": 2,
        "exclude": True,
        "fitted_on": "dev",
    }
