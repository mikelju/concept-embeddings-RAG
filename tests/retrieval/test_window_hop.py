"""Phase 13, S4: the window-seeded hop.

At `m = all (None)` every one of P1's entity nodes is a seed, so the gate removes nothing and
`WindowSeededHopStage` must equal `ColumnwiseEntityHopStage` bit for bit: that identity is
what lets the D5 grid contain P10-C itself. At `m = 1` only paragraphs holding the best-scored
entity enter, ordered by their overlap with all of P1's entities (D2). No scored entity gives
an empty list with `positives = 0`.
"""

import random
from collections.abc import Sequence

import numpy as np
import pytest

from concept_embeddings_rag.evaluation import phase13
from concept_embeddings_rag.evaluation.second_hop import gated_hop_columnwise, node_weights
from concept_embeddings_rag.nodes.index import build_node_index
from concept_embeddings_rag.nodes.local_extraction import record_from_forms
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.entity_hop import (
    CachedWindows,
    ColumnwiseEntityHopStage,
    EntityHopError,
    OnlineWindows,
    WindowSeededHopStage,
)


def a_random_index(seed: int, n_units: int = 300, vocabulary: int = 40):
    """Few forms over many rows, so shared-node sets repeat and scores tie often.

    The Phase 12 fixture (`tests/retrieval/test_seed_selection.py`), repeated here because the
    test directories are not packages.
    """
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


class NoWindows:
    """A window source the identity path must never ask for."""

    def windows_of(self, unit_id: str, p1_nodes: Sequence[int]):
        raise AssertionError(f"windows_of({unit_id!r}) called: m = None never needs a window")


def _no_question_vector(text: str) -> np.ndarray:
    raise AssertionError(f"question_vector({text!r}) called: m = None never needs one")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_m_all_equals_the_columnwise_phase_9_hop(seed):
    index = a_random_index(seed)
    weights = node_weights(index)
    columns = ColumnwiseEntityHopStage.columns_for(index)
    reference = ColumnwiseEntityHopStage(index, weights)
    stage = WindowSeededHopStage(
        index,
        weights,
        columns=columns,
        m=None,
        windows=NoWindows(),
        question_vector=_no_question_vector,
    )
    for start in range(0, len(index.unit_ids) - 12, 5):
        dense = dense_list(index.unit_ids, start)
        expected = reference.expand(dense, 100)
        got = stage.hop("q", dense, 100)
        assert got.candidates == expected.candidates
        assert got.positives == expected.positives


@pytest.mark.parametrize("seed", [1, 2])
def test_the_gated_hop_with_every_p1_node_as_seed_equals_the_phase_9_hop(seed):
    index = a_random_index(seed)
    weights = node_weights(index)
    columns = ColumnwiseEntityHopStage.columns_for(index)
    reference = ColumnwiseEntityHopStage(index, weights)
    incidence = index.incidence
    for start in range(0, len(index.unit_ids) - 12, 7):
        dense = dense_list(index.unit_ids, start)
        rows = [index.unit_ids.index(u) for u, _s in dense[:10]]
        p1 = rows[0]
        nodes = incidence.indices[incidence.indptr[p1] : incidence.indptr[p1 + 1]]
        got = gated_hop_columnwise(
            index, weights, columns, p1=p1, seeds=nodes, read=rows, depth=100
        )
        expected = reference.expand(dense, 100)
        assert tuple(got[0]) == expected.candidates and got[1] == expected.positives


# --- m = 1: the seed gates entry, the full overlap with P1 orders ----------------------------

FILLERS = [f"f{i}" for i in range(1, 10)]


def _gated_fixture():
    forms_by_unit = {
        "p1": ["Alpha", "Beta", "Gamma"],
        **{unit_id: ["Filler"] for unit_id in FILLERS},
        # Both hold the seed Alpha; c_b also shares Beta, so it must rank first although a
        # seed-only score (Phase 12's rule) would tie them and order by unit id.
        "c_a": ["Alpha"],
        "c_b": ["Alpha", "Beta"],
        "c_x": ["Beta", "Gamma"],  # no seed: never enters at m = 1
    }
    records = {
        unit_id: record_from_forms(unit_id, forms, model="fixture", configuration_digest="f")
        for unit_id, forms in forms_by_unit.items()
    }
    unit_ids = ["p1", *FILLERS, "c_a", "c_b", "c_x"]
    index = build_node_index(records, unit_ids, extraction_digest="0" * 16)
    node_of = {node.form: node.node_id for node in index.nodes}
    return index, node_of


class FixedWindows:
    """One window per P1 entity, a unit vector each."""

    def __init__(self, node_ids, vectors):
        self.node_ids = list(node_ids)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.asked: list[str] = []

    def windows_of(self, unit_id, p1_nodes):
        self.asked.append(unit_id)
        return self.node_ids, self.vectors


def _dense():
    return [("p1", 1.0)] + [(u, 0.9 - i / 100) for i, u in enumerate(FILLERS)]


def _stage(index, windows, m, question=(1.0, 0.0, 0.0)):
    weights = node_weights(index)
    return WindowSeededHopStage(
        index,
        weights,
        columns=ColumnwiseEntityHopStage.columns_for(index),
        m=m,
        windows=windows,
        question_vector=lambda _text: np.array(question, dtype=np.float32),
    )


def test_m1_admits_only_seed_holders_ordered_by_their_full_overlap_with_p1():
    index, node_of = _gated_fixture()
    windows = FixedWindows(
        [node_of["alpha"], node_of["beta"], node_of["gamma"]],
        np.eye(3),
    )
    got = _stage(index, windows, 1).hop("who", _dense(), 100)
    assert got.seeds == (node_of["alpha"],)
    assert [c.unit_id for c in got.candidates] == ["c_b", "c_a"]
    assert got.candidates[0].score > got.candidates[1].score
    assert got.positives == 2
    all_hop = _stage(index, windows, None).hop("who", _dense(), 100)
    assert "c_x" in [c.unit_id for c in all_hop.candidates]


def test_m2_takes_the_two_best_scored_entities():
    index, node_of = _gated_fixture()
    windows = FixedWindows(
        [node_of["alpha"], node_of["beta"], node_of["gamma"]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.6, 0.8, 0.0]],
    )
    got = _stage(index, windows, 2).hop("who", _dense(), 100)
    assert got.seeds == (node_of["beta"], node_of["gamma"])
    assert {c.unit_id for c in got.candidates} == {"c_b", "c_x"}


def test_no_scored_entity_gives_an_empty_list():
    index, _node_of = _gated_fixture()
    windows = FixedWindows([], np.zeros((0, 3)))
    got = _stage(index, windows, 1).hop("who", _dense(), 100)
    assert got.seeds == () and got.candidates == () and got.positives == 0
    assert got.p1_entity_nodes == 3 and got.scored_entity_nodes == 0


def test_the_gated_hop_with_no_seed_is_empty():
    index, _node_of = _gated_fixture()
    columns = ColumnwiseEntityHopStage.columns_for(index)
    got = gated_hop_columnwise(
        index,
        node_weights(index),
        columns,
        p1=0,
        seeds=np.array([], dtype=np.int64),
        read=[0],
        depth=100,
    )
    assert got == ([], 0)


# --- the window sources ----------------------------------------------------------------------


def _a_cache(tmp_path):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    unit_ids, node_ids, positions, first_words = ["u1", "u1"], [3, 4], [0, 1], [0, 2]
    key = phase13.window_cache_key("m", "r", "dev", ["u1", "u2"])
    phase13.write_window_cache(
        tmp_path, "dev", key, unit_ids, node_ids, positions, first_words, vectors
    )
    phase13.write_window_manifest(
        tmp_path,
        "dev",
        {
            "key": key,
            "vectors_digest": phase13.vectors_digest(vectors),
            "rows_digest": phase13.rows_digest(unit_ids, node_ids, positions, first_words),
        },
    )
    return phase13.read_window_cache(tmp_path, "dev")


def test_cached_windows_serve_known_units_and_refuse_unknown_ones(tmp_path):
    source = CachedWindows(_a_cache(tmp_path), units=["u1", "u2"])
    node_ids, vectors = source.windows_of("u1", [3, 4])
    assert node_ids == [3, 4] and vectors.shape == (2, 2)
    node_ids, vectors = source.windows_of("u2", [7])  # a P1 whose every window was empty
    assert node_ids == [] and vectors.shape == (0, 2)
    with pytest.raises(EntityHopError, match="not a P1"):
        source.windows_of("u9", [])


class CountingBackend:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2)


def test_online_windows_encode_each_unit_once_and_time_it():
    backend = CountingBackend()
    source = OnlineWindows(
        {"u1": ["Directed by James Cameron in 1997."]},
        {1: "james cameron", 2: "nowhere"},
        backend,
    )
    node_ids, vectors = source.windows_of("u1", [1, 2])
    assert node_ids == [1] and vectors.shape == (1, 2)
    assert backend.calls == [["directed by in 1997"]]
    again, _vectors = source.windows_of("u1", [1, 2])
    assert again == [1] and len(backend.calls) == 1
    assert len(source.calls) == 2 and source.calls[0] > 0.0 and source.calls[1] == 0.0
    assert source.windows_rows()[0] == ["u1"]
    with pytest.raises(EntityHopError, match="not in the corpus"):
        source.windows_of("u9", [])
