"""Phase 14, S2: the relevance-ordered hop.

At `alpha = 0` the stage must return `node_hop_columnwise`'s list and rarity scores unchanged,
without reading a vector: that identity is what lets the D4 grid contain P10-C itself (D2,
D3). For `alpha > 0` it scores the Phase 9 hop's whole candidate set `C(q)` by
`(1 - alpha) * r_hat + alpha * s_hat`, both terms min-max normalized over `C(q)`, so a
candidate below Phase 9's depth cut can enter the list.
"""

import random
from collections.abc import Sequence

import numpy as np
import pytest

from concept_embeddings_rag.evaluation import phase14
from concept_embeddings_rag.evaluation.second_hop import node_weights, relevance_hop_columnwise
from concept_embeddings_rag.nodes.index import build_node_index
from concept_embeddings_rag.nodes.local_extraction import record_from_forms
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.entity_hop import (
    ColumnwiseEntityHopStage,
    RelevanceHopStage,
)
from concept_embeddings_rag.retrieval.fusion import QueryTripleFusedRetriever

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


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


def random_vectors(seed: int, n_rows: int, dim: int = 4, distinct: int = 3) -> np.ndarray:
    """L2-normalized rows drawn from a few distinct vectors, so similarities tie often."""
    rng = np.random.default_rng(seed)
    choices = rng.standard_normal((distinct, dim)).astype(np.float32)
    choices /= np.linalg.norm(choices, axis=1, keepdims=True)
    return choices[rng.integers(0, distinct, size=n_rows)]


def dense_list(unit_ids: Sequence[str], start: int, n: int = 12) -> list[Hit]:
    ordered = sorted(unit_ids)
    return [(u, 1.0 - p / 100) for p, u in enumerate(ordered[start : start + n])]


class Untouchable:
    """A vector store the `alpha = 0` path must never read."""

    def __getitem__(self, key):
        raise AssertionError("the passage vectors were read: alpha = 0 never needs them")

    def __array__(self, *args, **kwargs):
        raise AssertionError("the passage vectors were read: alpha = 0 never needs them")


def _no_question_vector(text: str) -> np.ndarray:
    raise AssertionError(f"question_vector({text!r}) called: alpha = 0 never needs one")


def _stage(index, vectors, alpha, question):
    return RelevanceHopStage(
        index,
        node_weights(index),
        columns=ColumnwiseEntityHopStage.columns_for(index),
        vectors=vectors,
        question_vector=lambda _text: np.asarray(question, dtype=np.float32),
        alpha=alpha,
    )


# --- alpha = 0: the Phase 9 hop, unchanged ---------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_alpha_0_equals_the_columnwise_phase_9_hop_without_reading_a_vector(seed):
    index = a_random_index(seed)
    weights = node_weights(index)
    reference = ColumnwiseEntityHopStage(index, weights)
    stage = RelevanceHopStage(
        index,
        weights,
        columns=ColumnwiseEntityHopStage.columns_for(index),
        vectors=Untouchable(),  # type: ignore[arg-type]
        question_vector=_no_question_vector,
        alpha=0.0,
    )
    calls = 0
    for start in range(0, len(index.unit_ids) - 12, 5):
        dense = dense_list(index.unit_ids, start)
        expected = reference.expand(dense, 100)
        got = stage.hop("q", dense, 100)
        calls += 1
        assert got.candidates == expected.candidates
        assert got.positives == expected.positives
        assert stage.propose("q", dense, 100) == reference.propose(dense, 100)
        calls += 1
    assert stage.sim_seconds == [0.0] * calls


# --- alpha = 1: similarity alone, the entities only filter -----------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_alpha_1_orders_the_whole_candidate_set_by_similarity(seed):
    index = a_random_index(seed)
    vectors = random_vectors(seed, len(index.unit_ids))
    question = random_vectors(seed + 100, 1)[0]
    reference = ColumnwiseEntityHopStage(index, node_weights(index))
    stage = _stage(index, vectors, 1.0, question)
    row_of = {u: r for r, u in enumerate(index.unit_ids)}
    checked = 0
    for start in range(0, len(index.unit_ids) - 12, 5):
        dense = dense_list(index.unit_ids, start)
        read = {u for u, _s in dense[:10]}
        everything = reference.expand(dense, len(index.unit_ids)).candidates
        if not everything:
            continue
        sims = {c.unit_id: float(vectors[row_of[c.unit_id]] @ question) for c in everything}
        expected = sorted(sims, key=lambda u: (-sims[u], u))[:20]
        got = stage.hop("q", dense, 20)
        assert [c.unit_id for c in got.candidates] == expected
        assert not read & {c.unit_id for c in got.candidates}
        assert got.positives == len(everything)
        low, high = min(sims.values()), max(sims.values())
        for candidate in got.candidates:
            s_hat = 0.0 if high == low else (sims[candidate.unit_id] - low) / (high - low)
            assert candidate.score == pytest.approx(s_hat, abs=1e-6)
        checked += 1
    assert checked > 10


# --- a hand-built question: lifting a candidate, a single candidate --------------------------

FILLERS = [f"f{i}" for i in range(1, 10)]


def _fixture(candidates: dict[str, list[str]]):
    """P1 holds Alpha and Beta; nine fillers complete `read(q)`; `candidates` are the rest."""
    forms_by_unit = {
        "p1": ["Alpha", "Beta"],
        **{unit_id: ["Filler"] for unit_id in FILLERS},
        **candidates,
    }
    records = {
        unit_id: record_from_forms(unit_id, forms, model="fixture", configuration_digest="f")
        for unit_id, forms in forms_by_unit.items()
    }
    unit_ids = list(forms_by_unit)
    return build_node_index(records, unit_ids, extraction_digest="0" * 16)


def _dense() -> list[Hit]:
    return [("p1", 1.0)] + [(u, 0.9 - i / 100) for i, u in enumerate(FILLERS)]


def _vectors(index, similar: set[str]) -> np.ndarray:
    """`[1, 0]` for the units in `similar`, `[0, 1]` for every other one."""
    return np.array(
        [[1.0, 0.0] if u in similar else [0.0, 1.0] for u in index.unit_ids], dtype=np.float32
    )


def test_a_candidate_below_the_phase_9_cut_is_lifted_by_its_similarity():
    # c1..c4 share Alpha and Beta with P1; c_low shares only Beta, so Phase 9 ranks it last
    # and a depth-2 cut leaves it out. Its similarity is the highest, the others' the lowest.
    index = _fixture({**{f"c{i}": ["Alpha", "Beta"] for i in range(1, 5)}, "c_low": ["Beta"]})
    vectors = _vectors(index, {"c_low"})
    phase9 = _stage(index, vectors, 0.0, (1.0, 0.0)).hop("q", _dense(), 2)
    assert [c.unit_id for c in phase9.candidates] == ["c1", "c2"]
    lifted = _stage(index, vectors, 0.75, (1.0, 0.0)).hop("q", _dense(), 2)
    # r_hat(c_low) = 0, s_hat(c_low) = 1: 0.75; every other candidate 0.25.
    assert [(c.unit_id, c.score) for c in lifted.candidates] == [("c_low", 0.75), ("c1", 0.25)]
    assert lifted.positives == phase9.positives == 5


def test_a_single_candidate_scores_zero_on_both_terms():
    index = _fixture({"c1": ["Alpha"], "other": ["Gamma"]})
    vectors = _vectors(index, {"c1"})
    for alpha in (0.25, 0.5, 1.0):
        got = _stage(index, vectors, alpha, (1.0, 0.0)).hop("q", _dense(), 100)
        assert [(c.unit_id, c.score) for c in got.candidates] == [("c1", 0.0)]
    phase9 = _stage(index, vectors, 0.0, (1.0, 0.0)).hop("q", _dense(), 100)
    assert phase9.candidates[0].score > 0.0  # alpha = 0 keeps the raw rarity score


def test_no_candidate_gives_an_empty_list_and_no_similarity_work():
    index = _fixture({"other": ["Gamma"]})
    stage = _stage(index, _vectors(index, set()), 0.5, (1.0, 0.0))
    got = stage.hop("q", _dense(), 100)
    assert got.candidates == () and got.positives == 0
    assert stage.sim_seconds == [0.0]


# --- hop_all, the sim timing and the fused retriever -----------------------------------------


@pytest.mark.parametrize("seed", [1, 2])
def test_hop_all_equals_the_per_alpha_stages_and_times_sim_once_per_call(seed):
    index = a_random_index(seed)
    vectors = random_vectors(seed, len(index.unit_ids))
    question = random_vectors(seed + 100, 1)[0]
    stages = {alpha: _stage(index, vectors, alpha, question) for alpha in ALPHAS}
    everything = _stage(index, vectors, 0.5, question)
    calls = 0
    for start in range(0, len(index.unit_ids) - 12, 9):
        dense = dense_list(index.unit_ids, start)
        got = everything.hop_all("q", dense, 30)
        calls += 1
        assert list(got) == list(ALPHAS)
        for alpha in ALPHAS:
            assert got[alpha] == stages[alpha].hop("q", dense, 30)
    assert len(everything.sim_seconds) == calls
    assert all(seconds >= 0.0 for seconds in everything.sim_seconds)


def test_the_function_refuses_alpha_0_which_belongs_to_the_phase_9_hop():
    index = _fixture({"c1": ["Alpha"]})
    with pytest.raises(ValueError, match="alpha = 0"):
        relevance_hop_columnwise(
            index,
            node_weights(index),
            ColumnwiseEntityHopStage.columns_for(index),
            p1=0,
            read=[0],
            depth=10,
            vectors=_vectors(index, set()),
            question_vector=np.array([1.0, 0.0], dtype=np.float32),
            alphas=(0.0, 0.5),
        )


def test_the_stage_fills_the_third_slot_of_the_query_triple_retriever():
    index = _fixture({"c1": ["Alpha"]})
    stage = _stage(index, _vectors(index, {"c1"}), 0.5, (1.0, 0.0))

    class Dense:
        name = "dense"

        def retrieve(self, query: str, top_k: int) -> list[Hit]:
            return _dense()[:top_k]

    class BM25:
        name = "bm25"

        def retrieve(self, query: str, top_k: int) -> list[Hit]:
            return [("c1", 3.0)]

    system = QueryTripleFusedRetriever(
        Dense(), BM25(), stage, weights={"dense": 0.5, "bm25": 0.3, "seeded-hop": 0.2}
    )
    assert system.gather("q", 100)[2] == [("c1", 0.0)]
    assert phase14.alpha_key(stage.alpha) == "relevance-hop@alpha=0.50"
