"""Phase 6, T4: `FusedRetriever` admits a second-stage component (D4).

An entity hop conditioned on dense's list is not a `Retriever`: it never reads the question.
The generalized hybrid asks dense once, hands that very list to the stage, and fuses the two
lists with the unchanged primitives. What that must preserve, and what this module checks:

- the harness still sees a `Retriever` named by the rule, `hybrid-` plus the second name;
- dense is asked exactly once per query, and the stage receives what dense returned;
- only a declared stage name with a `propose` method is a stage, and only a declared signal
  name with a `retrieve` method is a signal: every other pairing is refused at construction;
- the declared consequences of the inherited fusion on an empty or flat stage list hold as
  the spec writes them, and nothing compensates them.
"""

from collections.abc import Sequence

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.retrieval.base import Hit, Retriever
from concept_embeddings_rag.retrieval.fusion import (
    ABSENT_FLOOR,
    SECOND_SIGNALS,
    SECOND_STAGES,
    FusedRetriever,
    SecondStage,
    rrf_scores,
)

DENSE_HITS: list[Hit] = [("d1", 0.9), ("d2", 0.7), ("d3", 0.5), ("d4", 0.2)]


class CountingDense:
    name = "dense"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        self.calls.append((query, top_k))
        return list(self.hits[:top_k])


class RecordingStage:
    name = "entity-hop"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.received: list[list[Hit]] = []

    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
        self.received.append(list(first))
        return list(self.hits[:top_k])


class NamedRetriever:
    def __init__(self, name: str, hits: list[Hit]) -> None:
        self.name = name
        self.hits = hits

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return list(self.hits[:top_k])


class NamedStage:
    def __init__(self, name: str) -> None:
        self.name = name

    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
        return []


def a_hybrid(stage_hits: list[Hit], scheme: str = "rrf", w: float | None = None):
    dense = CountingDense(DENSE_HITS)
    stage = RecordingStage(stage_hits)
    weights = None if w is None else {"dense": w, "entity-hop": 1.0 - w}
    return FusedRetriever([dense, stage], scheme=scheme, weights=weights), dense, stage


# --- The declared slot ---------------------------------------------------------------


def test_the_stage_names_are_declared_beside_the_untouched_signal_names():
    assert SECOND_STAGES == (config.ENTITY_HOP_NAME,) == ("entity-hop",)
    assert SECOND_SIGNALS == ("conceptual", "bm25")


def test_a_staged_hybrid_is_named_by_the_rule_and_is_a_retriever():
    hybrid, _, stage = a_hybrid([("e1", 2.0)])
    assert hybrid.name == "hybrid-entity-hop"
    assert hybrid.components == ["dense", "entity-hop"]
    assert isinstance(hybrid, Retriever)
    assert isinstance(stage, SecondStage)


def test_the_stage_may_arrive_first_and_is_still_fused_second():
    dense, stage = CountingDense(DENSE_HITS), RecordingStage([("e1", 1.0)])
    hybrid = FusedRetriever([stage, dense], scheme="rrf")
    assert hybrid.components == ["dense", "entity-hop"]


def test_dense_is_asked_once_per_query_and_the_stage_receives_that_list():
    hybrid, dense, stage = a_hybrid([("e1", 3.0), ("d3", 1.0)])

    hybrid.retrieve("first question", top_k=4)
    hybrid.retrieve("second question", top_k=3)

    assert dense.calls == [("first question", 4), ("second question", 3)]
    assert stage.received == [DENSE_HITS[:4], DENSE_HITS[:3]]


def test_the_stage_is_asked_for_the_top_k_the_hybrid_was_asked_for():
    asked: list[int] = []

    class DepthStage(RecordingStage):
        def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
            asked.append(top_k)
            return super().propose(first, top_k)

    hybrid = FusedRetriever([CountingDense(DENSE_HITS), DepthStage([])], scheme="rrf")
    hybrid.retrieve("q", top_k=config.EVALUATION_TOP_K)
    assert asked == [config.EVALUATION_TOP_K]


def test_a_stage_that_mutates_its_input_cannot_change_what_dense_contributes():
    class MutatingStage(RecordingStage):
        def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
            if isinstance(first, list):
                first.clear()
            return []

    hybrid = FusedRetriever([CountingDense(DENSE_HITS), MutatingStage([])], scheme="rrf")
    assert [unit for unit, _ in hybrid.retrieve("q", top_k=4)] == ["d1", "d2", "d3", "d4"]


# --- Refusals at construction ------------------------------------------------------------


def test_a_stage_whose_name_is_not_declared_is_refused():
    with pytest.raises(ValueError, match="unknown second signal"):
        FusedRetriever([CountingDense(DENSE_HITS), NamedStage("concept-hop")], scheme="rrf")


def test_a_retriever_under_a_stage_name_is_refused():
    with pytest.raises(ValueError, match="unknown second signal"):
        FusedRetriever([CountingDense(DENSE_HITS), NamedRetriever("entity-hop", [])], scheme="rrf")


def test_a_stage_under_a_signal_name_is_refused():
    with pytest.raises(ValueError, match="unknown second signal"):
        FusedRetriever([CountingDense(DENSE_HITS), NamedStage("bm25")], scheme="rrf")


def test_an_object_that_is_both_a_stage_and_a_retriever_is_refused_under_either_name():
    class Both(NamedRetriever):
        def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
            return []

    for name in ("bm25", "entity-hop"):
        with pytest.raises(ValueError, match="unknown second signal"):
            FusedRetriever([CountingDense(DENSE_HITS), Both(name, [])], scheme="rrf")


def test_dense_itself_must_be_a_retriever():
    with pytest.raises(ValueError, match="dense"):
        FusedRetriever([NamedStage("dense"), RecordingStage([])], scheme="rrf")


# --- Declared consequences, not repaired --------------------------------------------------


@pytest.mark.parametrize("w", [weight for weight in config.FUSION_WEIGHT_GRID if weight > 0.0])
def test_an_empty_stage_list_leaves_the_weighted_ranking_equal_to_dense(w: float):
    hybrid, _, _ = a_hybrid([], scheme="weighted", w=w)
    ranking = [unit for unit, _ in hybrid.retrieve("q", top_k=4)]
    assert ranking == [unit for unit, _ in DENSE_HITS]


def test_an_empty_stage_list_adds_no_rrf_term():
    hybrid, _, _ = a_hybrid([], scheme="rrf")
    fused = hybrid.retrieve("q", top_k=4)
    alone = rrf_scores([DENSE_HITS])
    assert fused == sorted(alone.items(), key=lambda entry: (-entry[1], entry[0]))


@pytest.mark.parametrize("w", [0.1, 0.5, 0.9])
def test_a_flat_stage_list_abstains_under_weighted(w: float):
    """Min-max has no range: every stage unit gets the floor, and dense orders the rest."""
    hybrid, _, _ = a_hybrid([("e2", 3.0), ("e1", 3.0), ("d4", 3.0)], scheme="weighted", w=w)
    fused = dict(hybrid.retrieve("q", top_k=10))

    assert fused["e1"] == fused["e2"] == ABSENT_FLOOR
    positive = [unit for unit, score in hybrid.retrieve("q", top_k=10) if score > 0.0]
    assert positive == ["d1", "d2", "d3"]
    # d4 is dense's worst hit, so min-max also puts it at the floor; the tie breaks by id.
    assert [unit for unit, _ in hybrid.retrieve("q", top_k=10)][3:] == ["d4", "e1", "e2"]


def test_a_single_candidate_abstains_under_weighted():
    hybrid, _, _ = a_hybrid([("e1", 5.0)], scheme="weighted", w=0.5)
    assert dict(hybrid.retrieve("q", top_k=10))["e1"] == ABSENT_FLOOR


def test_a_flat_stage_list_still_contributes_by_rank_under_rrf_with_ties_by_unit_id():
    hybrid, _, _ = a_hybrid([("e2", 3.0), ("e1", 3.0)], scheme="rrf")
    fused = dict(hybrid.retrieve("q", top_k=10))

    assert fused["e1"] == pytest.approx(1.0 / (config.RRF_K + 1))
    assert fused["e2"] == pytest.approx(1.0 / (config.RRF_K + 2))
    assert fused["e1"] > fused["e2"]


def test_the_bottom_tier_of_the_stage_list_gets_the_floor_under_weighted():
    hybrid, _, _ = a_hybrid([("e1", 4.0), ("e2", 1.0), ("e3", 1.0)], scheme="weighted", w=0.5)
    fused = dict(hybrid.retrieve("q", top_k=10))
    assert fused["e1"] == pytest.approx(0.5)
    assert fused["e2"] == fused["e3"] == ABSENT_FLOOR


def test_the_staged_hybrid_describes_itself_by_the_same_contract():
    hybrid, _, _ = a_hybrid([("e1", 1.0)], scheme="weighted", w=0.3)
    assert hybrid.describe() == {
        "name": "hybrid-entity-hop",
        "components": ["dense", "entity-hop"],
        "scheme": "weighted",
        "normalization": "min_max",
        "weights": {"dense": 0.3, "entity-hop": 0.7},
        "fitted_on": "dev",
    }
