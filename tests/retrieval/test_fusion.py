"""T8 and T9: putting two rankings on one scale without letting either one's units decide.

The conceptual score is a sum of products over `X`; the dense score is a cosine; BM25
is neither. They are not the same kind of number, and nothing makes them comparable
except a normalization declared in advance - which is what decision D6 does, instead
of leaving the question to whichever line of code happened to be written first.

Two schemes are measured because one of them costs something. The weighted scheme
buys a degree of freedom and has to earn it against RRF, which has none. What this
file pins down is what each of them promises: that the weighted scheme cannot be
moved by rescaling a component, that RRF reads ranks and nothing else, and that a
unit one component never returned is scored at the declared floor rather than
quietly dropped from the union - the difference between "this component ranked it
last" and "this component never saw it" being exactly what D6 refused to leave
implicit.

The worked example below is small enough to check by hand, and is checked by hand:
the expected fused orders in these tests were computed on paper from the same two
hit lists, not read off a first run.

T9 puts a `Retriever` around those primitives. What it adds is what the two hybrids
of this phase have to share to be comparable at all: one fixed component order, a
weight that belongs to the component it names rather than to a position, and a name
derived from the components instead of chosen by whoever built the object.
"""

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.harness import evaluate_retriever
from concept_embeddings_rag.retrieval.base import Hit, Retriever
from concept_embeddings_rag.retrieval.fusion import (
    ABSENT_FLOOR,
    FITTED_ON,
    MIN_MAX,
    NO_NORMALIZATION,
    FusedRetriever,
    fuse,
    min_max_normalize,
    normalization_for,
    rrf_scores,
    union_of,
    weighted_scores,
)

# --- One worked example, computed on paper ------------------------------------
#
# Component A (the dense side)      Component B (the other signal)
#   a  0.9   rank 1                   b  20.0  rank 1
#   b  0.5   rank 2                   d  10.0  rank 2
#   c  0.1   rank 3                   a   0.0  rank 3
#
# min-max A: lo 0.1, hi 0.9  ->  a 1.0, b 0.5, c 0.0, and d at the floor
# min-max B: lo 0.0, hi 20.0 ->  b 1.0, d 0.5, a 0.0, and c at the floor
# weighted at w = 0.5        ->  b 0.75, a 0.5, d 0.25, c 0.0
# RRF at k = 60              ->  b 1/62 + 1/61, a 1/61 + 1/63, d 1/62, c 1/63

A: list[Hit] = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
B: list[Hit] = [("b", 20.0), ("d", 10.0), ("a", 0.0)]
BOTH = [A, B]

BY_HAND_WEIGHTED = ["b", "a", "d", "c"]
BY_HAND_RRF = ["b", "a", "d", "c"]


def rescaled(hits: list[Hit], factor: float, shift: float = 0.0) -> list[Hit]:
    """The same ranking expressed in different units. Nothing about it is new."""
    return [(unit_id, factor * score + shift) for unit_id, score in hits]


def ranked(hits: list[Hit]) -> list[str]:
    return [unit_id for unit_id, _score in hits]


# --- The union, and what it is for --------------------------------------------


def test_the_union_is_every_unit_any_component_returned():
    assert union_of(BOTH) == ["a", "b", "c", "d"]


def test_the_union_is_ordered_deterministically_rather_than_by_arrival():
    """Ties later break by unit id, so the union must not carry an arbitrary order."""
    assert union_of([B, A]) == union_of(BOTH)


def test_the_union_of_nothing_is_empty_rather_than_an_error():
    assert union_of([[], []]) == []


# --- Per-query min-max --------------------------------------------------------


def test_min_max_puts_the_best_hit_at_one_and_the_worst_at_zero():
    normalized = min_max_normalize(A, union_of(BOTH))

    assert normalized["a"] == pytest.approx(1.0)
    assert normalized["c"] == pytest.approx(0.0)


def test_min_max_places_the_middle_hit_where_its_score_sits_in_the_range():
    normalized = min_max_normalize(A, union_of(BOTH))

    assert normalized["b"] == pytest.approx((0.5 - 0.1) / (0.9 - 0.1))


def test_a_unit_absent_from_a_component_gets_the_declared_floor_not_a_dropped_entry():
    """D6 declares the floor rather than leaving it to an implementation detail."""
    normalized = min_max_normalize(A, union_of(BOTH))

    assert "d" in normalized, "the absent unit was dropped instead of floored"
    assert normalized["d"] == ABSENT_FLOOR


def test_the_declared_floor_is_zero_and_the_code_says_so_rather_than_the_reader():
    assert ABSENT_FLOOR == 0.0


def test_every_unit_of_the_union_is_scored_by_every_component():
    for hits in BOTH:
        assert set(min_max_normalize(hits, union_of(BOTH))) == set(union_of(BOTH))


def test_min_max_is_invariant_to_multiplying_the_scores():
    plain = min_max_normalize(A, union_of(BOTH))
    loud = min_max_normalize(rescaled(A, 1000.0), union_of(BOTH))

    assert plain == pytest.approx(loud)


def test_min_max_is_invariant_to_shifting_the_scores():
    """An affine change of units is not a change of ranking, and must not read as one."""
    plain = min_max_normalize(A, union_of(BOTH))
    shifted = min_max_normalize(rescaled(A, 1.0, shift=17.0), union_of(BOTH))

    assert plain == pytest.approx(shifted)


def test_min_max_is_computed_per_query_and_not_over_some_corpus_wide_range():
    """Two queries with wildly different score ranges both land in [0, 1]."""
    small = min_max_normalize([("a", 0.02), ("b", 0.01)], ["a", "b"])
    large = min_max_normalize([("a", 200.0), ("b", 100.0)], ["a", "b"])

    assert small == large


def test_a_component_that_ranks_everything_equally_contributes_nothing():
    """The degenerate range D6 does not cover, declared here rather than improvised.

    With a flat list there is no "ranked last" to distinguish from "never seen": the
    component expressed no preference at all. Scoring its hits at the floor makes it
    abstain. The alternative - putting them all at 1.0 - would let whatever arbitrary
    order the component used to break its own ties decide the fused ranking.
    """
    flat = min_max_normalize([("a", 3.0), ("b", 3.0), ("c", 3.0)], ["a", "b", "c", "d"])

    assert set(flat.values()) == {ABSENT_FLOOR}


def test_a_component_that_returned_nothing_abstains_rather_than_raising():
    assert set(min_max_normalize([], ["a", "b"]).values()) == {ABSENT_FLOOR}


def test_a_component_cannot_report_a_unit_twice():
    with pytest.raises(ValueError, match="twice"):
        min_max_normalize([("a", 1.0), ("a", 0.5)], ["a"])


# --- RRF, the reference that has nothing to fit -------------------------------


def test_rrf_is_the_declared_reciprocal_rank_sum_at_the_standard_constant():
    k = config.RRF_K
    scores = rrf_scores(BOTH)

    assert scores["a"] == pytest.approx(1 / (k + 1) + 1 / (k + 3))
    assert scores["b"] == pytest.approx(1 / (k + 2) + 1 / (k + 1))
    assert scores["c"] == pytest.approx(1 / (k + 3))
    assert scores["d"] == pytest.approx(1 / (k + 2))


def test_the_rrf_constant_is_the_declared_one_and_is_not_fitted_anywhere():
    """Fitting it would make the parameter-free reference no longer parameter-free."""
    assert config.RRF_K == 60


def test_rrf_reads_ranks_alone_so_any_order_preserving_rescaling_leaves_it_alone():
    """The property that makes RRF the reference: it cannot be moved by units."""
    exotic = [("a", 1e9), ("b", 2.0), ("c", 1.999999)]

    assert rrf_scores([exotic, B]) == pytest.approx(rrf_scores(BOTH))


def test_a_unit_only_one_component_found_still_gets_that_components_term():
    """Absence contributes no term; it does not zero the unit out of the fusion."""
    assert rrf_scores(BOTH)["c"] > 0.0


def test_rrf_ranks_the_worked_example_the_way_it_was_computed_on_paper():
    assert ranked(fuse(BOTH, scheme="rrf", top_k=4)) == BY_HAND_RRF


def test_rrf_takes_no_weight_and_refuses_one_rather_than_ignoring_it():
    with pytest.raises(ValueError, match="no weight"):
        fuse(BOTH, scheme="rrf", weights=(0.5, 0.5), top_k=4)


def test_rrf_declares_that_it_normalizes_nothing():
    assert normalization_for("rrf") == NO_NORMALIZATION


# --- The weighted scheme ------------------------------------------------------


def test_the_weighted_scheme_is_the_declared_convex_combination():
    scores = weighted_scores(BOTH, (0.5, 0.5))

    assert scores["b"] == pytest.approx(0.75)
    assert scores["a"] == pytest.approx(0.5)
    assert scores["d"] == pytest.approx(0.25)
    assert scores["c"] == pytest.approx(0.0)


def test_the_weighted_scheme_ranks_the_worked_example_as_computed_on_paper():
    assert ranked(fuse(BOTH, scheme="weighted", weights=(0.5, 0.5), top_k=4)) == BY_HAND_WEIGHTED


def test_weight_one_is_the_first_component_alone():
    """An optimum at the endpoint says the second signal contributes nothing.

    D6 keeps the endpoints in the grid precisely so that finding can be read off the
    curve rather than inferred from its absence, which only works if the endpoint
    really is the component on its own.
    """
    fused = fuse(BOTH, scheme="weighted", weights=(1.0, 0.0), top_k=4)

    assert ranked(fused)[:3] == ranked(A)


def test_weight_zero_is_the_second_component_alone():
    fused = fuse(BOTH, scheme="weighted", weights=(0.0, 1.0), top_k=4)

    assert ranked(fused)[:3] == ranked(B)


def test_the_declared_grid_is_eleven_points_and_includes_both_endpoints():
    grid = config.FUSION_WEIGHT_GRID

    assert len(grid) == 11
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert np.allclose(np.diff(grid), 0.1)


def test_the_weighted_scheme_declares_the_normalization_it_used():
    assert normalization_for("weighted") == MIN_MAX


def test_weights_that_do_not_sum_to_one_are_refused():
    """`s = w * dense + (1 - w) * other` is a convex combination or it is not D6's."""
    with pytest.raises(ValueError, match="sum to 1"):
        weighted_scores(BOTH, (0.5, 0.9))


def test_a_negative_weight_is_refused():
    with pytest.raises(ValueError, match="negative"):
        weighted_scores(BOTH, (1.5, -0.5))


def test_there_must_be_one_weight_per_component():
    with pytest.raises(ValueError, match="one weight per component"):
        weighted_scores(BOTH, (1.0,))


def test_the_weighted_scheme_refuses_to_run_without_weights():
    with pytest.raises(ValueError, match="weights"):
        fuse(BOTH, scheme="weighted", top_k=4)


# --- The headline criterion of T8 ---------------------------------------------


@pytest.mark.parametrize("scheme", config.FUSION_SCHEMES)
@pytest.mark.parametrize("factor", [1000.0, 0.001])
def test_rescaling_one_components_raw_scores_leaves_the_fused_ranking_unchanged(
    scheme: str, factor: float
):
    """Neither signal gets to win on the size of its numbers.

    This is the whole reason a normalization is declared. Without it, `X @ c` and a
    cosine would be added together and the sum would be decided by whichever of them
    happens to live on the larger scale, which is a property of the arithmetic and
    not of the retrieval.
    """
    weights = (0.5, 0.5) if scheme == "weighted" else None

    plain = fuse(BOTH, scheme=scheme, weights=weights, top_k=4)
    loud = fuse([rescaled(A, factor), B], scheme=scheme, weights=weights, top_k=4)

    assert ranked(plain) == ranked(loud)


@pytest.mark.parametrize("scheme", config.FUSION_SCHEMES)
def test_rescaling_the_other_component_is_equally_powerless(scheme: str):
    weights = (0.5, 0.5) if scheme == "weighted" else None

    plain = fuse(BOTH, scheme=scheme, weights=weights, top_k=4)
    loud = fuse([A, rescaled(B, 1e6)], scheme=scheme, weights=weights, top_k=4)

    assert ranked(plain) == ranked(loud)


# --- What `fuse` returns ------------------------------------------------------


def test_the_fused_list_is_the_best_top_k_of_the_union():
    assert len(fuse(BOTH, scheme="rrf", top_k=2)) == 2
    assert ranked(fuse(BOTH, scheme="rrf", top_k=2)) == BY_HAND_RRF[:2]


def test_asking_for_more_than_the_union_holds_returns_the_union():
    assert len(fuse(BOTH, scheme="rrf", top_k=500)) == 4


def test_no_unit_of_the_union_is_dropped_when_the_budget_allows_it():
    assert set(ranked(fuse(BOTH, scheme="rrf", top_k=4))) == set(union_of(BOTH))


def test_the_fused_scores_come_back_beside_the_units_that_earned_them():
    fused = dict(fuse(BOTH, scheme="weighted", weights=(0.5, 0.5), top_k=4))

    assert fused == pytest.approx(weighted_scores(BOTH, (0.5, 0.5)))


def test_ties_are_broken_by_unit_id_rather_than_left_to_dictionary_order():
    """Same criterion as `dense.py` and `bm25.py`: a ranking must not drift by run."""
    tied = [[("z", 1.0), ("y", 1.0), ("x", 1.0)]]

    assert ranked(fuse(tied, scheme="rrf", top_k=3)) == ["x", "y", "z"]


def test_fusing_the_same_lists_twice_gives_the_same_list():
    first = fuse(BOTH, scheme="weighted", weights=(0.3, 0.7), top_k=4)
    second = fuse(BOTH, scheme="weighted", weights=(0.3, 0.7), top_k=4)

    assert first == second


def test_the_order_the_components_arrive_in_does_not_change_a_symmetric_fusion():
    forward = fuse(BOTH, scheme="rrf", top_k=4)
    backward = fuse([B, A], scheme="rrf", top_k=4)

    assert dict(forward) == pytest.approx(dict(backward))


def test_an_unknown_scheme_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="scheme"):
        fuse(BOTH, scheme="borda", top_k=4)


def test_an_unknown_scheme_has_no_normalization_to_report():
    with pytest.raises(ValueError, match="scheme"):
        normalization_for("borda")


def test_a_top_k_of_zero_or_less_is_refused():
    with pytest.raises(ValueError, match="top_k"):
        fuse(BOTH, scheme="rrf", top_k=0)


def test_fusing_needs_at_least_two_components_to_be_a_fusion():
    with pytest.raises(ValueError, match="two components"):
        weighted_scores([A], (1.0,))


# --- T9: the retriever that fuses two retrievers ------------------------------
#
# The phase builds two hybrids: dense+conceptual (System B) and dense+bm25 (the
# HU-5 control). They are the same object with a different second signal, which is
# the entire point of the control - if the control were fused by different code, a
# difference between them could be a difference of fusion rather than of signal.
#
# So the order of the components cannot be whatever order the caller passed, the
# weights cannot be positional, and the name cannot be a label the caller chooses:
# all three are derived from the components themselves, and these tests are what
# says so.


class StubRetriever:
    """A retriever with a fixed answer, recording what it was asked for."""

    def __init__(self, name: str, hits: list[Hit]) -> None:
        self.name = name
        self.hits = hits
        self.asked_for: list[int] = []

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        self.asked_for.append(top_k)
        return self.hits[:top_k]


def a_dense() -> StubRetriever:
    return StubRetriever("dense", A)


def a_conceptual() -> StubRetriever:
    return StubRetriever("conceptual", B)


def a_hybrid(scheme: str = "rrf", weights: dict[str, float] | None = None) -> FusedRetriever:
    return FusedRetriever([a_dense(), a_conceptual()], scheme=scheme, weights=weights)


def a_question() -> Question:
    return Question(
        qid="q1",
        question="a question",
        answer="an answer",
        gold_unit_ids=("g1", "g2"),
        supporting_facts=(("A", 0), ("B", 1)),
        split="dev",
    )


def a_run_config() -> dict:
    return {
        "model": "fake",
        "revision": "v0",
        "unit_set_hash": "abc123",
        "seed": 42,
        "tokenizer": "fake-tokenizer",
        "code_version": "0.1.0",
        "top_k": 4,
    }


# --- The harness cannot tell it apart from dense or BM25 ----------------------


def test_the_fused_retriever_implements_the_unchanged_retriever_protocol():
    assert isinstance(a_hybrid(), Retriever)


def test_the_harness_runs_the_hybrid_without_knowing_what_it_is():
    """The measurement path is the Phase 1 one, unmodified, or the numbers are not comparable."""
    hybrid = FusedRetriever(
        [
            StubRetriever("dense", [("x", 0.9), ("g1", 0.5)]),
            StubRetriever("conceptual", [("g2", 3.0), ("y", 1.0)]),
        ],
        scheme="rrf",
    )

    result = evaluate_retriever(
        hybrid,
        questions=[a_question()],
        token_counts={"g1": 100, "g2": 100, "x": 100, "y": 100},
        budgets=(400,),
        ks=(4,),
        top_k=4,
        config=a_run_config(),
    )

    assert result.system == "hybrid-conceptual"
    assert result.metrics["recall_at_4"] == 1.0


# --- What the hybrid records about itself -------------------------------------


def test_a_hybrid_is_named_after_the_signal_it_adds_to_dense():
    """The name is derived, not passed: no result can claim System B and have run the control."""
    assert a_hybrid().name == "hybrid-conceptual"
    assert FusedRetriever([a_dense(), StubRetriever("bm25", B)], scheme="rrf").name == "hybrid-bm25"


def test_describe_records_every_field_of_the_spec_contract():
    hybrid = a_hybrid(scheme="weighted", weights={"dense": 0.4, "conceptual": 0.6})

    assert hybrid.describe() == {
        "name": "hybrid-conceptual",
        "components": ["dense", "conceptual"],
        "scheme": "weighted",
        "normalization": MIN_MAX,
        "weights": {"dense": 0.4, "conceptual": 0.6},
        "fitted_on": "dev",
    }


def test_the_parameter_free_scheme_records_no_weight_and_no_normalization():
    described = a_hybrid(scheme="rrf").describe()

    assert described["weights"] is None
    assert described["normalization"] == NO_NORMALIZATION


def test_what_the_weights_were_fitted_on_is_recorded_rather_than_assumed():
    assert FITTED_ON == "dev"
    assert a_hybrid().describe()["fitted_on"] == FITTED_ON


# --- The fixed component order ------------------------------------------------


def test_the_component_order_is_recorded_and_starts_at_the_dense_signal():
    """D6 writes the weighted scheme as `w * dense + (1 - w) * other`; the order is that one."""
    assert a_hybrid().describe()["components"] == ["dense", "conceptual"]
    assert FusedRetriever([StubRetriever("bm25", B), a_dense()], scheme="rrf").describe()[
        "components"
    ] == ["dense", "bm25"]


@pytest.mark.parametrize("scheme", config.FUSION_SCHEMES)
def test_swapping_the_input_order_changes_neither_the_record_nor_the_ranking(scheme: str):
    weights = {"dense": 0.3, "conceptual": 0.7} if scheme == "weighted" else None

    forward = FusedRetriever([a_dense(), a_conceptual()], scheme=scheme, weights=weights)
    backward = FusedRetriever([a_conceptual(), a_dense()], scheme=scheme, weights=weights)

    assert forward.describe() == backward.describe()
    assert forward.retrieve("a query", top_k=4) == backward.retrieve("a query", top_k=4)


def test_a_weight_belongs_to_the_component_it_names_and_not_to_a_position():
    """All the weight on dense returns dense's own ranking, whichever order it arrived in."""
    backward = FusedRetriever(
        [a_conceptual(), a_dense()],
        scheme="weighted",
        weights={"dense": 1.0, "conceptual": 0.0},
    )

    assert ranked(backward.retrieve("a query", top_k=3)) == ["a", "b", "c"]


# --- What it returns ----------------------------------------------------------


def test_each_component_is_asked_for_the_top_k_the_hybrid_was_asked_for():
    dense, conceptual = a_dense(), a_conceptual()

    FusedRetriever([dense, conceptual], scheme="rrf").retrieve("a query", top_k=2)

    assert dense.asked_for == [2]
    assert conceptual.asked_for == [2]


def test_the_hybrid_returns_the_best_top_k_of_the_union():
    assert ranked(a_hybrid().retrieve("a query", top_k=4)) == BY_HAND_RRF
    assert ranked(a_hybrid().retrieve("a query", top_k=2)) == BY_HAND_RRF[:2]


def test_the_hybrid_returns_what_fusing_the_two_lists_by_hand_returns():
    weights = {"dense": 0.5, "conceptual": 0.5}
    hybrid = a_hybrid(scheme="weighted", weights=weights)

    assert hybrid.retrieve("a query", top_k=4) == fuse(
        BOTH, scheme="weighted", weights=(0.5, 0.5), top_k=4
    )


def test_ties_inside_the_hybrid_are_broken_by_unit_id():
    flat = [
        StubRetriever("dense", [("z", 1.0), ("y", 1.0), ("x", 1.0)]),
        StubRetriever("conceptual", [("z", 5.0), ("y", 5.0), ("x", 5.0)]),
    ]

    assert ranked(FusedRetriever(flat, scheme="rrf").retrieve("a query", top_k=3)) == [
        "x",
        "y",
        "z",
    ]


def test_the_same_query_twice_gives_the_same_ranking():
    hybrid = a_hybrid(scheme="weighted", weights={"dense": 0.4, "conceptual": 0.6})

    assert hybrid.retrieve("a query", top_k=4) == hybrid.retrieve("a query", top_k=4)


# --- What it refuses ----------------------------------------------------------


def test_a_hybrid_of_one_retriever_is_refused():
    with pytest.raises(ValueError, match="two components"):
        FusedRetriever([a_dense()], scheme="rrf")


def test_the_same_retriever_twice_is_not_a_fusion():
    with pytest.raises(ValueError, match="different"):
        FusedRetriever([a_dense(), a_dense()], scheme="rrf")


def test_a_hybrid_without_the_dense_component_is_refused():
    """HU-4 and HU-5 both fuse dense with a second signal; there is no third hybrid."""
    with pytest.raises(ValueError, match="dense"):
        FusedRetriever([StubRetriever("bm25", A), a_conceptual()], scheme="rrf")


def test_a_second_signal_the_phase_never_declared_is_refused():
    with pytest.raises(ValueError, match="second signal"):
        FusedRetriever([a_dense(), StubRetriever("random", B)], scheme="rrf")


def test_an_unknown_scheme_is_refused_when_the_hybrid_is_built():
    with pytest.raises(ValueError, match="scheme"):
        FusedRetriever([a_dense(), a_conceptual()], scheme="borda")


def test_the_weighted_scheme_is_refused_without_the_weights_it_needs():
    """A hybrid that only fails on its first query would fail in the middle of a sweep."""
    with pytest.raises(ValueError, match="weight"):
        FusedRetriever([a_dense(), a_conceptual()], scheme="weighted")


def test_the_parameter_free_scheme_refuses_a_weight_rather_than_ignoring_it():
    with pytest.raises(ValueError, match="rrf"):
        FusedRetriever(
            [a_dense(), a_conceptual()],
            scheme="rrf",
            weights={"dense": 0.5, "conceptual": 0.5},
        )


def test_a_weight_naming_a_component_that_is_not_in_the_fusion_is_refused():
    with pytest.raises(ValueError, match="bm25"):
        FusedRetriever(
            [a_dense(), a_conceptual()],
            scheme="weighted",
            weights={"dense": 0.5, "bm25": 0.5},
        )


def test_a_component_left_without_a_weight_is_refused():
    with pytest.raises(ValueError, match="weight"):
        FusedRetriever([a_dense(), a_conceptual()], scheme="weighted", weights={"dense": 1.0})


def test_weights_that_are_not_a_convex_combination_are_refused_at_construction():
    with pytest.raises(ValueError, match="sum to 1"):
        FusedRetriever(
            [a_dense(), a_conceptual()],
            scheme="weighted",
            weights={"dense": 0.5, "conceptual": 0.9},
        )


def test_a_top_k_of_zero_is_refused_before_any_component_is_asked():
    dense, conceptual = a_dense(), a_conceptual()
    hybrid = FusedRetriever([dense, conceptual], scheme="rrf")

    with pytest.raises(ValueError, match="top_k"):
        hybrid.retrieve("a query", top_k=0)

    assert dense.asked_for == []
    assert conceptual.asked_for == []
