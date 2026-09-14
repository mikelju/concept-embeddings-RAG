"""T13: fitting the fusion on dev, and what the curve has to be able to say afterwards.

HU-4 does not ask for a fused system. It asks for one whose weight was fitted and
whose fitting can be read back, which is three separate things: the declared grid
measured whole, the parameter-free scheme measured beside it, and both figures
surviving whatever the outcome. A weighted scheme that does not beat RRF has bought a
degree of freedom it did not need, and that is a finding this phase reports - which it
cannot be if the losing number is dropped on the way to the report.

The same function fits System B and the HU-5 control (decision D9). Handing the control
a shorter grid or a weaker reference is the easiest way there is to manufacture this
phase's headline, so what the tests below check is that the two come out of one code
path rather than out of two that look alike.

Nothing here measures anything real. Each component answers a fixed list per question,
every unit costs ten tokens, and the budget admits two of them - one, in the second
example - so the top of the fused ranking decides the figure and every number in this
file was computed on paper before it was measured. The curve included.
"""

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import selection
from concept_embeddings_rag.evaluation.harness import evaluate_retriever
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    FusionFit,
    SelectionError,
    fit_fusion_weight,
)
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.fusion import RRF, WEIGHTED

# --- The first worked example: two questions, one gold each -------------------
#
# Each component finds its own question's gold and buries the other's. Min-max over
# each component's own returned range (decision D6), then `w * dense + (1 - w) * other`:
#
#   question A, gold g1        dense norm      conceptual norm     fused
#     g1                          1.0              0.0             w
#     x                           0.889            1.0             1 - 0.111 w
#     y                           0.0              0.5             0.5 - 0.5 w
#   g1 enters the top two once it passes y, at w > 1/3: recall 1 from w = 0.4 up.
#
#   question B, gold g2        dense norm      conceptual norm     fused
#     x                           1.0              0.5             0.5 + 0.5 w
#     y                           0.889            0.0             0.889 w
#     g2                          0.0              1.0             1 - w
#   g2 holds a top-two place until y passes it, at w > 0.529: recall 1 up to w = 0.5.
#
# So the two endpoints score 0.5 - each component alone answers one question of the two -
# and only the middle of the grid answers both. An optimum that is neither endpoint is
# the case worth pinning: it is the one where the fitted weight is doing the work.

TOKENS = {"g1": 10, "g2": 10, "x": 10, "y": 10, "a": 10, "b": 10}
TWO_UNITS = 20
ONE_UNIT = 10

DENSE_HITS: dict[str, list[Hit]] = {
    "A": [("g1", 1.0), ("x", 0.9), ("y", 0.1)],
    "B": [("x", 1.0), ("y", 0.9), ("g2", 0.1)],
}
OTHER_HITS: dict[str, list[Hit]] = {
    "A": [("x", 9.0), ("y", 5.0), ("g1", 1.0)],
    "B": [("g2", 9.0), ("x", 5.0), ("y", 1.0)],
}

CURVE_BY_HAND = {
    0.0: 0.5,
    0.1: 0.5,
    0.2: 0.5,
    0.3: 0.5,
    0.4: 1.0,
    0.5: 1.0,
    0.6: 0.5,
    0.7: 0.5,
    0.8: 0.5,
    0.9: 0.5,
    1.0: 0.5,
}
# Ranks alone answer both questions here: g1 is dense's first and the other signal's
# third, which beats a unit ranked second by both. RRF reaches what the best weight
# reaches, and matching is not beating.
RRF_BY_HAND = 1.0

# --- The second worked example: one question where ranks mislead --------------
#
#   question C, gold g1        dense norm      conceptual norm     fused
#     g1                          1.0              0.0             w
#     a                           0.111            1.0             1 - 0.889 w
#     b                           0.0              0.875           0.875 - 0.875 w
#
# The budget admits one unit. `a` is ranked second by dense and first by the other
# signal, so RRF puts it on top and the question is lost; the weighted scheme recovers
# it once enough weight sits on dense, at w > 0.529. Weighted 1.0 against RRF 0.0.

MISLEADING_DENSE: dict[str, list[Hit]] = {"C": [("g1", 1.0), ("a", 0.2), ("b", 0.1)]}
MISLEADING_OTHER: dict[str, list[Hit]] = {"C": [("a", 9.0), ("b", 8.0), ("g1", 1.0)]}


class StubRetriever:
    """A retriever with one fixed answer per question, so the curve is checkable by hand."""

    def __init__(self, name: str, hits: dict[str, list[Hit]]) -> None:
        self.name = name
        self.hits = hits

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return self.hits[query][:top_k]


def a_question(qid: str, text: str, gold: tuple[str, ...], split: str = DEV_SPLIT) -> Question:
    return Question(
        qid=qid,
        question=text,
        answer="-",
        gold_unit_ids=gold,
        supporting_facts=(("A", 0),),
        split=split,
    )


def the_questions(split: str = DEV_SPLIT) -> list[Question]:
    return [
        a_question("q1", "A", ("g1",), split),
        a_question("q2", "B", ("g2",), split),
    ]


def a_config(**overrides) -> dict:
    base = {
        "model": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c",
        "unit_set_hash": "101f564fdcca620c",
        "seed": 42,
        "tokenizer": "cl100k_base",
        "code_version": "0.1.0",
        "top_k": 4,
    }
    base.update(overrides)
    return base


def fit(components=None, questions=None, budget: int = TWO_UNITS, **kwargs) -> FusionFit:
    """The fitting as the phase calls it, with the budget the worked example uses."""
    return fit_fusion_weight(
        components
        if components is not None
        else [StubRetriever("dense", DENSE_HITS), StubRetriever("conceptual", OTHER_HITS)],
        questions=questions if questions is not None else the_questions(),
        token_counts=TOKENS,
        base_config=a_config(),
        budgets=(budget,),
        ks=(2,),
        budget=budget,
        **kwargs,
    )


def a_misleading_fit(second_signal: str = "conceptual") -> FusionFit:
    return fit(
        components=[
            StubRetriever("dense", MISLEADING_DENSE),
            StubRetriever(second_signal, MISLEADING_OTHER),
        ],
        questions=[a_question("q1", "C", ("g1",))],
        budget=ONE_UNIT,
    )


# --- The grid, whole and declared ---------------------------------------------


def test_the_grid_fitted_over_is_the_one_declared_in_the_config():
    """D6 declares eleven points before anything is measured; nothing here shortens them."""
    assert fit().grid == config.FUSION_WEIGHT_GRID


def test_the_curve_carries_a_figure_for_every_point_of_the_grid():
    fitted = fit()

    assert sorted(fitted.curve) == sorted(fitted.grid)
    assert len(fitted.curve) == 11


def test_the_curve_is_the_one_computed_on_paper():
    """Not read off a first run: the fused scores above were worked out by hand."""
    assert fit().curve == pytest.approx(CURVE_BY_HAND)


def test_the_fit_records_how_many_dev_questions_stand_behind_the_curve():
    assert fit().n_questions == 2


def test_the_fit_records_the_metric_and_budget_the_curve_was_read_at():
    fitted = fit()

    assert fitted.metric == config.SELECTION_METRIC
    assert fitted.budget == TWO_UNITS


# --- The selected weight ------------------------------------------------------


def test_the_selected_weight_is_the_best_point_of_the_curve():
    fitted = fit()

    assert fitted.best_weight == 0.5
    assert fitted.best_weighted_score == pytest.approx(1.0)


def test_the_optimum_can_sit_between_the_endpoints_which_is_why_it_is_fitted():
    """Each component alone answers one question of the two; only a mixture answers both."""
    curve = fit().curve

    assert curve[0.0] < curve[0.5] > curve[1.0]


def test_the_selected_weight_always_belongs_to_the_curve_it_was_read_from():
    """Derived rather than stored: a field could disagree with the measurements."""
    fitted = fit()

    assert fitted.curve[fitted.best_weight] == fitted.best_weighted_score


def test_a_tie_on_the_curve_is_broken_toward_the_dense_side():
    """0.4 and 0.5 both reach 1.0 here, and the larger weight on dense takes it."""
    fitted = fit()

    assert fitted.curve[0.4] == fitted.curve[0.5]
    assert fitted.best_weight == 0.5


def test_a_curve_that_fusing_never_moved_selects_dense_alone():
    """A flat curve is a fusion that changed nothing, and must not read as a win.

    Breaking the tie toward the second signal would let "nothing happened" come back
    as evidence for the system under test. The rule is the conservative direction, and
    it is the same rule for the HU-5 control.
    """
    flat = FusionFit(
        components=("dense", "conceptual"),
        metric=config.SELECTION_METRIC,
        budget=config.SELECTION_BUDGET,
        n_questions=600,
        grid=config.FUSION_WEIGHT_GRID,
        curve=dict.fromkeys(config.FUSION_WEIGHT_GRID, 0.61),
        rrf_score=0.61,
    )

    assert flat.best_weight == 1.0


# --- The endpoints, and what they let the curve say ---------------------------


def test_weight_one_is_the_dense_component_measured_on_its_own():
    """HU-4 keeps the endpoint so "the second signal contributes nothing" is readable."""
    dense_alone = evaluate_retriever(
        StubRetriever("dense", MISLEADING_DENSE),
        questions=[a_question("q1", "C", ("g1",))],
        token_counts=TOKENS,
        budgets=(ONE_UNIT,),
        ks=(2,),
        top_k=4,
        config=a_config(),
        split=DEV_SPLIT,
    )

    assert a_misleading_fit().curve[1.0] == pytest.approx(
        dense_alone.metrics[f"budget_{ONE_UNIT}"][config.SELECTION_METRIC]
    )


def test_weight_zero_is_the_second_signal_measured_on_its_own():
    other_alone = evaluate_retriever(
        StubRetriever("conceptual", MISLEADING_OTHER),
        questions=[a_question("q1", "C", ("g1",))],
        token_counts=TOKENS,
        budgets=(ONE_UNIT,),
        ks=(2,),
        top_k=4,
        config=a_config(),
        split=DEV_SPLIT,
    )

    assert a_misleading_fit().curve[0.0] == pytest.approx(
        other_alone.metrics[f"budget_{ONE_UNIT}"][config.SELECTION_METRIC]
    )


def test_the_two_endpoints_are_different_measurements_and_not_one_repeated():
    misleading = a_misleading_fit()

    assert misleading.curve[1.0] == pytest.approx(1.0)
    assert misleading.curve[0.0] == pytest.approx(0.0)


# --- RRF, measured beside the curve rather than instead of it -----------------


def test_the_parameter_free_scheme_is_measured_on_the_same_questions():
    assert fit().rrf_score == pytest.approx(RRF_BY_HAND)


def test_a_weighted_scheme_that_only_matches_rrf_has_not_earned_its_degree_of_freedom():
    """The finding HU-4 asks for: matching the scheme that fits nothing is not beating it."""
    fitted = fit()

    assert fitted.best_weighted_score == pytest.approx(fitted.rrf_score)
    assert fitted.winning_scheme == RRF
    assert fitted.scheme_margin == pytest.approx(0.0)


def test_a_weighted_scheme_that_beats_rrf_takes_the_decision():
    fitted = a_misleading_fit()

    assert fitted.winning_scheme == WEIGHTED
    assert fitted.scheme_margin == pytest.approx(1.0)


def test_the_losing_figure_survives_whichever_scheme_lost():
    """Both numbers reach the report, or the finding cannot be stated in it."""
    matched, beaten = fit(), a_misleading_fit()

    assert (matched.rrf_score, matched.best_weighted_score) == (1.0, 1.0)
    assert (beaten.rrf_score, beaten.best_weighted_score) == (0.0, 1.0)


def test_the_two_schemes_are_the_two_the_config_declares():
    assert {fit().winning_scheme, a_misleading_fit().winning_scheme} == set(config.FUSION_SCHEMES)


# --- Dev, and only dev --------------------------------------------------------


def test_a_test_split_question_is_refused_by_the_fitting():
    """The same door the sweep has: a weight fitted on test would void the phase."""
    with pytest.raises(SelectionError, match="test"):
        fit(questions=the_questions(split="test"))


def test_one_test_question_among_the_dev_ones_is_enough_to_refuse():
    contaminated = [*the_questions(), a_question("q9", "A", ("g1",), split="test")]

    with pytest.raises(SelectionError, match="test"):
        fit(questions=contaminated)


def test_the_fitting_needs_questions_at_all():
    with pytest.raises(SelectionError, match="no question"):
        fit(questions=[])


def test_every_measurement_of_the_fitting_declares_the_dev_split(monkeypatch):
    splits = []
    real = selection.evaluate_retriever

    def spy(retriever, **kwargs):
        splits.append(kwargs["split"])
        return real(retriever, **kwargs)

    monkeypatch.setattr(selection, "evaluate_retriever", spy)
    fit()

    assert set(splits) == {DEV_SPLIT}


# --- Measured by the unchanged harness ----------------------------------------


def test_each_grid_point_and_the_reference_are_one_harness_call_each(monkeypatch):
    calls = []
    real = selection.evaluate_retriever

    def spy(retriever, **kwargs):
        calls.append((retriever, kwargs))
        return real(retriever, **kwargs)

    monkeypatch.setattr(selection, "evaluate_retriever", spy)
    fit()

    assert len(calls) == len(config.FUSION_WEIGHT_GRID) + 1
    schemes = [retriever.scheme for retriever, _kwargs in calls]
    assert schemes.count(RRF) == 1
    assert schemes.count(WEIGHTED) == len(config.FUSION_WEIGHT_GRID)


def test_every_measured_figure_records_the_scheme_and_the_weights_behind_it(monkeypatch):
    configs = []
    real = selection.evaluate_retriever

    def spy(retriever, **kwargs):
        configs.append(kwargs["config"])
        return real(retriever, **kwargs)

    monkeypatch.setattr(selection, "evaluate_retriever", spy)
    fit()

    weighted = [recorded for recorded in configs if recorded["fusion_scheme"] == WEIGHTED]
    assert [recorded["fusion_weights"]["dense"] for recorded in weighted] == pytest.approx(
        list(config.FUSION_WEIGHT_GRID)
    )
    assert [recorded for recorded in configs if recorded["fusion_scheme"] == RRF][0][
        "fusion_weights"
    ] is None


def test_the_top_k_comes_from_the_configuration_that_is_recorded():
    with pytest.raises(SelectionError, match="top_k"):
        fit_fusion_weight(
            [StubRetriever("dense", DENSE_HITS), StubRetriever("conceptual", OTHER_HITS)],
            questions=the_questions(),
            token_counts=TOKENS,
            base_config={key: value for key, value in a_config().items() if key != "top_k"},
        )


def test_a_configuration_that_could_not_reproduce_the_run_is_refused():
    with pytest.raises(SelectionError, match="seed"):
        fit_fusion_weight(
            [StubRetriever("dense", DENSE_HITS), StubRetriever("conceptual", OTHER_HITS)],
            questions=the_questions(),
            token_counts=TOKENS,
            base_config={key: value for key, value in a_config().items() if key != "seed"},
        )


def test_refitting_reproduces_the_curve_exactly():
    """A curve that drifts between runs cannot justify a weight, let alone a freeze."""
    first, second = fit(), fit()

    assert first.curve == second.curve
    assert (first.best_weight, first.rrf_score) == (second.best_weight, second.rrf_score)


# --- One function, System B and the control it has to be read against ---------


def test_the_control_is_fitted_by_the_same_function_over_the_same_grid():
    """Decision D9: `[dense, bm25]` goes through this code path, not through a copy of it."""
    control = fit(
        components=[StubRetriever("dense", DENSE_HITS), StubRetriever("bm25", OTHER_HITS)]
    )

    assert control.components == ("dense", "bm25")
    assert control.grid == config.FUSION_WEIGHT_GRID


def test_a_control_whose_signal_ranks_the_same_is_fitted_to_the_same_curve():
    """Same lists, same numbers: nothing but the second signal's name may differ."""
    system_b = fit()
    control = fit(
        components=[StubRetriever("dense", DENSE_HITS), StubRetriever("bm25", OTHER_HITS)]
    )

    assert control.curve == system_b.curve
    assert control.rrf_score == system_b.rrf_score
    assert control.best_weight == system_b.best_weight


def test_the_component_order_belongs_to_the_fusion_and_not_to_the_caller():
    reversed_in = fit(
        components=[StubRetriever("conceptual", OTHER_HITS), StubRetriever("dense", DENSE_HITS)]
    )

    assert reversed_in.components == ("dense", "conceptual")
    assert reversed_in.curve == fit().curve


def test_the_weight_names_the_dense_component_rather_than_a_position():
    """`w` is the weight on dense, so a curve read the other way round is a different claim."""
    misleading = a_misleading_fit()
    reversed_in = fit(
        components=[
            StubRetriever("conceptual", MISLEADING_OTHER),
            StubRetriever("dense", MISLEADING_DENSE),
        ],
        questions=[a_question("q1", "C", ("g1",))],
        budget=ONE_UNIT,
    )

    assert reversed_in.curve[1.0] == misleading.curve[1.0] == pytest.approx(1.0)


# --- What a fit refuses to be -------------------------------------------------


def a_fit(**overrides) -> FusionFit:
    fields = {
        "components": ("dense", "conceptual"),
        "metric": config.SELECTION_METRIC,
        "budget": config.SELECTION_BUDGET,
        "n_questions": 600,
        "grid": config.FUSION_WEIGHT_GRID,
        "curve": dict.fromkeys(config.FUSION_WEIGHT_GRID, 0.61),
        "rrf_score": 0.60,
    }
    fields.update(overrides)
    return FusionFit(**fields)


def test_a_curve_that_does_not_cover_its_grid_is_refused():
    partial = {weight: 0.61 for weight in config.FUSION_WEIGHT_GRID[:5]}

    with pytest.raises(SelectionError, match="curve"):
        a_fit(curve=partial)


def test_a_curve_carrying_a_point_the_grid_never_declared_is_refused():
    smuggled = dict.fromkeys(config.FUSION_WEIGHT_GRID, 0.61) | {0.55: 0.99}

    with pytest.raises(SelectionError, match="curve"):
        a_fit(curve=smuggled)


def test_a_grid_that_repeats_a_point_is_refused():
    with pytest.raises(SelectionError, match="repeats"):
        a_fit(grid=(0.0, 0.5, 0.5, 1.0), curve={0.0: 0.1, 0.5: 0.2, 1.0: 0.3})


def test_a_fit_over_no_grid_at_all_is_refused():
    with pytest.raises(SelectionError, match="grid"):
        a_fit(grid=(), curve={})


def test_fitting_over_an_empty_grid_is_refused_before_anything_is_measured():
    with pytest.raises(SelectionError, match="grid"):
        fit(grid=())
