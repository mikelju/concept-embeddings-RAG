"""Phase 6, T6: `fit_fusion_weight` fits a staged hybrid and keeps every point's metrics (D10).

Decision D9 of Phase 3 makes one fitting function serve every hybrid, and Phase 6 keeps it
that way: the only changes are the type of the components it accepts and an observe-only
hook that sees every measurement. The curve, the RRF reading, the tie rules and the returned
`FusionFit` are the function's own, and the hook cannot change them.
"""

import json
from collections.abc import Sequence

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.harness import RunResult
from concept_embeddings_rag.evaluation.selection import (
    DEV_SPLIT,
    FusionFit,
    fit_fusion_weight,
    fusion_decisions,
    load_selection,
)
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.fusion import RRF, WEIGHTED, FusedRetriever

# One unit costs half the selection budget: at 2,048 tokens exactly two units fit.
TOKENS = dict.fromkeys(["g1", "g2", "x", "y"], 1024)
DENSE_HITS: list[Hit] = [("g1", 1.0), ("x", 0.5), ("y", 0.0)]
STAGE_HITS: list[Hit] = [("g2", 2.0), ("x", 1.0)]


class StubDense:
    name = "dense"

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return DENSE_HITS[:top_k]


class StubStage:
    name = "entity-hop"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.received: list[list[Hit]] = []

    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
        self.received.append(list(first))
        return self.hits[:top_k]


def a_question() -> Question:
    return Question(
        qid="q1",
        question="a question",
        answer="-",
        gold_unit_ids=("g1", "g2"),
        supporting_facts=(("g1", 0),),
        split=DEV_SPLIT,
    )


def a_config() -> dict:
    return {
        "model": "fake",
        "revision": "v0",
        "unit_set_hash": "poolhash0001",
        "seed": 42,
        "tokenizer": "fake-tokenizer",
        "code_version": "0.1.0",
        "top_k": 3,
    }


def a_fit(stage_hits: list[Hit] = STAGE_HITS, on_measure=None) -> FusionFit:
    kwargs = {} if on_measure is None else {"on_measure": on_measure}
    return fit_fusion_weight(
        [StubDense(), StubStage(stage_hits)],
        questions=[a_question()],
        token_counts=TOKENS,
        base_config=a_config(),
        **kwargs,
    )


def test_a_staged_hybrid_is_fitted_over_the_declared_grid_and_rrf():
    fit = a_fit()
    assert fit.components == ("dense", "entity-hop")
    assert fit.grid == config.FUSION_WEIGHT_GRID
    assert fit.metric == config.SELECTION_METRIC
    assert fit.budget == config.SELECTION_BUDGET
    assert fit.n_questions == 1


def test_the_staged_curve_is_the_one_computed_on_paper():
    """Min-max: dense g1 1, x 0.5, y 0; stage g2 1, x 0 (its bottom tier). Two units fit.

    fused(w): g1 = w, x = w / 2, g2 = 1 - w. Up to w = 0.6 the context is {g1, g2} (at 0.5
    the two tie and break by id, both still in); from w = 0.7, x at 0.35 passes g2 at 0.3
    and the context is {g1, x}. RRF: x scores 2/62 and leads; g1 and g2 tie at 1/61 and g1
    takes the second place by id, so RRF recovers half the gold.
    """
    fit = a_fit()
    expected = {w: (1.0 if w <= 0.6 else 0.5) for w in config.FUSION_WEIGHT_GRID}

    assert fit.curve == expected
    assert fit.rrf_score == 0.5
    assert fit.best_weight == 0.6
    assert fit.winning_scheme == WEIGHTED
    assert fit.weights == {"dense": 0.6, "entity-hop": pytest.approx(0.4)}


def test_ties_go_to_the_larger_dense_weight_on_a_staged_fit():
    fit = a_fit()
    tied = [w for w, value in fit.curve.items() if value == fit.best_weighted_score]
    assert len(tied) > 1
    assert fit.best_weight == max(tied)


def test_an_empty_stage_gives_a_flat_curve_and_rrf_wins_the_tie():
    """The strict-win rule: weighted matching RRF is not weighted beating it."""
    fit = a_fit(stage_hits=[])

    assert len(set(fit.curve.values())) == 1
    assert fit.best_weight == 1.0
    assert fit.best_weighted_score == fit.rrf_score
    assert fit.winning_scheme == RRF
    assert fit.weights is None
    assert [decision.name for decision in fusion_decisions(fit)] == ["fusion_scheme"]


def test_the_hook_sees_rrf_then_every_grid_weight_with_all_four_budgets():
    seen: list[tuple[dict, RunResult]] = []
    a_fit(on_measure=lambda hybrid, result: seen.append((hybrid.describe(), result)))

    assert len(seen) == 1 + len(config.FUSION_WEIGHT_GRID) == 12
    assert seen[0][0]["scheme"] == RRF
    assert [described["weights"]["dense"] for described, _ in seen[1:]] == list(
        config.FUSION_WEIGHT_GRID
    )
    for described, result in seen:
        assert described["name"] == "hybrid-entity-hop"
        assert result.split == DEV_SPLIT
        assert sorted(result.metrics) == sorted(
            [f"budget_{b}" for b in config.CONTEXT_BUDGETS]
            + [f"recall_at_{k}" for k in config.RECALL_AT_K]
        )


def test_the_hook_readings_match_the_curve_it_observed():
    seen: list[tuple[FusedRetriever, RunResult]] = []
    fit = a_fit(on_measure=lambda hybrid, result: seen.append((hybrid, result)))
    budget = f"budget_{config.SELECTION_BUDGET}"

    assert seen[0][1].metrics[budget][config.SELECTION_METRIC] == fit.rrf_score
    for hybrid, result in seen[1:]:
        assert hybrid.weights is not None
        weight = hybrid.weights["dense"]
        assert result.metrics[budget][config.SELECTION_METRIC] == fit.curve[weight]


def test_removing_the_hook_changes_no_field_of_the_fit():
    with_hook = a_fit(on_measure=lambda hybrid, result: None)
    without = a_fit()

    assert with_hook == without
    assert with_hook.curve == without.curve
    assert with_hook.rrf_score == without.rrf_score


def test_a_hook_that_tampers_with_what_it_is_shown_cannot_move_the_curve():
    def vandal(hybrid: FusedRetriever, result: RunResult) -> None:
        result.metrics[f"budget_{config.SELECTION_BUDGET}"][config.SELECTION_METRIC] = -1.0

    assert a_fit(on_measure=vandal) == a_fit()


def test_the_stage_is_conditioned_on_the_dense_list_at_every_point():
    stage = StubStage(STAGE_HITS)
    fit_fusion_weight(
        [StubDense(), stage],
        questions=[a_question()],
        token_counts=TOKENS,
        base_config=a_config(),
    )
    assert len(stage.received) == 12
    assert all(received == DENSE_HITS for received in stage.received)


def test_the_frozen_phase_3_selection_still_verifies():
    path = config.SELECTION_DIR / "selection.json"
    if not path.exists():
        pytest.skip("the Phase 3 selection is not on disk")
    load_selection(config.SELECTION_DIR, expected_unit_set_hash=config.PHASE_1_UNIT_SET_HASH)
    assert json.loads(path.read_text(encoding="utf-8"))["digest"] == config.PINNED_SELECTION_DIGEST
