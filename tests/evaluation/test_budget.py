"""T10: filling the context up to a fixed token budget.

This is the rule that makes systems retrieving different numbers of chunks
comparable, so it has to be boring and identical for everyone: fill in rank
order, skip whatever does not fit whole, never truncate, never exceed.
"""

from concept_embeddings_rag.evaluation.budget import fill_context, total_tokens

COUNTS = {"a": 100, "big": 900, "b": 200, "c": 50}


def test_fills_in_rank_order_until_the_budget_is_exhausted():
    selected = fill_context(["a", "b", "c"], COUNTS, budget=350)
    assert selected == ["a", "b", "c"]


def test_a_unit_that_does_not_fit_is_skipped_and_the_next_is_tried():
    selected = fill_context(["a", "big", "b"], COUNTS, budget=350)
    assert selected == ["a", "b"]


def test_the_context_never_exceeds_the_budget():
    selected = fill_context(["a", "big", "b", "c"], COUNTS, budget=350)
    assert total_tokens(selected, COUNTS) <= 350


def test_no_unit_is_ever_truncated():
    """A unit is either included whole or not at all."""
    selected = fill_context(["big"], COUNTS, budget=100)
    assert selected == []


def test_the_same_ranking_and_budget_always_give_the_same_context():
    first = fill_context(["a", "big", "b", "c"], COUNTS, budget=400)
    second = fill_context(["a", "big", "b", "c"], COUNTS, budget=400)
    assert first == second


def test_a_unit_without_a_known_token_count_raises():
    import pytest

    with pytest.raises(KeyError):
        fill_context(["unknown"], COUNTS, budget=1000)


def test_an_empty_ranking_yields_an_empty_context():
    assert fill_context([], COUNTS, budget=1000) == []
