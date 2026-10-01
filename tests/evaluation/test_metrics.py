"""T11: the four metrics.

Full Support Rate is the one that decides: on HotpotQA a question needs both gold
paragraphs, so retrieving one of two is worth nothing in practice even though it
scores 0.5 on plain recall.
"""

import math

import pytest

from concept_embeddings_rag.evaluation.metrics import (
    context_precision,
    full_support,
    gold_recall,
    ndcg_at_k,
    recall_at_k,
)

GOLD = ("g1", "g2")


def test_both_gold_present_is_full_recall_and_full_support():
    context = ["x", "g1", "g2"]
    assert gold_recall(context, GOLD) == 1.0
    assert full_support(context, GOLD) == 1


def test_one_of_two_is_half_recall_and_no_support():
    context = ["g1", "x", "y"]
    assert gold_recall(context, GOLD) == 0.5
    assert full_support(context, GOLD) == 0


def test_neither_present_scores_zero():
    assert gold_recall(["x", "y"], GOLD) == 0.0
    assert full_support(["x", "y"], GOLD) == 0


def test_precision_counts_gold_among_included_units():
    assert context_precision(["g1", "g2", "x", "y"], GOLD) == 0.5
    assert context_precision(["g1"], GOLD) == 1.0


def test_precision_of_an_empty_context_is_zero_not_undefined():
    assert context_precision([], GOLD) == 0.0


def test_recall_at_k_only_looks_at_the_first_k_hits():
    ranking = ["x", "g1", "y", "g2", "z"]
    assert recall_at_k(ranking, GOLD, k=2) == 0.5
    assert recall_at_k(ranking, GOLD, k=4) == 1.0
    assert recall_at_k(ranking, GOLD, k=1) == 0.0


def test_duplicated_gold_in_the_context_does_not_inflate_recall():
    assert gold_recall(["g1", "g1", "g1"], GOLD) == 0.5


# --- Phase 17 (D6): nDCG@10, binary relevance, the BEIR metric -----------------


def test_ndcg_at_10_by_hand_with_a_sentinel_gold_in_the_ideal():
    """Gold at ranks 2 and 4; the third gold is a sentinel no unit carries, so the ideal
    DCG still counts three relevant positions and the score cannot reach 1."""
    ranking = ["x", "g1", "y", "g2", "z"]
    gold = ("g1", "g2", "sentinel")
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranking, gold, 10) == pytest.approx(dcg / ideal)


def test_ndcg_ideal_is_capped_at_k_and_a_perfect_order_scores_one():
    assert ndcg_at_k(["g1", "g2"], GOLD, 10) == pytest.approx(1.0)
    # Only one position fits at k = 1: the ideal holds one gold, not two.
    assert ndcg_at_k(["g1", "g2"], GOLD, 1) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "g1"], GOLD, 1) == 0.0


def test_ndcg_counts_a_repeated_gold_once_and_ignores_ranks_past_k():
    once = 1 / math.log2(2)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["g1", "g1"], GOLD, 10) == pytest.approx(once / ideal)
    assert ndcg_at_k(["x"] * 10 + ["g1", "g2"], GOLD, 10) == 0.0


def test_ndcg_of_a_question_without_gold_is_zero():
    assert ndcg_at_k(["g1"], (), 10) == 0.0
