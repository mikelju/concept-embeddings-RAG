"""T11: the four metrics.

Full Support Rate is the one that decides: on HotpotQA a question needs both gold
paragraphs, so retrieving one of two is worth nothing in practice even though it
scores 0.5 on plain recall.
"""

from concept_embeddings_rag.evaluation.metrics import (
    context_precision,
    full_support,
    gold_recall,
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
