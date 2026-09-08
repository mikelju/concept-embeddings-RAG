"""The four retrieval metrics.

Gold Paragraph Recall is the headline number, but Full Support Rate is the one
that decides: a HotpotQA question needs both of its gold paragraphs, so half the
evidence answers nothing. A system can lift average recall by reliably finding
the easy paragraph while never finding the second one - which is precisely the
failure mode this project claims to fix.
"""

from collections.abc import Sequence


def gold_recall(context_unit_ids: Sequence[str], gold_unit_ids: Sequence[str]) -> float:
    """Fraction of the gold paragraphs present in the context."""
    if not gold_unit_ids:
        return 0.0
    present = set(context_unit_ids) & set(gold_unit_ids)
    return len(present) / len(set(gold_unit_ids))


def full_support(context_unit_ids: Sequence[str], gold_unit_ids: Sequence[str]) -> int:
    """1 if every gold paragraph is present, 0 otherwise."""
    return int(set(gold_unit_ids) <= set(context_unit_ids))


def context_precision(context_unit_ids: Sequence[str], gold_unit_ids: Sequence[str]) -> float:
    """Fraction of the included units that are gold. Empty context scores 0."""
    if not context_unit_ids:
        return 0.0
    gold = set(gold_unit_ids)
    hits = sum(1 for unit_id in context_unit_ids if unit_id in gold)
    return hits / len(context_unit_ids)


def recall_at_k(
    ranked_unit_ids: Sequence[str],
    gold_unit_ids: Sequence[str],
    k: int,
) -> float:
    """Gold recall considering only the first `k` hits of the ranking."""
    return gold_recall(list(ranked_unit_ids)[:k], gold_unit_ids)
