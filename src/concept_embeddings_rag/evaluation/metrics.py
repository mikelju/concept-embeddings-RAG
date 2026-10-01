"""The four retrieval metrics.

Gold Paragraph Recall is the headline number, but Full Support Rate is the one
that decides: a HotpotQA question needs both of its gold paragraphs, so half the
evidence answers nothing. A system can lift average recall by reliably finding
the easy paragraph while never finding the second one - which is precisely the
failure mode this project claims to fix.
"""

import math
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


def ndcg_at_k(ranked_unit_ids: Sequence[str], gold_unit_ids: Sequence[str], k: int) -> float:
    """nDCG over the first `k` hits with binary relevance, the BEIR metric (Phase 17, D6).

    A gold id counts once, at its first rank. The ideal DCG places min(|gold|, k) distinct
    gold ids first; a sentinel gold, which no unit carries, counts in it as in every recall of
    the line, so a question with unreachable gold cannot score 1. No gold scores 0.
    """
    gold = set(gold_unit_ids)
    if not gold:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for position, unit_id in enumerate(list(ranked_unit_ids)[:k], start=1):
        if unit_id in gold and unit_id not in seen:
            seen.add(unit_id)
            dcg += 1.0 / math.log2(position + 1)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(gold), k) + 1))
    return dcg / ideal
