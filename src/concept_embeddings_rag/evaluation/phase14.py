"""Phase 14: ordering the hop's candidates by their similarity to the question.

`14.spec.md` (approved 2026-09-28) fixes every rule here before any Phase 14 number exists.
The Phase 9 hop's candidate set is unchanged: every paragraph outside `read(q)` sharing an
entity node with P1. What is new is the order and the score the fusion reads (D2): each
candidate's rarity overlap with P1 and its BGE similarity to the question (D1) are min-max
normalized over the question's whole candidate set and mixed as
`(1 - alpha) * r_hat + alpha * s_hat`. At `alpha = 0` the Phase 9 hop is returned unchanged.

This module holds those rules as pure functions over plain arrays, plus the D4 grid and tie
rule, the D5 gate, the D1 integrity check, the D8 Dense-rank split and the D6 label. It
imports no evaluation runner, because `evaluation.second_hop` and `retrieval.entity_hop`
import it. The runtime stage lives in `retrieval/entity_hop.py`.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import phase12

SUPPORTED = "CANDIDATE_RELEVANCE_SUPPORTED"
NOT_SUPPORTED = "CANDIDATE_RELEVANCE_NOT_SUPPORTED"
REGRESSION = "CANDIDATE_RELEVANCE_REGRESSION"
DEV_STOP = "DEV_STOP"


class Phase14Error(Exception):
    """A Phase 14 input is missing, modified, or not the one this run may use."""


# --- D1, D2: similarity, normalization, mixing and the ranking ------------------------------


def candidate_similarity(
    vectors: np.ndarray, rows: np.ndarray, question_vector: np.ndarray
) -> np.ndarray:
    """`sim(c) = v_q . v_c` in float32 over the gathered rows (D1).

    Both sides are L2-normalized, so the dot product is the cosine. This is the one function
    dev and `test-11` share; it never reuses Dense's own score vector.
    """
    gathered = vectors[rows].astype(np.float32, copy=False)
    return gathered @ np.asarray(question_vector, dtype=np.float32)


def min_max(values: np.ndarray) -> np.ndarray:
    """D2's per-question normalization in float64; 0 everywhere when max equals min."""
    values = np.asarray(values, dtype=np.float64)
    low, high = values.min(), values.max()
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def mixed_scores(r_hat: np.ndarray, s_hat: np.ndarray, alpha: float) -> np.ndarray:
    """`score_alpha = (1 - alpha) * r_hat + alpha * s_hat`, both terms already normalized."""
    return (1.0 - alpha) * r_hat + alpha * s_hat


def rank_cut(
    rows: np.ndarray, scores: np.ndarray, unit_ids: Sequence[str], depth: int
) -> list[tuple[int, float]]:
    """`(row, score)` by score descending, then unit id ascending, cut at `depth` (D2).

    `scores` is aligned with `rows`; `unit_ids` is indexed by row. As in
    `second_hop.node_hop_columnwise`, the rows strictly above the depth-th best score, plus
    the tied rows at it, are sorted and cut: the head of a full sort, without sorting the rest.
    """
    kept = np.arange(rows.size)
    if rows.size > depth:
        threshold = -np.partition(-scores, depth - 1)[depth - 1]
        kept = np.nonzero(scores >= threshold)[0]
    ordered = sorted(
        (int(i) for i in kept), key=lambda i: (-float(scores[i]), unit_ids[int(rows[i])])
    )
    return [(int(rows[i]), float(scores[i])) for i in ordered[:depth]]


# --- D4, D5: the grid, the tie rule and the dev gate ------------------------------------------


def alpha_key(alpha: float) -> str:
    """The dev-list / grid key for one alpha, e.g. `relevance-hop@alpha=0.25`."""
    return f"{config.PHASE_14_COMPONENT_NAMES[2]}@alpha={alpha:.2f}"


def grid_points(
    alphas: Sequence[float] = config.PHASE_14_ALPHAS,
) -> list[tuple[float, tuple[float, float, float]]]:
    """The 5 alphas x the 66 convex weight triples: 330 points (D4)."""
    triples = phase12.weight_grid(config.PHASE_14_GRID_TENTHS)
    return [(alpha, weights) for alpha in alphas for weights in triples]


def selection_key(point: Mapping[str, Any]) -> tuple[Any, ...]:
    """D4, in order: Full Support; gold recall; smaller alpha; larger w_dense, larger w_bm25."""
    w_dense, w_bm25, _w_hop = point["weights"]
    return (point["supported"], point["gold_recall_sum"], -point["alpha"], w_dense, w_bm25)


def choose_point(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best: Mapping[str, Any] = max(curve, key=selection_key)
    return dict(best)


def dev_gate(chosen: Mapping[str, Any]) -> str | None:
    """D5: `DEV_STOP` below the bar (P10-C's 4,801 plus 34, 4,835)."""
    return None if chosen["supported"] >= config.PHASE_14_DEV_BAR else DEV_STOP


# --- D1: the integrity check ------------------------------------------------------------------


def integrity_verdict(
    recomputed: Mapping[str, float],
    recorded: Mapping[str, float],
    tolerance: float = config.PHASE_14_SIM_TOLERANCE,
) -> dict[str, Any]:
    """D1: every recomputed `sim(P1)` within `tolerance` of Dense's recorded score for P1."""
    if set(recomputed) != set(recorded):
        raise Phase14Error("the integrity check must compare the same questions")
    differences = {qid: abs(float(recomputed[qid]) - float(recorded[qid])) for qid in recorded}
    offending = sorted(qid for qid, difference in differences.items() if difference > tolerance)
    return {
        "questions": len(differences),
        "tolerance": tolerance,
        "max_abs_difference": max(differences.values(), default=0.0),
        "offending_qids": offending,
        "passed": not offending,
    }


# --- D8: where the gains come from ------------------------------------------------------------


def dense_rank_class(rank: int | None) -> str:
    """A 1-based Dense rank's class: `1-10`, `11-100`, or `beyond-100` (absent from depth 100)."""
    if rank is None or rank > 100:
        return "beyond-100"
    return "1-10" if rank <= 10 else "11-100"


def coverage_changes(
    *,
    gold: Sequence[str],
    candidate_context: Collection[str],
    control_context: Collection[str],
    dense_ranks: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Each gold paragraph in one system's 2,048-token context and not the other's (D8).

    `gained` when only the candidate's context holds it, `lost` when only the control's does;
    `dense_ranks` maps a unit to its 1-based rank in the Dense list (depth 100).
    """
    changes: list[dict[str, Any]] = []
    for unit_id in gold:
        in_candidate, in_control = unit_id in candidate_context, unit_id in control_context
        if in_candidate == in_control:
            continue
        rank = dense_ranks.get(unit_id)
        changes.append(
            {
                "unit_id": unit_id,
                "change": "gained" if in_candidate else "lost",
                "dense_rank": rank,
                "dense_class": dense_rank_class(rank),
            }
        )
    return changes


DENSE_RANK_CLASSES: tuple[str, ...] = ("1-10", "11-100", "beyond-100")


def dense_rank_split(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """D8's split of the won and lost questions, one entry per question.

    Each entry holds `qid`, `gold`, both 2,048-token contexts (`candidate_context`,
    `control_context`) and `dense_ids`, the Dense list in rank order (depth 100). A question
    is won when only the candidate's context holds every gold paragraph, lost when only the
    control's does; ties are not counted, whatever moved inside them. For each won or lost
    question, every gold paragraph whose coverage changed is tallied by its Dense rank class.
    """

    def tally() -> dict[str, Any]:
        return {
            "questions": 0,
            "gold_changed": dict.fromkeys(DENSE_RANK_CLASSES, 0),
            "gold_changed_total": 0,
        }

    split = {"won": tally(), "lost": tally(), "total": tally()}
    per_question: list[dict[str, Any]] = []
    for entry in entries:
        gold = list(entry["gold"])
        candidate, control = set(entry["candidate_context"]), set(entry["control_context"])
        candidate_full, control_full = set(gold) <= candidate, set(gold) <= control
        if candidate_full == control_full:
            continue
        outcome = "won" if candidate_full else "lost"
        ranks = {unit_id: rank for rank, unit_id in enumerate(entry["dense_ids"], start=1)}
        changes = coverage_changes(
            gold=gold, candidate_context=candidate, control_context=control, dense_ranks=ranks
        )
        for bucket in (split[outcome], split["total"]):
            bucket["questions"] += 1
            for change in changes:
                bucket["gold_changed"][change["dense_class"]] += 1
                bucket["gold_changed_total"] += 1
        per_question.append({"qid": entry["qid"], "outcome": outcome, "changes": changes})
    return {**split, "per_question": per_question}


RECORDED_RANK_CLASSES: tuple[str, ...] = ("1-10", "beyond-10", "undetermined")


def recorded_rank_split(
    candidate: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
    dense: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """D8 on `test-11` from the recorded outcome records alone, which hold no ranking.

    A question is won or lost on Full Support @2,048. The winner's context holds every gold
    paragraph, so the winner gives the gold count (`context_precision * units_included`) and
    the loser's gold recall gives how many of them changed coverage: exact counts. Which
    paragraph changed is not recorded, so its Dense rank is bounded from Dense's recorded
    gold recall @10 (how many of the question's gold paragraphs Dense ranks 1-10): changed
    paragraphs certainly in Dense's first 10 count `1-10`, certainly outside it `beyond-10`,
    the rest `undetermined`. Ranks 11-100 and beyond 100 cannot be told apart here.
    """
    budget = str(config.PHASE_9_PRIMARY_BUDGET)
    by_qid = [{r["qid"]: r for r in rows} for rows in (candidate, control, dense)]
    if not (set(by_qid[0]) == set(by_qid[1]) == set(by_qid[2])):
        raise Phase14Error("the recorded split must read three runs over the same questions")

    def tally() -> dict[str, Any]:
        return {
            "questions": 0,
            "gold_changed": dict.fromkeys(RECORDED_RANK_CLASSES, 0),
            "gold_changed_total": 0,
        }

    split = {"won": tally(), "lost": tally(), "total": tally()}
    for qid in (r["qid"] for r in candidate):
        cand, ctrl, base = by_qid[0][qid], by_qid[1][qid], by_qid[2][qid]
        cand_full = cand["budgets"][budget]["full_support"] == 1.0
        ctrl_full = ctrl["budgets"][budget]["full_support"] == 1.0
        if cand_full == ctrl_full:
            continue
        outcome = "won" if cand_full else "lost"
        winner, loser = (cand, ctrl) if cand_full else (ctrl, cand)
        reading = winner["budgets"][budget]
        n_gold = round(reading["context_precision"] * reading["units_included"])
        changed = round(n_gold * (1.0 - loser["budgets"][budget]["gold_recall"]))
        in_top_10 = round(n_gold * base["gpr_at_k"]["10"])
        surely_in = max(0, in_top_10 - (n_gold - changed))
        maybe_in = min(changed, in_top_10)
        counts = {
            "1-10": surely_in,
            "beyond-10": changed - maybe_in,
            "undetermined": maybe_in - surely_in,
        }
        for bucket in (split[outcome], split["total"]):
            bucket["questions"] += 1
            bucket["gold_changed_total"] += changed
            for name, count in counts.items():
                bucket["gold_changed"][name] += count
    return split


# --- D6: the label ----------------------------------------------------------------------------


def label(wins: int, losses: int, p: float) -> str:
    """D6, from P14 against P10-C only; the secondary comparison (D7) takes no part in it."""
    alpha = config.PHASE_9_ALPHA
    if wins > losses and p < alpha:
        return SUPPORTED
    if losses > wins and p < alpha:
        return REGRESSION
    return NOT_SUPPORTED


# --- D3: the reproduction ---------------------------------------------------------------------


def reproduction_verdict(
    observed: int, differing_qids: Sequence[str], lists_differing_qids: Sequence[str]
) -> dict[str, Any]:
    """D3: exactly P10-C's dev count, no question moved, no `alpha = 0` hop list moved.

    The shape of `phase13.reproduction_verdict`, with Phase 14's point.
    """
    weights = dict(zip(config.PHASE_14_COMPONENT_NAMES, config.PHASE_14_P10C_WEIGHTS, strict=True))
    recorded = config.PHASE_14_P10C_DEV_SUPPORTED
    return {
        "metric": "full_support@2048_tokens over the 7,405 dev questions",
        "point": {"alpha": config.PHASE_14_P10C_ALPHA, "weights": weights},
        "observed": int(observed),
        "recorded": recorded,
        "differing_qids": sorted(differing_qids),
        "hop_list_differing_qids": sorted(lists_differing_qids),
        "passed": observed == recorded and not differing_qids and not lists_differing_qids,
    }
