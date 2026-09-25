"""Phase 11: a better use of the same entity signal, against P10-C, at FullWiki scale.

`11.spec.md` (approved 2026-09-24, amended and re-approved 2026-09-25) fixes every rule here
before any Phase 11 number exists. Two new entity components - the Phase 9 hop from P1 with
the hubs above a DF cap dropped, and the same hop seeded by the entities GLiNER reads in the
question - join Dense and BM25 in one four-component fusion fitted on the 7,405 dev
questions. The single configuration dev selects is measured once on `test-11`. Every Phase 9
and Phase 10 artifact is read-only.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import digest_of, write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import fullwiki as phase9
from concept_embeddings_rag.evaluation import phase10
from concept_embeddings_rag.nodes.index import NodeIndex
from concept_embeddings_rag.nodes.normalization import normalize

DEV_STOP = "DEV_STOP"
SUPPORTED = "ENTITY_USE_SUPPORTED"
NOT_SUPPORTED = "ENTITY_USE_NOT_SUPPORTED"
REGRESSION = "ENTITY_USE_REGRESSION"
QUAD_NAMES: tuple[str, ...] = ("dense", "bm25", config.ENTITY_HOP_NAME, "question-hop")


class Phase11Error(Exception):
    """A Phase 11 input does not match what the spec or a recorded artifact requires."""


def load_test_11(phase10_dir: Path | str, *, corpus_unit_set_hash: str) -> list[Question]:
    """`test-11`, as Phase 10 froze it, digests recomputed. Only Phase 11 reads it."""
    return phase10._read_set(
        phase10_dir, config.PHASE_11_TEST, corpus_unit_set_hash=corpus_unit_set_hash
    )


# --- S1 (HU-1): the question entities -------------------------------------------------------


def entity_forms(index: NodeIndex) -> dict[str, int]:
    """Node form -> node id over the entity nodes the hop reads (`ENTITY_HOP_TYPES`)."""
    wanted = set(config.ENTITY_HOP_TYPES)
    return {node.form: node.node_id for node in index.nodes if node.type in wanted}


def match_spans(spans: Sequence[str], forms: Mapping[str, int]) -> dict[str, Any]:
    """One question's spans, normalized as the index was, matched to entity nodes (D1).

    A span that normalizes to nothing is dropped; one whose form is not a node is kept in
    `unmatched`. Node ids are unique and ascending.
    """
    normalized = [form for form in (normalize(span) for span in spans) if form]
    matched = sorted({forms[form] for form in normalized if form in forms})
    unmatched = sorted({form for form in normalized if form not in forms})
    return {"spans": list(spans), "node_ids": matched, "unmatched": unmatched}


def entities_filename(set_name: str) -> str:
    return f"question-entities-{set_name}.json"


def write_question_entities(
    directory: Path | str,
    set_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    extractor: Mapping[str, Any],
) -> Path:
    """One set's frozen question entities, written once, with a digest over the rows."""
    if set_name not in config.PHASE_11_QUESTION_SETS:
        raise Phase11Error(f"unknown Phase 11 question set {set_name!r}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / entities_filename(set_name)
    if target.exists():
        raise Phase11Error(f"{target} already freezes the {set_name} question entities")
    canonical = json.dumps([dict(row) for row in rows], sort_keys=True, ensure_ascii=True)
    body = {
        "set": set_name,
        "n_questions": len(rows),
        "extractor": dict(extractor),
        "rows_digest": digest_of(canonical),
        "questions": [dict(row) for row in rows],
    }
    write_text_atomic(target, json.dumps(body, indent=2, sort_keys=True, ensure_ascii=True))
    return target


def load_question_entities(directory: Path | str, set_name: str) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(
        (Path(directory) / entities_filename(set_name)).read_text(encoding="utf-8")
    )
    canonical = json.dumps(body["questions"], sort_keys=True, ensure_ascii=True)
    if digest_of(canonical) != body["rows_digest"]:
        raise Phase11Error(f"the {set_name} question entities do not match their digest")
    return body


def nodes_by_question(
    body: Mapping[str, Any], questions: Sequence[Question]
) -> dict[str, list[int]]:
    """Question text -> matched node ids, in the set's own question order, checked by qid."""
    rows = body["questions"]
    if [row["qid"] for row in rows] != [q.qid for q in questions]:
        raise Phase11Error("the question entities are not in the set's question order")
    mapping: dict[str, list[int]] = {}
    for row, question in zip(rows, questions, strict=True):
        nodes = [int(node) for node in row["node_ids"]]
        if mapping.get(question.question, nodes) != nodes:
            raise Phase11Error(f"one question text with two entity sets: {question.question!r}")
        mapping[question.question] = nodes
    return mapping


# --- S4 (D3): the reproduction of P10-C on dev --------------------------------------------


def reproduction_verdict(
    observed: int, differing_qids: Sequence[str], lists_differing_qids: Sequence[str]
) -> dict[str, Any]:
    """D3: exactly 4,801, no question whose outcome moved, no P1 list that moved."""
    recorded = config.PHASE_11_P10C_DEV_SUPPORTED
    return {
        "metric": "full_support@2048_tokens over the 7,405 dev questions",
        "point": {"p1_df_cap": None, "weights": dict(config.PHASE_11_P10C_WEIGHTS)},
        "observed": int(observed),
        "recorded": recorded,
        "differing_qids": sorted(differing_qids),
        "uncapped_list_differing_qids": sorted(lists_differing_qids),
        "passed": observed == recorded and not differing_qids and not lists_differing_qids,
    }


# --- S5 (D2, D4): the grid, the tie rule and the dev gate -----------------------------------


def weight_grid(tenths: int = config.PHASE_11_GRID_TENTHS) -> list[tuple[float, ...]]:
    """Every convex quadruple on a grid of `1 / tenths`: 286 points for tenths."""
    return [
        (d / tenths, b / tenths, p / tenths, (tenths - d - b - p) / tenths)
        for d in range(tenths, -1, -1)
        for b in range(tenths - d, -1, -1)
        for p in range(tenths - d - b, -1, -1)
    ]


def grid_points(
    caps: Iterable[int | None] = config.PHASE_11_DF_CAPS,
) -> list[tuple[int | None, tuple[float, ...]]]:
    return [(cap, weights) for cap in caps for weights in weight_grid()]


def _cap_rank(cap: int | None) -> float:
    return float("inf") if cap is None else float(cap)


def selection_key(point: Mapping[str, Any]) -> tuple[Any, ...]:
    """D2, in order: Full Support; gold recall; no question weight; larger cap; larger
    `w_dense`, `w_bm25`, `w_p1`. `max` over this key is the selected point."""
    w_dense, w_bm25, w_p1, w_question = point["weights"]
    return (
        point["supported"],
        point["gold_recall_sum"],
        w_question == 0.0,
        _cap_rank(point["p1_df_cap"]),
        w_dense,
        w_bm25,
        w_p1,
    )


def choose_point(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best: Mapping[str, Any] = max(curve, key=selection_key)
    return dict(best)


def dev_gate(chosen: Mapping[str, Any]) -> str | None:
    """D4: `DEV_STOP` below P10-C's 4,801 plus 37 questions."""
    bar = config.PHASE_11_P10C_DEV_SUPPORTED + config.PHASE_11_DEV_MARGIN
    return None if chosen["supported"] >= bar else DEV_STOP


def uses_question(chosen: Mapping[str, Any]) -> bool:
    return bool(chosen["weights"][3] > 0.0)


# --- S7 (D5, D6): the paired comparisons and the label --------------------------------------


def paired(
    control: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Full Support @2,048 of two runs on the same `test-11` questions, exact McNemar."""
    budget = str(config.PHASE_9_PRIMARY_BUDGET)
    a = {r["qid"]: r["budgets"][budget]["full_support"] for r in control}
    c = {r["qid"]: r["budgets"][budget]["full_support"] for r in candidate}
    if set(a) != set(c) or len(a) != config.PHASE_10_SET_SIZES[config.PHASE_11_TEST]:
        raise Phase11Error("the two runs do not hold the same test-11 questions")
    wins = sum(1 for q in a if c[q] == 1.0 and a[q] == 0.0)
    losses = sum(1 for q in a if a[q] == 1.0 and c[q] == 0.0)
    n = len(a)
    control_successes, candidate_successes = int(sum(a.values())), int(sum(c.values()))
    return {
        "n_questions": n,
        "control_successes": control_successes,
        "candidate_successes": candidate_successes,
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
        "delta_percentage_points": 100.0 * (candidate_successes - control_successes) / n,
        "exact_two_sided_p": phase9.exact_two_sided_p(wins, losses),
    }


def label(primary: Mapping[str, Any]) -> str:
    """D5, from P11 against P10-C only."""
    alpha = config.PHASE_9_ALPHA
    if primary["wins"] > primary["losses"] and primary["exact_two_sided_p"] < alpha:
        return SUPPORTED
    if primary["losses"] > primary["wins"] and primary["exact_two_sided_p"] < alpha:
        return REGRESSION
    return NOT_SUPPORTED
