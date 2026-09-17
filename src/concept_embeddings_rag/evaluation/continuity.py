"""Continuity with Phase 5: the entity stage reproduces the frozen hop run (HU-3, D11).

Four checks over the pilot questions, each computed from Phase 6's own `D(q)` and `E(q)` and
compared with what Phase 5 froze. Nothing here re-runs Phase 5 or reads its code path: the
pilot, the hop run and the traces are the verified artifacts.

(a) `read(q)` equals the pilot's `read`, so `p1(q)` equals its `p1`. Phase 5 broke dense ties by
    pool row and Phase 6 by unit id; this check turns "no exact float tie changes the head of
    the list" into a verified fact.
(b) `P(q)` equals the hop run's `positives` for the entity arm.
(c) hit@10 and hit@100 of `E(q)` over the pilot's missing paragraphs equal the entity arm's
    `scores`, with `==`: the same integer counts divided the same way give the same float.
(d) every entity trace is found at its rank in `E(q)`: the same unit, the same nodes (id, type,
    form), and the score and each node weight within the relative tolerance decided in OI-2,
    with the node weights summed by `math.fsum` against the score.

The pilot's `missing` is the only annotation read, and only to compute (c). A mismatch is
recorded, never repaired; any failed check stops the phase before a B figure exists.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.navigation import ARM_HOPS
from concept_embeddings_rag.evaluation.pilot import PilotSet
from concept_embeddings_rag.retrieval.base import Retriever
from concept_embeddings_rag.retrieval.entity_hop import EntityExpansion, EntityHopStage

ENTITY_HOP = ARM_HOPS["entity"]
ENTITY_ARM = "entity"
CHECK_NAMES: tuple[str, ...] = ("a_read", "b_positives", "c_hits", "d_traces")
MAX_EXAMPLES = 20


@dataclass(frozen=True)
class ContinuityCheck:
    name: str
    agreements: int
    total: int
    mismatches: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatches and self.agreements == self.total

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "agreements": self.agreements,
            "total": self.total,
            "n_mismatches": len(self.mismatches),
            "mismatches": list(self.mismatches[:MAX_EXAMPLES]),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ContinuityReport:
    checks: tuple[ContinuityCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_payload(self) -> dict[str, Any]:
        return {"checks": [check.as_payload() for check in self.checks], "passed": self.passed}


class _Tally:
    def __init__(self, name: str) -> None:
        self.name = name
        self.agreements = 0
        self.total = 0
        self.mismatches: list[str] = []

    def record(self, agrees: bool, mismatch: str) -> None:
        self.total += 1
        if agrees:
            self.agreements += 1
        else:
            self.mismatches.append(mismatch)

    def done(self) -> ContinuityCheck:
        return ContinuityCheck(self.name, self.agreements, self.total, tuple(self.mismatches))


def _found(expansion: EntityExpansion, missing: Sequence[str], depth: int) -> int:
    top = {candidate.unit_id for candidate in expansion.candidates[:depth]}
    return sum(1 for unit_id in missing if unit_id in top)


def _close(a: float, b: float, rel_tol: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=0.0)


def _trace_disagreements(
    trace: Mapping[str, Any], expansion: EntityExpansion, rel_tol: float
) -> list[str]:
    rank = int(trace["rank"])
    if not 1 <= rank <= len(expansion.candidates):
        return [f"rank {rank} is beyond the {len(expansion.candidates)} candidates of E(q)"]
    candidate = expansion.candidates[rank - 1]
    problems: list[str] = []
    if candidate.unit_id != trace["unit_id"]:
        problems.append(f"rank {rank} holds {candidate.unit_id}, the trace {trace['unit_id']}")
    if not _close(candidate.score, trace["score"], rel_tol):
        problems.append(f"score {candidate.score!r} against {trace['score']!r}")
    recorded = list(trace["nodes"])
    ours = list(candidate.nodes)
    identities = [(node.node_id, node.type, node.form) for node in ours]
    theirs = [(int(node["node_id"]), str(node["type"]), str(node["form"])) for node in recorded]
    if identities != theirs:
        problems.append(f"nodes {identities} against {theirs}")
    else:
        for node, entry in zip(ours, recorded, strict=True):
            if not _close(node.weight, entry["weight"], rel_tol):
                problems.append(
                    f"node {node.node_id} weight {node.weight!r} against {entry['weight']!r}"
                )
    weight_sum = math.fsum(float(entry["weight"]) for entry in recorded)
    if not _close(weight_sum, trace["score"], rel_tol):
        problems.append(f"the trace's node weights sum to {weight_sum!r}, not {trace['score']!r}")
    return problems


def check_continuity(
    pilot: PilotSet,
    hop_run: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    *,
    dense: Retriever,
    stage: EntityHopStage,
    question_text: Mapping[str, str],
    top_k: int = config.EVALUATION_TOP_K,
    rel_tol: float = config.TRACE_WEIGHT_REL_TOL,
) -> ContinuityReport:
    """Checks (a)-(d) for every pilot question, each with its agreements and mismatches."""
    read_check, positives_check = _Tally(CHECK_NAMES[0]), _Tally(CHECK_NAMES[1])
    hits_check, traces_check = _Tally(CHECK_NAMES[2]), _Tally(CHECK_NAMES[3])
    recorded = {str(entry["qid"]): entry for entry in hop_run["per_question"]}
    depths = [str(depth) for depth in config.SECOND_HOP_DEPTHS]

    expansions: dict[str, EntityExpansion] = {}
    for entry in pilot.questions:
        first = dense.retrieve(question_text[entry.qid], top_k)
        expansion = stage.expand(first, top_k)
        expansions[entry.qid] = expansion

        read_check.record(
            expansion.read == tuple(entry.read) and expansion.p1 == entry.p1,
            f"{entry.qid}: read {list(expansion.read)} against the pilot's {list(entry.read)}",
        )

        frozen = recorded.get(entry.qid)
        if frozen is None:
            positives_check.record(False, f"{entry.qid}: not in the hop run")
            for depth in depths:
                hits_check.record(False, f"{entry.qid} @{depth}: not in the hop run")
            continue
        positives = frozen["positives"][ENTITY_HOP]
        positives_check.record(
            expansion.positives == positives,
            f"{entry.qid}: P(q) {expansion.positives} against {positives}",
        )
        for depth in depths:
            ours = _found(expansion, entry.missing, int(depth)) / len(entry.missing)
            theirs = frozen["scores"][ENTITY_HOP][depth]
            hits_check.record(ours == theirs, f"{entry.qid} @{depth}: {ours!r} against {theirs!r}")

    for trace in traces:
        if trace.get("arm") != ENTITY_ARM:
            continue
        qid = str(trace["qid"])
        traced = expansions.get(qid)
        if traced is None:
            traces_check.record(False, f"{qid}: the trace names a question outside the pilot")
            continue
        problems = _trace_disagreements(trace, traced, rel_tol)
        traces_check.record(not problems, f"{qid} {trace.get('unit_id')}: {'; '.join(problems)}")

    return ContinuityReport(
        checks=(read_check.done(), positives_check.done(), hits_check.done(), traces_check.done())
    )
