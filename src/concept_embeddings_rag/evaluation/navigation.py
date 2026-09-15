"""The navigation run of Phase 5: every second hop, over the frozen pilot, in one pass.

HU-5 and HU-6 of the Phase 5 spec, decision D10 of its plan. For each pilot question the run
recomputes dense's order and checks it against the read list the pilot froze - a pilot that
no longer matches the embeddings is refused, not silently re-derived - and then asks every
hop where a hop from `p1` leads:

- the four hops of the diagnostic that owe nothing to a concept space,
- the pooled-embedding concept hop at the space Phase 3 selected, as a reference,
- and the three text-derived arms: entity-only, concept-only, entity + concept.

What counts as found is decided here, and only here, from the pilot's missing paragraphs. A
question's score at a depth is the share of its missing paragraphs within that depth; rates
are reported per question, the statistical unit of the gate, and per paragraph, for
continuity with the diagnostic.

The run is written once, with a digest, and its traces are written beside it bound to that
digest: a trace read against another run explains a ranking that was never produced.
"""

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation import second_hop
from concept_embeddings_rag.evaluation.pilot import PilotSet, QueryEncoder, check_pilot_against
from concept_embeddings_rag.evaluation.second_hop import (
    BASELINE_HOPS,
    BM25_FOLLOW,
    BM25_QUESTION,
    DENSE_CONTINUE,
    DENSE_NEIGHBOURS,
    TextRanker,
)
from concept_embeddings_rag.evaluation.selection import (
    check_dev_only,
    digest_of_payload,
    serialize_payload,
)
from concept_embeddings_rag.nodes.index import NodeIndex

REFERENCE_HOP = f"concept-hop(p1) k={config.REFERENCE_CONCEPT_K}"
ARM_HOPS: dict[str, str] = {arm: f"node-hop(p1) {arm}" for arm in config.NAVIGATION_ARMS}
ARM_TYPES: dict[str, tuple[str, ...]] = {
    "entity": ("entity",),
    "concept": ("concept",),
    "entity+concept": ("entity", "concept"),
}
HOPS: tuple[str, ...] = (*BASELINE_HOPS, REFERENCE_HOP, *ARM_HOPS.values())

HOP_RUN_FILENAME = "hop-run.json"
TRACES_FILENAME = "traces.json"


class NavigationError(Exception):
    """The navigation run or its traces are missing, modified, or inconsistent."""


@dataclass(frozen=True)
class NavigationRun:
    payload: dict[str, Any]
    traces: list[dict[str, Any]]


def _found(missing_rows: Sequence[int], ranking: Sequence[int], depth: int) -> int:
    top = set(ranking[:depth])
    return sum(1 for row in missing_rows if row in top)


def run_navigation(
    pilot: PilotSet,
    questions_by_id: Mapping[str, Question],
    *,
    unit_ids: Sequence[str],
    texts: Sequence[str],
    vectors: np.ndarray,
    query_backend: QueryEncoder,
    bm25: TextRanker,
    index: NodeIndex,
    concept_matrix: sparse.csr_matrix,
    concept_weights: np.ndarray,
    provenance: Mapping[str, Any],
    depths: Sequence[int] = config.SECOND_HOP_DEPTHS,
) -> NavigationRun:
    """Run every hop for every pilot question, and record what each found and how."""
    check_pilot_against(pilot, list(questions_by_id.values()))
    check_dev_only([questions_by_id[entry.qid] for entry in pilot.questions])
    if tuple(index.unit_ids) != tuple(unit_ids):
        raise NavigationError("the node index was built over another pool than this run's")
    if len(texts) != len(unit_ids) or vectors.shape[0] != len(unit_ids):
        raise NavigationError("the texts, embeddings and unit ids describe different pools")

    depths = tuple(sorted(depths))
    shallow, deepest = depths[0], depths[-1]
    row_of = {unit_id: row for row, unit_id in enumerate(unit_ids)}
    node_weights = second_hop.node_weights(index)

    per_question: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    found_paragraphs = {hop: dict.fromkeys(depths, 0) for hop in HOPS}

    for entry in pilot.questions:
        question = questions_by_id[entry.qid]
        order = second_hop.dense_order(vectors, query_backend.encode([question.question])[0])
        read = second_hop.read_rows(order, pilot.read_depth)
        if [unit_ids[row] for row in read] != list(entry.read):
            raise NavigationError(
                f"question {entry.qid}: the read list recomputed from the embeddings differs "
                "from the one the pilot froze; the pilot no longer describes these inputs"
            )
        read_set = set(read)
        p1 = second_hop.origin(read)
        missing_rows = [row_of[unit_id] for unit_id in entry.missing]

        rankings: dict[str, list[int]] = {
            DENSE_CONTINUE: second_hop.dense_continue(
                order, read_depth=pilot.read_depth, depth=deepest
            ),
            BM25_QUESTION: second_hop.rank_bm25(
                bm25,
                row_of,
                question.question,
                read_set,
                read_depth=pilot.read_depth,
                depth=deepest,
            ),
            DENSE_NEIGHBOURS: second_hop.dense_neighbours(vectors, p1, read_set, depth=deepest),
            BM25_FOLLOW: second_hop.rank_bm25(
                bm25, row_of, texts[p1], read_set, read_depth=pilot.read_depth, depth=deepest
            ),
        }
        rankings[REFERENCE_HOP], _reach = second_hop.concept_hop(
            concept_matrix, concept_weights, p1=p1, read=read_set, depth=deepest
        )

        positives: dict[str, int] = {}
        missing = set(entry.missing)
        for arm, hop in ARM_HOPS.items():
            candidates, count = second_hop.node_hop(
                index, node_weights, types=ARM_TYPES[arm], p1=p1, read=read_set, depth=deepest
            )
            rankings[hop] = [candidate.row for candidate in candidates]
            positives[hop] = count
            for rank, candidate in enumerate(candidates, start=1):
                if candidate.unit_id in missing:
                    traces.append(
                        {
                            "qid": entry.qid,
                            "hop": hop,
                            "arm": arm,
                            "unit_id": candidate.unit_id,
                            "p1": entry.p1,
                            "rank": rank,
                            "score": candidate.score,
                            "nodes": [
                                {
                                    "node_id": node.node_id,
                                    "type": node.type,
                                    "form": node.form,
                                    "weight": node.weight,
                                }
                                for node in candidate.nodes
                            ],
                        }
                    )

        scores: dict[str, dict[str, float]] = {}
        for hop in HOPS:
            scores[hop] = {}
            for depth in depths:
                found = _found(missing_rows, rankings[hop], depth)
                found_paragraphs[hop][depth] += found
                scores[hop][str(depth)] = found / len(missing_rows)

        dense_reach = read_set | set(rankings[DENSE_CONTINUE][:deepest])
        bm25_reach = set(rankings[BM25_QUESTION][:deepest])
        new_relevant = {
            hop: [
                unit_ids[row]
                for row in missing_rows
                if row in set(rankings[hop][:shallow])
                and row not in dense_reach
                and row not in bm25_reach
            ]
            for hop in ARM_HOPS.values()
        }

        per_question.append(
            {
                "qid": entry.qid,
                "p1": entry.p1,
                "n_missing": len(missing_rows),
                "exactly_one_gold_read": len(question.gold_unit_ids) - len(missing_rows) == 1,
                "scores": scores,
                "positives": positives,
                "new_relevant": new_relevant,
            }
        )

    n_missing = sum(entry["n_missing"] for entry in per_question)
    subgroup = [entry for entry in per_question if entry["exactly_one_gold_read"]]
    aggregates: dict[str, dict[str, Any]] = {}
    for hop in HOPS:
        figures: dict[str, Any] = {
            "per_question": {
                str(depth): float(
                    statistics.fmean(entry["scores"][hop][str(depth)] for entry in per_question)
                )
                for depth in depths
            },
            "per_paragraph": {
                str(depth): found_paragraphs[hop][depth] / n_missing for depth in depths
            },
            "subgroup_per_question": (
                {
                    str(depth): float(
                        statistics.fmean(entry["scores"][hop][str(depth)] for entry in subgroup)
                    )
                    for depth in depths
                }
                if subgroup
                else None
            ),
        }
        if hop in ARM_HOPS.values():
            counts = [entry["positives"][hop] for entry in per_question]
            figures["new_relevant"] = sum(len(entry["new_relevant"][hop]) for entry in per_question)
            figures["positive_candidates"] = {
                "mean": float(statistics.fmean(counts)),
                "median": float(statistics.median(counts)),
                "max": int(max(counts)),
                "mean_share_of_pool": float(statistics.fmean(counts)) / len(unit_ids),
            }
        aggregates[hop] = figures

    payload = {
        "provenance": {
            **dict(provenance),
            "read_depth": pilot.read_depth,
            "depths": list(depths),
            "hops": list(HOPS),
            "n_questions": len(per_question),
            "n_missing": n_missing,
            "n_subgroup": len(subgroup),
            "n_units": len(unit_ids),
            "split": pilot.split,
        },
        "per_question": per_question,
        "aggregates": aggregates,
    }
    return NavigationRun(payload=payload, traces=traces)


def save_navigation(run: NavigationRun, directory: Path | str) -> tuple[Path, Path]:
    """Write the run and its traces once. The gate is computed from this run and no other."""
    directory = Path(directory)
    run_path = directory / HOP_RUN_FILENAME
    traces_path = directory / TRACES_FILENAME
    if run_path.exists() or traces_path.exists():
        raise NavigationError(
            f"a navigation run is already written in {directory}; this phase measures once, and "
            "the gate may not be re-run on a second measurement"
        )
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(run.payload)
    payload["digest"] = digest_of_payload(payload)
    trace_payload: dict[str, Any] = {"hop_run_digest": payload["digest"], "traces": run.traces}
    trace_payload["digest"] = digest_of_payload(trace_payload)
    write_text_atomic(traces_path, serialize_payload(trace_payload))
    write_text_atomic(run_path, serialize_payload(payload))
    return run_path, traces_path


def load_hop_run(directory: Path | str) -> dict[str, Any]:
    path = Path(directory) / HOP_RUN_FILENAME
    if not path.exists():
        raise NavigationError(f"no navigation run in {directory}: run `cer navigate`")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("digest") != digest_of_payload(payload):
        raise NavigationError(f"{path.name} does not match its digest; it has been modified")
    return payload


def load_traces(directory: Path | str, *, hop_run_digest: str) -> list[dict[str, Any]]:
    path = Path(directory) / TRACES_FILENAME
    if not path.exists():
        raise NavigationError(f"no traces in {directory}: run `cer navigate`")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("digest") != digest_of_payload(payload):
        raise NavigationError(f"{path.name} does not match its digest; it has been modified")
    if payload.get("hop_run_digest") != hop_run_digest:
        raise NavigationError(
            f"{path.name} explains another run than the one asked for; refusing to read it"
        )
    traces: list[dict[str, Any]] = payload["traces"]
    return traces


def diagnostic_mismatches(hop_run: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> list[str]:
    """HU-5's continuity check: the non-concept hops against the Phase 4 diagnostic.

    Both are rates over the same missing paragraphs, so they must agree exactly. The
    pooled-embedding reference is compared too, against the diagnostic's all-concepts row.
    """
    pairs = [(hop, hop) for hop in BASELINE_HOPS]
    pairs.append((REFERENCE_HOP, f"{REFERENCE_HOP}/m=all"))
    mismatches = []
    for ours, theirs in pairs:
        for depth in hop_run["provenance"]["depths"]:
            mine = hop_run["aggregates"][ours]["per_paragraph"][str(depth)]
            recorded = diagnostic["hit_rates"][theirs][f"all@{depth}"]
            if mine != recorded:
                mismatches.append(f"{ours} @{depth}: {mine!r} here, {recorded!r} in the diagnostic")
    return mismatches
