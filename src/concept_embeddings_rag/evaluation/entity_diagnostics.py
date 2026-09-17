"""Recording wrappers, `EntityHopDiagnostics` and `EntityHopTrace` (Phase 6, HU-8, D17).

Diagnostics and traces explain B's rankings from the lists the run itself produced, so nothing is
retrieved a second time (on test, a second retrieval would be a second read). Three wrappers keep
those lists, in call order, which is the harness's question order:

- `RecordingRetriever` around dense (and BM25): each returned list;
- `RecordingStage` around the entity stage: each `EntityExpansion` - `p1`, the read list, the
  ranked candidates with their shared nodes, `P(q)`;
- `RecordingHybrid` around the fused retriever: each fused list.

Each exposes the inner `name`, delegates unchanged and returns what the inner object returned.
They are used for the selected-configuration runs, never inside a fit.

From the recorded lists and the token counts, per question:

- **Diagnostics**: `p1`, its entity-node count, whether it is a failed extraction, `P(q)`,
  `|E(q)|`, whether it reaches the depth, empty, single candidate, abstained under the selected
  scheme (weighted only; `null` under RRF), bottom-tier size, overlap of `E(q)` with ranks 11-100
  of `D(q)`, and how many units of B's and of A's context at 2,048 lie outside `D(q)`.
- **Traces**: for every unit in B's context at 2,048 that is not in dense's context at 2,048,
  `origin = entity` if the unit is in `E(q)` (its entity rank, raw score and shared nodes,
  heaviest first) and `dense-reorder` otherwise; its fused rank in every case.

The context re-derived from each recorded fused list must reproduce the recorded outcome at
2,048, or the build stops. Both artifacts are bound to the freeze digest, the digests of the runs
they explain, and the node-index and extraction digests, and carry their own digest. They are
descriptive only: nothing in the decision path reads them.
"""

import json
import math
import re
import statistics
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.artifacts import write_text_atomic
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.decision_parameters import DECISION_BUDGET
from concept_embeddings_rag.evaluation.budget import fill_context
from concept_embeddings_rag.evaluation.harness import SAFE_NAME, RunResult
from concept_embeddings_rag.evaluation.metrics import context_precision, full_support, gold_recall
from concept_embeddings_rag.evaluation.outcomes import PerQuestionOutcomes
from concept_embeddings_rag.evaluation.selection import digest_of_payload, serialize_payload
from concept_embeddings_rag.nodes.index import NodeIndex
from concept_embeddings_rag.retrieval.base import Hit, Retriever
from concept_embeddings_rag.retrieval.entity_hop import EntityExpansion, EntityHopStage
from concept_embeddings_rag.retrieval.fusion import WEIGHTED, FusedRetriever

ENTITY_ORIGIN = "entity"
REORDER_ORIGIN = "dense-reorder"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ENTITY_TYPE = config.ENTITY_HOP_TYPES[0]
SYSTEM_A, SYSTEM_B = "hybrid-bm25", "hybrid-entity-hop"


class DiagnosticsError(Exception):
    """A recording, a diagnostic or a trace does not agree with the run it claims to explain."""


# --- Recording wrappers -----------------------------------------------------------------------


class RecordingRetriever:
    """A retriever that keeps a copy of every list it returns."""

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.name = inner.name
        self.calls: list[list[Hit]] = []

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        hits = self.inner.retrieve(query, top_k)
        self.calls.append(list(hits))
        return hits


class RecordingStage:
    """The entity stage, keeping the whole expansion behind every list it proposes."""

    def __init__(self, inner: EntityHopStage) -> None:
        self.inner = inner
        self.name = inner.name
        self.expansions: list[EntityExpansion] = []

    def propose(self, first: Sequence[Hit], top_k: int) -> list[Hit]:
        expansion = self.inner.expand(first, top_k)
        self.expansions.append(expansion)
        return [(candidate.unit_id, candidate.score) for candidate in expansion.candidates]


class RecordingHybrid:
    """A fused retriever over recording components, keeping every fused list."""

    def __init__(
        self,
        dense: Retriever,
        second: Retriever | EntityHopStage,
        *,
        scheme: str,
        weights: Mapping[str, float] | None,
    ) -> None:
        self.dense = RecordingRetriever(dense)
        self.second: RecordingRetriever | RecordingStage = (
            RecordingStage(second)
            if isinstance(second, EntityHopStage)
            else RecordingRetriever(second)
        )
        self.inner = FusedRetriever([self.dense, self.second], scheme=scheme, weights=weights)
        self.name = self.inner.name
        self.scheme = self.inner.scheme
        self.fused: list[list[Hit]] = []

    def describe(self) -> dict:
        return self.inner.describe()

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        hits = self.inner.retrieve(query, top_k)
        self.fused.append(list(hits))
        return hits


@dataclass(frozen=True)
class QuestionRecord:
    qid: str
    dense: tuple[Hit, ...]
    fused: tuple[Hit, ...]
    expansion: EntityExpansion | None


def records_of(hybrid: RecordingHybrid, questions: Sequence[Question]) -> list[QuestionRecord]:
    """The recorded lists, paired with the questions in the harness's order."""
    n = len(questions)
    expansions = hybrid.second.expansions if isinstance(hybrid.second, RecordingStage) else None
    lengths = {len(hybrid.dense.calls), len(hybrid.fused)}
    if expansions is not None:
        lengths.add(len(expansions))
    if lengths != {n}:
        raise DiagnosticsError(
            f"the recording holds {sorted(lengths)} calls for {n} questions; it is not this run's"
        )
    return [
        QuestionRecord(
            qid=question.qid,
            dense=tuple(hybrid.dense.calls[position]),
            fused=tuple(hybrid.fused[position]),
            expansion=None if expansions is None else expansions[position],
        )
        for position, question in enumerate(questions)
    ]


def run_digest(result: RunResult) -> str:
    """A run's digest, over the serialization its file holds (`RunResult` has no digest field)."""
    return digest_of_payload(json.loads(json.dumps(asdict(result), sort_keys=True)))


def run_digest_of_file(path: Path) -> str:
    return digest_of_payload(json.loads(Path(path).read_text(encoding="utf-8")))


# --- Building ---------------------------------------------------------------------------------


def _ids(hits: Sequence[Hit]) -> list[str]:
    return [unit_id for unit_id, _score in hits]


def _check_context(
    label: str,
    context: Sequence[str],
    question: Question,
    outcomes: Mapping[str, Mapping[str, Any]],
    budget: int,
) -> None:
    entry = outcomes.get(question.qid)
    if entry is None:
        raise DiagnosticsError(f"{label}: question {question.qid} has no recorded outcome")
    recorded = entry[f"budget_{budget}"]
    derived = {
        "units_included": len(context),
        "gold_recall": gold_recall(context, question.gold_unit_ids),
        "full_support": float(full_support(context, question.gold_unit_ids)),
        "precision": context_precision(context, question.gold_unit_ids),
    }
    for name, value in derived.items():
        if recorded[name] != value:
            raise DiagnosticsError(
                f"{label}: the context re-derived for {question.qid} gives {name} {value!r}, and "
                f"the recorded outcome says {recorded[name]!r}; the build stops"
            )


def _bindings(
    freeze_digest: str,
    run_digests: Mapping[str, str],
    node_index_digest: str,
    extraction_digest: str,
) -> dict[str, Any]:
    for label, value in (
        ("freeze", freeze_digest),
        ("node index", node_index_digest),
        ("extraction", extraction_digest),
        *((f"run of {system}", digest) for system, digest in run_digests.items()),
    ):
        if not SHA256.match(str(value)):
            raise DiagnosticsError(f"the {label} digest is not a sha256")
    return {
        "freeze_digest": freeze_digest,
        "run_digests": dict(run_digests),
        "node_index_digest": node_index_digest,
        "extraction_digest": extraction_digest,
    }


def build_diagnostics_and_traces(
    *,
    split: str,
    questions: Sequence[Question],
    b_records: Sequence[QuestionRecord],
    a_records: Sequence[QuestionRecord],
    b_outcomes: PerQuestionOutcomes,
    a_outcomes: PerQuestionOutcomes,
    token_counts: Mapping[str, int],
    failed_units: Collection[str],
    b_scheme: str,
    freeze_digest: str,
    run_digests: Mapping[str, str],
    node_index_digest: str,
    extraction_digest: str,
    budget: int = DECISION_BUDGET,
    max_depth: int = config.ENTITY_HOP_MAX_DEPTH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The two payloads for one split, from the recorded lists of B's and A's runs."""
    if not SAFE_NAME.match(split):
        raise DiagnosticsError(f"split {split!r} is not usable in a filename")
    if b_outcomes.system != SYSTEM_B or a_outcomes.system != SYSTEM_A:
        raise DiagnosticsError("the outcomes handed in are not B's and A's")
    if {b_outcomes.split, a_outcomes.split} != {split}:
        raise DiagnosticsError(f"the outcomes handed in are not on {split!r}")
    if set(run_digests) != {SYSTEM_A, SYSTEM_B}:
        raise DiagnosticsError("the run digests must name A's and B's runs")
    bindings = _bindings(freeze_digest, run_digests, node_index_digest, extraction_digest)
    qids = [question.qid for question in questions]
    if [record.qid for record in b_records] != qids or [r.qid for r in a_records] != qids:
        raise DiagnosticsError("the recordings are not in the order of the questions handed in")
    failed = set(failed_units)
    b_by_qid, a_by_qid = b_outcomes.by_qid(), a_outcomes.by_qid()

    per_question: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for question, b, a in zip(questions, b_records, a_records, strict=True):
        expansion = b.expansion
        if expansion is None:
            raise DiagnosticsError(f"B's recording of {question.qid} holds no entity expansion")
        if a.dense != b.dense:
            raise DiagnosticsError(f"A and B did not see the same D(q) for {question.qid}")
        dense_ids = _ids(b.dense)
        if list(expansion.read) != dense_ids[: len(expansion.read)]:
            raise DiagnosticsError(f"the expansion of {question.qid} was not taken from D(q)")

        b_context = fill_context(_ids(b.fused), token_counts, budget)
        a_context = fill_context(_ids(a.fused), token_counts, budget)
        _check_context(SYSTEM_B, b_context, question, b_by_qid, budget)
        _check_context(SYSTEM_A, a_context, question, a_by_qid, budget)

        candidates = expansion.candidates
        scores = [candidate.score for candidate in candidates]
        flat = len(set(scores)) <= 1
        entity_ids = {candidate.unit_id for candidate in candidates}
        dense_set = set(dense_ids)
        per_question.append(
            {
                "qid": question.qid,
                "p1": expansion.p1,
                "p1_entity_nodes": expansion.p1_entity_nodes,
                "p1_failed_extraction": expansion.p1 in failed,
                "positives": expansion.positives,
                "size": len(candidates),
                "reaches_depth": len(candidates) == max_depth,
                "empty": not candidates,
                "single_candidate": len(candidates) == 1,
                "abstained": flat if b_scheme == WEIGHTED else None,
                "bottom_tier": scores.count(min(scores)) if scores else 0,
                "overlap_ranks_11_100": len(entity_ids & set(dense_ids[10:100])),
                "b_context_outside_dense_list": sum(1 for u in b_context if u not in dense_set),
                "a_context_outside_dense_list": sum(1 for u in a_context if u not in dense_set),
            }
        )

        dense_context = set(fill_context(dense_ids, token_counts, budget))
        fused_rank = {unit_id: rank for rank, unit_id in enumerate(_ids(b.fused), start=1)}
        entity_rank = {c.unit_id: (rank, c) for rank, c in enumerate(candidates, start=1)}
        for unit_id in b_context:
            if unit_id in dense_context:
                continue
            ranked = entity_rank.get(unit_id)
            if ranked is None:
                traces.append(
                    {
                        "qid": question.qid,
                        "unit_id": unit_id,
                        "origin": REORDER_ORIGIN,
                        "p1": expansion.p1,
                        "entity_rank": None,
                        "score": None,
                        "fused_rank": fused_rank[unit_id],
                        "nodes": [],
                    }
                )
                continue
            rank, candidate = ranked
            traces.append(
                {
                    "qid": question.qid,
                    "unit_id": unit_id,
                    "origin": ENTITY_ORIGIN,
                    "p1": expansion.p1,
                    "entity_rank": rank,
                    "score": candidate.score,
                    "fused_rank": fused_rank[unit_id],
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

    diagnostics = {
        "split": split,
        "budget": budget,
        "scheme": b_scheme,
        "descriptive": True,
        **bindings,
        "questions": per_question,
        "aggregates": aggregate_diagnostics(per_question, b_scheme),
    }
    trace_payload = {
        "split": split,
        "budget": budget,
        **bindings,
        "n_traces": len(traces),
        "traces": traces,
    }
    return diagnostics, trace_payload


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def aggregate_diagnostics(per_question: Sequence[Mapping[str, Any]], scheme: str) -> dict:
    def count(name: str) -> int:
        return sum(1 for entry in per_question if entry[name])

    return {
        "n_questions": len(per_question),
        "positives": _summary([entry["positives"] for entry in per_question]),
        "size": _summary([entry["size"] for entry in per_question]),
        "reaching_depth": count("reaches_depth"),
        "empty": count("empty"),
        "single_candidate": count("single_candidate"),
        "abstained": count("abstained") if scheme == WEIGHTED else None,
        "bottom_tier": _summary([entry["bottom_tier"] for entry in per_question]),
        "p1_without_entity_node": sum(1 for e in per_question if e["p1_entity_nodes"] == 0),
        "p1_failed_extraction": count("p1_failed_extraction"),
        "overlap_ranks_11_100": _summary([e["overlap_ranks_11_100"] for e in per_question]),
        "b_context_outside_dense_list": _summary(
            [entry["b_context_outside_dense_list"] for entry in per_question]
        ),
        "a_context_outside_dense_list": _summary(
            [entry["a_context_outside_dense_list"] for entry in per_question]
        ),
    }


# --- Writing and loading ----------------------------------------------------------------------


def _indexed(stem: str, split: str, index: int) -> str:
    if not SAFE_NAME.match(split):
        raise DiagnosticsError(f"split {split!r} is not usable in a filename")
    return f"{stem}-{split}.json" if index == 1 else f"{stem}-{split}-{index}.json"


def diagnostics_path(directory: Path | str, split: str, index: int = 1) -> Path:
    """`diagnostics-{split}.json`, or `-{index}` under a superseding freeze (OI-4)."""
    return Path(directory) / _indexed("diagnostics", split, index)


def traces_path(directory: Path | str, split: str, index: int = 1) -> Path:
    return Path(directory) / _indexed("traces", split, index)


def _save_once(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise DiagnosticsError(f"{path} already exists; it is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = {key: value for key, value in payload.items() if key != "digest"}
    sealed["digest"] = digest_of_payload(sealed)
    write_text_atomic(path, serialize_payload(sealed))
    return path


def save_diagnostics_and_traces(
    diagnostics: Mapping[str, Any],
    traces: Mapping[str, Any],
    directory: Path | str,
    index: int = 1,
) -> tuple[Path, Path]:
    split = str(diagnostics["split"])
    return (
        _save_once(diagnostics_path(directory, split, index), diagnostics),
        _save_once(traces_path(directory, split, index), traces),
    )


def _load_bound(
    path: Path,
    *,
    freeze_digest: str,
    run_digests: Mapping[str, str],
    node_index_digest: str,
    extraction_digest: str,
) -> dict[str, Any]:
    if not path.exists():
        raise DiagnosticsError(f"no artifact at {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("digest") != digest_of_payload(payload):
        raise DiagnosticsError(f"{path.name} does not match its digest; it has been modified")
    expected = {
        "freeze_digest": freeze_digest,
        "run_digests": dict(run_digests),
        "node_index_digest": node_index_digest,
        "extraction_digest": extraction_digest,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise DiagnosticsError(f"{path.name} is bound to another {key.replace('_', ' ')}")
    return payload


def load_diagnostics(
    directory: Path | str,
    split: str,
    *,
    freeze_digest: str,
    run_digests: Mapping[str, str],
    node_index_digest: str,
    extraction_digest: str,
) -> dict[str, Any]:
    return _load_bound(
        diagnostics_path(directory, split),
        freeze_digest=freeze_digest,
        run_digests=run_digests,
        node_index_digest=node_index_digest,
        extraction_digest=extraction_digest,
    )


def load_traces(
    directory: Path | str,
    split: str,
    *,
    freeze_digest: str,
    run_digests: Mapping[str, str],
    index: NodeIndex,
    extraction_digest: str,
    weights: np.ndarray | None = None,
    rel_tol: float = config.TRACE_WEIGHT_REL_TOL,
) -> dict[str, Any]:
    """Read traces back: bound to their runs, one per unit, every node resolving in the index."""
    payload = _load_bound(
        traces_path(directory, split),
        freeze_digest=freeze_digest,
        run_digests=run_digests,
        node_index_digest=index.digest,
        extraction_digest=extraction_digest,
    )
    seen: set[tuple[str, str]] = set()
    for trace in payload["traces"]:
        key = (str(trace["qid"]), str(trace["unit_id"]))
        if key in seen:
            raise DiagnosticsError(f"two traces for unit {key[1]} of question {key[0]}")
        seen.add(key)
        if trace["origin"] == REORDER_ORIGIN:
            if trace["nodes"] or trace["score"] is not None or trace["entity_rank"] is not None:
                raise DiagnosticsError(f"a dense-reorder trace of {key[0]} carries an entity score")
            continue
        if trace["origin"] != ENTITY_ORIGIN or not trace["nodes"]:
            raise DiagnosticsError(f"the entity trace of {key} has no shared node")
        for node in trace["nodes"]:
            node_id = int(node["node_id"])
            if not 0 <= node_id < len(index.nodes):
                raise DiagnosticsError(f"node {node_id} of {key} is not in the node index")
            known = index.nodes[node_id]
            if known.type != ENTITY_TYPE or node["type"] != ENTITY_TYPE:
                raise DiagnosticsError(f"node {node_id} of {key} is not an entity node")
            if known.form != node["form"]:
                raise DiagnosticsError(f"node {node_id} of {key} has another form in the index")
            if weights is not None and not math.isclose(
                float(weights[node_id]), float(node["weight"]), rel_tol=rel_tol, abs_tol=0.0
            ):
                raise DiagnosticsError(f"node {node_id} of {key} carries another weight")
        total = math.fsum(float(node["weight"]) for node in trace["nodes"])
        if not math.isclose(total, float(trace["score"]), rel_tol=rel_tol, abs_tol=0.0):
            raise DiagnosticsError(f"the node weights of {key} sum to {total!r}, not its score")
    return payload
