"""Phase 6, T14: recording wrappers, `EntityHopDiagnostics` and `EntityHopTrace` (HU-8, D17).

The toy below is a node index and a set of dense lists built so that each diagnostic is hit on
purpose: an empty entity list, a failed `p1`, a single candidate, a flat list, a bottom tier, an
overlap with ranks 11-100 of `D(q)`, a list that reaches the depth, and - through a unit too large
to fit - both kinds of trace, `entity` and `dense-reorder`. B and A are measured through the
unchanged harness with and without the wrappers, and the wrappers change nothing.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.evaluation.entity_diagnostics import (
    DiagnosticsError,
    RecordingHybrid,
    build_diagnostics_and_traces,
    load_diagnostics,
    load_traces,
    records_of,
    run_digest,
    run_digest_of_file,
    save_diagnostics_and_traces,
)
from concept_embeddings_rag.evaluation.harness import QuestionOutcome, evaluate_retriever
from concept_embeddings_rag.evaluation.outcomes import build_outcomes, load_outcomes, save_outcomes
from concept_embeddings_rag.evaluation.second_hop import node_weights
from concept_embeddings_rag.nodes.extraction import FAILED, OK, PROMPT_DIGEST, ExtractionRecord
from concept_embeddings_rag.nodes.index import build_node_index
from concept_embeddings_rag.retrieval.base import Hit
from concept_embeddings_rag.retrieval.entity_hop import EntityHopStage
from concept_embeddings_rag.retrieval.fusion import FusedRetriever

FILLERS = [f"f{index:02d}" for index in range(9)]
HUB = [f"h{index:03d}" for index in range(105)]
FREEZE = "f" * 64
NODES = "d" * 64
EXTRACTION = "e" * 64


def record(unit_id: str, entities=(), concepts=(), status: str = OK) -> ExtractionRecord:
    ok = status == OK
    return ExtractionRecord(
        unit_id=unit_id,
        model=config.EXTRACTION_MODEL,
        prompt_digest=PROMPT_DIGEST,
        status=status,
        entities=tuple(entities) if ok else (),
        concepts=tuple(concepts) if ok else (),
        failure=None if ok else "invalid_json",
        input_tokens=1,
        output_tokens=1,
    )


SPECIAL = {
    "pe": record("pe", [], ["river"]),
    "pf": record("pf", status=FAILED),
    "ps": record("ps", ["Solo"]),
    "s1": record("s1", ["Solo"]),
    "pl": record("pl", ["Flat"]),
    "fa": record("fa", ["Flat"]),
    "fb": record("fb", ["Flat"]),
    "pa": record("pa", ["Alpha", "Beta"]),
    "t1": record("t1", ["Alpha", "Beta"]),
    "t2": record("t2", ["Alpha"]),
    "t3": record("t3", ["Alpha"]),
    "ph": record("ph", ["Hub"]),
    "pr": record("pr", ["Re"]),
    "r1": record("r1", ["Re"]),
    "dbig": record("dbig"),
    "dsmall": record("dsmall"),
    "x0": record("x0"),
}
UNIT_IDS = sorted({*FILLERS, *HUB, *SPECIAL})


def the_index():
    records = {unit_id: record(unit_id) for unit_id in UNIT_IDS}
    records.update(SPECIAL)
    for unit_id in HUB:
        records[unit_id] = record(unit_id, ["Hub"])
    return build_node_index(records, UNIT_IDS, extraction_digest="toy")


def dense_list(units: Sequence[str]) -> list[Hit]:
    return [(unit, 1.0 - index / 1000) for index, unit in enumerate(units)]


HEADS = {
    "q_empty": "pe",
    "q_failed": "pf",
    "q_single": "ps",
    "q_flat": "pl",
    "q_tier": "pa",
    "q_depth": "ph",
    "q_reorder": "pr",
}
TAILS = {"q_tier": ["x0", "t2"], "q_reorder": ["dbig", "dsmall"]}
DENSE = {qid: dense_list([head, *FILLERS, *TAILS.get(qid, [])]) for qid, head in HEADS.items()}
TOKENS = dict.fromkeys(UNIT_IDS, 100)
TOKENS.update({"dbig": 1040, "dsmall": 40, "r1": 1000})
QUESTIONS = [
    Question(qid, f"question {qid}", "-", (HEADS[qid], FILLERS[0]), (("t", 0),), "dev")
    for qid in HEADS
]


class ScriptedDense:
    name = "dense"

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return DENSE[query.removeprefix("question ")][:top_k]


class ScriptedBM25:
    name = "bm25"

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        return [("x0", 2.0), ("t3", 1.0)]


def a_config() -> dict:
    return {
        "model": "toy",
        "revision": "rev",
        "unit_set_hash": "pool",
        "seed": 42,
        "tokenizer": "toy",
        "code_version": "0.1.0",
        "top_k": 100,
    }


def measure(retriever, records: list[QuestionOutcome] | None = None):
    return evaluate_retriever(
        retriever,
        questions=QUESTIONS,
        token_counts=TOKENS,
        budgets=config.CONTEXT_BUDGETS,
        ks=config.RECALL_AT_K,
        top_k=100,
        config=a_config(),
        split="dev",
        outcomes=records,
    )


def stage() -> EntityHopStage:
    index = the_index()
    return EntityHopStage(index, node_weights(index))


def weights_for(scheme: str, second: str):
    return None if scheme == "rrf" else {"dense": 0.5, second: 0.5}


def a_run(tmp_path: Path, scheme: str = "rrf"):
    """B and A through recording wrappers, their outcomes on disk, and their records."""
    b_hybrid = RecordingHybrid(
        ScriptedDense(), stage(), scheme=scheme, weights=weights_for(scheme, "entity-hop")
    )
    a_hybrid = RecordingHybrid(
        ScriptedDense(), ScriptedBM25(), scheme="weighted", weights={"dense": 0.5, "bm25": 0.5}
    )
    order = [question.qid for question in QUESTIONS]
    loaded = {}
    results = {}
    for hybrid in (b_hybrid, a_hybrid):
        records: list[QuestionOutcome] = []
        result = measure(hybrid, records)
        path = result.save(tmp_path)
        payload = build_outcomes(
            result, records, run_file=path.name, split_order=order, freeze_digest=None
        )
        loaded[hybrid.name] = load_outcomes(
            save_outcomes(payload, tmp_path), split_order=order, run_dir=tmp_path
        )
        results[hybrid.name] = (result, path)
    return b_hybrid, a_hybrid, loaded, results


def build(tmp_path: Path, scheme: str = "rrf", b_records=None):
    b_hybrid, a_hybrid, loaded, results = a_run(tmp_path, scheme)
    return build_diagnostics_and_traces(
        split="dev",
        questions=QUESTIONS,
        b_records=b_records or records_of(b_hybrid, QUESTIONS),
        a_records=records_of(a_hybrid, QUESTIONS),
        b_outcomes=loaded["hybrid-entity-hop"],
        a_outcomes=loaded["hybrid-bm25"],
        token_counts=TOKENS,
        failed_units={"pf"},
        b_scheme=scheme,
        freeze_digest=FREEZE,
        run_digests={
            "hybrid-entity-hop": run_digest(results["hybrid-entity-hop"][0]),
            "hybrid-bm25": run_digest(results["hybrid-bm25"][0]),
        },
        node_index_digest=NODES,
        extraction_digest=EXTRACTION,
    ), results


def diagnostics_of(tmp_path: Path, scheme: str = "rrf") -> dict:
    return build(tmp_path, scheme)[0][0]


def by_qid(diagnostics) -> dict:
    return {entry["qid"]: entry for entry in diagnostics["questions"]}


# --- The wrappers change nothing --------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["rrf", "weighted"])
def test_the_same_run_with_and_without_wrappers_gives_identical_outcomes(scheme):
    plain = FusedRetriever(
        [ScriptedDense(), stage()], scheme=scheme, weights=weights_for(scheme, "entity-hop")
    )
    wrapped = RecordingHybrid(
        ScriptedDense(), stage(), scheme=scheme, weights=weights_for(scheme, "entity-hop")
    )
    plain_records: list[QuestionOutcome] = []
    wrapped_records: list[QuestionOutcome] = []
    plain_result = measure(plain, plain_records)
    wrapped_result = measure(wrapped, wrapped_records)

    assert wrapped.name == plain.name == "hybrid-entity-hop"
    assert wrapped_records == plain_records
    assert wrapped_result.metrics == plain_result.metrics
    assert wrapped_result.cost["mean_units_included"] == plain_result.cost["mean_units_included"]
    assert [list(hits) for hits in wrapped.fused] == [
        plain.retrieve(question.question, 100) for question in QUESTIONS
    ]


def test_the_recordings_are_in_the_harness_question_order():
    wrapped = RecordingHybrid(ScriptedDense(), stage(), scheme="rrf", weights=None)
    measure(wrapped)
    records = records_of(wrapped, QUESTIONS)
    assert [r.qid for r in records] == [q.qid for q in QUESTIONS]
    assert records[0].dense == tuple(DENSE["q_empty"])
    assert records[4].expansion is not None and records[4].expansion.p1 == "pa"


def test_a_recording_of_another_length_is_refused():
    wrapped = RecordingHybrid(ScriptedDense(), stage(), scheme="rrf", weights=None)
    measure(wrapped)
    with pytest.raises(DiagnosticsError, match="calls"):
        records_of(wrapped, QUESTIONS[:-1])


# --- Each diagnostic, on a question built to hit it -------------------------------------------


def test_an_empty_list_and_a_p1_with_no_entity_node(tmp_path):
    entry = by_qid(diagnostics_of(tmp_path))["q_empty"]
    assert entry["empty"] is True and entry["size"] == 0 and entry["positives"] == 0
    assert entry["p1_entity_nodes"] == 0
    assert entry["p1_failed_extraction"] is False


def test_a_failed_p1(tmp_path):
    entry = by_qid(diagnostics_of(tmp_path))["q_failed"]
    assert entry["p1_failed_extraction"] is True
    assert entry["empty"] is True and entry["p1_entity_nodes"] == 0


def test_a_single_candidate_abstains_under_weighted_and_is_null_under_rrf(tmp_path):
    weighted = by_qid(diagnostics_of(tmp_path / "w", "weighted"))["q_single"]
    rrf = by_qid(diagnostics_of(tmp_path / "r", "rrf"))["q_single"]
    assert weighted["single_candidate"] is True and weighted["abstained"] is True
    assert rrf["single_candidate"] is True and rrf["abstained"] is None


def test_a_flat_list_abstains_under_weighted(tmp_path):
    entry = by_qid(diagnostics_of(tmp_path, "weighted"))["q_flat"]
    assert entry["size"] == 2 and entry["abstained"] is True and entry["bottom_tier"] == 2


def test_a_list_with_a_range_does_not_abstain_and_reports_its_bottom_tier(tmp_path):
    entry = by_qid(diagnostics_of(tmp_path, "weighted"))["q_tier"]
    assert entry["size"] == 3
    assert entry["abstained"] is False
    assert entry["bottom_tier"] == 2


def test_the_overlap_with_ranks_11_to_100_of_the_dense_list(tmp_path):
    diagnostics = by_qid(diagnostics_of(tmp_path))
    assert diagnostics["q_tier"]["overlap_ranks_11_100"] == 1
    assert diagnostics["q_single"]["overlap_ranks_11_100"] == 0


def test_a_list_that_reaches_the_depth(tmp_path):
    entry = by_qid(diagnostics_of(tmp_path))["q_depth"]
    assert entry["positives"] == 105
    assert entry["size"] == 100
    assert entry["reaches_depth"] is True


def test_context_units_outside_the_dense_list_for_b_and_for_a(tmp_path):
    diagnostics = by_qid(diagnostics_of(tmp_path))
    assert diagnostics["q_reorder"]["b_context_outside_dense_list"] == 1  # r1
    assert diagnostics["q_empty"]["b_context_outside_dense_list"] == 0
    assert diagnostics["q_empty"]["a_context_outside_dense_list"] == 2  # BM25's x0 and t3
    assert diagnostics["q_tier"]["a_context_outside_dense_list"] == 1  # t3; t2 is in D(q)


def test_the_aggregates_count_the_diagnostics(tmp_path):
    aggregates = diagnostics_of(tmp_path, "weighted")["aggregates"]
    assert aggregates["n_questions"] == len(QUESTIONS)
    assert aggregates["empty"] == 2
    assert aggregates["single_candidate"] == 2  # q_single and q_reorder
    assert aggregates["reaching_depth"] == 1
    assert aggregates["p1_failed_extraction"] == 1
    assert aggregates["p1_without_entity_node"] == 2
    # empty, failed, single, flat, reorder (one candidate), depth (105 hub units, one score)
    assert aggregates["abstained"] == 6
    assert diagnostics_of(tmp_path / "rrf", "rrf")["aggregates"]["abstained"] is None


# --- Traces -----------------------------------------------------------------------------------


def test_one_trace_per_unit_in_bs_context_and_not_in_denses(tmp_path):
    from concept_embeddings_rag.evaluation.budget import fill_context

    b_hybrid, *_ = a_run(tmp_path / "lists")
    (_diagnostics, traces), _ = build(tmp_path / "build")
    expected = set()
    for question, fused in zip(QUESTIONS, b_hybrid.fused, strict=True):
        b_context = fill_context([unit for unit, _ in fused], TOKENS, 2048)
        dense_context = set(fill_context([u for u, _ in DENSE[question.qid]], TOKENS, 2048))
        expected |= {(question.qid, unit) for unit in b_context if unit not in dense_context}
    keys = [(trace["qid"], trace["unit_id"]) for trace in traces["traces"]]
    assert len(keys) == len(set(keys))
    assert set(keys) == expected
    assert expected, "the toy must produce traces"


def test_entity_and_dense_reorder_traces_on_a_toy_with_both(tmp_path):
    b_hybrid, a_hybrid, loaded, results = a_run(tmp_path)
    _, traces = build_diagnostics_and_traces(
        split="dev",
        questions=QUESTIONS,
        b_records=records_of(b_hybrid, QUESTIONS),
        a_records=records_of(a_hybrid, QUESTIONS),
        b_outcomes=loaded["hybrid-entity-hop"],
        a_outcomes=loaded["hybrid-bm25"],
        token_counts=TOKENS,
        failed_units={"pf"},
        b_scheme="rrf",
        freeze_digest=FREEZE,
        run_digests={name: run_digest(result) for name, (result, _) in results.items()},
        node_index_digest=NODES,
        extraction_digest=EXTRACTION,
    )
    reorder = {(t["qid"], t["unit_id"]): t for t in traces["traces"]}
    assert reorder[("q_reorder", "dsmall")]["origin"] == "dense-reorder"
    assert reorder[("q_reorder", "dsmall")]["nodes"] == []
    assert reorder[("q_reorder", "dsmall")]["score"] is None
    entity = reorder[("q_reorder", "r1")]
    assert entity["origin"] == "entity" and entity["entity_rank"] == 1 and entity["p1"] == "pr"
    assert [node["form"] for node in entity["nodes"]] == ["re"]
    assert math.fsum(node["weight"] for node in entity["nodes"]) == pytest.approx(entity["score"])
    assert len(reorder) == len(traces["traces"])

    fused = dict(zip(QUESTIONS, b_hybrid.fused, strict=True))
    for trace in traces["traces"]:
        question = next(q for q in QUESTIONS if q.qid == trace["qid"])
        ranking = [unit for unit, _ in fused[question]]
        assert ranking.index(trace["unit_id"]) + 1 == trace["fused_rank"]


def test_a_recorded_context_that_disagrees_with_the_recorded_outcome_stops_the_build(tmp_path):
    b_hybrid, *_ = a_run(tmp_path / "records")
    records = records_of(b_hybrid, QUESTIONS)
    tampered = [
        replace(record, fused=tuple(reversed(record.fused)))
        if record.qid == "q_reorder"
        else record
        for record in records
    ]
    with pytest.raises(DiagnosticsError, match="stops"):
        build(tmp_path / "build", b_records=tampered)


# --- Written once, verified on load -----------------------------------------------------------


def saved(tmp_path: Path):
    (diagnostics, traces), results = build(tmp_path)
    save_diagnostics_and_traces(diagnostics, traces, tmp_path)
    digests = {name: run_digest_of_file(path) for name, (_result, path) in results.items()}
    return diagnostics, traces, digests


def test_the_run_digest_is_recomputed_from_the_file(tmp_path):
    (_d, _t), results = build(tmp_path)
    for result, path in results.values():
        assert run_digest(result) == run_digest_of_file(path)


def test_diagnostics_and_traces_round_trip_bound_to_their_runs(tmp_path):
    diagnostics, traces, digests = saved(tmp_path)
    index = the_index()
    loaded = load_diagnostics(
        tmp_path,
        "dev",
        freeze_digest=FREEZE,
        run_digests=digests,
        node_index_digest=NODES,
        extraction_digest=EXTRACTION,
    )
    assert loaded["questions"] == diagnostics["questions"]
    loaded_traces = load_traces(
        tmp_path,
        "dev",
        freeze_digest=FREEZE,
        run_digests=digests,
        index=replace(index, digest=NODES),
        extraction_digest=EXTRACTION,
        weights=node_weights(index),
    )
    assert loaded_traces["traces"] == traces["traces"]
    with pytest.raises(DiagnosticsError, match="already"):
        save_diagnostics_and_traces(diagnostics, traces, tmp_path)


def test_a_wrong_freeze_or_run_digest_is_refused(tmp_path):
    _, _, digests = saved(tmp_path)
    with pytest.raises(DiagnosticsError, match="freeze"):
        load_diagnostics(
            tmp_path,
            "dev",
            freeze_digest="0" * 64,
            run_digests=digests,
            node_index_digest=NODES,
            extraction_digest=EXTRACTION,
        )
    wrong = {**digests, "hybrid-entity-hop": "0" * 64}
    with pytest.raises(DiagnosticsError, match="run"):
        load_traces(
            tmp_path,
            "dev",
            freeze_digest=FREEZE,
            run_digests=wrong,
            index=replace(the_index(), digest=NODES),
            extraction_digest=EXTRACTION,
        )


def reseal(path: Path, change) -> None:
    from concept_embeddings_rag.evaluation.selection import digest_of_payload

    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    payload["digest"] = digest_of_payload(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def first_entity(payload) -> dict:
    return next(trace for trace in payload["traces"] if trace["origin"] == "entity")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: first_entity(p)["nodes"][0].update({"node_id": 10**6}), "not in the node index"),
        (lambda p: first_entity(p)["nodes"][0].update({"type": "concept"}), "entity node"),
        (lambda p: first_entity(p).update({"score": first_entity(p)["score"] * 1.001}), "sum"),
        (lambda p: p["traces"].append(dict(p["traces"][0])), "two traces"),
    ],
)
def test_a_trace_that_does_not_resolve_or_sum_is_refused_on_load(tmp_path, change, message):
    _, _, digests = saved(tmp_path)
    reseal(tmp_path / "traces-dev.json", change)
    with pytest.raises(DiagnosticsError, match=message):
        load_traces(
            tmp_path,
            "dev",
            freeze_digest=FREEZE,
            run_digests=digests,
            index=replace(the_index(), digest=NODES),
            extraction_digest=EXTRACTION,
        )


def test_a_trace_node_of_type_concept_in_the_index_is_refused(tmp_path):
    _, _, digests = saved(tmp_path)
    index = the_index()
    concept = next(node for node in index.nodes if node.type == "concept")

    def point_at_concept(payload):
        node = first_entity(payload)["nodes"][0]
        node.update({"node_id": concept.node_id})

    reseal(tmp_path / "traces-dev.json", point_at_concept)
    with pytest.raises(DiagnosticsError, match="entity node"):
        load_traces(
            tmp_path,
            "dev",
            freeze_digest=FREEZE,
            run_digests=digests,
            index=replace(index, digest=NODES),
            extraction_digest=EXTRACTION,
        )
