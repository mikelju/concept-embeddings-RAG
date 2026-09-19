"""Phase 6, T8: continuity with Phase 5, checked from the frozen artifacts (HU-3, D11).

The entity stage is Phase 5's hop, and that is shown rather than assumed: for every pilot
question, `D(q)` from the dense retriever and `E(q)` from the stage must reproduce what the
frozen pilot, hop run and traces recorded, without re-running Phase 5:

(a) `read(q)` equals the pilot's read list, so `p1` equals its `p1`;
(b) `P(q)` equals the hop run's `positives` for the entity arm;
(c) hit@10 and hit@100 over the pilot's missing paragraphs equal the entity arm's `scores`,
    compared with `==`;
(d) every entity trace is found at its rank with the same unit and nodes, its score and node
    weights within the OI-2 tolerance (`math.fsum`, relative 1e-12).

Each check fails on its own when the value it reads is altered. On the real pilot the function
is only run here; its outcome is the T21 checkpoint, not an assertion of this test.
"""

import copy
from dataclasses import replace

import pytest
from replacement_fixtures import build_toy_pipeline, toy_counter

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation.continuity import (
    CHECK_NAMES,
    ContinuityReport,
    check_continuity,
)
from concept_embeddings_rag.evaluation.navigation import ARM_HOPS
from concept_embeddings_rag.evaluation.replacement_inputs import (
    InputPaths,
    InputPins,
    load_verified_inputs,
    offline_token_counter,
)
from concept_embeddings_rag.evaluation.second_hop import node_weights
from concept_embeddings_rag.retrieval.dense import DenseRetriever
from concept_embeddings_rag.retrieval.entity_hop import EntityHopStage

ENTITY = ARM_HOPS["entity"]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    pipeline = build_toy_pipeline(tmp_path_factory.mktemp("continuity"))
    verified = load_verified_inputs(
        pipeline.paths, pins=pipeline.pins, backend=pipeline.backend, token_counter=toy_counter
    )
    dense = DenseRetriever(verified.vectors, verified.unit_ids, verified.dev_query_backend)
    stage = EntityHopStage(verified.node_index, node_weights(verified.node_index))
    texts = {question.qid: question.question for question in verified.dev_questions}
    return verified, dense, stage, texts


def run(toy, *, pilot=None, hop_run=None, traces=None) -> ContinuityReport:
    verified, dense, stage, texts = toy
    return check_continuity(
        verified.pilot if pilot is None else pilot,
        verified.hop_run if hop_run is None else hop_run,
        verified.navigation_traces if traces is None else traces,
        dense=dense,
        stage=stage,
        question_text=texts,
    )


def outcomes(report: ContinuityReport) -> dict[str, bool]:
    return {check.name: check.passed for check in report.checks}


def failing_only(report: ContinuityReport, name: str) -> bool:
    return outcomes(report) == {other: other != name for other in CHECK_NAMES}


def entity_traces(toy) -> list[dict]:
    return [trace for trace in toy[0].navigation_traces if trace["arm"] == "entity"]


# --- All four pass on a consistent pipeline ---------------------------------------------------


def test_the_four_checks_pass_on_a_consistent_toy_pipeline(toy):
    report = run(toy)
    verified = toy[0]

    assert [check.name for check in report.checks] == list(CHECK_NAMES)
    assert report.passed
    n_pilot = len(verified.pilot.questions)
    by_name = {check.name: check for check in report.checks}
    assert by_name["a_read"].total == n_pilot
    assert by_name["b_positives"].total == n_pilot
    assert by_name["c_hits"].total == 2 * n_pilot
    assert by_name["d_traces"].total == len(entity_traces(toy)) > 0
    for check in report.checks:
        assert check.agreements == check.total
        assert check.mismatches == ()


def test_the_report_is_json_ready_with_counts_and_outcomes(toy):
    payload = run(toy).as_payload()
    assert payload["passed"] is True
    assert [entry["name"] for entry in payload["checks"]] == list(CHECK_NAMES)
    for entry in payload["checks"]:
        assert set(entry) == {"name", "agreements", "total", "n_mismatches", "mismatches", "passed"}


# --- Each check fails on its own --------------------------------------------------------------


def test_an_altered_read_list_fails_check_a_alone(toy):
    pilot = toy[0].pilot
    first = pilot.questions[0]
    swapped = replace(first, read=(first.read[1], first.read[0], *first.read[2:]))
    altered = replace(pilot, questions=(swapped, *pilot.questions[1:]))

    report = run(toy, pilot=altered)
    assert failing_only(report, "a_read")


def test_altered_positives_fail_check_b_alone(toy):
    hop_run = copy.deepcopy(toy[0].hop_run)
    hop_run["per_question"][0]["positives"][ENTITY] += 1

    assert failing_only(run(toy, hop_run=hop_run), "b_positives")


@pytest.mark.parametrize("depth", ["10", "100"])
def test_an_altered_score_fails_check_c_alone(toy, depth):
    hop_run = copy.deepcopy(toy[0].hop_run)
    scores = hop_run["per_question"][0]["scores"][ENTITY]
    scores[depth] = scores[depth] + 1e-15 if scores[depth] < 1.0 else scores[depth] - 1e-15

    assert failing_only(run(toy, hop_run=hop_run), "c_hits")


def altered_trace(toy, change) -> list[dict]:
    traces = copy.deepcopy(toy[0].navigation_traces)
    target = next(trace for trace in traces if trace["arm"] == "entity")
    change(target)
    return traces


def test_an_altered_trace_rank_fails_check_d_alone(toy):
    traces = altered_trace(toy, lambda trace: trace.update({"rank": trace["rank"] + 1}))
    assert failing_only(run(toy, traces=traces), "d_traces")


def test_an_altered_trace_score_beyond_the_tolerance_fails_check_d_alone(toy):
    traces = altered_trace(toy, lambda trace: trace.update({"score": trace["score"] * (1 + 1e-9)}))
    assert failing_only(run(toy, traces=traces), "d_traces")


def test_a_trace_score_within_the_tolerance_passes(toy):
    traces = altered_trace(toy, lambda trace: trace.update({"score": trace["score"] * (1 + 1e-14)}))
    assert run(toy, traces=traces).passed


def test_an_altered_trace_node_fails_check_d_alone(toy):
    def rename(trace):
        trace["nodes"][0]["form"] = trace["nodes"][0]["form"] + " altered"

    assert failing_only(run(toy, traces=altered_trace(toy, rename)), "d_traces")


def test_an_altered_trace_node_weight_fails_check_d_alone(toy):
    def reweigh(trace):
        trace["nodes"][0]["weight"] = trace["nodes"][0]["weight"] * (1 + 1e-9)

    assert failing_only(run(toy, traces=altered_trace(toy, reweigh)), "d_traces")


def test_a_trace_of_a_question_outside_the_pilot_fails_check_d(toy):
    traces = altered_trace(toy, lambda trace: trace.update({"qid": "f" * 24}))
    assert not outcomes(run(toy, traces=traces))["d_traces"]


def test_the_scores_are_compared_exactly_and_the_tolerance_is_the_decided_one():
    assert config.TRACE_WEIGHT_REL_TOL == 1e-12


def test_the_continuity_module_reads_no_gold_annotation():
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "concept_embeddings_rag"
        / "evaluation"
        / "continuity.py"
    ).read_text(encoding="utf-8")
    names = {
        node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
        for node in ast.walk(ast.parse(source))
    }
    assert "gold_unit_ids" not in names
    assert "supporting_facts" not in names


# --- The real pilot (data-dependent, dev only; outcome not asserted) ----------------------------


def test_the_function_runs_on_the_real_dev_pilot():
    paths = InputPaths.from_config()
    if not (paths.data_dir / "pool.json").exists() or not paths.nodes_dir.exists():
        pytest.skip("the real pipeline artifacts are not on disk")
    from concept_embeddings_rag.embeddings.backend import SentenceTransformerBackend

    try:
        counter = offline_token_counter()
        counter([])
    except OSError as error:
        pytest.skip(f"the pinned tokenizer is not in the local cache: {error}")
    verified = load_verified_inputs(
        paths,
        pins=InputPins.from_config(),
        backend=SentenceTransformerBackend(),
        token_counter=counter,
    )
    dense = DenseRetriever(verified.vectors, verified.unit_ids, verified.dev_query_backend)
    stage = EntityHopStage(verified.node_index, node_weights(verified.node_index))
    report = check_continuity(
        verified.pilot,
        verified.hop_run,
        verified.navigation_traces,
        dense=dense,
        stage=stage,
        question_text={q.qid: q.question for q in verified.dev_questions},
    )

    assert [check.name for check in report.checks] == list(CHECK_NAMES)
