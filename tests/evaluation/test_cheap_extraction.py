"""Phase 7 dev measurement, the selection rule and the single held-out run.

The dev half runs real indexing, fusion fitting and measurement on a small bridge
corpus; the rule is exercised against dev readings written by hand, so a bar can be
landed on exactly rather than approached by luck.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.cache import unit_set_hash
from concept_embeddings_rag.evaluation.cheap_extraction import (
    CheapEvaluationError,
    evaluate_candidate,
    measure_held_out,
    retention_bar,
    select_extractor,
)
from concept_embeddings_rag.evaluation.selection import SelectionError, digest_of_payload
from concept_embeddings_rag.nodes.index import (
    NodeIndexError,
    build_node_index,
    load_node_index,
    save_node_index,
)
from concept_embeddings_rag.nodes.local_extraction import (
    LocalExtractionError,
    LocalExtractor,
    build_manifest,
    projection,
    record_from_forms,
    write_extraction,
)


class DenseFixture:
    name = "dense"

    def retrieve(self, query, top_k):
        ids = [f"u{i:03d}" for i in range(120)]
        if query == "no bridge":
            ids[0], ids[1] = ids[1], ids[0]
        return [(uid, 1.0 - i / 120) for i, uid in enumerate(ids[:top_k])]


@pytest.fixture
def candidate(tmp_path):
    ids = [f"u{i:03d}" for i in range(120)]
    extractor = LocalExtractor(
        extractor_id="gliner",
        model="fixture",
        revision="r1",
        labels=("person",),
        parameters={},
        library_versions={"fixture": "1"},
        spans=lambda texts: [[] for _ in texts],
    )
    forms = {"u000": ["Bridge", "Common"], "u118": ["Common"], "u119": ["Bridge", "Common"]}
    records = [
        record_from_forms(
            uid,
            forms.get(uid, []),
            model=extractor.model,
            configuration_digest=extractor.configuration_digest,
        )
        for uid in ids
    ]
    manifest = build_manifest(records, extractor, seconds=2.0, hardware={"device": "cpu"})
    directory = tmp_path / "phase7" / "gliner"
    write_extraction(records, manifest, directory)
    questions = [
        Question("q1", "bridge", "", ("u000", "u119"), (), "dev"),
        Question("q2", "no bridge", "", ("u001",), (), "dev"),
    ]
    arguments = {
        "extractor_id": "gliner",
        "unit_ids": ids,
        "dense": DenseFixture(),
        "questions": questions,
        "token_counts": dict.fromkeys(ids, 1024),
        "run_config": {
            "model": "fixture",
            "revision": "r1",
            "unit_set_hash": unit_set_hash(ids),
            "seed": config.DEFAULT_SEED,
            "tokenizer": "fixture",
            "code_version": "test",
            "top_k": config.EVALUATION_TOP_K,
        },
    }
    return directory, arguments, records, manifest


def test_real_hop_fit_and_harness_recover_evidence_outside_dense(candidate):
    directory, arguments, _, manifest = candidate
    path = evaluate_candidate(directory, **arguments)
    payload = json.loads(path.read_text())
    assert payload["metrics"]["budget_2048"]["full_support"] == 1.0
    assert payload["fit"]["metric"] == "gold_recall"
    assert payload["fit"]["budget"] == 2048
    assert payload["fit"]["grid"] == list(config.FUSION_WEIGHT_GRID)
    assert len(payload["fit"]["curve"]) == len(config.FUSION_WEIGHT_GRID)
    assert "rrf_score" in payload["fit"]
    for budget in config.CONTEXT_BUDGETS:
        assert "gold_recall" in payload["metrics"][f"budget_{budget}"]
    for depth in config.RECALL_AT_K:
        assert f"recall_at_{depth}" in payload["metrics"]
    assert payload["hop"]["positive_candidates_mean"] == 1.0
    assert payload["hop"]["positive_candidates_median"] == 1.0
    assert payload["hop"]["zero_candidate_share"] == 0.5
    assert payload["hop"]["p1s_without_entity"] == 1
    assert payload["hop"]["introduced_context_units"] == 1
    assert payload["index"]["distinct_nodes"] == 2
    assert payload["index"]["entity_incidences"] == 5
    assert payload["index"]["paragraphs_without_node"] == 117
    assert payload["index"]["bytes"] > 0
    assert payload["extraction"] == manifest
    assert payload["digest"] == digest_of_payload(payload)
    run = json.loads((directory / payload["run_file"]).read_text())
    assert payload["run_digest"] == digest_of_payload(run)
    assert run["metrics"] == payload["metrics"]
    assert run["config"]["extraction_digest"] == manifest["digest"]
    assert run["split"] == "dev"
    assert not (directory.parent / "selection.json").exists()
    assert not (directory.parent / "test.json").exists()
    index = load_node_index(directory, expected_unit_ids=arguments["unit_ids"])
    assert index.digest == payload["index"]["digest"]


def test_test_questions_are_refused_before_any_index_or_measurement(candidate):
    directory, arguments, _, _ = candidate
    arguments["questions"] = [replace(arguments["questions"][0], split="test")]
    with pytest.raises(SelectionError, match="dev"):
        evaluate_candidate(directory, **arguments)
    assert not list(directory.glob("nodes-*"))
    assert not (directory / "dev.json").exists()


def test_existing_index_from_another_pool_is_refused(candidate):
    directory, arguments, records, manifest = candidate
    index = build_node_index(
        {record.unit_id: record for record in records},
        list(reversed(arguments["unit_ids"])),
        extraction_digest=manifest["digest"],
    )
    save_node_index(index, directory)
    with pytest.raises(NodeIndexError, match="another pool"):
        evaluate_candidate(directory, **arguments)


def test_changed_extraction_fails_before_a_dev_number_is_written(candidate):
    directory, arguments, _, _ = candidate
    manifest_path = directory / "extraction.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["digest"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(LocalExtractionError, match="digest"):
        evaluate_candidate(directory, **arguments)
    assert not (directory / "dev.json").exists()


def test_measured_dev_artifact_is_never_overwritten(candidate):
    directory, arguments, _, _ = candidate
    target = directory / "dev.json"
    target.write_text("existing measurement")
    with pytest.raises(CheapEvaluationError, match="not overwriting"):
        evaluate_candidate(directory, **arguments)
    assert target.read_text() == "existing measurement"


@pytest.mark.parametrize("name", ["selection.json", "test.json"])
def test_fitting_stops_after_selection(candidate, name):
    directory, arguments, _, _ = candidate
    (directory.parent / name).write_text("{}")
    with pytest.raises(CheapEvaluationError, match="fitting is closed"):
        evaluate_candidate(directory, **arguments)


# --- S5: the rule, and the one held-out run --------------------------------------------


def a_dev_payload(
    extractor_id: str,
    full_support: float,
    *,
    n_questions: int = config.N_DEV,
    paragraphs_per_second: float = 60.0,
    usd: float = 0.0,
) -> dict:
    """A dev reading with the fields the rule reads, signed like the real artifact."""
    payload = {
        "extractor_id": extractor_id,
        "split": "dev",
        "n_questions": n_questions,
        "config": {"phase": 7},
        "fit": {"scheme": "weighted", "weights": {"dense": 0.7, "entity-hop": 0.3}},
        "metrics": {
            f"budget_{config.SELECTION_BUDGET}": {
                "full_support": full_support,
                "gold_recall": full_support,
            }
        },
        "cost": {"mean_latency_ms": 1.0},
        "extraction": {
            "model": f"{extractor_id}-model",
            "revision": "r1",
            "labels": ["person"],
            "configuration_digest": "c" * 16,
            "digest": "d" * 64,
            "seconds": 19366 / paragraphs_per_second,
            "paragraphs_per_second": paragraphs_per_second,
            "usd": usd,
            "failures": {},
            "failure_rate": 0.0,
            "hardware": {"device": "cpu"},
            "configuration": {"extractor_id": extractor_id},
            "projection_5m": projection(19366, paragraphs_per_second),
        },
        "index": {"distinct_nodes": 10, "entity_incidences": 20, "bytes": 1, "digest": "e" * 64},
        "hop": {"positive_candidates_mean": 1.0},
        "run_file": "run.json",
        "run_digest": "f" * 64,
    }
    payload["digest"] = digest_of_payload(payload)
    return payload


def write_dev(phase_7_dir, extractor_id: str, payload: dict) -> None:
    directory = phase_7_dir / extractor_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dev.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_the_bar_is_the_spec_s_own_number_in_whole_questions():
    share, required = retention_bar(config.N_DEV)

    assert required == config.PHASE_7_RETENTION_BAR_QUESTIONS
    assert round(share, 4) == config.PHASE_7_RETENTION_BAR


def test_a_candidate_on_the_bar_passes_and_one_below_it_does_not(tmp_path):
    write_dev(tmp_path, "gliner", a_dev_payload("gliner", 517 / config.N_DEV))
    write_dev(tmp_path, "spacy", a_dev_payload("spacy", 516 / config.N_DEV))

    selection = json.loads(select_extractor(tmp_path).read_text())

    rows = {row["extractor_id"]: row for row in selection["candidates"]}
    assert rows["gliner"]["retention"]["supported_questions"] == 517
    assert rows["gliner"]["retention"]["passed"] is True
    assert rows["spacy"]["retention"]["supported_questions"] == 516
    assert rows["spacy"]["retention"]["passed"] is False
    assert selection["selected"] == "gliner"
    assert selection["ranking"] == ["gliner"]
    assert selection["digest"] == digest_of_payload(selection)


def test_a_candidate_that_cannot_finish_fullwiki_in_time_fails_the_economic_bar(tmp_path):
    slow = config.PROJECTION_PARAGRAPHS / (config.PHASE_7_FULLWIKI_HOURS_CEILING * 3600) * 0.99
    write_dev(tmp_path, "gliner", a_dev_payload("gliner", 0.95, paragraphs_per_second=slow))

    selection = json.loads(select_extractor(tmp_path).read_text())

    economics = selection["candidates"][0]["economics"]
    assert economics["projected_fullwiki_hours"] > config.PHASE_7_FULLWIKI_HOURS_CEILING
    assert economics["passed"] is False
    assert selection["selected"] is None
    assert "negative result" in selection["reason"]


def test_an_infrastructure_charge_over_the_ceiling_fails_the_economic_bar(tmp_path):
    over = config.PHASE_7_INFRASTRUCTURE_CEILING_USD + 0.01
    write_dev(tmp_path, "gliner", a_dev_payload("gliner", 0.95, usd=over))

    selection = json.loads(select_extractor(tmp_path).read_text())

    assert selection["candidates"][0]["economics"]["infrastructure_usd"] == over
    assert selection["candidates"][0]["economics"]["passed"] is False
    assert selection["selected"] is None


def test_the_cheaper_projection_wins_and_the_bm25_line_is_a_qualification(tmp_path):
    write_dev(tmp_path, "gliner", a_dev_payload("gliner", 0.95, paragraphs_per_second=40.0))
    write_dev(tmp_path, "spacy", a_dev_payload("spacy", 0.87, paragraphs_per_second=900.0))

    selection = json.loads(select_extractor(tmp_path).read_text())

    assert selection["ranking"] == ["spacy", "gliner"]
    assert selection["selected"] == "spacy"
    rows = {row["extractor_id"]: row for row in selection["candidates"]}
    assert rows["spacy"]["bm25_dev_line"]["below_line"] is False
    assert rows["gliner"]["bm25_dev_line"]["below_line"] is False
    assert rows["spacy"]["passed_both_bars"] and rows["gliner"]["passed_both_bars"]


def test_a_candidate_with_no_dev_reading_is_named_rather_than_assumed(tmp_path):
    write_dev(tmp_path, "gliner", a_dev_payload("gliner", 0.95))

    selection = json.loads(select_extractor(tmp_path).read_text())

    assert selection["candidates_without_dev_measurement"] == ["spacy"]
    assert [row["extractor_id"] for row in selection["candidates"]] == ["gliner"]


def test_an_edited_dev_reading_is_refused_rather_than_selected_from(tmp_path):
    payload = a_dev_payload("gliner", 0.50)
    payload["metrics"][f"budget_{config.SELECTION_BUDGET}"]["full_support"] = 0.99
    write_dev(tmp_path, "gliner", payload)

    with pytest.raises(CheapEvaluationError, match="digest"):
        select_extractor(tmp_path)


def test_the_selection_is_written_once(tmp_path):
    write_dev(tmp_path, "gliner", a_dev_payload("gliner", 0.95))
    select_extractor(tmp_path)

    with pytest.raises(CheapEvaluationError, match="written once"):
        select_extractor(tmp_path)


def test_selection_without_any_dev_reading_names_the_stage_to_run(tmp_path):
    with pytest.raises(CheapEvaluationError, match="cheap-eval"):
        select_extractor(tmp_path)


def held_out_arguments(arguments: dict) -> dict:
    """The same inputs, on the split the phase reads exactly once."""
    return {
        "unit_ids": arguments["unit_ids"],
        "dense": arguments["dense"],
        "questions": [
            Question("t1", "bridge", "", ("u000", "u119"), (), "test"),
            Question("t2", "no bridge", "", ("u001",), (), "test"),
        ],
        "token_counts": arguments["token_counts"],
        "run_config": arguments["run_config"],
    }


def test_the_held_out_run_measures_the_selected_extractor_once(candidate):
    directory, arguments, _, manifest = candidate
    evaluate_candidate(directory, **arguments)
    selection = json.loads(select_extractor(directory.parent).read_text())
    assert selection["selected"] == "gliner"

    path = measure_held_out(directory.parent, **held_out_arguments(arguments))

    payload = json.loads(path.read_text())
    assert payload["split"] == "test"
    assert payload["extractor_id"] == "gliner"
    assert payload["config"]["extraction_digest"] == manifest["digest"]
    assert payload["fusion"] == {
        "scheme": selection["candidates"][0]["fusion"]["scheme"],
        "weights": selection["candidates"][0]["fusion"]["weights"],
    }
    assert payload["inherited_test_full_support"] == config.PHASE_7_REFERENCE_HELD_OUT_FULL_SUPPORT
    assert payload["selection_digest"] == selection["digest"]
    assert payload["digest"] == digest_of_payload(payload)
    run = json.loads((directory / Path(payload["run_file"]).name).read_text())
    assert run["split"] == "test"
    assert run["metrics"] == payload["metrics"]

    with pytest.raises(CheapEvaluationError, match="read once"):
        measure_held_out(directory.parent, **held_out_arguments(arguments))


def test_the_held_out_run_refuses_without_a_selection(candidate):
    directory, arguments, _, _ = candidate
    evaluate_candidate(directory, **arguments)

    with pytest.raises(CheapEvaluationError, match="cheap-eval"):
        measure_held_out(directory.parent, **held_out_arguments(arguments))


def test_a_negative_selection_reads_no_test_figure_at_all(tmp_path):
    write_dev(tmp_path, "gliner", a_dev_payload("gliner", 0.50))
    selection = json.loads(select_extractor(tmp_path).read_text())
    assert selection["selected"] is None

    with pytest.raises(CheapEvaluationError, match="names no extractor"):
        measure_held_out(
            tmp_path,
            unit_ids=["u000"],
            dense=DenseFixture(),
            questions=[Question("t1", "bridge", "", ("u000",), (), "test")],
            token_counts={"u000": 10},
            run_config={"unit_set_hash": unit_set_hash(["u000"]), "top_k": 10},
        )


def test_dev_questions_are_refused_by_the_held_out_run(candidate):
    directory, arguments, _, _ = candidate
    evaluate_candidate(directory, **arguments)
    select_extractor(directory.parent)
    held_out = held_out_arguments(arguments)
    held_out["questions"] = list(arguments["questions"])

    with pytest.raises(CheapEvaluationError, match="measures test"):
        measure_held_out(directory.parent, **held_out)
    assert not (directory.parent / "test.json").exists()
