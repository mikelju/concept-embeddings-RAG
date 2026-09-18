"""S4 exercises real indexing, fusion fitting and measurement on a small bridge corpus."""

import json
from dataclasses import replace

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import Question
from concept_embeddings_rag.embeddings.cache import unit_set_hash
from concept_embeddings_rag.evaluation.cheap_extraction import (
    CheapEvaluationError,
    evaluate_candidate,
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
