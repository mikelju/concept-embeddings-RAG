"""Phase 6, T7: every inherited input is verified on load, or the phase does not run (D2, D3).

HU-1 asks the run to refuse when the pool, the split, the tokenizer, the embeddings, the node
index, the extraction, the Phase 5 references or `selection.json` do not verify. Each refusal
is shown on a toy pipeline written by the project's own writers, one input at a time.

The four historical result files carry no digest and are pinned by name. They are opened in
two ways only before the test stage: the **identity reader** (name, sha256 of the bytes,
system, split, creation time and the configuration keys HU-2 lists, and nothing else) for all
four, and the **dev figures reader** for the two dev files. No reader for the figures of a
test file exists in this module; the accesses are recorded to prove the identity reader never
touches `metrics` or `cost`.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
from replacement_fixtures import (
    ToyPipeline,
    build_toy_pipeline,
    rewrite_json,
    toy_counter,
    with_pins,
)

from concept_embeddings_rag import config
from concept_embeddings_rag.embeddings.cache import EmbeddingCache, cache_key
from concept_embeddings_rag.evaluation import replacement_inputs as inputs_module
from concept_embeddings_rag.evaluation.replacement_inputs import (
    HistoricalIdentity,
    InputPaths,
    InputPins,
    ReplacementInputError,
    identify_result_file,
    identity_of,
    load_verified_inputs,
    offline_token_counter,
    read_dev_figures,
)


class RecordingMapping(Mapping):
    """A mapping that counts every access to its contents."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)
        self.accesses = 0

    def __getitem__(self, key: str) -> Any:
        self.accesses += 1
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        self.accesses += 1
        return iter(self._data)

    def __len__(self) -> int:
        self.accesses += 1
        return len(self._data)


class RecordingParser:
    """`json.loads` that hands the identity reader recording mappings for metrics and cost."""

    def __init__(self) -> None:
        self.recorded: dict[str, tuple[RecordingMapping, RecordingMapping]] = {}

    def __call__(self, text: str) -> dict[str, Any]:
        document = json.loads(text)
        metrics, cost = RecordingMapping(document["metrics"]), RecordingMapping(document["cost"])
        document["metrics"], document["cost"] = metrics, cost
        self.recorded[f"{document['system']}/{document['split']}"] = (metrics, cost)
        return document


@pytest.fixture(scope="module")
def pristine(tmp_path_factory) -> ToyPipeline:
    return build_toy_pipeline(tmp_path_factory.mktemp("pristine"))


@pytest.fixture
def toy(tmp_path) -> ToyPipeline:
    return build_toy_pipeline(tmp_path)


def load(pipeline: ToyPipeline, **overrides: Any):
    arguments: dict[str, Any] = {
        "pins": pipeline.pins,
        "backend": pipeline.backend,
        "token_counter": toy_counter,
    }
    arguments.update(overrides)
    return load_verified_inputs(pipeline.paths, **arguments)


# --- The toy pipeline verifies ---------------------------------------------------------------


def test_a_consistent_toy_pipeline_verifies_whole(pristine):
    verified = load(pristine)

    assert verified.pool_hash == pristine.pins.unit_set_hash
    assert len(verified.units) == pristine.pins.n_units
    assert set(verified.split_orders) == {"dev", "test"}
    assert len(verified.split_orders["dev"]) == 6
    assert verified.counts["entity_nodes"] == pristine.pins.entity_nodes
    assert verified.counts["pilot_questions"] == pristine.pins.pilot_questions
    assert verified.selection.control is not None
    assert verified.node_index.digest == pristine.pins.node_index_digest
    assert verified.token_counts_sha256 == inputs_module.sha256_of(
        pristine.paths.data_dir / "token_counts.json"
    )


def test_the_split_orders_are_the_pool_order_and_hashed(pristine):
    verified = load(pristine)
    dev_in_pool_order = [q.qid for q in pristine.questions if q.split == "dev"]
    assert list(verified.split_orders["dev"]) == dev_in_pool_order
    assert set(verified.question_set_hashes) == {"dev", "test"}


def test_only_a_dev_query_backend_is_built(pristine):
    verified = load(pristine)
    names = {field.name for field in fields(verified)}
    assert "dev_query_backend" in names
    assert not any("test" in name and "backend" in name for name in names)
    dev = [q for q in pristine.questions if q.split == "dev"]
    assert verified.dev_query_backend.encode([dev[0].question]).shape == (1, 16)
    test = next(q for q in pristine.questions if q.split == "test")
    with pytest.raises(KeyError):
        verified.dev_query_backend.encode([test.question])


def test_all_four_historical_files_are_identified_by_name_hash_and_config(pristine):
    verified = load(pristine)
    assert set(verified.historical) == {
        ("dense", "dev"),
        ("dense", "test"),
        ("hybrid-bm25", "dev"),
        ("hybrid-bm25", "test"),
    }
    for (system, split), identity in verified.historical.items():
        assert identity.system == system and identity.split == split
        assert len(identity.sha256) == 64
        assert set(identity.config) >= set(config.HISTORICAL_CONFIG_KEYS)
    assert set(verified.historical[("hybrid-bm25", "dev")].config) >= set(
        config.HISTORICAL_HYBRID_CONFIG_KEYS
    )


def test_the_identity_reader_never_touches_metrics_or_cost_of_any_file(pristine):
    parser = RecordingParser()
    load(pristine, parse=parser)

    assert set(parser.recorded) == {
        "dense/dev",
        "dense/test",
        "hybrid-bm25/dev",
        "hybrid-bm25/test",
    }
    for metrics, cost in parser.recorded.values():
        assert metrics.accesses == 0
        assert cost.accesses == 0


# --- One refusal per input ------------------------------------------------------------------


def test_a_modified_pool_is_refused(toy):
    def edit(payload):
        payload["units"][0]["sentences"][0] += " edited"

    rewrite_json(toy.paths.data_dir / "pool.json", edit)
    with pytest.raises(ReplacementInputError, match="pool"):
        load(toy)


def test_a_pool_other_than_the_pinned_one_is_refused(pristine):
    with pytest.raises(ReplacementInputError, match="pool"):
        load(with_pins(pristine, unit_set_hash="ffffffffffffffff"))


def test_a_question_on_the_wrong_side_of_the_split_rule_is_refused(toy):
    def swap(payload):
        dev = next(q for q in payload["questions"] if q["split"] == "dev")
        test = next(q for q in payload["questions"] if q["split"] == "test")
        dev["split"], test["split"] = "test", "dev"

    rewrite_json(toy.paths.data_dir / "pool.json", swap)
    with pytest.raises(ReplacementInputError, match="split"):
        load(toy)


def test_a_token_count_differing_from_the_recount_is_refused(toy):
    def edit(payload):
        first = sorted(payload)[0]
        payload[first] += 1

    rewrite_json(toy.paths.data_dir / "token_counts.json", edit)
    with pytest.raises(ReplacementInputError, match="token"):
        load(toy)


def test_a_foreign_embedding_cache_is_refused(toy):
    key = cache_key(toy.pins.model, toy.pins.revision, toy.pins.unit_set_hash, True)
    cache = EmbeddingCache(toy.paths.cache_dir)
    vectors, unit_ids = cache.load(key)
    cache.save(key, vectors[::-1], unit_ids[::-1], {"model": toy.pins.model})
    with pytest.raises(ReplacementInputError, match="embedding"):
        load(toy)


def test_a_missing_dev_question_cache_is_refused_rather_than_embedded(toy):
    for path in toy.paths.question_cache_dir.glob("*"):
        payload = path.read_text(encoding="utf-8") if path.suffix == ".json" else None
        if payload is not None and json.loads(payload).get("split") == "dev":
            path.unlink()
            path.with_suffix(".npz").unlink()
    with pytest.raises(ReplacementInputError, match="question"):
        load(toy)


def test_a_selection_with_another_digest_is_refused(pristine):
    with pytest.raises(ReplacementInputError, match="selection"):
        load(with_pins(pristine, selection_digest="0" * 64))


def test_a_tampered_selection_is_refused(toy):
    rewrite_json(toy.paths.selection_dir / "selection.json", lambda p: p.update({"seed": 7}))
    with pytest.raises(ReplacementInputError, match="selection"):
        load(toy)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seed", 7),
        ("top_k", 50),
        ("tokenizer", "another-tokenizer"),
        ("fusion_weights", {"dense": 0.9, "bm25": 0.1}),
        ("unit_set_hash", "ffffffffffffffff"),
        ("revision", "main"),
    ],
)
def test_a_historical_file_with_another_configuration_is_refused(toy, key, value):
    name = toy.pins.historical_dev_files["hybrid-bm25"]
    rewrite_json(toy.paths.results_dir / name, lambda p: p["config"].update({key: value}))
    with pytest.raises(ReplacementInputError, match=key):
        load(toy)


def test_a_historical_test_file_with_another_configuration_is_refused_by_identity(toy):
    name = toy.pins.historical_test_files["dense"]
    rewrite_json(toy.paths.results_dir / name, lambda p: p["config"].update({"seed": 7}))
    with pytest.raises(ReplacementInputError, match="seed"):
        load(toy)


def test_a_missing_historical_file_is_refused(toy):
    (toy.paths.results_dir / toy.pins.historical_test_files["hybrid-bm25"]).unlink()
    with pytest.raises(ReplacementInputError, match="historical"):
        load(toy)


def test_a_historical_file_under_the_wrong_name_is_refused(pristine):
    swapped = {"dense": pristine.pins.historical_dev_files["hybrid-bm25"]}
    swapped["hybrid-bm25"] = pristine.pins.historical_dev_files["dense"]
    with pytest.raises(ReplacementInputError, match="historical"):
        load(with_pins(pristine, historical_dev_files=swapped))


@pytest.mark.parametrize(
    "pin",
    ["pilot_digest", "hop_run_digest", "traces_digest", "extraction_digest", "node_index_digest"],
)
def test_a_phase_5_artifact_with_another_digest_is_refused(pristine, pin):
    with pytest.raises(ReplacementInputError):
        load(with_pins(pristine, **{pin: "a" * 64}))


def test_a_tampered_pilot_is_refused(toy):
    rewrite_json(toy.paths.pilot_dir / "pilot.json", lambda p: p.update({"read_depth": 9}))
    with pytest.raises(ReplacementInputError, match="pilot"):
        load(toy)


def test_a_tampered_hop_run_is_refused(toy):
    rewrite_json(toy.paths.navigation_dir / "hop-run.json", lambda p: p.update({"extra": 1}))
    with pytest.raises(ReplacementInputError, match="hop"):
        load(toy)


def test_a_tampered_extraction_archive_is_refused(toy):
    archive = toy.paths.extraction_dir / f"extraction-{toy.pins.extraction_prompt_digest}.jsonl.gz"
    import gzip

    text = gzip.decompress(archive.read_bytes()).decode("utf-8").replace("Entity 1", "Entity 2", 1)
    archive.write_bytes(gzip.compress(text.encode("utf-8")))
    with pytest.raises(ReplacementInputError, match="extraction"):
        load(toy)


def test_a_normalization_version_other_than_the_pinned_one_is_refused(pristine):
    with pytest.raises(ReplacementInputError, match="normalization"):
        load(with_pins(pristine, normalization_version="normalization-v2"))


# --- SEC-022 is not reachable -----------------------------------------------------------------


def test_the_node_index_path_comes_from_the_pin_not_from_the_summary(toy, monkeypatch):
    """A summary whose digest is a path fragment is refused before any archive or index opens."""
    summary = toy.paths.extraction_dir / f"extraction-{toy.pins.extraction_prompt_digest}.json"
    rewrite_json(summary, lambda p: p.update({"digest": "../nodes/../../elsewhere"}))
    opened: list[str] = []
    monkeypatch.setattr(inputs_module, "load_extraction", lambda *a, **k: opened.append("x"))
    monkeypatch.setattr(inputs_module, "load_node_index", lambda *a, **k: opened.append("n"))

    with pytest.raises(ReplacementInputError, match="extraction"):
        load(toy)
    assert opened == []


def test_a_pinned_extraction_digest_that_is_not_a_sha256_is_refused_before_any_file(
    pristine, monkeypatch
):
    opened: list[str] = []
    monkeypatch.setattr(
        inputs_module, "load_extraction_summary", lambda *a, **k: opened.append("s")
    )
    with pytest.raises(ReplacementInputError, match="sha256"):
        load(with_pins(pristine, extraction_digest="../../x"))
    assert opened == []


# --- Counts ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pin",
    [
        "n_units",
        "entity_nodes",
        "failed_extractions",
        "units_without_entity_node",
        "pilot_questions",
        "entity_traces",
    ],
)
def test_a_count_differing_from_the_expected_one_stops_with_both_numbers(pristine, pin):
    expected = getattr(pristine.pins, pin) + 1
    with pytest.raises(ReplacementInputError) as raised:
        load(with_pins(pristine, **{pin: expected}))
    message = str(raised.value)
    assert str(expected) in message
    assert str(expected - 1) in message


# --- The three readers ------------------------------------------------------------------------


def a_document(split: str = "dev") -> dict[str, Any]:
    return {
        "system": "dense",
        "split": split,
        "created_at": "2026-09-11T13:31:54+00:00",
        "config": {
            "unit_set_hash": "p",
            "tokenizer": "t",
            "seed": 42,
            "top_k": 100,
            "model": "m",
            "revision": "r",
            "code_version": "0.1.0",
        },
        "metrics": RecordingMapping({"budget_2048": {"full_support": 0.5}}),
        "cost": RecordingMapping({"mean_units_included": 3.0}),
    }


def test_the_identity_has_no_metrics_or_cost_field():
    names = {field.name for field in fields(HistoricalIdentity)}
    assert names == {"name", "sha256", "system", "split", "created_at", "config"}


def test_identifying_a_document_accesses_neither_its_metrics_nor_its_cost():
    document = a_document("test")
    identity = identity_of(document, name="run-dense-test-x.json", sha256="0" * 64)

    assert identity.split == "test"
    assert identity.config["seed"] == 42
    assert "code_version" not in identity.config
    assert document["metrics"].accesses == 0
    assert document["cost"].accesses == 0


def test_the_identity_reader_hashes_the_raw_bytes(tmp_path):
    path = tmp_path / "run-dense-dev-x.json"
    document = a_document()
    document["metrics"], document["cost"] = {"a": 1}, {"b": 2}
    path.write_text(json.dumps(document), encoding="utf-8")

    identity = identify_result_file(path)
    assert identity.sha256 == inputs_module.sha256_of(path)
    assert identity.name == path.name


def test_the_dev_figures_reader_refuses_a_test_file(tmp_path):
    path = tmp_path / "run-dense-test-x.json"
    document = a_document("test")
    document["metrics"], document["cost"] = {"a": 1}, {"b": 2}
    path.write_text(json.dumps(document), encoding="utf-8")
    identity = identify_result_file(path)

    with pytest.raises(ReplacementInputError, match="dev"):
        read_dev_figures(tmp_path, identity)


def test_the_dev_figures_reader_refuses_bytes_that_changed_since_identification(tmp_path):
    path = tmp_path / "run-dense-dev-x.json"
    document = a_document()
    document["metrics"], document["cost"] = {"a": 1}, {"b": 2}
    path.write_text(json.dumps(document), encoding="utf-8")
    identity = identify_result_file(path)
    path.write_text(json.dumps({**document, "cost": {"b": 3}}), encoding="utf-8")

    with pytest.raises(ReplacementInputError, match="sha256"):
        read_dev_figures(tmp_path, identity)


def test_the_dev_figures_reader_returns_metrics_and_cost_of_a_dev_file(pristine):
    verified = load(pristine)
    figures = read_dev_figures(pristine.paths.results_dir, verified.historical[("dense", "dev")])
    assert "budget_2048" in figures.metrics
    assert "mean_units_included" in figures.cost


def test_this_module_holds_no_reader_of_test_figures():
    """D2: the reader of historical test figures is written in T18, in replacement_run.py."""
    public = [name for name in dir(inputs_module) if not name.startswith("_")]
    assert not [name for name in public if "test" in name.lower() and "figure" in name.lower()]


# --- The real inputs (data-dependent, dev only) --------------------------------------------------


def test_the_real_inputs_verify_and_the_historical_test_files_are_only_identified():
    paths = InputPaths.from_config()
    if not (paths.data_dir / "pool.json").exists() or not (paths.nodes_dir).exists():
        pytest.skip("the real pipeline artifacts are not on disk")
    from concept_embeddings_rag.embeddings.backend import SentenceTransformerBackend

    try:
        counter = offline_token_counter()
        counter([])
    except OSError as error:
        pytest.skip(f"the pinned tokenizer is not in the local cache: {error}")

    parser = RecordingParser()
    verified = load_verified_inputs(
        paths,
        pins=InputPins.from_config(),
        backend=SentenceTransformerBackend(),
        token_counter=counter,
        parse=parser,
    )

    assert verified.pool_hash == config.PHASE_1_UNIT_SET_HASH
    assert verified.counts == {
        "n_units": config.EXPECTED_N_UNITS,
        "entity_nodes": config.EXPECTED_ENTITY_NODES,
        "failed_extractions": config.EXPECTED_FAILED_EXTRACTIONS,
        "units_without_entity_node": config.EXPECTED_UNITS_WITHOUT_ENTITY_NODE,
        "pilot_questions": config.EXPECTED_PILOT_QUESTIONS,
        "entity_traces": config.EXPECTED_ENTITY_TRACES,
    }
    for split in ("test",):
        for system in ("dense", "hybrid-bm25"):
            identity = verified.historical[(system, split)]
            assert identity.name == config.HISTORICAL_TEST_RESULT_FILES[system]
            metrics, cost = parser.recorded[f"{system}/{split}"]
            assert metrics.accesses == 0
            assert cost.accesses == 0
    assert Path(paths.results_dir).is_dir()
