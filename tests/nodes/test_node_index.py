"""T7 of Phase 5: the node index and its fragmentation, and `cer nodes` (HU-3)."""

import json
from pathlib import Path

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import IndexingUnit, Question, save_pool, unit_id_for
from concept_embeddings_rag.nodes.extraction import (
    FAILED,
    OK,
    PROMPT_DIGEST,
    ExtractionRecord,
    RawResponse,
    run_full,
    run_sample,
)
from concept_embeddings_rag.nodes.index import (
    NodeIndexError,
    build_node_index,
    fragmentation,
    load_node_index,
    save_node_index,
)

MODEL = config.EXTRACTION_MODEL
DIGEST = "extraction-digest-for-tests"


def record(unit_id: str, entities=(), concepts=(), status=OK) -> ExtractionRecord:
    return ExtractionRecord(
        unit_id=unit_id,
        model=MODEL,
        prompt_digest=PROMPT_DIGEST,
        status=status,
        entities=tuple(entities),
        concepts=tuple(concepts),
        failure=None if status == OK else "refusal",
        input_tokens=1,
        output_tokens=1,
    )


UNIT_IDS = ["u0", "u1", "u2", "u3"]


def the_records() -> dict[str, ExtractionRecord]:
    return {
        "u0": record(
            "u0", ["The Beatles", "Beatles", "John Lennon"], ["rock band", "Rock Band", "beatles"]
        ),
        "u1": record("u1", ["John Lennon"], ["rock band"]),
        "u2": record("u2", status=FAILED),
        "u3": record("u3", [], ["", "river"]),
    }


def an_index():
    return build_node_index(the_records(), UNIT_IDS, extraction_digest=DIGEST)


def forms(index) -> list[tuple[str, str]]:
    return [(node.type, node.form) for node in index.nodes]


def row(index, unit_id: str) -> set[tuple[str, str]]:
    position = index.unit_ids.index(unit_id)
    start, end = index.incidence.indptr[position], index.incidence.indptr[position + 1]
    return {
        (index.nodes[column].type, index.nodes[column].form)
        for column in index.incidence.indices[start:end]
    }


def test_nodes_are_normalized_typed_and_numbered_deterministically():
    index = an_index()

    assert forms(index) == [
        ("entity", "beatles"),
        ("entity", "john lennon"),
        ("concept", "beatles"),
        ("concept", "river"),
        ("concept", "rock band"),
    ]
    assert [node.node_id for node in index.nodes] == [0, 1, 2, 3, 4]
    reordered = dict(reversed(list(the_records().items())))
    assert forms(build_node_index(reordered, UNIT_IDS, extraction_digest=DIGEST)) == forms(index)


def test_the_same_string_as_entity_and_as_concept_is_two_nodes():
    index = an_index()

    assert ("entity", "beatles") in row(index, "u0")
    assert ("concept", "beatles") in row(index, "u0")


def test_a_paragraphs_node_set_has_no_duplicates_after_normalization():
    index = an_index()
    position = index.unit_ids.index("u0")
    start, end = index.incidence.indptr[position], index.incidence.indptr[position + 1]
    columns = index.incidence.indices[start:end].tolist()

    assert len(columns) == len(set(columns)) == 4
    assert np.all(index.incidence.data == 1.0)


def test_a_failed_extraction_is_an_empty_row_and_is_counted():
    index = an_index()

    assert row(index, "u2") == set()
    assert index.failed_units == 1
    assert index.dropped_empty == {"entity": 0, "concept": 1}


def test_the_rows_follow_the_pool_order():
    index = an_index()

    assert index.unit_ids == tuple(UNIT_IDS)
    assert index.incidence.shape == (4, 5)


def test_a_pool_unit_without_a_record_is_refused():
    records = the_records()
    del records["u3"]

    with pytest.raises(NodeIndexError, match="cover"):
        build_node_index(records, UNIT_IDS, extraction_digest=DIGEST)


def test_fragmentation_matches_a_hand_count():
    report = fragmentation(an_index())

    assert report["entity"] == {
        "distinct_nodes": 2,
        "singleton_share": 0.5,
        "paragraphs_per_node": {"mean": 1.5, "median": 1.5, "max": 2},
        "nodes_per_paragraph": {"mean": 0.75, "median": 0.5, "max": 2},
        "paragraphs_without_node": 2,
    }
    assert report["concept"]["distinct_nodes"] == 3
    assert report["concept"]["singleton_share"] == pytest.approx(2 / 3)
    assert report["concept"]["paragraphs_per_node"] == {
        "mean": pytest.approx(4 / 3),
        "median": 1.0,
        "max": 2,
    }
    assert report["concept"]["nodes_per_paragraph"] == {"mean": 1.0, "median": 1.0, "max": 2}
    assert report["concept"]["paragraphs_without_node"] == 1


def test_an_index_round_trips_through_disk(tmp_path: Path):
    index = an_index()

    save_node_index(index, tmp_path)
    loaded = load_node_index(tmp_path, extraction_digest=DIGEST, expected_unit_ids=UNIT_IDS)

    assert loaded.nodes == index.nodes
    assert loaded.unit_ids == index.unit_ids
    assert (loaded.incidence != index.incidence).nnz == 0
    assert loaded.digest == index.digest


def test_a_modified_sidecar_is_refused(tmp_path: Path):
    save_node_index(an_index(), tmp_path)
    sidecar = next(tmp_path.glob("nodes-*.json"))
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["nodes"][0]["form"] = "the rolling stones"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NodeIndexError, match="digest"):
        load_node_index(tmp_path, extraction_digest=DIGEST)


def test_an_index_built_with_another_normalization_is_refused(tmp_path: Path):
    save_node_index(an_index(), tmp_path)
    sidecar = next(tmp_path.glob("nodes-*.json"))
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["normalization_version"] = "normalization-v0"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NodeIndexError):
        load_node_index(tmp_path, extraction_digest=DIGEST)


def test_an_index_over_another_pool_is_refused(tmp_path: Path):
    save_node_index(an_index(), tmp_path)

    with pytest.raises(NodeIndexError, match="pool"):
        load_node_index(tmp_path, extraction_digest=DIGEST, expected_unit_ids=["u0", "u1"])


# --- `cer nodes` -----------------------------------------------------------------------


class FakeSyncClient:
    model = MODEL

    def __init__(self, refuse: set[str] | None = None) -> None:
        self.refuse = refuse or set()

    def extract(self, request: dict) -> RawResponse:
        content = request["messages"][0]["content"]
        if any(marker in content for marker in self.refuse):
            return RawResponse("refusal", "", 10, 0)
        text = json.dumps({"entities": ["Some Person"], "concepts": ["river"]})
        return RawResponse("end_turn", text, 100, 20)


class NoBatches:
    """Every paragraph is cached by the sample, so the full run must submit nothing."""

    model = MODEL

    def submit(self, requests):
        raise AssertionError("nothing should be submitted")

    def status(self, batch_id):
        raise AssertionError("nothing should be polled")

    def results(self, batch_id):
        raise AssertionError("nothing should be read")


def a_workspace(tmp_path: Path, refuse: set[str] | None = None):
    units = []
    for index in range(10):
        title = f"Article {index}"
        sentences = (f"Paragraph {index} about a river.",)
        units.append(
            IndexingUnit(unit_id=unit_id_for(title, sentences), title=title, sentences=sentences)
        )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    question = Question("q0", "Unused?", "-", (units[0].unit_id,), ((units[0].title, 0),), "dev")
    save_pool(units, [question], data_dir / "pool.json")
    counts = {unit.unit_id: 20 for unit in units}
    extraction_dir = tmp_path / "extraction"
    run_sample(
        units,
        FakeSyncClient(refuse),
        cache_dir=tmp_path / "cache",
        extraction_dir=extraction_dir,
        token_counts=counts,
        size=len(units),
    )
    run_full(
        units,
        NoBatches(),
        cache_dir=tmp_path / "cache",
        extraction_dir=extraction_dir,
        wait=lambda: None,
    )
    return data_dir, units


def nodes(tmp_path: Path, data_dir: Path, **overrides):
    from concept_embeddings_rag.cli import cmd_nodes

    arguments = {
        "data_dir": data_dir,
        "extraction_dir": tmp_path / "extraction",
        "nodes_dir": tmp_path / "nodes",
    }
    arguments.update(overrides)
    return cmd_nodes(**arguments)


def test_nodes_without_an_extraction_names_the_extract_stage(tmp_path: Path):
    data_dir, _ = a_workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        nodes(tmp_path, data_dir, extraction_dir=tmp_path / "nowhere")

    assert "extract" in str(excinfo.value)


def test_nodes_refuses_an_extraction_whose_failures_are_a_finding(tmp_path: Path):
    data_dir, _ = a_workspace(tmp_path, refuse={"Article 3"})

    with pytest.raises(SystemExit) as excinfo:
        nodes(tmp_path, data_dir)

    assert "finding" in str(excinfo.value)
    assert not (tmp_path / "nodes").exists() or not any((tmp_path / "nodes").iterdir())


def test_nodes_writes_the_index_and_prints_fragmentation_without_node_forms(tmp_path: Path, capsys):
    data_dir, units = a_workspace(tmp_path)

    path = nodes(tmp_path, data_dir)

    out = capsys.readouterr().out
    assert out.isascii()
    assert "entity" in out and "concept" in out
    assert "some person" not in out.lower()
    index = load_node_index(tmp_path / "nodes", expected_unit_ids=[unit.unit_id for unit in units])
    assert index.incidence.shape == (10, 2)
    assert path.exists()
