"""S2: spans to records, the sentence window, and the local extraction artifact.

Result-changing code only, per the simplification rule. What is asserted here is what
would silently move a Phase 7 number: the de-duplication that makes a tagger's output
comparable with Claude's one-item-per-name answer, the bounds rule that fails rather than
repairs, the window rule that lets a long paragraph be read at all, and the digest that
makes a modified archive unusable.

No model is downloaded and neither extractor library is imported: both adapters are
exercised through a fake that returns fixed spans, which is the only thing the conversion
ever sees.
"""

import gzip
import json
from collections.abc import Sequence

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import IndexingUnit
from concept_embeddings_rag.nodes import extraction as phase_5
from concept_embeddings_rag.nodes import local_extraction as local
from concept_embeddings_rag.nodes.extraction import FAILED, OK

MODEL = "fake/extractor-v1"
DIGEST = "0123456789abcdef"


def a_unit(unit_id: str, title: str, *sentences: str) -> IndexingUnit:
    return IndexingUnit(unit_id=unit_id, title=title, sentences=tuple(sentences))


def a_fake_extractor(
    answers: dict[str, list[str]],
    *,
    max_tokens: int | None = None,
) -> local.LocalExtractor:
    """An extractor that answers by text, so a test declares what the model returned."""

    def spans(texts: Sequence[str]) -> list[list[str]]:
        return [list(answers.get(text, [])) for text in texts]

    def count_tokens(text: str) -> int:
        return len(text.split())

    return local.LocalExtractor(
        extractor_id="fake",
        model=MODEL,
        revision="r1",
        labels=("person", "location"),
        parameters={"threshold": 0.5},
        library_versions={"fake": "1.0"},
        spans=spans,
        max_tokens=max_tokens,
        count_tokens=count_tokens if max_tokens is not None else None,
    )


# --- 1. The conversion to an ExtractionRecord (D6) -------------------------------------


def test_forms_are_deduplicated_by_exact_surface_in_first_occurrence_order():
    """A tagger emits one span per mention; Claude was asked for one per distinct name."""
    record = local.record_from_forms(
        "u1",
        ["Kaley Cuoco", "CBS", "Kaley Cuoco", " CBS ", "The Big Bang Theory"],
        model=MODEL,
        configuration_digest=DIGEST,
    )

    assert record.status == OK
    assert record.entities == ("Kaley Cuoco", "CBS", "The Big Bang Theory")


def test_the_surface_is_taken_verbatim_and_never_normalized_here():
    """normalization-v1 is this project's only normalizer and it runs in build_node_index."""
    record = local.record_from_forms(
        "u1", ["  Universite de Montreal  ", "the BBC"], model=MODEL, configuration_digest=DIGEST
    )

    assert record.entities == ("Universite de Montreal", "the BBC")


def test_a_paragraph_with_no_span_is_ok_with_no_entity_rather_than_failed():
    record = local.record_from_forms("u1", [], model=MODEL, configuration_digest=DIGEST)

    assert record.status == OK
    assert record.entities == ()
    assert record.failure is None


def test_more_forms_than_the_bound_is_a_failure_and_nothing_is_truncated():
    forms = [f"name {index}" for index in range(config.MAX_NODES_PER_TYPE + 1)]

    record = local.record_from_forms("u1", forms, model=MODEL, configuration_digest=DIGEST)

    assert record.status == FAILED
    assert record.failure == local.BOUNDS
    assert record.entities == ()


def test_the_bound_counts_distinct_forms_because_deduplication_happens_first():
    """Fifty distinct names repeated a hundred times is fifty names, not five thousand."""
    forms = [f"name {index}" for index in range(config.MAX_NODES_PER_TYPE)] * 100

    record = local.record_from_forms("u1", forms, model=MODEL, configuration_digest=DIGEST)

    assert record.status == OK
    assert len(record.entities) == config.MAX_NODES_PER_TYPE


def test_one_form_longer_than_the_character_bound_fails_the_whole_record():
    record = local.record_from_forms(
        "u1",
        ["Kaley Cuoco", "x" * (config.MAX_NODE_CHARS + 1)],
        model=MODEL,
        configuration_digest=DIGEST,
    )

    assert record.status == FAILED
    assert record.failure == local.BOUNDS
    assert record.entities == ()


def test_a_record_carries_no_concept_and_no_token_cost():
    record = local.record_from_forms("u1", ["CBS"], model=MODEL, configuration_digest=DIGEST)

    assert record.concepts == ()
    assert record.input_tokens == 0
    assert record.output_tokens == 0
    assert record.model == MODEL
    assert record.prompt_digest == DIGEST


# --- The sentence window (D6.5) ---------------------------------------------------------


def test_a_paragraph_inside_the_window_is_read_in_one_pass_as_indexable_text():
    unit = a_unit("u1", "Kaley Cuoco", "She is an actress.", "She starred in a sitcom.")
    extractor = a_fake_extractor({}, max_tokens=100)

    assert local.windows_for(unit, extractor) == [unit.indexable_text]


def test_an_extractor_with_no_window_never_splits():
    unit = a_unit("u1", "Title", *[f"Sentence {index} is here." for index in range(50)])
    extractor = a_fake_extractor({})

    assert local.windows_for(unit, extractor) == [unit.indexable_text]


def test_a_long_paragraph_is_split_at_its_own_sentence_boundaries_and_covers_the_text():
    unit = a_unit("u1", "Title", "One two three.", "Four five six.", "Seven eight nine.")
    # Four words per window: the title segment is one and each sentence is three.
    extractor = a_fake_extractor({}, max_tokens=4)

    windows = local.windows_for(unit, extractor)

    assert windows == ["Title. One two three.", "Four five six.", "Seven eight nine."]
    assert " ".join(windows) == unit.indexable_text


def test_the_windowing_takes_the_fewest_consecutive_windows_that_fit():
    unit = a_unit("u1", "T", "a b.", "c d.", "e f.")
    extractor = a_fake_extractor({}, max_tokens=3)

    windows = local.windows_for(unit, extractor)

    assert windows == ["T. a b.", "c d.", "e f."]


def test_a_single_sentence_longer_than_the_window_still_becomes_one_window():
    """The rule never splits a sentence; the model truncates it internally instead."""
    unit = a_unit("u1", "T", "a b c d e f g h.", "i j.")
    extractor = a_fake_extractor({}, max_tokens=3)

    windows = local.windows_for(unit, extractor)

    assert windows == ["T.", "a b c d e f g h.", "i j."]


def test_the_forms_of_every_window_are_unioned_into_one_record():
    unit = a_unit("u1", "T", "a b.", "c d.")
    extractor = a_fake_extractor({"T. a b.": ["First"], "c d.": ["Second", "First"]}, max_tokens=3)

    records, seconds = local.run_extraction([unit], extractor, chunk_size=8)

    assert seconds >= 0.0
    assert records[0].entities == ("First", "Second")


# --- The pass -------------------------------------------------------------------------


def test_the_pass_covers_every_unit_in_unit_id_order():
    units = [a_unit("u2", "B", "b."), a_unit("u1", "A", "a."), a_unit("u3", "C", "c.")]
    extractor = a_fake_extractor({units[1].indexable_text: ["A name"]})

    records, _ = local.run_extraction(units, extractor, chunk_size=2)

    assert [record.unit_id for record in records] == ["u1", "u2", "u3"]
    assert records[0].entities == ("A name",)
    assert records[1].entities == ()


def test_the_pass_refuses_an_extractor_that_does_not_answer_every_text():
    def short(texts: Sequence[str]) -> list[list[str]]:
        return [[] for _ in texts][:-1]

    extractor = a_fake_extractor({})
    extractor = local.LocalExtractor(
        extractor_id=extractor.extractor_id,
        model=extractor.model,
        revision=extractor.revision,
        labels=extractor.labels,
        parameters=extractor.parameters,
        library_versions=extractor.library_versions,
        spans=short,
    )

    with pytest.raises(local.LocalExtractionError, match="answered"):
        local.run_extraction([a_unit("u1", "A", "a."), a_unit("u2", "B", "b.")], extractor)


def test_the_pass_refuses_an_empty_pool():
    with pytest.raises(local.LocalExtractionError, match="paragraphs"):
        local.run_extraction([], a_fake_extractor({}))


# --- Identity ---------------------------------------------------------------------------


def test_the_configuration_digest_changes_with_anything_that_decides_what_is_extracted():
    base = a_fake_extractor({})
    other = local.LocalExtractor(
        extractor_id=base.extractor_id,
        model=base.model,
        revision=base.revision,
        labels=("person",),
        parameters=base.parameters,
        library_versions=base.library_versions,
        spans=base.spans,
    )

    assert len(base.configuration_digest) == 16
    assert base.configuration_digest != other.configuration_digest
    assert base.configuration_digest == a_fake_extractor({}).configuration_digest


def test_build_extractor_refuses_a_candidate_this_phase_never_declared():
    with pytest.raises(local.LocalExtractionError, match="two candidates"):
        local.build_extractor("bert-base", "/nowhere")


# --- 2. The artifact round-trips, and refuses a modified archive -------------------------


def an_extraction(tmp_path, entities=("CBS",)):
    units = [a_unit("u1", "A", "a."), a_unit("u2", "B", "b.")]
    extractor = a_fake_extractor({units[0].indexable_text: list(entities)})
    records, seconds = local.run_extraction(units, extractor, chunk_size=4)
    manifest = local.build_manifest(
        records, extractor, seconds=max(seconds, 1e-6), hardware={"device": "cpu"}
    )
    local.write_extraction(records, manifest, tmp_path)
    return records, manifest


def test_an_extraction_round_trips_through_its_archive(tmp_path):
    written, manifest = an_extraction(tmp_path)

    loaded, read_manifest = local.load_extraction(tmp_path, expected_unit_ids=["u1", "u2"])

    assert sorted(loaded) == ["u1", "u2"]
    assert loaded["u1"].entities == written[0].entities
    assert loaded["u1"].concepts == ()
    assert loaded["u1"].prompt_digest == manifest["configuration_digest"]
    assert read_manifest["digest"] == manifest["digest"]


def test_the_digest_is_stable_across_two_writes_of_the_same_records(tmp_path):
    _records, first = an_extraction(tmp_path)
    _records, second = an_extraction(tmp_path / "again")

    assert first["digest"] == second["digest"]


def test_a_modified_archive_raises_rather_than_loading(tmp_path):
    an_extraction(tmp_path)
    archive = tmp_path / local.RECORDS_NAME
    text = gzip.decompress(archive.read_bytes()).decode("utf-8")
    archive.write_bytes(gzip.compress(text.replace("CBS", "NBC").encode("utf-8")))

    with pytest.raises(local.LocalExtractionError, match="digest"):
        local.load_extraction(tmp_path)


def test_an_extraction_that_does_not_cover_the_pool_is_refused(tmp_path):
    an_extraction(tmp_path)

    with pytest.raises(local.LocalExtractionError, match="does not cover"):
        local.load_extraction(tmp_path, expected_unit_ids=["u1", "u2", "u3"])


def test_a_missing_extraction_says_which_stage_to_run(tmp_path):
    with pytest.raises(local.LocalExtractionError, match="cheap-extract"):
        local.load_extraction(tmp_path)


def test_the_line_shape_is_the_phase_5_one_so_one_reader_reads_either_archive():
    record = local.record_from_forms("u1", ["CBS"], model=MODEL, configuration_digest=DIGEST)

    assert local.record_line(record) == phase_5._record_line(record)


def test_the_manifest_carries_the_cost_block_and_the_projection(tmp_path):
    _records, manifest = an_extraction(tmp_path)

    assert manifest["usd"] == 0.0
    assert manifest["n_units"] == 2
    assert manifest["paragraphs_per_second"] > 0.0
    projection = manifest["projection_5m"]
    assert projection["paragraphs"] == config.PROJECTION_PARAGRAPHS
    assert projection["usd"] == 0.0
    assert projection["method"].startswith("linear from measured throughput")
    assert projection["hours"] == pytest.approx(
        config.PROJECTION_PARAGRAPHS / manifest["paragraphs_per_second"] / 3600.0
    )


def test_the_manifest_counts_failures_by_kind(tmp_path):
    units = [a_unit("u1", "A", "a."), a_unit("u2", "B", "b.")]
    extractor = a_fake_extractor(
        {
            units[0].indexable_text: [
                f"name {index}" for index in range(config.MAX_NODES_PER_TYPE + 1)
            ]
        }
    )
    records, seconds = local.run_extraction(units, extractor)

    manifest = local.build_manifest(
        records, extractor, seconds=max(seconds, 1e-6), hardware={"device": "cpu"}
    )

    assert manifest["failures"] == {local.BOUNDS: 1}
    assert manifest["failure_rate"] == 0.5
    assert manifest["units_without_entity"] == 2


def test_a_manifest_whose_digest_does_not_match_its_records_is_refused(tmp_path):
    records, manifest = an_extraction(tmp_path / "source")
    manifest = {**manifest, "digest": "0" * 64}

    with pytest.raises(local.LocalExtractionError, match="digest"):
        local.write_extraction(records, manifest, tmp_path / "elsewhere")


def test_the_archive_is_plain_json_lines_and_holds_no_question_or_split(tmp_path):
    an_extraction(tmp_path)

    text = gzip.decompress((tmp_path / local.RECORDS_NAME).read_bytes()).decode("utf-8")
    lines = [json.loads(line) for line in text.splitlines()]

    assert [line["unit_id"] for line in lines] == ["u1", "u2"]
    for line in lines:
        assert sorted(line) == [
            "concepts",
            "entities",
            "failure",
            "input_tokens",
            "output_tokens",
            "status",
            "unit_id",
        ]


def test_a_projection_needs_a_measured_throughput():
    with pytest.raises(local.LocalExtractionError, match="positive throughput"):
        local.projection(10, 0.0)


def test_the_hardware_block_records_the_machine_rather_than_assuming_one():
    block = local.hardware_block()

    assert block["device"] == "cpu"
    assert block["machine"]
    assert block["python"]
    assert "cpu_count" in block


def test_the_pickle_refusal_names_the_file_it_found(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"not really a pickle")

    with pytest.raises(local.LocalExtractionError, match="pytorch_model.bin"):
        local._refuse_executable_weights(tmp_path)


def test_the_pickle_refusal_passes_a_safetensors_only_directory(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "gliner_config.json").write_text("{}", encoding="utf-8")

    local._refuse_executable_weights(tmp_path)
