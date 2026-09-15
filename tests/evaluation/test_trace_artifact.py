"""T8: the trace as an artifact, bound to the run it describes.

A trace is a diagnostic, and a diagnostic read against the wrong run is worse than
no diagnostic at all: it explains a ranking that was never produced, in a document
that looks exactly as authoritative as a correct one. So the file names the
configuration that walked, and the reader refuses anything else.

The rest is the rule Phase 1's audit left behind and this project has applied to
every artifact since: verify on load, do not merely record. The digest is over the
serialization minus itself, so a tampered byte anywhere in the file is caught, and
the write is atomic so an interrupted run leaves the previous file rather than half
of this one.
"""

import json

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.evaluation import traces
from concept_embeddings_rag.evaluation.traces import (
    TraceArtifactError,
    load_traces,
    save_traces,
    traces_path,
)
from concept_embeddings_rag.retrieval.diffusion import (
    CAP,
    THRESHOLD,
    DiffusionError,
    ExpansionTrace,
    IterationTrace,
)

DIGEST = "a1b2c3d4e5f60718"


def a_row(index: int = 1, stopped: bool = True) -> IterationTrace:
    return IterationTrace(
        index=index,
        concepts=((7, 0.61), (12, 0.22)),
        units=(("5a7bbc50_p3", 0.4), ("5a7bbc50_p7", 0.1)),
        new_mass=0.0301,
        stopped=stopped,
    )


def a_trace(qid: str = "q1", *, arm: str = "dense", digest: str = DIGEST) -> ExpansionTrace:
    return ExpansionTrace(
        qid=qid,
        seed_arm=arm,
        config_digest=digest,
        iterations=(a_row(1, stopped=False), a_row(2, stopped=True)),
        stop_reason=THRESHOLD,
    )


# --- The round trip -----------------------------------------------------------


def test_the_traces_survive_the_round_trip_with_every_field_intact(tmp_path):
    written = [a_trace("q1"), a_trace("q2")]
    save_traces(written, tmp_path, seed_arm="dense", config_digest=DIGEST)

    read = load_traces(tmp_path, seed_arm="dense", config_digest=DIGEST)

    assert read == written


def test_one_file_per_arm_and_digest(tmp_path):
    save_traces([a_trace()], tmp_path, seed_arm="dense", config_digest=DIGEST)
    save_traces([a_trace(arm="conceptual")], tmp_path, seed_arm="conceptual", config_digest=DIGEST)
    other = "ffffffffffffffff"
    save_traces([a_trace(digest=other)], tmp_path, seed_arm="dense", config_digest=other)

    assert len(sorted(tmp_path.glob("*.json"))) == 3
    assert traces_path(tmp_path, seed_arm="dense", config_digest=DIGEST).exists()


def test_the_artifact_is_plain_json_a_human_can_read(tmp_path):
    save_traces([a_trace()], tmp_path, seed_arm="dense", config_digest=DIGEST)
    payload = json.loads(
        traces_path(tmp_path, seed_arm="dense", config_digest=DIGEST).read_text(encoding="utf-8")
    )

    assert payload["seed_arm"] == "dense"
    assert payload["config_digest"] == DIGEST
    assert payload["n_traces"] == 1
    assert payload["traces"][0]["qid"] == "q1"


def test_the_written_artifact_lands_whole_leaving_no_temporary_behind(tmp_path):
    save_traces([a_trace()], tmp_path, seed_arm="dense", config_digest=DIGEST)

    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".json"]


def test_a_rerun_of_the_same_configuration_replaces_its_own_file(tmp_path):
    """Unlike a selection, a trace is a rerunnable diagnostic: nothing was decided by it."""
    save_traces([a_trace("q1")], tmp_path, seed_arm="dense", config_digest=DIGEST)
    save_traces([a_trace("q1"), a_trace("q2")], tmp_path, seed_arm="dense", config_digest=DIGEST)

    assert len(load_traces(tmp_path, seed_arm="dense", config_digest=DIGEST)) == 2


# --- Bound to the run it describes --------------------------------------------


def test_a_trace_file_read_against_another_configuration_is_refused(tmp_path):
    save_traces([a_trace()], tmp_path, seed_arm="dense", config_digest=DIGEST)
    path = traces_path(tmp_path, seed_arm="dense", config_digest=DIGEST)
    path.rename(traces_path(tmp_path, seed_arm="dense", config_digest="0" * 16))

    with pytest.raises(TraceArtifactError, match="digest"):
        load_traces(tmp_path, seed_arm="dense", config_digest="0" * 16)


def test_a_trace_whose_own_digest_differs_from_its_file_is_refused(tmp_path):
    """The inner check: one trace of the file claims a run the file does not."""
    mixed = [a_trace("q1"), a_trace("q2", digest="0" * 16)]

    with pytest.raises(TraceArtifactError, match="digest"):
        save_traces(mixed, tmp_path, seed_arm="dense", config_digest=DIGEST)


def test_a_trace_of_another_arm_cannot_be_filed_under_this_one(tmp_path):
    with pytest.raises(TraceArtifactError, match="arm"):
        save_traces([a_trace(arm="conceptual")], tmp_path, seed_arm="dense", config_digest=DIGEST)


def test_reading_a_file_that_was_never_written_names_what_is_missing(tmp_path):
    with pytest.raises(TraceArtifactError, match="no trace"):
        load_traces(tmp_path, seed_arm="dense", config_digest=DIGEST)


def test_two_traces_of_the_same_question_are_refused(tmp_path):
    with pytest.raises(TraceArtifactError, match="twice"):
        save_traces(
            [a_trace("q1"), a_trace("q1")], tmp_path, seed_arm="dense", config_digest=DIGEST
        )


def test_a_file_holding_no_trace_at_all_is_refused(tmp_path):
    with pytest.raises(TraceArtifactError, match="empty"):
        save_traces([], tmp_path, seed_arm="dense", config_digest=DIGEST)


# --- Verified on load ---------------------------------------------------------


def test_a_tampered_byte_is_caught_by_the_digest(tmp_path):
    save_traces([a_trace()], tmp_path, seed_arm="dense", config_digest=DIGEST)
    path = traces_path(tmp_path, seed_arm="dense", config_digest=DIGEST)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["traces"][0]["iterations"][0]["new_mass"] = 0.9999
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(TraceArtifactError, match="modified"):
        load_traces(tmp_path, seed_arm="dense", config_digest=DIGEST)


def test_the_digest_covers_the_count_as_well_as_the_traces(tmp_path):
    save_traces([a_trace()], tmp_path, seed_arm="dense", config_digest=DIGEST)
    path = traces_path(tmp_path, seed_arm="dense", config_digest=DIGEST)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["n_traces"] = 99
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(TraceArtifactError, match="modified"):
        load_traces(tmp_path, seed_arm="dense", config_digest=DIGEST)


# --- Concept ids, never labels ------------------------------------------------


def test_a_label_smuggled_into_the_artifact_is_refused_on_load(tmp_path):
    """HU-5's last criterion, enforced at the reader as well as at the writer.

    The digest is recomputed after the edit on purpose: a tampered file already
    fails on its hash, which would let this pass without the label check ever
    running. What is asserted here is the check itself, against an artifact whose
    digest verifies perfectly.
    """
    save_traces([a_trace()], tmp_path, seed_arm="dense", config_digest=DIGEST)
    path = traces_path(tmp_path, seed_arm="dense", config_digest=DIGEST)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["traces"][0]["iterations"][0]["concepts"][0][0] = "Florida baseball"
    payload["digest"] = traces._payload_digest(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(TraceArtifactError, match="id"):
        load_traces(tmp_path, seed_arm="dense", config_digest=DIGEST)


def test_no_concept_reaches_the_file_as_anything_but_a_number(tmp_path):
    save_traces([a_trace()], tmp_path, seed_arm="dense", config_digest=DIGEST)
    payload = json.loads(
        traces_path(tmp_path, seed_arm="dense", config_digest=DIGEST).read_text(encoding="utf-8")
    )

    for trace in payload["traces"]:
        for row in trace["iterations"]:
            for concept, mass in row["concepts"]:
                assert isinstance(concept, int)
                assert isinstance(mass, float)


def test_the_traces_directory_is_the_one_config_declares():
    assert config.TRACES_DIR.name == "traces"


def test_a_stop_reason_this_phase_does_not_have_is_refused_on_load(tmp_path):
    """Rehashed after the edit, so it is the check that fires rather than the digest."""
    save_traces([a_trace()], tmp_path, seed_arm="dense", config_digest=DIGEST)
    path = traces_path(tmp_path, seed_arm="dense", config_digest=DIGEST)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["traces"][0]["stop_reason"] = "ran out of ideas"
    payload["digest"] = traces._payload_digest(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(DiffusionError, match="stop reason"):
        load_traces(tmp_path, seed_arm="dense", config_digest=DIGEST)


def test_a_trace_that_stopped_at_the_cap_round_trips_too(tmp_path):
    capped = ExpansionTrace(
        qid="q1",
        seed_arm="dense",
        config_digest=DIGEST,
        iterations=tuple(
            a_row(i, stopped=i == config.MAX_ITERATIONS)
            for i in range(1, config.MAX_ITERATIONS + 1)
        ),
        stop_reason=CAP,
    )
    save_traces([capped], tmp_path, seed_arm="dense", config_digest=DIGEST)

    assert load_traces(tmp_path, seed_arm="dense", config_digest=DIGEST) == [capped]
