"""T5 of Phase 5: what an extraction request may hold, and what an answer must be (HU-2, HU-8)."""

import inspect
import json
from pathlib import Path

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.corpus.pool import IndexingUnit, unit_id_for
from concept_embeddings_rag.nodes import extraction
from concept_embeddings_rag.nodes.extraction import (
    FAILED,
    OK,
    ExtractionError,
    RawResponse,
    build_request,
    read_cached,
    record_from_response,
    render_prompt,
    write_cached,
)

MODEL = config.EXTRACTION_MODEL


def a_unit(title: str = "Some Article", text: str = "A sentence. Another one.") -> IndexingUnit:
    sentences = tuple(part.strip() + "." for part in text.split(".") if part.strip())
    return IndexingUnit(unit_id=unit_id_for(title, sentences), title=title, sentences=sentences)


def answer(entities: list, concepts: list) -> RawResponse:
    return RawResponse(
        stop_reason="end_turn",
        text=json.dumps({"entities": entities, "concepts": concepts}),
        input_tokens=120,
        output_tokens=40,
    )


# --- The request ----------------------------------------------------------------------


def test_the_prompt_is_the_template_filled_with_the_indexable_text_and_nothing_else():
    unit = a_unit("Braces {in} a title", "Text with {braces} and 100% symbols.")

    prompt = render_prompt(unit)

    assert prompt == extraction.PROMPT_TEMPLATE.format(paragraph=unit.indexable_text)
    assert prompt.endswith(unit.indexable_text)


def test_the_request_builder_accepts_a_unit_and_nothing_that_could_carry_a_question():
    parameters = inspect.signature(build_request).parameters
    assert list(parameters) == ["unit", "model"]
    assert parameters["model"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_request_holds_one_user_message_and_the_declared_shape():
    unit = a_unit()

    request = build_request(unit)

    assert set(request) == {"model", "messages", "thinking", "output_config", "max_tokens"}
    assert request["model"] == MODEL
    assert request["messages"] == [{"role": "user", "content": render_prompt(unit)}]
    assert request["thinking"] == {"type": "disabled"}
    assert request["max_tokens"] == config.EXTRACTION_MAX_TOKENS
    schema = request["output_config"]["format"]["schema"]
    assert schema["required"] == ["entities", "concepts"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"entities", "concepts"}


def test_the_prompt_template_is_ascii_and_its_digest_is_fixed_by_its_text():
    assert extraction.PROMPT_TEMPLATE.isascii()
    assert extraction.PROMPT_TEMPLATE.count("{paragraph}") == 1
    assert len(extraction.PROMPT_DIGEST) == 16


# --- The answer -----------------------------------------------------------------------


def test_a_valid_answer_becomes_an_ok_record_exactly_as_returned():
    entities = ["Some Person", "  An Organization ", "Some Person"]
    concepts = ["television series", ""]

    record = record_from_response("u1", answer(entities, concepts), model=MODEL)

    assert record.status == OK
    assert record.failure is None
    # Raw strings, untouched: duplicates, spaces and empties are normalization's business.
    assert record.entities == tuple(entities)
    assert record.concepts == tuple(concepts)
    assert (record.input_tokens, record.output_tokens) == (120, 40)
    assert record.prompt_digest == extraction.PROMPT_DIGEST


def failure_of(response: RawResponse) -> str:
    record = record_from_response("u1", response, model=MODEL)
    assert record.status == FAILED
    assert record.entities == () and record.concepts == ()
    assert record.failure is not None
    return record.failure


def test_a_refusal_is_a_failure():
    assert failure_of(RawResponse("refusal", "", 10, 0)) == "refusal"


def test_an_answer_cut_by_max_tokens_is_a_failure_even_if_it_parses():
    text = json.dumps({"entities": [], "concepts": []})
    assert failure_of(RawResponse("max_tokens", text, 10, 1024)) == "max_tokens"


def test_non_json_is_a_failure():
    assert failure_of(RawResponse("end_turn", "not json at all", 10, 5)) == "not_json"


@pytest.mark.parametrize(
    "payload",
    [
        ["entities", "concepts"],
        {"entities": []},
        {"entities": [], "concepts": [], "relations": []},
        {"entities": "Some Person", "concepts": []},
        {"entities": [3], "concepts": []},
    ],
)
def test_an_answer_that_does_not_match_the_schema_is_a_failure(payload):
    response = RawResponse("end_turn", json.dumps(payload), 10, 5)
    assert failure_of(response) == "schema"


def test_too_many_items_is_a_failure_not_a_truncation():
    too_many = [f"thing {index}" for index in range(config.MAX_NODES_PER_TYPE + 1)]
    assert failure_of(answer(too_many, [])) == "bounds"
    at_limit = [f"thing {index}" for index in range(config.MAX_NODES_PER_TYPE)]
    assert record_from_response("u1", answer(at_limit, []), model=MODEL).status == OK


def test_an_over_long_item_is_a_failure_not_a_truncation():
    too_long = "x" * (config.MAX_NODE_CHARS + 1)
    assert failure_of(answer([], [too_long])) == "bounds"
    at_limit = "x" * config.MAX_NODE_CHARS
    assert record_from_response("u1", answer([], [at_limit]), model=MODEL).status == OK


# --- The cache ------------------------------------------------------------------------


def test_a_record_round_trips_through_the_cache_unchanged(tmp_path: Path):
    record = record_from_response("u1", answer(["Some Person"], ["river"]), model=MODEL)

    write_cached(tmp_path, record)

    assert read_cached(tmp_path, "u1", model=MODEL) == record


def test_a_failed_record_round_trips_too(tmp_path: Path):
    record = record_from_response("u2", RawResponse("refusal", "", 10, 0), model=MODEL)

    write_cached(tmp_path, record)

    assert read_cached(tmp_path, "u2", model=MODEL) == record


def test_nothing_is_cached_for_a_unit_never_extracted(tmp_path: Path):
    assert read_cached(tmp_path, "never", model=MODEL) is None


def test_an_entry_for_another_model_or_prompt_is_not_reused(tmp_path: Path):
    record = record_from_response("u1", answer(["Some Person"], []), model=MODEL)
    write_cached(tmp_path, record)

    assert read_cached(tmp_path, "u1", model="claude-haiku-4-5") is None
    assert read_cached(tmp_path, "u1", model=MODEL, prompt_digest="0" * 16) is None


def test_an_entry_edited_to_claim_another_unit_is_refused(tmp_path: Path):
    record = record_from_response("u1", answer(["Some Person"], []), model=MODEL)
    path = write_cached(tmp_path, record)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unit_id"] = "u9"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExtractionError, match="delete"):
        read_cached(tmp_path, "u1", model=MODEL)


def test_an_unreadable_entry_is_refused_rather_than_re_requested_silently(tmp_path: Path):
    record = record_from_response("u1", answer([], []), model=MODEL)
    path = write_cached(tmp_path, record)
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ExtractionError, match="delete"):
        read_cached(tmp_path, "u1", model=MODEL)


def test_a_cache_entry_holds_only_the_declared_fields_and_no_credential(tmp_path: Path):
    record = record_from_response("u1", answer(["Some Person"], ["river"]), model=MODEL)

    payload = json.loads(write_cached(tmp_path, record).read_text(encoding="utf-8"))

    assert set(payload) == {
        "unit_id",
        "model",
        "prompt_digest",
        "status",
        "entities",
        "concepts",
        "failure",
        "input_tokens",
        "output_tokens",
    }
    text = json.dumps(payload).lower()
    for marker in ("sk-ant", "api_key", "x-api-key", "authorization", "bearer"):
        assert marker not in text
