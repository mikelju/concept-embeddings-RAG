"""T14: naming the concepts, for readers only, with every call cached on disk.

The label is interpretability and nothing else. A concept's embedding is its
dictionary atom, so retrieval never reads a name - which is what makes it
acceptable that this one stage is not reproducible. Current models reject
`temperature`, so the same prompt can return a different name on a different day;
the on-disk cache is therefore not an optimization but the thing that fixes the
result, and the artifact is what the report cites.

Every test here runs against a fake client. No network, no key, no spend.
"""

import json
import os

import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.concepts.labeling import (
    PROMPT_TEMPLATE_HASH,
    LabelResponse,
    build_prompt,
    label_concepts,
    load_labels,
    prompt_cache_key,
    save_labels,
)

EVIDENCE = {
    0: [
        ("unit0001", "The Douro flows through northern Portugal."),
        ("unit0002", "Port wine is produced in the Douro valley."),
    ],
    1: [("unit0003", "Ada Lovelace wrote the first published algorithm.")],
    2: [("unit0004", "The 1969 Apollo 11 mission landed on the Moon.")],
}


class FakeClient:
    """Records what it was asked, answers deterministically, spends nothing."""

    def __init__(self, model: str = "fake-model") -> None:
        self.model = model
        self.prompts: list[str] = []

    def label(self, prompt: str) -> LabelResponse:
        self.prompts.append(prompt)
        return LabelResponse(
            name=f"Concept {len(self.prompts)}",
            gloss="What the units shown have in common.",
            input_tokens=1000,
            output_tokens=50,
        )


def test_every_concept_is_labelled_from_its_own_units(tmp_path):
    client = FakeClient()

    labels = label_concepts(
        dictionary_key="abc123", evidence=EVIDENCE, client=client, cache_dir=tmp_path
    )

    assert set(labels.labels) == {"0", "1", "2"}
    assert labels.labels["0"]["evidence_unit_ids"] == ["unit0001", "unit0002"]
    assert labels.labels["1"]["name"]
    assert labels.labels["1"]["gloss"]


def test_the_prompt_shows_the_units_that_activate_that_concept():
    prompt = build_prompt(0, EVIDENCE[0])

    assert "The Douro flows through northern Portugal." in prompt
    assert "Port wine is produced in the Douro valley." in prompt
    assert "Ada Lovelace wrote the first published algorithm." not in prompt


def test_a_second_run_over_an_unchanged_dictionary_makes_zero_calls(tmp_path):
    first = FakeClient()
    label_concepts(dictionary_key="abc123", evidence=EVIDENCE, client=first, cache_dir=tmp_path)

    second = FakeClient()
    again = label_concepts(
        dictionary_key="abc123", evidence=EVIDENCE, client=second, cache_dir=tmp_path
    )

    assert len(first.prompts) == 3
    assert second.prompts == []
    assert again.cost["calls"] == 0
    assert again.cost["cached"] == 3


def test_the_cache_survives_as_the_record_of_what_was_answered(tmp_path):
    client = FakeClient()
    first = label_concepts(
        dictionary_key="abc123", evidence=EVIDENCE, client=client, cache_dir=tmp_path
    )

    again = label_concepts(
        dictionary_key="abc123", evidence=EVIDENCE, client=FakeClient(), cache_dir=tmp_path
    )

    assert again.labels == first.labels


def test_changed_evidence_is_a_different_prompt_and_a_different_key():
    """The units shown are part of the key, so new evidence is a new question."""
    base = prompt_cache_key("m", build_prompt(0, EVIDENCE[0]))
    fewer = prompt_cache_key("m", build_prompt(0, EVIDENCE[0][:1]))
    other_model = prompt_cache_key("other", build_prompt(0, EVIDENCE[0]))

    assert base != fewer
    assert base != other_model


def test_the_cache_key_holds_no_key_material_and_no_environment(monkeypatch):
    prompt = build_prompt(0, EVIDENCE[0])

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-one")
    with_one = prompt_cache_key("m", prompt)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-two")
    with_two = prompt_cache_key("m", prompt)

    assert with_one == with_two
    assert "secret" not in with_one
    assert os.environ["ANTHROPIC_API_KEY"] not in with_one


def test_the_artifact_holds_no_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    labels = label_concepts(
        dictionary_key="abc123", evidence=EVIDENCE, client=FakeClient(), cache_dir=tmp_path
    )

    path = save_labels(labels, tmp_path)
    written = path.read_text(encoding="utf-8")

    assert "sk-ant-secret-value" not in written
    assert "ANTHROPIC_API_KEY" not in written


def test_the_cache_files_hold_no_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    label_concepts(
        dictionary_key="abc123", evidence=EVIDENCE, client=FakeClient(), cache_dir=tmp_path
    )

    for cached in tmp_path.rglob("*.json"):
        assert "sk-ant-secret-value" not in cached.read_text(encoding="utf-8")


def test_the_artifact_records_what_a_later_relabelling_would_be_compared_against(tmp_path):
    labels = label_concepts(
        dictionary_key="abc123",
        evidence=EVIDENCE,
        client=FakeClient(model="fake-model"),
        cache_dir=tmp_path,
        n_units_shown=2,
    )

    assert labels.model == "fake-model"
    assert labels.prompt_hash == PROMPT_TEMPLATE_HASH
    assert labels.n_units_shown == 2
    assert labels.created_at


def test_the_artifact_says_labelling_is_not_reproducible(tmp_path):
    labels = label_concepts(
        dictionary_key="abc123", evidence=EVIDENCE, client=FakeClient(), cache_dir=tmp_path
    )

    path = save_labels(labels, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    note = payload["reproducibility_note"].lower()
    assert "not reproducible" in note
    assert "cache" in note


def test_cost_accumulates_the_real_tokens_at_the_declared_rates(tmp_path):
    labels = label_concepts(
        dictionary_key="abc123", evidence=EVIDENCE, client=FakeClient(), cache_dir=tmp_path
    )

    assert labels.cost["calls"] == 3
    assert labels.cost["input_tokens"] == 3000
    assert labels.cost["output_tokens"] == 150
    expected = 3000 / 1e6 * config.LABELING_INPUT_USD_PER_MTOK
    expected += 150 / 1e6 * config.LABELING_OUTPUT_USD_PER_MTOK
    assert labels.cost["actual_usd"] == pytest.approx(expected)


def test_an_estimate_is_recorded_beside_the_actual_spend(tmp_path):
    """HU-6 asks for both, since this is the first stage that spends money."""
    labels = label_concepts(
        dictionary_key="abc123", evidence=EVIDENCE, client=FakeClient(), cache_dir=tmp_path
    )

    assert labels.cost["estimated_usd"] > 0


def test_the_labels_round_trip_as_an_artifact(tmp_path):
    original = label_concepts(
        dictionary_key="abc123", evidence=EVIDENCE, client=FakeClient(), cache_dir=tmp_path
    )
    save_labels(original, tmp_path)

    loaded = load_labels("abc123", tmp_path)

    assert loaded.labels == original.labels
    assert loaded.model == original.model
    assert loaded.cost == original.cost
    assert loaded.dictionary_key == "abc123"


def test_only_the_requested_number_of_units_is_shown(tmp_path):
    client = FakeClient()

    label_concepts(
        dictionary_key="abc123",
        evidence=EVIDENCE,
        client=client,
        cache_dir=tmp_path,
        n_units_shown=1,
    )

    assert "Port wine is produced in the Douro valley." not in client.prompts[0]


# --- The real client: only its guards, never a call ---------------------------


def test_the_client_refuses_to_be_built_without_a_key(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    assert anthropic  # the optional group is installed, so the guard is reachable

    from concept_embeddings_rag.concepts.labeling import (
        API_KEY_VARIABLE,
        AnthropicLabelingClient,
        MissingAPIKey,
    )

    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)

    with pytest.raises(MissingAPIKey, match=API_KEY_VARIABLE):
        AnthropicLabelingClient()


def test_the_typed_exception_chain_is_most_specific_first():
    """Decision D9: a bad key aborts; a rate limit or a 5xx is retried by the SDK."""
    pytest.importorskip("anthropic")

    from concept_embeddings_rag.concepts.labeling import EXCEPTION_ORDER

    names = [name for name, _ in EXCEPTION_ORDER]
    assert names.index("AuthenticationError") < names.index("APIStatusError")
    assert names.index("RateLimitError") < names.index("APIStatusError")
    assert "APIConnectionError" in names


def test_the_request_disables_thinking_and_asks_for_a_structured_answer():
    """Naming a concept from ten paragraphs needs no reasoning budget."""
    pytest.importorskip("anthropic")

    from concept_embeddings_rag.concepts.labeling import LABEL_SCHEMA, REQUEST_SHAPE

    assert REQUEST_SHAPE["thinking"] == {"type": "disabled"}
    assert REQUEST_SHAPE["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in REQUEST_SHAPE
    assert set(LABEL_SCHEMA["required"]) == {"name", "gloss"}
