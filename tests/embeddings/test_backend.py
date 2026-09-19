"""T6: the embedding backend sits behind an interface.

These tests never download a model: a fake backend proves the contract, and the
real one is only checked for interface compliance. A test suite that pulls
hundreds of megabytes is a test suite nobody runs.
"""

import hashlib
import sys
import types

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.embeddings.backend import (
    BackendError,
    EmbeddingBackend,
    QueryPromptError,
    SentenceTransformerBackend,
    check_query_prompt,
    l2_normalize,
    weights_sha256,
)


class FakeBackend:
    """Deterministic stand-in: vectors derived from the text length."""

    name = "fake"
    revision = "v0"
    dim = 4

    def encode(self, texts):
        return np.array([[len(t), 1.0, 0.0, 0.0] for t in texts], dtype=np.float32)


def test_a_fake_backend_satisfies_the_interface():
    backend: EmbeddingBackend = FakeBackend()
    vectors = backend.encode(["one", "two words"])
    assert vectors.shape == (2, 4)


def test_the_real_backend_declares_the_interface_without_loading_a_model():
    backend = SentenceTransformerBackend()
    assert backend.name
    assert backend.revision
    assert backend.dim == 384
    assert hasattr(backend, "encode")


def test_l2_normalize_gives_unit_rows():
    vectors = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)

    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0)


def test_l2_normalize_leaves_zero_rows_alone_instead_of_dividing_by_zero():
    vectors = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)

    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized[0], 0.0)


# --- Phase 8 (S2): the query prompt, read from the model and never assumed ---
#
# Restriction R2: `Qwen/Qwen3-Embedding-0.6B` is asymmetric, so queries carry an
# instruction prefix and documents do not. The prompt is a declared configuration item:
# the backend reads `model.prompts["query"]` at load time and aborts on any inexact
# match **before a single vector exists**, because a silently different prompt would
# change every query vector while leaving the artifact's provenance looking correct.
#
# No model is downloaded here: `sentence_transformers` is replaced in `sys.modules`, so
# the seam under test is the one the real run goes through.


class FakeModel:
    """A stand-in for a loaded SentenceTransformer, recording what it was asked."""

    def __init__(self, prompts: dict | None = None, dim: int = 4, max_seq_length: int = 32768):
        self.prompts = prompts if prompts is not None else {}
        self.max_seq_length = max_seq_length
        self.calls: list[dict] = []
        self._dim = dim

    def encode(self, texts, **arguments):
        self.calls.append({"texts": list(texts), **arguments})
        return np.array([[1.0] * self._dim for _ in texts], dtype=np.float32)


def install_fake_model(monkeypatch, model: FakeModel) -> list[dict]:
    """Make `SentenceTransformer(...)` return `model`, and record how it was constructed."""
    constructed: list[dict] = []

    def build(name, revision=None, **arguments):
        constructed.append({"name": name, "revision": revision, **arguments})
        return model

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = build
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return constructed


def a_prompted_backend(prompt: str = config.PHASE_8_QUERY_PROMPT) -> SentenceTransformerBackend:
    return SentenceTransformerBackend(
        name=config.PHASE_8_DENSE_MODEL,
        revision=config.PHASE_8_DENSE_REVISION,
        dim=config.PHASE_8_DENSE_DIM,
        query_prompt=prompt,
    )


def test_a_backend_declares_no_query_prompt_by_default():
    assert SentenceTransformerBackend().query_prompt == ""
    assert SentenceTransformerBackend().resolved_query_prompt == ""


@pytest.mark.parametrize(
    "prompts",
    [
        {},
        {"query": "Instruct: something else\nQuery:"},
        {"query": config.PHASE_8_QUERY_PROMPT + " "},
        {"document": config.PHASE_8_QUERY_PROMPT},
        None,
    ],
)
def test_a_prompt_the_model_does_not_carry_aborts_before_any_vector_exists(monkeypatch, prompts):
    model = FakeModel(prompts=prompts)
    install_fake_model(monkeypatch, model)
    backend = a_prompted_backend()

    with pytest.raises(QueryPromptError):
        backend.encode(["a query"])

    assert model.calls == []


def test_the_matching_prompt_is_recorded_and_queries_are_encoded_through_it(monkeypatch):
    model = FakeModel(prompts={"query": config.PHASE_8_QUERY_PROMPT, "document": ""})
    install_fake_model(monkeypatch, model)
    backend = a_prompted_backend()

    backend.encode(["a query"])

    assert model.calls[0]["prompt_name"] == "query"
    assert backend.resolved_query_prompt == config.PHASE_8_QUERY_PROMPT


def test_documents_are_encoded_with_no_prompt_at_all(monkeypatch):
    model = FakeModel(prompts={"query": config.PHASE_8_QUERY_PROMPT, "document": ""})
    install_fake_model(monkeypatch, model)
    backend = SentenceTransformerBackend(
        name=config.PHASE_8_DENSE_MODEL,
        revision=config.PHASE_8_DENSE_REVISION,
        dim=config.PHASE_8_DENSE_DIM,
    )

    backend.encode(["a paragraph"])

    assert "prompt_name" not in model.calls[0]
    assert backend.resolved_query_prompt == ""


def test_the_resolved_prompt_is_the_string_read_from_the_model(monkeypatch):
    """Provenance records what was used, not what was expected - they are equal by proof."""
    model = FakeModel(prompts={"query": config.PHASE_8_QUERY_PROMPT})
    install_fake_model(monkeypatch, model)
    backend = a_prompted_backend()

    backend.encode(["a query"])

    assert backend.resolved_query_prompt is model.prompts["query"]


def test_the_prompt_check_is_exact_string_equality():
    prompt = config.PHASE_8_QUERY_PROMPT

    assert check_query_prompt({"query": prompt}, prompt) == prompt
    assert check_query_prompt({}, "") == ""
    for wrong in (prompt + " ", prompt.replace("\n", " "), prompt.lower(), " " + prompt):
        with pytest.raises(QueryPromptError):
            check_query_prompt({"query": wrong}, prompt)


def test_no_call_path_ever_asks_for_remote_code(monkeypatch):
    """Security: the approved model needs no `trust_remote_code`, and none is passed."""
    model = FakeModel(prompts={"query": config.PHASE_8_QUERY_PROMPT})
    constructed = install_fake_model(monkeypatch, model)

    a_prompted_backend().encode(["a query"])

    assert constructed[0]["revision"] == config.PHASE_8_DENSE_REVISION
    assert "trust_remote_code" not in constructed[0]


def test_the_effective_max_sequence_length_is_read_from_the_loaded_model(monkeypatch):
    """Decision 6: the pinned revision's own files disagree, so nothing is taken on trust."""
    model = FakeModel(prompts={"query": config.PHASE_8_QUERY_PROMPT}, max_seq_length=1234)
    install_fake_model(monkeypatch, model)

    assert a_prompted_backend().max_seq_length() == 1234


def test_the_weight_digest_is_taken_over_the_bytes_that_were_actually_read(tmp_path):
    """R7 / SEC-031: a revision names a commit, only a digest binds the weights."""
    (tmp_path / config.PHASE_8_WEIGHTS_FILE).write_bytes(b"not really a tensor")

    digest = weights_sha256(tmp_path)

    assert digest == hashlib.sha256(b"not really a tensor").hexdigest()


def test_a_snapshot_without_the_declared_weight_file_is_refused(tmp_path):
    """What is never read cannot be recorded: the digest is not invented from absence."""
    (tmp_path / "pytorch_model.bin").write_bytes(b"a pickle archive")

    with pytest.raises(BackendError):
        weights_sha256(tmp_path)
