"""T6: the embedding backend sits behind an interface.

These tests never download a model: a fake backend proves the contract, and the
real one is only checked for interface compliance. A test suite that pulls
hundreds of megabytes is a test suite nobody runs.
"""

import numpy as np

from concept_embeddings_rag.embeddings.backend import (
    EmbeddingBackend,
    SentenceTransformerBackend,
    l2_normalize,
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
