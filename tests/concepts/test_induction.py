"""T3: inducing the dictionary by non-negative sparse coding.

The dimensions of the concept space have to come from the corpus, and the codes
have to be non-negative and multi-active: that is what gives co-activation, and
co-activation is the whole substrate Phase 4 diffuses over. What is *not*
constrained is the dictionary itself - see decision D1 of the phase plan: half
the energy of these embeddings is negative, and a non-negative dictionary drives
the space straight back into the one-hot regime that disqualified clustering.

Runs on small synthetic vectors: no model download, no cached corpus.
"""

import numpy as np
import pytest

from concept_embeddings_rag.concepts.dictionary import induce_dictionary


def synthetic_vectors(n: int = 80, dim: int = 12, seed: int = 0) -> np.ndarray:
    """Signed, L2-normalized vectors, like the real embeddings."""
    rng = np.random.default_rng(seed)
    parts = rng.normal(size=(4, dim))
    weights = rng.random(size=(n, 4))
    vectors = (weights @ parts + 0.05 * rng.normal(size=(n, dim))).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_atoms_come_back_unit_norm_float32_and_of_the_requested_shape():
    vectors = synthetic_vectors()

    dictionary = induce_dictionary(vectors, k=6, seed=42, alpha=0.05, max_iter=3, batch_size=16)

    assert dictionary.atoms.shape == (6, vectors.shape[1])
    assert dictionary.atoms.dtype == np.float32
    assert np.allclose(np.linalg.norm(dictionary.atoms, axis=1), 1.0, atol=1e-5)


def test_the_dictionary_records_the_configuration_that_produced_it():
    dictionary = induce_dictionary(
        synthetic_vectors(),
        k=6,
        seed=42,
        alpha=0.05,
        max_iter=3,
        batch_size=16,
        unit_set_hash="pool-hash",
        model="fake-model",
        revision="fake-rev",
    )

    assert dictionary.k == 6
    assert dictionary.seed == 42
    assert dictionary.sparsity_param == 0.05
    assert dictionary.max_iter == 3
    assert dictionary.unit_set_hash == "pool-hash"
    assert dictionary.model == "fake-model"
    assert dictionary.revision == "fake-rev"
    assert dictionary.code_version


def test_the_code_is_non_negative_and_not_one_hot():
    """`positive_code=True` is the constraint the hypothesis actually needs."""
    vectors = synthetic_vectors()

    dictionary = induce_dictionary(vectors, k=6, seed=42, alpha=0.01, max_iter=5, batch_size=16)
    code = dictionary.encode(vectors)

    assert (code >= 0).all()
    assert code.max() > 0
    # More than one atom per unit on average: a one-hot code has no co-activation
    # and nothing for expansion to travel across.
    assert (code > 0).sum(axis=1).mean() > 1.0


def test_the_dictionary_is_not_constrained_to_be_non_negative():
    """Decision D1: constraining it too would floor the reconstruction error."""
    dictionary = induce_dictionary(
        synthetic_vectors(), k=6, seed=42, alpha=0.05, max_iter=3, batch_size=16
    )

    assert (dictionary.atoms < 0).any()


def test_a_larger_alpha_yields_a_sparser_code():
    vectors = synthetic_vectors()

    loose = induce_dictionary(vectors, k=8, seed=42, alpha=0.005, max_iter=5, batch_size=16)
    tight = induce_dictionary(vectors, k=8, seed=42, alpha=0.5, max_iter=5, batch_size=16)

    assert (tight.encode(vectors) > 0).sum() < (loose.encode(vectors) > 0).sum()


def test_the_estimator_object_is_never_kept_on_the_artifact():
    """Only the learned array travels; a fitted estimator would mean pickle."""
    dictionary = induce_dictionary(
        synthetic_vectors(), k=6, seed=42, alpha=0.05, max_iter=3, batch_size=16
    )

    for value in vars(dictionary).values():
        assert not hasattr(value, "fit"), "a fitted estimator is stored on the artifact"


def test_requesting_more_atoms_than_samples_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="samples"):
        induce_dictionary(synthetic_vectors(n=5), k=64, seed=42, alpha=0.05, max_iter=2)
