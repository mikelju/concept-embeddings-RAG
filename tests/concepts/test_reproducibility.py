"""T4: the same seed must reproduce the same dictionary, bit for bit.

Without this the concept space is not a result: every number Phase 3 and Phase 5
derive from it would belong to one particular run and to nothing else. The test
also has to be able to fail, which is why the different-seed case is here too.
"""

import numpy as np

from concept_embeddings_rag.concepts.dictionary import induce_dictionary


def synthetic_vectors(n: int = 80, dim: int = 12, seed: int = 0) -> np.ndarray:
    """Signed, L2-normalized vectors, like the real embeddings."""
    rng = np.random.default_rng(seed)
    parts = rng.normal(size=(4, dim))
    weights = rng.random(size=(n, 4))
    vectors = (weights @ parts + 0.05 * rng.normal(size=(n, dim))).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def induce(seed: int):
    return induce_dictionary(
        synthetic_vectors(), k=6, seed=seed, alpha=0.05, max_iter=5, batch_size=16
    )


def test_two_inductions_with_the_same_seed_hash_to_the_same_digest():
    assert induce(42).digest() == induce(42).digest()


def test_the_atoms_themselves_are_bit_identical():
    assert np.array_equal(induce(42).atoms, induce(42).atoms)


def test_a_different_seed_produces_a_different_digest():
    """Proves the equality above is a property of the seed, not of the assertion."""
    assert induce(42).digest() != induce(7).digest()
