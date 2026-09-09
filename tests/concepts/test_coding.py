"""T6: coding the corpus into X.

`X` is the output of the coding against the dictionary - not a similarity matrix
computed afterwards, and not an assignment produced by any other rule. Every
entry is non-negative and the matrix is stored sparse, because a dense
19,366 x 4,096 block is 634 MB for a result that is over 99% zeros.
"""

import numpy as np
from scipy import sparse

from concept_embeddings_rag.concepts.coding import code_corpus
from concept_embeddings_rag.concepts.dictionary import induce_dictionary


def synthetic_vectors(n: int = 120, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = rng.normal(size=(8, dim))
    weights = rng.random(size=(n, 8))
    vectors = (weights @ parts + 0.05 * rng.normal(size=(n, dim))).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def a_dictionary(k: int = 24):
    return induce_dictionary(
        synthetic_vectors(), k=k, seed=42, alpha=0.05, max_iter=5, batch_size=32
    )


def unit_ids_for(n: int) -> list[str]:
    return [f"unit{i:04d}" for i in range(n)]


def test_x_is_sparse_non_negative_and_of_the_right_shape():
    vectors = synthetic_vectors()
    dictionary = a_dictionary()

    matrix = code_corpus(dictionary, vectors, unit_ids_for(len(vectors)), alpha=0.01)

    assert sparse.issparse(matrix.X)
    assert matrix.X.shape == (len(vectors), dictionary.k)
    assert (matrix.X.data >= 0).all()


def test_the_batch_size_does_not_change_the_result():
    """Batching is a memory strategy; it must not be a numerical one."""
    vectors = synthetic_vectors()
    dictionary = a_dictionary()
    ids = unit_ids_for(len(vectors))

    whole = code_corpus(dictionary, vectors, ids, alpha=0.01, batch_size=10_000)
    batched = code_corpus(dictionary, vectors, ids, alpha=0.01, batch_size=7)

    assert np.array_equal(whole.X.toarray(), batched.X.toarray())


def test_rows_line_up_with_the_unit_ids_they_are_given():
    vectors = synthetic_vectors(n=20)
    ids = unit_ids_for(20)

    matrix = code_corpus(a_dictionary(), vectors, ids, alpha=0.01)

    assert matrix.unit_ids == ids
    assert matrix.X.shape[0] == len(ids)


def test_achieved_sparsity_and_reconstruction_error_are_measured():
    vectors = synthetic_vectors()

    matrix = code_corpus(a_dictionary(), vectors, unit_ids_for(len(vectors)), alpha=0.01)

    active = np.diff(matrix.X.tocsr().indptr)
    assert matrix.mean_active_per_unit == active.mean()
    assert 0.0 <= matrix.reconstruction_error <= 1.0


def test_a_looser_penalty_yields_a_denser_matrix():
    vectors = synthetic_vectors()
    dictionary = a_dictionary()
    ids = unit_ids_for(len(vectors))

    loose = code_corpus(dictionary, vectors, ids, alpha=0.001)
    tight = code_corpus(dictionary, vectors, ids, alpha=0.3)

    assert loose.mean_active_per_unit > tight.mean_active_per_unit


def test_the_matrix_carries_the_dictionary_it_was_coded_against():
    dictionary = a_dictionary()
    vectors = synthetic_vectors()

    matrix = code_corpus(dictionary, vectors, unit_ids_for(len(vectors)), alpha=0.01)

    assert matrix.dictionary_key == dictionary.key
    assert matrix.view == "raw"
