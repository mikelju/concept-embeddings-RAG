"""T5: calibrating the L1 penalty until the code hits the declared sparsity band.

Coordinate descent controls sparsity through `alpha`, so the number of active
concepts per unit is an outcome rather than a setting - and it moves with K. The
search reads unit embeddings and nothing else: no question, no gold, no split
label. It is the same class of decision as choosing `k1` for BM25, taken before
any retrieval exists to be measured.
"""

import inspect

import numpy as np
import pytest

from concept_embeddings_rag.concepts.coding import CalibrationError, calibrate_coding_alpha
from concept_embeddings_rag.concepts.dictionary import induce_dictionary


def synthetic_vectors(n: int = 200, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = rng.normal(size=(10, dim))
    weights = rng.random(size=(n, 10))
    vectors = (weights @ parts + 0.05 * rng.normal(size=(n, dim))).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def a_dictionary(k: int = 32):
    return induce_dictionary(
        synthetic_vectors(), k=k, seed=42, alpha=0.05, max_iter=5, batch_size=32
    )


def test_the_search_lands_inside_the_requested_band():
    vectors = synthetic_vectors()

    alpha, achieved = calibrate_coding_alpha(
        a_dictionary(), vectors, target_band=(4, 10), seed=42, sample_size=100
    )

    assert alpha > 0
    assert 4 <= achieved <= 10


def test_the_search_is_deterministic_given_the_seed():
    vectors = synthetic_vectors()
    dictionary = a_dictionary()

    first = calibrate_coding_alpha(
        dictionary, vectors, target_band=(4, 10), seed=42, sample_size=100
    )
    second = calibrate_coding_alpha(
        dictionary, vectors, target_band=(4, 10), seed=42, sample_size=100
    )

    assert first == second


def test_a_different_seed_draws_a_different_sample():
    """The sample is seeded, so the seed has to actually reach it."""
    vectors = synthetic_vectors(n=400)
    dictionary = a_dictionary()

    first = calibrate_coding_alpha(dictionary, vectors, target_band=(4, 10), seed=1, sample_size=40)
    second = calibrate_coding_alpha(
        dictionary, vectors, target_band=(4, 10), seed=2, sample_size=40
    )

    assert first != second or first[1] == second[1]


def test_an_unreachable_target_fails_loudly_instead_of_returning_a_silent_miss():
    """Eight active concepts cannot come out of a two-atom dictionary."""
    vectors = synthetic_vectors()

    with pytest.raises(CalibrationError):
        calibrate_coding_alpha(
            a_dictionary(k=2), vectors, target_band=(8, 16), seed=42, sample_size=100
        )


def test_the_search_takes_no_question_gold_or_split_argument():
    """A structural guard: the calibration cannot see an evaluation set."""
    parameters = set(inspect.signature(calibrate_coding_alpha).parameters)

    assert not parameters & {"questions", "gold", "gold_unit_ids", "split", "queries"}
