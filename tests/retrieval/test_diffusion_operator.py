"""T2: one round of diffusion, checked against arithmetic rather than against itself.

HU-1 asks for the propagation to be two sparse matrix-vector products and for the
unit x unit matrix never to be materialized. Both are properties of the code, so
both are asserted here: the first against a reference computed with plain dense
numpy on a hand-built matrix, the second by watching what the operator multiplies.

The reference is written out longhand on purpose. Calling the module under test to
produce the number the module under test is checked against proves only that it is
self-consistent, which is exactly what decision D2 says it must not rest on.
"""

import numpy as np
import pytest
from scipy import sparse

from concept_embeddings_rag import config
from concept_embeddings_rag.retrieval.diffusion import (
    NONE,
    STOCHASTIC,
    SYMMETRIC,
    DiffusionError,
    DiffusionOperator,
)

# Three units, two concepts, chosen so that every unit and every concept has a
# different degree: a bug that confuses rows with columns cannot survive it.
DENSE_X = np.array(
    [
        [1.0, 0.0],
        [2.0, 1.0],
        [0.0, 3.0],
    ],
    dtype=np.float64,
)


def _matrix() -> sparse.csr_matrix:
    return sparse.csr_matrix(DENSE_X)


def _l1(vector: np.ndarray) -> np.ndarray:
    return vector / vector.sum()


def _reference(mass: np.ndarray, arm: str, weights: np.ndarray) -> np.ndarray:
    """One round, written out in dense numpy, independently of the module under test."""
    degree_units = DENSE_X.sum(axis=1)
    degree_concepts = DENSE_X.sum(axis=0)

    if arm == NONE:
        left = DENSE_X
        right = DENSE_X
    elif arm == SYMMETRIC:
        scaled = DENSE_X / np.sqrt(np.outer(degree_units, degree_concepts))
        left = scaled
        right = scaled
    elif arm == STOCHASTIC:
        left = DENSE_X / degree_units[:, None]
        right = DENSE_X / degree_concepts[None, :]
    else:
        raise AssertionError(f"the test does not know arm {arm!r}")

    concepts = left.T @ mass
    units = right @ (weights * concepts)
    return _l1(units)


@pytest.mark.parametrize("arm", config.NORMALIZATION_ARMS)
def test_one_round_matches_a_hand_computed_reference(arm):
    mass = np.array([0.5, 0.5, 0.0])
    weights = np.ones(2)

    operator = DiffusionOperator(_matrix(), normalization=arm)

    assert operator.propagate(mass) == pytest.approx(_reference(mass, arm, weights))


@pytest.mark.parametrize("arm", config.NORMALIZATION_ARMS)
def test_the_damping_is_applied_to_the_concept_vector_in_the_middle(arm):
    """Decision D2: the rarity weight multiplies the concepts, not the units."""
    mass = np.array([0.2, 0.3, 0.5])
    weights = np.array([4.0, 0.5])

    operator = DiffusionOperator(_matrix(), normalization=arm, concept_weights=weights)

    assert operator.propagate(mass) == pytest.approx(_reference(mass, arm, weights))


def test_the_concept_vector_is_exposed_so_a_trace_need_not_recompute_it():
    mass = np.array([0.5, 0.5, 0.0])
    weights = np.array([4.0, 0.5])
    operator = DiffusionOperator(_matrix(), normalization=NONE, concept_weights=weights)

    concepts = operator.to_concepts(mass)

    assert concepts == pytest.approx(weights * (DENSE_X.T @ mass))


def test_the_unit_by_unit_matrix_is_never_formed():
    """The whole point of D2: two sparse products, never the 375-million-entry product."""
    operator = DiffusionOperator(_matrix(), normalization=NONE)
    n_units = DENSE_X.shape[0]

    for operand in (operator.left, operator.right):
        assert sparse.issparse(operand)
        assert operand.shape != (n_units, n_units)
        assert operand.shape == DENSE_X.shape


def test_the_stochastic_arm_conserves_mass_before_it_is_renormalized():
    """A real random walk moves mass, it does not create or destroy it.

    Undamped, because the damping deliberately breaks conservation: a weight of 4.0
    on one concept multiplies the mass that passes through it. That is the reason
    the renormalization of D3 exists, and the reason this test declares its scope.
    """
    mass = np.array([0.2, 0.3, 0.5])
    operator = DiffusionOperator(_matrix(), normalization=STOCHASTIC)

    assert operator.propagate_raw(mass).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("arm", config.NORMALIZATION_ARMS)
def test_every_arm_returns_a_non_negative_distribution(arm):
    mass = np.array([0.2, 0.3, 0.5])
    operator = DiffusionOperator(_matrix(), normalization=arm)

    propagated = operator.propagate(mass)

    assert (propagated >= 0.0).all()
    assert propagated.sum() == pytest.approx(1.0)


def test_a_unit_with_no_concepts_raises_rather_than_dividing_by_zero():
    """Phase 2 measured zero orphan units at every K, so this should hold, not be handled."""
    orphaned = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 3.0]]))

    with pytest.raises(DiffusionError, match="unit"):
        DiffusionOperator(orphaned, normalization=STOCHASTIC)


def test_a_concept_no_unit_activates_raises_rather_than_dividing_by_zero():
    dead = sparse.csr_matrix(np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))

    with pytest.raises(DiffusionError, match="concept"):
        DiffusionOperator(dead, normalization=STOCHASTIC)


def test_an_unknown_normalization_arm_is_refused():
    with pytest.raises(DiffusionError, match="normalization"):
        DiffusionOperator(_matrix(), normalization="row_normalized")


def test_concept_weights_must_have_one_weight_per_concept():
    with pytest.raises(DiffusionError, match="weights"):
        DiffusionOperator(_matrix(), normalization=NONE, concept_weights=np.ones(5))


def test_the_mass_must_be_one_value_per_unit():
    operator = DiffusionOperator(_matrix(), normalization=NONE)

    with pytest.raises(DiffusionError, match="mass"):
        operator.propagate(np.array([0.5, 0.5]))


def test_propagation_is_deterministic_to_the_bit():
    mass = np.array([0.2, 0.3, 0.5])
    operator = DiffusionOperator(_matrix(), normalization=SYMMETRIC)

    first = operator.propagate(mass)
    second = operator.propagate(mass)

    assert np.array_equal(first, second)
