"""T4: how a question becomes a concept vector.

Decision D4 of the phase plan has three arms and this module is all of them:
projection with truncation, projection without it, and sparse coding at the same
penalty the corpus was coded with. Truncation is part of the operator rather than
a knob swept beside it, which is why the arm names carry it.

The operator's only inputs are a vector and a dictionary. That is the property the
whole phase leans on: the question enters the concept space by geometry, never
through a generated name, a split label or a gold annotation.
"""

import inspect

import numpy as np
import pytest

from concept_embeddings_rag import config
from concept_embeddings_rag.concepts.dictionary import ConceptDictionary
from concept_embeddings_rag.retrieval.conceptual import (
    PROJECTION,
    SPARSE_CODING,
    QueryOperator,
    query_concepts,
    resolve_query_operator,
)


def dictionary_of(atoms: np.ndarray) -> ConceptDictionary:
    atoms = np.asarray(atoms, dtype=np.float32)
    atoms = atoms / np.linalg.norm(atoms, axis=1, keepdims=True)
    return ConceptDictionary(
        atoms=atoms,
        k=atoms.shape[0],
        seed=42,
        sparsity_param=0.05,
        max_iter=3,
        unit_set_hash="101f564fdcca620c",
        model="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c",
    )


def signed_basis(dim: int = 6) -> ConceptDictionary:
    """Every basis direction and its negative: half the atoms must project negative."""
    eye = np.eye(dim, dtype=np.float32)
    return dictionary_of(np.vstack([eye, -eye]))


def graded_query(dim: int = 6) -> np.ndarray:
    """A query whose projections onto `signed_basis` are 6 > 5 > 4 > 3 > 2 > 1 > 0."""
    vector = np.arange(dim, 0, -1, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def random_dictionary(k: int = 32, dim: int = 16) -> ConceptDictionary:
    return dictionary_of(np.random.default_rng(0).normal(size=(k, dim)))


def random_query(dim: int = 16) -> np.ndarray:
    vector = np.random.default_rng(1).normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


# --- The three arms of decision D4 --------------------------------------------


def test_every_declared_arm_resolves_and_none_is_left_unhandled():
    for arm in config.QUERY_OPERATORS:
        resolved = resolve_query_operator(arm)
        assert isinstance(resolved, QueryOperator)
        assert resolved.arm == arm
        assert resolved.operator in (PROJECTION, SPARSE_CODING)


def test_truncation_travels_with_the_arm_name_instead_of_beside_it():
    assert resolve_query_operator("projection_top16").top_m == 16
    assert resolve_query_operator("projection_full").top_m is None
    assert resolve_query_operator("sparse_coding").top_m is None


def test_the_truncating_arm_agrees_with_the_declared_constant():
    """If the two ever disagreed, the arm name in a recorded result would be a lie."""
    truncating = [
        resolve_query_operator(arm)
        for arm in config.QUERY_OPERATORS
        if resolve_query_operator(arm).top_m is not None
    ]
    assert [resolved.top_m for resolved in truncating] == [config.QUERY_TOP_M]


def test_the_arms_map_onto_the_two_operators_the_spec_records():
    assert resolve_query_operator("projection_top16").operator == PROJECTION
    assert resolve_query_operator("projection_full").operator == PROJECTION
    assert resolve_query_operator("sparse_coding").operator == SPARSE_CODING


def test_an_unknown_arm_is_refused_rather_than_silently_treated_as_projection():
    with pytest.raises(ValueError, match="unknown query operator"):
        resolve_query_operator("projection_top")


# --- Projection ---------------------------------------------------------------


def test_projection_is_the_atoms_against_the_query_clipped_at_zero():
    dictionary = signed_basis()
    query = graded_query()

    concepts = query_concepts(query, dictionary, PROJECTION, top_m=None, alpha=None)

    assert np.allclose(concepts, np.maximum(dictionary.atoms @ query, 0.0))
    assert (concepts >= 0.0).all()
    assert concepts.shape == (dictionary.k,)


def test_truncation_keeps_exactly_top_m_non_zeros():
    concepts = query_concepts(graded_query(), signed_basis(), PROJECTION, top_m=3, alpha=None)

    assert int(np.count_nonzero(concepts)) == 3


def test_truncation_keeps_the_largest_coordinates_and_zeroes_every_other():
    concepts = query_concepts(graded_query(), signed_basis(), PROJECTION, top_m=3, alpha=None)

    assert np.count_nonzero(concepts[:3]) == 3
    assert not np.count_nonzero(concepts[3:])


def test_truncation_never_invents_a_non_zero_where_the_projection_was_not_positive():
    """Six of the twelve atoms project negative; asking for nine cannot conjure them."""
    concepts = query_concepts(graded_query(), signed_basis(), PROJECTION, top_m=9, alpha=None)

    assert int(np.count_nonzero(concepts)) == 6
    assert (concepts >= 0.0).all()


def test_a_truncation_outside_the_dictionary_is_refused():
    dictionary = signed_basis()
    for bad in (0, -1, dictionary.k + 1):
        with pytest.raises(ValueError, match="top_m"):
            query_concepts(graded_query(), dictionary, PROJECTION, top_m=bad, alpha=None)


def test_projection_refuses_a_coding_penalty_it_would_have_ignored():
    with pytest.raises(ValueError, match="alpha"):
        query_concepts(graded_query(), signed_basis(), PROJECTION, top_m=None, alpha=0.1)


# --- Sparse coding ------------------------------------------------------------


def test_sparse_coding_output_is_non_negative_and_sparse():
    dictionary = random_dictionary()

    concepts = query_concepts(random_query(), dictionary, SPARSE_CODING, top_m=None, alpha=0.1)

    assert concepts.shape == (dictionary.k,)
    assert (concepts >= 0.0).all()
    assert 0 < int(np.count_nonzero(concepts)) < dictionary.k


def test_sparse_coding_without_a_penalty_is_refused_rather_than_defaulted():
    """The corpus coding alpha lives on the matrix; this function must be told it."""
    with pytest.raises(ValueError, match="alpha"):
        query_concepts(random_query(), random_dictionary(), SPARSE_CODING, top_m=None, alpha=None)


def test_sparse_coding_refuses_a_truncation_that_belongs_to_the_other_arm():
    with pytest.raises(ValueError, match="top_m"):
        query_concepts(random_query(), random_dictionary(), SPARSE_CODING, top_m=16, alpha=0.1)


def test_the_penalty_is_what_decides_how_sparse_the_query_code_is():
    dictionary = random_dictionary()
    query = random_query()

    light = query_concepts(query, dictionary, SPARSE_CODING, top_m=None, alpha=0.01)
    heavy = query_concepts(query, dictionary, SPARSE_CODING, top_m=None, alpha=0.5)

    assert np.count_nonzero(heavy) < np.count_nonzero(light)


# --- What the whole phase leans on --------------------------------------------


def test_repeated_calls_return_a_bit_identical_vector():
    dictionary = random_dictionary()
    query = random_query()

    for operator, top_m, alpha in (
        (PROJECTION, None, None),
        (PROJECTION, 8, None),
        (SPARSE_CODING, None, 0.1),
    ):
        first = query_concepts(query, dictionary, operator, top_m=top_m, alpha=alpha)
        second = query_concepts(query, dictionary, operator, top_m=top_m, alpha=alpha)
        assert first.tobytes() == second.tobytes()


def test_the_operator_takes_a_vector_and_a_dictionary_and_nothing_else():
    """No question text, no split label, no gold: the query enters by geometry."""
    parameters = list(inspect.signature(query_concepts).parameters)

    assert parameters == ["vector", "dictionary", "operator", "top_m", "alpha"]


def test_an_unknown_operator_is_refused():
    with pytest.raises(ValueError, match="unknown query operator"):
        query_concepts(graded_query(), signed_basis(), "cosine", top_m=None, alpha=None)


def test_a_query_of_the_wrong_dimension_is_refused_rather_than_broadcast():
    with pytest.raises(ValueError, match="dimension"):
        query_concepts(np.ones(5, dtype=np.float32), signed_basis(), PROJECTION, None, None)


def test_a_block_of_queries_is_refused_because_the_contract_is_one_question():
    with pytest.raises(ValueError, match="1-D"):
        query_concepts(np.ones((2, 6), dtype=np.float32), signed_basis(), PROJECTION, None, None)
